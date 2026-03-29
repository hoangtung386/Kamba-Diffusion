"""VAE training script with GAN, EMA, and mixed precision support.

Usage:
    python scripts/train_vae.py --config configs/vae_train.yaml
    python scripts/train_vae.py --config configs/vae_train.yaml --data_root /path/to/data
"""

import argparse
import logging
import os
import sys

import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kamba.data.imagenet import ImageNetDataset
from kamba.models.vae import VAE
from kamba.models.vae.loss import VAELoss
from kamba.utils.checkpoint import save_checkpoint
from kamba.utils.ema import EMA
from kamba.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def train_epoch(
    model: VAE,
    dataloader: DataLoader,
    criterion: VAELoss,
    optimizer_vae: torch.optim.Optimizer,
    optimizer_disc: torch.optim.Optimizer | None,
    scaler: torch.amp.GradScaler,
    ema: EMA | None,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    log_every: int = 100,
) -> dict[str, float]:
    """Train for one epoch with optional GAN discriminator."""
    model.train()

    stats = {
        "vae_loss": 0.0,
        "disc_loss": 0.0,
        "recon_loss": 0.0,
        "kl_loss": 0.0,
        "perc_loss": 0.0,
    }

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)

        # === Update Discriminator ===
        if optimizer_disc is not None:
            optimizer_disc.zero_grad()

            with torch.amp.autocast("cuda", enabled=use_amp):
                with torch.no_grad():
                    recon, mean, logvar = model(images)
                disc_loss, disc_log = criterion(
                    recon, images, mean, logvar, optimizer_idx=1
                )

            scaler.scale(disc_loss).backward()
            scaler.unscale_(optimizer_disc)
            torch.nn.utils.clip_grad_norm_(
                criterion.discriminator.parameters(), max_norm=grad_clip_norm
            )
            scaler.step(optimizer_disc)

        # === Update VAE ===
        if batch_idx % gradient_accumulation_steps == 0:
            optimizer_vae.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            recon, mean, logvar = model(images)
            vae_loss, vae_log = criterion(
                recon, images, mean, logvar, optimizer_idx=0
            )
            vae_loss = vae_loss / gradient_accumulation_steps

        scaler.scale(vae_loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer_vae)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=grad_clip_norm
            )
            scaler.step(optimizer_vae)
            scaler.update()

            if ema is not None:
                ema.update(model)

        # Logging
        stats["vae_loss"] += vae_log["total_loss"]
        stats["disc_loss"] += disc_log.get("gan/d_loss", 0) if optimizer_disc else 0
        stats["recon_loss"] += vae_log["recon_loss"]
        stats["kl_loss"] += vae_log["kl_loss"]
        stats["perc_loss"] += vae_log.get("perceptual_loss", 0)

        if batch_idx % log_every == 0:
            pbar.set_postfix({
                "vae": f"{vae_log['total_loss']:.4f}",
                "recon": f"{vae_log['recon_loss']:.4f}",
                "kl": f"{vae_log['kl_loss']:.6f}",
            })

    n = len(dataloader)
    return {k: v / n for k, v in stats.items()}


@torch.no_grad()
def validate(
    model: VAE,
    dataloader: DataLoader,
    criterion: VAELoss,
    device: torch.device,
    ema: EMA | None = None,
) -> dict[str, float]:
    """Validate model with optional EMA weights."""
    if ema is not None:
        ema.apply_shadow(model)

    model.eval()
    stats = {"total_loss": 0.0, "recon_loss": 0.0, "kl_loss": 0.0}

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        recon, mean, logvar = model(images, sample=False)
        loss, loss_dict = criterion(recon, images, mean, logvar, optimizer_idx=0)

        stats["total_loss"] += loss_dict["total_loss"]
        stats["recon_loss"] += loss_dict["recon_loss"]
        stats["kl_loss"] += loss_dict["kl_loss"]

    if ema is not None:
        ema.restore(model)

    n = len(dataloader)
    return {k: v / n for k, v in stats.items()}


@torch.no_grad()
def save_samples(
    model: VAE,
    val_loader: DataLoader,
    save_dir: str,
    epoch: int,
    device: torch.device,
    num_samples: int = 8,
) -> None:
    """Save reconstruction comparison images."""
    from torchvision.utils import make_grid, save_image

    model.eval()
    batch = next(iter(val_loader))
    images = batch["image"][:num_samples].to(device)
    recon, _, _ = model(images, sample=False)

    comparison = torch.cat([images, recon])
    grid = make_grid(comparison, nrow=num_samples, normalize=True)
    save_path = os.path.join(save_dir, f"recon_epoch{epoch:03d}.png")
    save_image(grid, save_path)
    logger.info("Saved samples to %s", save_path)


def main(config: dict) -> None:
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # Directories
    exp_dir = os.path.join(
        config["training"].get("output_dir", "experiments"),
        config["training"].get("exp_name", "vae_default"),
    )
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Dataset
    mc = config["model"]
    tc = config["training"]
    dc = config.get("data", {})
    lc = config.get("logging", tc)

    image_size = mc.get("image_size", 256)
    data_root = dc.get("data_root", tc.get("data_root", ""))

    train_dataset = ImageNetDataset(
        data_root=data_root, split="train", image_size=image_size, center_crop=False
    )
    val_dataset = ImageNetDataset(
        data_root=data_root, split="val", image_size=image_size, center_crop=True
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc.get("num_workers", 4),
        pin_memory=True,
        persistent_workers=tc.get("num_workers", 4) > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc.get("num_workers", 4),
        pin_memory=True,
    )
    logger.info("Train: %d images, Val: %d images", len(train_dataset), len(val_dataset))

    # Model
    model = VAE(
        in_channels=mc.get("in_channels", 3),
        latent_channels=mc.get("latent_channels", 4),
        hidden_dims=tuple(mc.get("hidden_dims", [128, 256, 512, 512])),
        image_size=image_size,
        num_res_blocks=mc.get("num_res_blocks", 2),
        dropout=mc.get("dropout", 0.0),
        use_kan_decoder=mc.get("use_kan_decoder", True),
    ).to(device)

    # Loss
    loss_cfg = config.get("loss", {})
    use_gan = loss_cfg.get("use_gan", True)
    criterion = VAELoss(
        kl_weight=loss_cfg.get("kl_weight", 1e-6),
        perceptual_weight=loss_cfg.get("perceptual_weight", 1.0),
        use_perceptual=loss_cfg.get("use_perceptual", True),
        use_gan=use_gan,
        gan_weight=loss_cfg.get("gan_weight", 0.5),
        disc_start_epoch=loss_cfg.get("disc_start_epoch", 10),
    )
    if use_gan:
        criterion.discriminator.to(device)

    # Optimizers
    optimizer_vae = AdamW(
        model.parameters(), lr=tc["lr"], weight_decay=tc.get("weight_decay", 0.01)
    )
    optimizer_disc = None
    if use_gan:
        disc_lr = tc["lr"] * loss_cfg.get("disc_lr_factor", 0.5)
        optimizer_disc = AdamW(
            criterion.discriminator.parameters(),
            lr=disc_lr,
            weight_decay=tc.get("weight_decay", 0.01),
        )

    scheduler_vae = CosineAnnealingLR(
        optimizer_vae, T_max=tc["epochs"], eta_min=tc["lr"] * 0.01
    )
    scheduler_disc = None
    if use_gan:
        scheduler_disc = CosineAnnealingLR(
            optimizer_disc, T_max=tc["epochs"], eta_min=disc_lr * 0.01
        )

    # EMA
    ema = None
    if tc.get("use_ema", True):
        ema = EMA(model, decay=tc.get("ema_decay", 0.9999), device=device)
        logger.info("EMA enabled with decay=%s", tc.get("ema_decay", 0.9999))

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=tc.get("use_amp", True))

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if tc.get("resume_from"):
        ckpt = torch.load(tc["resume_from"], map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer_vae.load_state_dict(ckpt["optimizer_vae_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("val_loss", float("inf"))
        logger.info("Resumed from epoch %d", start_epoch - 1)

    # Training
    logger.info("Starting training for %d epochs", tc["epochs"])

    for epoch in range(start_epoch, tc["epochs"] + 1):
        criterion.set_epoch(epoch)

        train_stats = train_epoch(
            model, train_loader, criterion,
            optimizer_vae, optimizer_disc,
            scaler, ema, device, epoch,
            use_amp=tc.get("use_amp", True),
            grad_clip_norm=tc.get("grad_clip_norm", 1.0),
            gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
            log_every=lc.get("log_every", 100),
        )

        logger.info(
            "Epoch %d/%d - VAE: %.4f, Disc: %.4f, Recon: %.4f, KL: %.6f",
            epoch, tc["epochs"],
            train_stats["vae_loss"], train_stats["disc_loss"],
            train_stats["recon_loss"], train_stats["kl_loss"],
        )

        # Validate
        if epoch % lc.get("val_every", 5) == 0:
            val_stats = validate(model, val_loader, criterion, device, ema)
            logger.info(
                "Val - Loss: %.4f, Recon: %.4f, KL: %.6f",
                val_stats["total_loss"], val_stats["recon_loss"], val_stats["kl_loss"],
            )

            if val_stats["total_loss"] < best_val_loss:
                best_val_loss = val_stats["total_loss"]
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_vae_state_dict": optimizer_vae.state_dict(),
                        "val_loss": best_val_loss,
                        **({"ema_state_dict": ema.state_dict()} if ema else {}),
                        **(
                            {
                                "discriminator_state_dict": criterion.discriminator.state_dict(),
                                "optimizer_disc_state_dict": optimizer_disc.state_dict(),
                            }
                            if use_gan
                            else {}
                        ),
                    },
                    ckpt_dir,
                    filename="vae_best.pth",
                )
                logger.info("Saved best model (val_loss: %.4f)", best_val_loss)

        # Samples
        if epoch % lc.get("sample_every", 10) == 0:
            save_samples(model, val_loader, sample_dir, epoch, device)

        # Periodic checkpoint
        if epoch % lc.get("save_every", 10) == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_vae_state_dict": optimizer_vae.state_dict(),
                    "scheduler_vae_state_dict": scheduler_vae.state_dict(),
                },
                ckpt_dir,
                filename=f"vae_epoch{epoch}.pth",
            )

        scheduler_vae.step()
        if scheduler_disc is not None:
            scheduler_disc.step()

    logger.info("Training completed. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train VAE")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_root", type=str, help="Override data root path")
    parser.add_argument("--exp_name", type=str, help="Override experiment name")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    # CLI overrides
    if args.data_root:
        config.setdefault("data", {})["data_root"] = args.data_root
    if args.exp_name:
        config.setdefault("training", {})["exp_name"] = args.exp_name

    setup_logger("kamba", config["training"].get("output_dir", "experiments"))
    main(config)
