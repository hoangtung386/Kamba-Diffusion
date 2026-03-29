"""LDM training script (Stage 2): Train text-conditional diffusion with frozen VAE.

Usage:
    python scripts/train_ldm.py --config configs/ldm_train.yaml
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
from torchvision.utils import save_image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kamba.data.coco import COCODataset
from kamba.models.pipeline import LatentDiffusionModel
from kamba.utils.checkpoint import save_checkpoint
from kamba.utils.ema import EMA
from kamba.utils.logger import setup_logger

logger = logging.getLogger(__name__)


def train_epoch(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    ema: EMA | None,
    device: torch.device,
    epoch: int,
    use_amp: bool = True,
    grad_clip_norm: float = 1.0,
    gradient_accumulation_steps: int = 1,
    log_every: int = 100,
) -> float:
    """Train for one epoch."""
    model.train()
    model.vae.eval()

    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        captions = batch["caption"]

        if batch_idx % gradient_accumulation_steps == 0:
            optimizer.zero_grad()

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = model(images, captions)
            loss = loss / gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.denoiser.parameters(), max_norm=grad_clip_norm
            )
            scaler.step(optimizer)
            scaler.update()

            if ema is not None:
                ema.update(model.denoiser)

        total_loss += loss.item() * gradient_accumulation_steps

        if batch_idx % log_every == 0:
            pbar.set_postfix({"loss": f"{loss.item() * gradient_accumulation_steps:.4f}"})

    return total_loss / len(dataloader)


@torch.no_grad()
def validate(
    model: LatentDiffusionModel,
    dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> float:
    """Validate model."""
    model.eval()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Validating"):
        images = batch["image"].to(device)
        captions = batch["caption"]

        with torch.amp.autocast("cuda", enabled=use_amp):
            loss = model(images, captions)
        total_loss += loss.item()

    return total_loss / len(dataloader)


@torch.no_grad()
def generate_samples(
    model: LatentDiffusionModel,
    prompts: list[str],
    save_dir: str,
    epoch: int,
    config: dict,
) -> None:
    """Generate sample images for visualization."""
    model.eval()
    images = model.generate(
        captions=prompts,
        num_steps=50,
        guidance_scale=7.5,
        height=config.get("model", {}).get("vae", {}).get("image_size", 256),
        width=config.get("model", {}).get("vae", {}).get("image_size", 256),
    )

    for i, img in enumerate(images):
        img = (img + 1) / 2
        img = torch.clamp(img, 0, 1)
        save_image(img, os.path.join(save_dir, f"epoch{epoch}_sample{i}.png"))

    logger.info("Generated %d samples at epoch %d", len(prompts), epoch)


def main(config: dict) -> None:
    """Main training loop."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    tc = config["training"]
    mc = config["model"]
    pc = config.get("paths", {})
    lc = config.get("logging", tc)

    # Directories
    exp_dir = os.path.join(pc.get("output_dir", "experiments"), pc.get("exp_name", "ldm"))
    ckpt_dir = os.path.join(exp_dir, "checkpoints")
    sample_dir = os.path.join(exp_dir, "samples")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)

    # Dataset
    vae_cfg = mc.get("vae", {})
    image_size = vae_cfg.get("image_size", 256)

    train_dataset = COCODataset(
        data_root=pc.get("data_root", ""), split="train", image_size=image_size
    )
    val_dataset = COCODataset(
        data_root=pc.get("data_root", ""), split="val", image_size=image_size
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=tc["batch_size"],
        shuffle=True,
        num_workers=tc.get("num_workers", 4),
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=tc["batch_size"],
        shuffle=False,
        num_workers=tc.get("num_workers", 4),
        pin_memory=True,
    )

    # Model
    diff_cfg = mc.get("diffusion", {})
    den_cfg = mc.get("denoiser", {})

    model = LatentDiffusionModel(
        vae_config={
            "in_channels": vae_cfg.get("in_channels", 3),
            "latent_channels": vae_cfg.get("latent_channels", 4),
            "hidden_dims": tuple(vae_cfg.get("hidden_dims", [128, 256, 512, 512])),
            "image_size": image_size,
            "use_kan_decoder": vae_cfg.get("use_kan_decoder", True),
        },
        vae_checkpoint=pc.get("vae_checkpoint"),
        text_encoder_model=pc.get("text_encoder_model", "openai/clip-vit-large-patch14"),
        denoiser_config={
            "in_channels": den_cfg.get("in_channels", 4),
            "out_channels": den_cfg.get("out_channels", 4),
            "model_channels": den_cfg.get("model_channels", 320),
            "channel_mult": tuple(den_cfg.get("channel_mult", [1, 2, 4, 4])),
            "num_res_blocks": den_cfg.get("num_res_blocks", 2),
            "attention_resolutions": tuple(den_cfg.get("attention_resolutions", [1, 2, 3])),
            "context_dim": den_cfg.get("context_dim", 768),
            "num_heads": den_cfg.get("num_heads", 8),
            "use_cross_attn": den_cfg.get("use_cross_attn", True),
        },
        timesteps=diff_cfg.get("timesteps", 1000),
        beta_schedule=diff_cfg.get("beta_schedule", "linear"),
        prediction_type=diff_cfg.get("prediction_type", "epsilon"),
        min_snr_gamma=diff_cfg.get("min_snr_gamma", 5.0),
        unconditional_prob=tc.get("unconditional_prob", 0.1),
        device=str(device),
    ).to(device)

    # Optimizer (only denoiser)
    optimizer = AdamW(
        model.denoiser.parameters(),
        lr=tc["lr"],
        weight_decay=tc.get("weight_decay", 0.01),
    )
    scheduler = CosineAnnealingLR(
        optimizer, T_max=tc["epochs"], eta_min=tc["lr"] * 0.1
    )

    # EMA
    ema = None
    if tc.get("use_ema", True):
        ema = EMA(model.denoiser, decay=tc.get("ema_decay", 0.9999), device=device)

    # AMP
    scaler = torch.amp.GradScaler("cuda", enabled=tc.get("use_amp", True))

    # Resume
    start_epoch = 1
    best_val_loss = float("inf")
    if tc.get("resume_from"):
        ckpt = torch.load(tc["resume_from"], map_location=device, weights_only=False)
        model.denoiser.load_state_dict(ckpt["denoiser_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0) + 1
        logger.info("Resumed from epoch %d", start_epoch - 1)

    # Sample prompts
    sample_prompts = [
        "A cat sitting on a table",
        "A beautiful sunset over mountains",
        "A dog playing in the park",
        "A bird flying in the sky",
    ]

    # Training
    logger.info("Starting LDM training for %d epochs", tc["epochs"])

    for epoch in range(start_epoch, tc["epochs"] + 1):
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler, ema,
            device, epoch,
            use_amp=tc.get("use_amp", True),
            grad_clip_norm=tc.get("grad_clip_norm", 1.0),
            gradient_accumulation_steps=tc.get("gradient_accumulation_steps", 1),
            log_every=lc.get("log_every", 100),
        )
        logger.info("Epoch %d/%d - Train Loss: %.4f", epoch, tc["epochs"], train_loss)

        # Validate
        if epoch % lc.get("val_every", 5) == 0:
            val_loss = validate(model, val_loader, device, use_amp=tc.get("use_amp", True))
            logger.info("Val Loss: %.4f", val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(
                    {
                        "epoch": epoch,
                        "denoiser_state_dict": model.denoiser.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": best_val_loss,
                    },
                    ckpt_dir,
                    filename="ldm_best.pth",
                )
                logger.info("Saved best model (val_loss: %.4f)", best_val_loss)

        # Generate samples
        if epoch % lc.get("sample_every", 10) == 0:
            generate_samples(model, sample_prompts, sample_dir, epoch, config)

        # Periodic checkpoint
        if epoch % lc.get("save_every", 50) == 0:
            save_checkpoint(
                {
                    "epoch": epoch,
                    "denoiser_state_dict": model.denoiser.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                },
                ckpt_dir,
                filename=f"ldm_epoch{epoch}.pth",
            )

        scheduler.step()

    logger.info("Training completed. Best val loss: %.4f", best_val_loss)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LDM (Stage 2)")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--data_root", type=str, help="Override data root")
    parser.add_argument("--vae_checkpoint", type=str, help="Override VAE checkpoint path")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.data_root:
        config.setdefault("paths", {})["data_root"] = args.data_root
    if args.vae_checkpoint:
        config.setdefault("paths", {})["vae_checkpoint"] = args.vae_checkpoint

    setup_logger("kamba", config.get("paths", {}).get("output_dir", "experiments"))
    main(config)
