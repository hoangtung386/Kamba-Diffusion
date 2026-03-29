"""VAE loss components including perceptual loss and PatchGAN discriminator.

Provides a composite loss for training Variational Autoencoders with:
- Pixel-level reconstruction (L1 or L2)
- VGG-based perceptual loss with optional LPIPS-style normalisation
- KL divergence (properly normalised per element)
- PatchGAN adversarial loss with hinge formulation
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PatchGANDiscriminator(nn.Module):
    """PatchGAN discriminator that classifies overlapping image patches.

    Outputs a spatial grid of real/fake predictions rather than a single
    scalar.  Architecture follows Pix2Pix (Isola et al., 2017).

    Args:
        in_channels: Number of input image channels.
        ndf: Base number of discriminator filters.
        n_layers: Number of downsampling layers.
        use_sigmoid: If ``True``, apply sigmoid to the output.
    """

    def __init__(
        self,
        in_channels: int = 3,
        ndf: int = 64,
        n_layers: int = 3,
        use_sigmoid: bool = False,
    ) -> None:
        super().__init__()

        sequence: list[nn.Module] = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        nf_mult = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(
                    ndf * nf_mult_prev,
                    ndf * nf_mult,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, inplace=True),
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(
                ndf * nf_mult_prev,
                ndf * nf_mult,
                kernel_size=4,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, inplace=True),
        ]

        sequence.append(
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)
        )

        if use_sigmoid:
            sequence.append(nn.Sigmoid())

        self.model = nn.Sequential(*sequence)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute patch-level real/fake predictions.

        Args:
            x: Input image of shape ``(B, C, H, W)``.

        Returns:
            Patch predictions of shape ``(B, 1, H', W')``.
        """
        return self.model(x)


class PerceptualLoss(nn.Module):
    """VGG-based perceptual loss with optional LPIPS-style normalisation.

    Extracts features from multiple VGG-16 layers and compares them
    between predicted and target images.

    Args:
        layers: VGG-16 layer indices to extract features from.
        weights: Per-layer loss weights.
        use_lpips_normalisation: If ``True``, L2-normalise features before
            computing the squared difference (LPIPS-style).
    """

    def __init__(
        self,
        layers: Optional[Tuple[int, ...]] = None,
        weights: Optional[Tuple[float, ...]] = None,
        use_lpips_normalisation: bool = True,
    ) -> None:
        super().__init__()

        if layers is None:
            layers = (3, 8, 15, 22)
        if weights is None:
            weights = (1.0, 1.0, 1.0, 1.0)

        vgg = models.vgg16(weights="IMAGENET1K_V1").features

        self.blocks = nn.ModuleList()
        prev_idx = 0
        for idx in layers:
            block = nn.Sequential(
                *[vgg[i] for i in range(prev_idx, idx + 1)]
            )
            self.blocks.append(block)
            prev_idx = idx + 1

        # Freeze all VGG parameters.
        for param in self.parameters():
            param.requires_grad = False

        self.weights = weights
        self.use_lpips_normalisation = use_lpips_normalisation

        # ImageNet normalisation constants.
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        )

        if use_lpips_normalisation:
            self.register_buffer(
                "lin_weights", torch.ones(len(layers), 1, 1, 1, 1)
            )

    @staticmethod
    def _normalise_features(x: torch.Tensor) -> torch.Tensor:
        """L2-normalise feature maps along the channel dimension."""
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + 1e-10)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute perceptual loss between predicted and target images.

        Inputs may be in ``[-1, 1]`` or ``[0, 1]`` range; they are
        automatically normalised to ImageNet statistics.

        Args:
            pred: Predicted image of shape ``(B, 3, H, W)``.
            target: Target image of shape ``(B, 3, H, W)``.

        Returns:
            Scalar perceptual loss.
        """
        # Normalise to [0, 1] if inputs are in [-1, 1].
        if pred.min() < 0:
            pred = (pred + 1) / 2
        if target.min() < 0:
            target = (target + 1) / 2

        # Apply ImageNet normalisation.
        pred = (pred - self.mean) / self.std
        target = (target - self.mean) / self.std

        loss = torch.tensor(0.0, device=pred.device)
        x_pred = pred
        x_target = target

        for i, block in enumerate(self.blocks):
            x_pred = block(x_pred)
            x_target = block(x_target)

            if self.use_lpips_normalisation:
                x_pred_norm = self._normalise_features(x_pred)
                x_target_norm = self._normalise_features(x_target)
                diff = (x_pred_norm - x_target_norm) ** 2
                diff = diff.mean(dim=[2, 3], keepdim=True)
                loss = loss + self.weights[i] * (self.lin_weights[i] * diff).mean()
            else:
                loss = loss + self.weights[i] * F.l1_loss(x_pred, x_target)

        return loss


class VAELoss(nn.Module):
    """Composite VAE training loss.

    Combines reconstruction, perceptual, KL-divergence, and optional GAN
    losses into a single module:

    ``L = L_recon + w_p * L_perceptual + w_kl * L_kl + w_gan * L_gan``

    Args:
        recon_loss_type: Reconstruction loss type, ``"l1"`` or ``"l2"``.
        perceptual_weight: Weight for the perceptual loss term.
        use_perceptual: If ``True``, include perceptual loss.
        use_lpips_norm: If ``True``, use LPIPS-style feature normalisation.
        kl_weight: Weight for the KL divergence term.
        use_gan: If ``True``, include adversarial loss and create a
            ``PatchGANDiscriminator`` as ``self.discriminator``.
        gan_weight: Weight for the adversarial loss term.
        disc_start_epoch: Epoch at which to start using the discriminator.
        disc_factor: Multiplicative factor applied to the GAN weight.
        use_feature_matching: If ``True``, enable feature matching loss
            (currently a placeholder).
        feature_matching_weight: Weight for the feature matching loss.
    """

    def __init__(
        self,
        recon_loss_type: str = "l1",
        perceptual_weight: float = 1.0,
        use_perceptual: bool = True,
        use_lpips_norm: bool = True,
        kl_weight: float = 1e-6,
        use_gan: bool = False,
        gan_weight: float = 0.5,
        disc_start_epoch: int = 0,
        disc_factor: float = 1.0,
        use_feature_matching: bool = False,
        feature_matching_weight: float = 10.0,
    ) -> None:
        super().__init__()

        self.recon_loss_type = recon_loss_type
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        self.use_gan = use_gan
        self.gan_weight = gan_weight
        self.disc_start_epoch = disc_start_epoch
        self.disc_factor = disc_factor
        self.use_feature_matching = use_feature_matching
        self.feature_matching_weight = feature_matching_weight

        if use_perceptual:
            self.perceptual_loss = PerceptualLoss(
                use_lpips_normalisation=use_lpips_norm
            )

        if use_gan:
            self.discriminator = PatchGANDiscriminator(
                in_channels=3, ndf=64, n_layers=3
            )

        self.current_epoch: int = 0

    def set_epoch(self, epoch: int) -> None:
        """Update the current epoch for discriminator scheduling.

        Args:
            epoch: Current training epoch.
        """
        self.current_epoch = epoch

    @staticmethod
    def kl_divergence(mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Compute element-normalised KL divergence from a unit Gaussian.

        ``KL(q(z|x) || p(z))`` where ``p(z) = N(0, I)``.

        Args:
            mean: Mean of the approximate posterior.
            logvar: Log-variance of the approximate posterior.

        Returns:
            Scalar KL divergence normalised by the total number of elements.
        """
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        return kl / mean.numel()

    def reconstruction_loss(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        """Compute pixel-level reconstruction loss.

        Args:
            pred: Predicted image.
            target: Target image.

        Returns:
            Scalar reconstruction loss.
        """
        if self.recon_loss_type == "l1":
            return F.l1_loss(pred, target)
        if self.recon_loss_type == "l2":
            return F.mse_loss(pred, target)
        raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type!r}")

    def gan_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        optimizer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute GAN loss for the generator or discriminator.

        Args:
            pred: Reconstructed images.
            target: Real images.
            optimizer_idx: ``0`` for the generator, ``1`` for the
                discriminator.

        Returns:
            Tuple of ``(loss, log_dict)`` where ``log_dict`` contains
            scalar metrics for logging.
        """
        if not self.use_gan or self.current_epoch < self.disc_start_epoch:
            return torch.tensor(0.0, device=pred.device), {}

        if optimizer_idx == 0:
            logits_fake = self.discriminator(pred)
            g_loss = -torch.mean(logits_fake)
            return g_loss, {
                "gan/g_loss": g_loss.item(),
                "gan/logits_fake": logits_fake.mean().item(),
            }

        # Discriminator loss (hinge formulation).
        logits_real = self.discriminator(target.detach())
        logits_fake = self.discriminator(pred.detach())

        d_loss_real = torch.mean(F.relu(1.0 - logits_real))
        d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        return d_loss, {
            "gan/d_loss": d_loss.item(),
            "gan/d_loss_real": d_loss_real.item(),
            "gan/d_loss_fake": d_loss_fake.item(),
            "gan/logits_real": logits_real.mean().item(),
            "gan/logits_fake": logits_fake.mean().item(),
        }

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        mean: torch.Tensor,
        logvar: torch.Tensor,
        optimizer_idx: int = 0,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the composite VAE loss.

        Args:
            pred: Reconstructed image of shape ``(B, C, H, W)``.
            target: Original image of shape ``(B, C, H, W)``.
            mean: Latent mean of shape ``(B, latent_ch, H', W')``.
            logvar: Latent log-variance of shape ``(B, latent_ch, H', W')``.
            optimizer_idx: ``0`` for the VAE/generator, ``1`` for the
                discriminator.

        Returns:
            Tuple of ``(total_loss, loss_dict)`` where ``loss_dict``
            contains per-component scalar losses for logging.
        """
        loss_dict: Dict[str, float] = {}

        if optimizer_idx == 0:
            recon_loss = self.reconstruction_loss(pred, target)
            loss_dict["recon_loss"] = recon_loss.item()

            if self.use_perceptual:
                perc_loss = self.perceptual_loss(pred, target)
                loss_dict["perceptual_loss"] = perc_loss.item()
            else:
                perc_loss = torch.tensor(0.0, device=pred.device)
                loss_dict["perceptual_loss"] = 0.0

            kl_loss = self.kl_divergence(mean, logvar)
            loss_dict["kl_loss"] = kl_loss.item()

            g_loss, gan_log = self.gan_loss(pred, target, optimizer_idx=0)
            loss_dict.update(gan_log)

            total_loss = (
                recon_loss
                + self.perceptual_weight * perc_loss
                + self.kl_weight * kl_loss
                + self.gan_weight * self.disc_factor * g_loss
            )
            loss_dict["total_loss"] = total_loss.item()

        else:
            total_loss, disc_log = self.gan_loss(pred, target, optimizer_idx=1)
            loss_dict.update(disc_log)
            loss_dict["total_loss"] = total_loss.item()

        return total_loss, loss_dict
