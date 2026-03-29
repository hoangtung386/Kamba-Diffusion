"""Variational Autoencoder with KL-regularised latent space."""

from typing import Tuple

import torch
import torch.nn as nn

from kamba.models.vae.encoder import Encoder
from kamba.models.vae.decoder import KANDecoder


class VAE(nn.Module):
    """Variational Autoencoder with KL-regularised latent space.

    Architecture:
        - Encoder: ResNet-style downsampling (8x spatial reduction).
        - Latent: Gaussian distribution parameterised by mean and log-variance.
        - Decoder: KAN-based upsampling (8x spatial expansion).

    Includes quantisation and post-quantisation 1x1 convolutions following
    the Stable Diffusion convention.

    Args:
        in_channels: Number of input/output image channels.
        latent_channels: Number of latent channels.
        hidden_dims: Channel dimensions for the encoder (reversed for decoder).
        image_size: Spatial size of the input image (assumes square).
        num_res_blocks: Number of residual/KAN blocks per level.
        dropout: Dropout probability for encoder residual blocks.
        use_kan_decoder: If ``True``, use KAN blocks in the decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 4,
        hidden_dims: Tuple[int, ...] = (128, 256, 512, 512),
        image_size: int = 256,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
        use_kan_decoder: bool = True,
    ) -> None:
        super().__init__()
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.latent_size = image_size // 8

        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout,
        )

        decoder_dims = tuple(reversed(hidden_dims))
        self.decoder = KANDecoder(
            latent_channels=latent_channels,
            hidden_dims=decoder_dims,
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            use_kan=use_kan_decoder,
        )

        self.quant_conv = nn.Conv2d(
            latent_channels * 2, latent_channels * 2, kernel_size=1
        )
        self.post_quant_conv = nn.Conv2d(
            latent_channels, latent_channels, kernel_size=1
        )

    def encode(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image to latent distribution parameters.

        Args:
            x: Input image of shape ``(B, C, H, W)``.

        Returns:
            Tuple of ``(mean, logvar)``, each of shape
            ``(B, latent_channels, H//8, W//8)``.
        """
        mean, logvar = self.encoder(x)
        moments = self.quant_conv(torch.cat([mean, logvar], dim=1))
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent code to an image.

        Args:
            z: Latent tensor of shape ``(B, latent_channels, H//8, W//8)``.

        Returns:
            Reconstructed image of shape ``(B, C, H, W)``.
        """
        z = self.post_quant_conv(z)
        return self.decoder(z)

    @staticmethod
    def reparameterize(
        mean: torch.Tensor, logvar: torch.Tensor
    ) -> torch.Tensor:
        """Sample from the latent distribution via the reparameterization trick.

        ``z = mean + std * epsilon`` where ``epsilon ~ N(0, I)``.

        Args:
            mean: Mean of the latent distribution.
            logvar: Log-variance of the latent distribution.

        Returns:
            Sampled latent tensor with the same shape as ``mean``.
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mean + eps * std

    def forward(
        self, x: torch.Tensor, sample: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Full forward pass: encode, sample, decode.

        Args:
            x: Input image of shape ``(B, C, H, W)``.
            sample: If ``True``, sample from the latent distribution.
                If ``False``, use the mean deterministically.

        Returns:
            Tuple of ``(reconstruction, mean, logvar)``.
        """
        mean, logvar = self.encode(x)
        z = self.reparameterize(mean, logvar) if sample else mean
        recon = self.decode(z)
        return recon, mean, logvar

    def sample(self, num_samples: int, device: torch.device) -> torch.Tensor:
        """Generate images by sampling from the prior ``p(z) = N(0, I)``.

        Args:
            num_samples: Number of images to generate.
            device: Device on which to create tensors.

        Returns:
            Generated images of shape ``(num_samples, C, H, W)``.
        """
        z = torch.randn(
            num_samples,
            self.latent_channels,
            self.latent_size,
            self.latent_size,
            device=device,
        )
        return self.decode(z)
