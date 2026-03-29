"""Encoder components for the VAE."""

from typing import Tuple

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    """Residual block with GroupNorm, SiLU activation, and optional dropout.

    Applies two convolution layers with a skip connection. When input and
    output channel counts differ, a 1x1 convolution is used for the shortcut.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        dropout: Dropout probability applied between convolutions.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply residual block.

        Args:
            x: Input tensor of shape ``(B, in_channels, H, W)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)


class Encoder(nn.Module):
    """ResNet-style downsampling encoder for the VAE.

    Progressively downsamples the input image through residual blocks and
    strided convolutions, then projects to mean and log-variance for the
    latent Gaussian distribution.

    With default ``hidden_dims``, spatial dimensions are reduced by 8x:
    ``256x256 -> 128x128 -> 64x64 -> 32x32``.

    Args:
        in_channels: Number of input image channels.
        hidden_dims: Channel dimensions at each encoder level.
        latent_channels: Number of latent channels (output has
            ``2 * latent_channels`` for mean and log-variance).
        num_res_blocks: Number of residual blocks per level.
        dropout: Dropout probability for residual blocks.
    """

    def __init__(
        self,
        in_channels: int = 3,
        hidden_dims: Tuple[int, ...] = (128, 256, 512, 512),
        latent_channels: int = 4,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i - 1] if i > 0 else hidden_dims[0]
            out_ch = hidden_dims[i]

            blocks: list[nn.Module] = []
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch

            # Downsample at every level except the last.
            if i < len(hidden_dims) - 1:
                blocks.append(
                    nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1)
                )

            self.down_blocks.append(nn.Sequential(*blocks))

        final_ch = hidden_dims[-1]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_ch, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(
            final_ch, 2 * latent_channels, kernel_size=3, padding=1
        )

    def forward(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode an image to latent distribution parameters.

        Args:
            x: Input image of shape ``(B, C, H, W)``.

        Returns:
            Tuple of ``(mean, logvar)``, each of shape
            ``(B, latent_channels, H', W')`` where ``H'`` and ``W'`` are the
            spatially downsampled dimensions.
        """
        h = self.input_conv(x)
        for down_block in self.down_blocks:
            h = down_block(h)

        h = self.act_out(self.norm_out(h))
        moments = self.conv_out(h)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        return mean, logvar
