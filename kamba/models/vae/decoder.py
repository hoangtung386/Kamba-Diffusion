"""KAN-based decoder for the VAE."""

from typing import Tuple

import torch
import torch.nn as nn

from kamba.models.vae.encoder import ResBlock


class KANDecoder(nn.Module):
    """KAN-based decoder for the VAE.

    Uses KAN blocks for interpretable channel mixing combined with
    nearest-neighbor upsampling and convolution for spatial upsampling.
    Falls back to standard ``ResBlock`` when ``use_kan=False``.

    With default ``hidden_dims``, spatial dimensions are increased by 8x:
    ``32x32 -> 64x64 -> 128x128 -> 256x256``.

    Args:
        latent_channels: Number of input latent channels.
        hidden_dims: Channel dimensions at each decoder level.
        out_channels: Number of output image channels.
        num_res_blocks: Number of KAN/residual blocks per level.
        use_kan: If ``True``, use ``KANBlock2d``; otherwise use ``ResBlock``.
    """

    def __init__(
        self,
        latent_channels: int = 4,
        hidden_dims: Tuple[int, ...] = (512, 512, 256, 128),
        out_channels: int = 3,
        num_res_blocks: int = 2,
        use_kan: bool = True,
    ) -> None:
        super().__init__()
        self._use_kan = use_kan

        if use_kan:
            from kamba.models.blocks.kan_blocks import KANBlock2d

        self.input_conv = nn.Conv2d(
            latent_channels, hidden_dims[0], kernel_size=3, padding=1
        )

        self.up_blocks = nn.ModuleList()
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i + 1] if i < len(hidden_dims) - 1 else hidden_dims[-1]

            blocks: list[nn.Module] = []
            for _ in range(num_res_blocks):
                if use_kan:
                    blocks.append(KANBlock2d(in_ch))
                else:
                    blocks.append(ResBlock(in_ch, in_ch))

            # Upsample at every level except the last.
            if i < len(hidden_dims) - 1:
                blocks.append(
                    nn.Sequential(
                        nn.Upsample(scale_factor=2, mode="nearest"),
                        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                    )
                )

            self.up_blocks.append(nn.Sequential(*blocks))

        final_ch = hidden_dims[-1]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_ch, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(final_ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Decode a latent code to an image.

        Args:
            z: Latent tensor of shape ``(B, latent_channels, H, W)``.

        Returns:
            Reconstructed image of shape ``(B, out_channels, H', W')``.
        """
        h = self.input_conv(z)
        for up_block in self.up_blocks:
            h = up_block(h)

        h = self.act_out(self.norm_out(h))
        return self.conv_out(h)
