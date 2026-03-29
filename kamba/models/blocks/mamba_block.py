"""Vision Mamba blocks for 2-D feature processing.

Provides components that adapt the 1-D Mamba SSM to operate on spatial
feature maps by flattening to sequences and unflattening the output.
"""

from typing import Any

import torch
import torch.nn as nn

try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None


class LayerNorm2d(nn.LayerNorm):
    """Layer normalization for channel-first tensors ``(B, C, H, W)``.

    Standard ``nn.LayerNorm`` expects the normalized dimension last.
    This wrapper permutes the input so that normalization is applied
    over the channel dimension and then permutes back.

    Args:
        normalized_shape: Size of the channel dimension.
        **kwargs: Forwarded to ``nn.LayerNorm``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply layer normalization over channels.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Normalized tensor of shape ``(B, C, H, W)``.
        """
        x = x.permute(0, 2, 3, 1)  # (B, H, W, C)
        x = super().forward(x)
        x = x.permute(0, 3, 1, 2)  # (B, C, H, W)
        return x


class MambaVisionBlock(nn.Module):
    """Vision Mamba block for 2-D feature maps.

    Flattens spatial dimensions into a sequence, processes with a Mamba
    SSM layer, and reshapes back. Includes a pre-norm and residual
    connection.

    Args:
        d_model: Number of input/output channels.
        d_state: SSM state expansion factor.
        d_conv: Width of the local convolution inside Mamba.
        expand: Channel expansion factor for the inner projection.

    Raises:
        ImportError: If ``mamba_ssm`` is not installed.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ) -> None:
        super().__init__()
        if Mamba is None:
            raise ImportError(
                "mamba_ssm is not installed. "
                "Please install it to use MambaVisionBlock."
            )

        self.d_model = d_model
        self.norm = LayerNorm2d(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a spatial feature map through Mamba.

        Args:
            x: Tensor of shape ``(B, C, H, W)`` where ``C == d_model``.

        Returns:
            Tensor of shape ``(B, C, H, W)`` with residual connection.
        """
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.reshape(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        x = self.mamba(x)
        x = x.transpose(1, 2).reshape(b, c, h, w)  # (B, C, H, W)

        return x + residual


class MambaStage(nn.Module):
    """A stage composed of multiple sequential Mamba vision blocks.

    Args:
        dim: Channel dimensionality for each block.
        depth: Number of ``MambaVisionBlock`` layers.
        **kwargs: Additional keyword arguments forwarded to each
            ``MambaVisionBlock`` (e.g. ``d_state``, ``d_conv``,
            ``expand``).
    """

    def __init__(self, dim: int, depth: int, **kwargs: Any) -> None:
        super().__init__()
        self.blocks = nn.ModuleList(
            [MambaVisionBlock(d_model=dim, **kwargs) for _ in range(depth)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all blocks sequentially.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, C, H, W)``.
        """
        for block in self.blocks:
            x = block(x)
        return x
