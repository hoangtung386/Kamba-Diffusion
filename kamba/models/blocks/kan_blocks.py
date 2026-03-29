"""Kolmogorov-Arnold Network (KAN) building blocks.

Provides B-spline basis computation, a KAN linear layer that combines a
standard linear path with a learned spline path, and a 2-D spatial KAN
block suitable for use in image decoders.
"""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BSplineBasis(nn.Module):
    """Vectorized B-spline basis evaluation via Cox-de Boor recursion.

    Constructs an extended knot vector and evaluates all basis functions
    at the given input points. Uses a soft boundary mapping (tanh) to
    gracefully handle inputs outside the grid range.

    Args:
        grid_size: Number of interior grid intervals.
        spline_order: Polynomial order of the B-spline (e.g. 3 for cubic).
        grid_range: Two-element list ``[lo, hi]`` defining the active
            domain of the spline.
    """

    def __init__(
        self,
        grid_size: int = 5,
        spline_order: int = 3,
        grid_range: Optional[List[float]] = None,
    ) -> None:
        super().__init__()
        if grid_range is None:
            grid_range = [-1.0, 1.0]

        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range: Tuple[float, float] = (grid_range[0], grid_range[1])

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid)

        self.num_basis: int = len(grid) - spline_order - 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate B-spline basis functions.

        Args:
            x: Tensor of shape ``(B, in_features)`` with values ideally
                in ``[grid_range[0], grid_range[1]]``.

        Returns:
            Tensor of shape ``(B, in_features, num_basis)`` containing
            the value of each basis function at each input point.
        """
        lo, hi = self.grid_range
        range_size = hi - lo

        # Soft boundary mapping via tanh.
        x_norm = (x - lo) / range_size
        x_norm = torch.tanh(x_norm * 2.0 - 1.0) * 0.5 + 0.5
        x = x_norm * range_size + lo

        x = x.unsqueeze(-1)  # (B, in_features, 1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, num_knots)

        # Order-0 indicator basis.
        basis = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        # Include right boundary.
        basis[:, :, -1] = basis[:, :, -1] + (
            x[:, :, 0] == grid[0, 0, -1]
        ).float()

        # Cox-de Boor recursion for orders 1 .. spline_order.
        for k in range(1, self.spline_order + 1):
            left_num = x - grid[:, :, : -(k + 1)]
            left_den = grid[:, :, k:-1] - grid[:, :, : -(k + 1)]
            left_den = torch.where(
                torch.abs(left_den) < 1e-8,
                torch.ones_like(left_den),
                left_den,
            )
            left_coef = left_num / left_den

            right_num = grid[:, :, k + 1 :] - x
            right_den = grid[:, :, k + 1 :] - grid[:, :, 1:-k]
            right_den = torch.where(
                torch.abs(right_den) < 1e-8,
                torch.ones_like(right_den),
                right_den,
            )
            right_coef = right_num / right_den

            basis = left_coef * basis[:, :, :-1] + right_coef * basis[:, :, 1:]

        return basis


class KANLinear(nn.Module):
    """KAN linear layer combining a standard linear path with a B-spline path.

    The output is the sum of a conventional linear transformation
    (with SiLU activation) and a spline-parameterised transformation
    that enables learnable, per-feature activation functions.

    Args:
        in_features: Number of input features.
        out_features: Number of output features.
        grid_size: Number of interior grid intervals for the spline.
        spline_order: Polynomial order of the B-spline.
        scale_base: Scaling factor for the base (linear) path.
        scale_spline: Scaling factor for the spline path.
        grid_range: Two-element list for the spline domain.
        dropout: Dropout probability applied after combining paths.
        use_layernorm: Whether to apply layer normalization to the input.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        grid_range: Optional[List[float]] = None,
        dropout: float = 0.0,
        use_layernorm: bool = False,
    ) -> None:
        super().__init__()
        if grid_range is None:
            grid_range = [-1.0, 1.0]

        self.in_features = in_features
        self.out_features = out_features
        self.scale_base = scale_base
        self.scale_spline = scale_spline

        # Optional input normalization.
        self.norm: Optional[nn.LayerNorm] = (
            nn.LayerNorm(in_features) if use_layernorm else None
        )

        # Base (linear) path.
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = nn.SiLU()

        # Spline path.
        self.bspline = BSplineBasis(grid_size, spline_order, grid_range)
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, self.bspline.num_basis)
        )

        # Regularization.
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        """Initialize weights with appropriate strategies."""
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        with torch.no_grad():
            std = 1.0 / math.sqrt(self.in_features * self.bspline.grid_size)
            nn.init.normal_(self.spline_weight, mean=0.0, std=std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the KAN linear transformation.

        Args:
            x: Tensor of shape ``(..., in_features)``.

        Returns:
            Tensor of shape ``(..., out_features)``.
        """
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        if self.norm is not None:
            x = self.norm(x)

        # Base path.
        base_output = F.linear(self.base_activation(x), self.base_weight)
        base_output = base_output * self.scale_base

        # Spline path.
        basis = self.bspline(x)  # (batch, in_features, num_basis)
        spline_output = torch.einsum(
            "bik,oik->bo", basis, self.spline_weight,
        )
        spline_output = spline_output * self.scale_spline

        output = self.dropout(base_output + spline_output)
        return output.reshape(*original_shape[:-1], self.out_features)


class KANBlock2d(nn.Module):
    """2-D KAN block with depthwise spatial mixing and KAN channel mixing.

    Applies a depthwise convolution for spatial context, followed by two
    KAN linear layers for channel mixing (replacing a standard MLP).
    Includes group normalization, layer scaling, and an optional residual
    connection.

    Args:
        channels: Number of input/output channels.
        expansion: Channel expansion ratio for the hidden KAN layer.
        grid_size: Grid size for the internal KAN linear layers.
        dropout: Dropout probability inside KAN layers.
        use_residual: Whether to add the input back (residual connection).
    """

    def __init__(
        self,
        channels: int,
        expansion: int = 2,
        grid_size: int = 5,
        dropout: float = 0.0,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        self.use_residual = use_residual
        hidden_dim = channels * expansion

        # Spatial mixing.
        self.dwconv = nn.Conv2d(
            channels,
            channels,
            kernel_size=7,
            padding=3,
            groups=channels,
            bias=False,
        )

        self.norm1 = nn.GroupNorm(
            num_groups=min(32, channels),
            num_channels=channels,
            eps=1e-6,
        )

        # KAN channel mixing.
        self.kan1 = KANLinear(
            channels,
            hidden_dim,
            grid_size=grid_size,
            dropout=dropout,
            use_layernorm=False,
        )

        self.norm2 = nn.GroupNorm(
            num_groups=min(32, hidden_dim),
            num_channels=hidden_dim,
            eps=1e-6,
        )

        self.kan2 = KANLinear(
            hidden_dim,
            channels,
            grid_size=grid_size,
            dropout=dropout,
            use_layernorm=False,
        )

        # Layer scale for training stability.
        self.layer_scale = nn.Parameter(torch.ones(channels, 1, 1) * 1e-6)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply the 2-D KAN block.

        Args:
            x: Tensor of shape ``(B, C, H, W)``.

        Returns:
            Tensor of shape ``(B, C, H, W)``.
        """
        residual = x

        # Spatial mixing.
        out = self.norm1(self.dwconv(x))

        # Channel mixing via KAN: (B, C, H, W) -> (B, H, W, C).
        out = out.permute(0, 2, 3, 1)
        out = self.kan1(out)

        # Intermediate norm: back to (B, C, H, W) for GroupNorm.
        out = out.permute(0, 3, 1, 2)
        out = self.norm2(out)
        out = out.permute(0, 2, 3, 1)

        out = self.kan2(out)
        out = out.permute(0, 3, 1, 2)

        out = out * self.layer_scale

        if self.use_residual:
            out = out + residual

        return out
