"""Multi-head attention modules for text-to-image conditioning.

Provides cross-attention, self-attention, and combined attention blocks
used throughout the U-Net architecture. Uses ``torch.nn.functional.scaled_dot_product_attention``
(PyTorch 2.0+) for fused, memory-efficient attention kernels.
"""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """Multi-head cross-attention layer.

    Queries come from image features; keys and values come from a
    conditioning context (e.g. CLIP text embeddings).

    Args:
        query_dim: Dimensionality of the query (image) features.
        context_dim: Dimensionality of the context (text) features.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        dropout: Dropout probability applied after the output projection.
    """

    def __init__(
        self,
        query_dim: int,
        context_dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = head_dim * num_heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply cross-attention.

        Args:
            x: Query tensor of shape ``(B, N, query_dim)``.
            context: Key/value tensor of shape ``(B, M, context_dim)``.
            mask: Optional boolean attention mask of shape
                ``(B, N, M)`` or broadcastable. ``True`` values are
                *attended to*; ``False`` values are masked out.

        Returns:
            Output tensor of shape ``(B, N, query_dim)``.
        """
        batch_size = x.shape[0]

        q = self._reshape_heads(self.to_q(x), batch_size)
        k = self._reshape_heads(self.to_k(context), batch_size)
        v = self._reshape_heads(self.to_v(context), batch_size)

        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1)  # broadcast over heads

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0,
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        return self.to_out(out)

    def _reshape_heads(
        self, tensor: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Reshape ``(B, seq, inner)`` to ``(B, heads, seq, head_dim)``."""
        return (
            tensor.view(batch_size, -1, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )


class SelfAttention(nn.Module):
    """Multi-head self-attention layer.

    All queries, keys, and values are derived from the same input.

    Args:
        dim: Input and output feature dimensionality.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        dropout: Dropout probability applied after the output projection.
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = head_dim * num_heads

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply self-attention.

        Args:
            x: Input tensor of shape ``(B, N, dim)``.
            mask: Optional boolean attention mask.

        Returns:
            Output tensor of shape ``(B, N, dim)``.
        """
        batch_size = x.shape[0]

        qkv = self.to_qkv(x)
        qkv = qkv.view(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(dim=0)

        attn_mask: Optional[torch.Tensor] = None
        if mask is not None:
            attn_mask = mask.unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=0.0,
        )

        out = (
            out.transpose(1, 2)
            .contiguous()
            .view(batch_size, -1, self.num_heads * self.head_dim)
        )
        return self.to_out(out)


class AttentionBlock(nn.Module):
    """Pre-norm transformer block with self-attention, optional cross-attention, and FFN.

    Args:
        dim: Feature dimensionality.
        context_dim: Dimensionality of the conditioning context. Required
            when ``use_cross_attn`` is ``True``.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        dropout: Dropout probability.
        use_cross_attn: Whether to include a cross-attention sub-layer.
    """

    def __init__(
        self,
        dim: int,
        context_dim: Optional[int] = None,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
    ) -> None:
        super().__init__()
        self.use_cross_attn = use_cross_attn

        # Self-attention sub-layer.
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, head_dim, dropout)

        # Cross-attention sub-layer (optional).
        if use_cross_attn and context_dim is not None:
            self.norm2 = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(
                dim, context_dim, num_heads, head_dim, dropout,
            )
        else:
            self.norm2 = None
            self.cross_attn = None

        # Feed-forward sub-layer.
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Apply the attention block.

        Args:
            x: Input tensor of shape ``(B, N, dim)``.
            context: Optional conditioning tensor of shape
                ``(B, M, context_dim)``.

        Returns:
            Output tensor of shape ``(B, N, dim)``.
        """
        x = x + self.self_attn(self.norm1(x))

        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)

        x = x + self.ff(self.norm3(x))
        return x


class SpatialCrossAttention(nn.Module):
    """Cross-attention adapter for 2-D spatial feature maps.

    Converts ``(B, C, H, W)`` spatial features to a sequence, applies
    cross-attention with a conditioning context, and reshapes back.

    Args:
        channels: Number of input channels.
        context_dim: Dimensionality of the conditioning context.
        num_heads: Number of attention heads.
        head_dim: Dimensionality of each attention head.
        dropout: Dropout probability.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int = 768,
        num_heads: int = 8,
        head_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(
            num_groups=32, num_channels=channels, eps=1e-6,
        )
        self.attn = CrossAttention(
            channels, context_dim, num_heads, head_dim, dropout,
        )

    def forward(
        self, x: torch.Tensor, context: torch.Tensor,
    ) -> torch.Tensor:
        """Apply spatial cross-attention.

        Args:
            x: Spatial feature tensor of shape ``(B, C, H, W)``.
            context: Conditioning tensor of shape ``(B, M, context_dim)``.

        Returns:
            Output tensor of shape ``(B, C, H, W)``.
        """
        b, c, h, w = x.shape
        residual = x

        x = self.norm(x)
        x = x.view(b, c, h * w).transpose(1, 2)  # (B, H*W, C)
        x = self.attn(x, context)
        x = x.transpose(1, 2).view(b, c, h, w)

        return x + residual
