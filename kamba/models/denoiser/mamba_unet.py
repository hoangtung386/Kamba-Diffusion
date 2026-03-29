"""Mamba U-Net denoiser with optional gradient checkpointing.

This module provides the main denoiser architecture for Kamba Diffusion,
combining residual blocks with Mamba-based attention and cross-attention
mechanisms in a U-Net structure.
"""

import logging
from typing import Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from kamba.models.blocks.attention import SpatialCrossAttention
from kamba.models.blocks.embedding import SinusoidalTimeEmbedding
from kamba.models.blocks.mamba_block import MambaVisionBlock

logger = logging.getLogger(__name__)


class ResBlock(nn.Module):
    """Residual block with optional time embedding and gradient checkpointing.

    Applies two convolution layers with group normalization, SiLU activation,
    and an optional time embedding projection between them. Supports gradient
    checkpointing to reduce memory usage during training.

    Args:
        in_channels: Number of input channels.
        out_channels: Number of output channels.
        time_emb_dim: Dimension of the time embedding. ``None`` to disable.
        dropout: Dropout probability.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: Optional[int] = None,
        dropout: float = 0.0,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.out_channels = out_channels

        # First convolution path.
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

        # Time embedding projection.
        self.time_emb_proj: Optional[nn.Linear] = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)

        # Second convolution path.
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        # Shortcut connection.
        if in_channels != out_channels:
            self.shortcut: nn.Module = nn.Conv2d(
                in_channels, out_channels, kernel_size=1
            )
        else:
            self.shortcut = nn.Identity()

        self.act = nn.SiLU()

    def _forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the forward computation without checkpointing.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            time_emb: Optional time embedding of shape ``(B, D)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.
        """
        h = self.act(self.norm1(x))
        h = self.conv1(h)

        if time_emb is not None and self.time_emb_proj is not None:
            t = self.time_emb_proj(self.act(time_emb))
            h = h + t[:, :, None, None]

        h = self.act(self.norm2(h))
        h = self.dropout(h)
        h = self.conv2(h)

        return h + self.shortcut(x)

    def forward(
        self, x: torch.Tensor, time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            time_emb: Optional time embedding of shape ``(B, D)``.

        Returns:
            Output tensor of shape ``(B, out_channels, H, W)``.
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, time_emb, use_reentrant=False)
        return self._forward(x, time_emb)


class MambaAttentionBlock(nn.Module):
    """Combined Mamba and cross-attention block.

    Applies a :class:`MambaVisionBlock` followed by an optional
    :class:`SpatialCrossAttention` layer. Supports gradient checkpointing.

    Args:
        channels: Number of input/output channels.
        context_dim: Dimension of the cross-attention context vectors.
        mamba_d_state: State dimension for the Mamba block.
        use_cross_attn: Whether to include the cross-attention layer.
        num_heads: Number of attention heads for cross-attention.
        use_checkpoint: Whether to use gradient checkpointing.
    """

    def __init__(
        self,
        channels: int,
        context_dim: int = 768,
        mamba_d_state: int = 16,
        use_cross_attn: bool = True,
        num_heads: int = 8,
        use_checkpoint: bool = False,
    ) -> None:
        super().__init__()

        self.use_cross_attn = use_cross_attn
        self.use_checkpoint = use_checkpoint

        try:
            self.mamba: nn.Module = MambaVisionBlock(
                d_model=channels, d_state=mamba_d_state
            )
        except ImportError:
            logger.warning(
                "mamba_ssm not available, using ResBlock fallback for "
                "MambaAttentionBlock."
            )
            self.mamba = ResBlock(channels, channels)

        if use_cross_attn:
            self.cross_attn = SpatialCrossAttention(
                channels=channels,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=channels // num_heads,
            )

    def _forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Run the forward computation without checkpointing.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            context: Optional context tensor of shape ``(B, M, D)``.

        Returns:
            Output tensor of shape ``(B, C, H, W)``.
        """
        x = self.mamba(x)

        if self.use_cross_attn and context is not None:
            x = self.cross_attn(x, context)

        return x

    def forward(
        self, x: torch.Tensor, context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional gradient checkpointing.

        Args:
            x: Input tensor of shape ``(B, C, H, W)``.
            context: Optional context tensor of shape ``(B, M, D)``.

        Returns:
            Output tensor of shape ``(B, C, H, W)``.
        """
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, context, use_reentrant=False)
        return self._forward(x, context)


class MambaUNet(nn.Module):
    """Mamba U-Net denoiser with gradient checkpointing support.

    A U-Net architecture that uses :class:`ResBlock` for feature extraction
    and :class:`MambaAttentionBlock` for sequence modelling and text
    conditioning. Designed for latent-space diffusion models.

    Args:
        in_channels: Number of input latent channels.
        out_channels: Number of output latent channels.
        model_channels: Base channel count.
        channel_mult: Per-level channel multipliers.
        num_res_blocks: Number of residual blocks per encoder level.
        attention_resolutions: Encoder levels that include attention blocks.
        context_dim: Dimension of the text-conditioning context.
        num_heads: Number of attention heads.
        mamba_d_state: State dimension for Mamba blocks.
        dropout: Dropout probability.
        use_cross_attn: Whether to use cross-attention.
        use_checkpoint: Global flag to enable gradient checkpointing.
        checkpoint_res_blocks: Checkpoint residual blocks when enabled.
        checkpoint_attn_blocks: Checkpoint attention blocks when enabled.
    """

    def __init__(
        self,
        in_channels: int = 4,
        out_channels: int = 4,
        model_channels: int = 320,
        channel_mult: Sequence[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        attention_resolutions: Sequence[int] = (4, 2, 1),
        context_dim: int = 768,
        num_heads: int = 8,
        mamba_d_state: int = 16,
        dropout: float = 0.0,
        use_cross_attn: bool = True,
        use_checkpoint: bool = False,
        checkpoint_res_blocks: bool = True,
        checkpoint_attn_blocks: bool = True,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = tuple(channel_mult)
        self.num_resolutions = len(self.channel_mult)
        self.use_checkpoint = use_checkpoint

        # Time embedding.
        time_emb_dim = model_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(model_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # Input projection.
        self.input_conv = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, padding=1
        )

        # --- Encoder ---
        self.encoder_blocks = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()

        ch = model_channels
        input_block_channels: list[int] = [ch]

        attention_resolutions_set = set(attention_resolutions)

        for level in range(self.num_resolutions):
            for _ in range(num_res_blocks):
                layers: list[nn.Module] = [
                    ResBlock(
                        ch,
                        model_channels * self.channel_mult[level],
                        time_emb_dim,
                        dropout,
                        use_checkpoint=checkpoint_res_blocks and use_checkpoint,
                    )
                ]
                ch = model_channels * self.channel_mult[level]

                if level in attention_resolutions_set:
                    layers.append(
                        MambaAttentionBlock(
                            ch,
                            context_dim=context_dim,
                            mamba_d_state=mamba_d_state,
                            use_cross_attn=use_cross_attn,
                            num_heads=num_heads,
                            use_checkpoint=checkpoint_attn_blocks and use_checkpoint,
                        )
                    )

                self.encoder_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)

            # Downsample (except at the last level).
            if level != self.num_resolutions - 1:
                self.encoder_downs.append(
                    nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1)
                )
                input_block_channels.append(ch)
            else:
                self.encoder_downs.append(None)  # type: ignore[arg-type]

        # --- Bottleneck ---
        self.bottleneck = nn.ModuleList(
            [
                ResBlock(ch, ch, time_emb_dim, dropout, use_checkpoint=use_checkpoint),
                MambaAttentionBlock(
                    ch,
                    context_dim=context_dim,
                    mamba_d_state=mamba_d_state,
                    use_cross_attn=use_cross_attn,
                    num_heads=num_heads,
                    use_checkpoint=use_checkpoint,
                ),
                ResBlock(ch, ch, time_emb_dim, dropout, use_checkpoint=use_checkpoint),
            ]
        )

        # --- Decoder ---
        self.decoder_blocks = nn.ModuleList()

        for level in reversed(range(self.num_resolutions)):
            for block_idx in range(num_res_blocks + 1):
                skip_ch = input_block_channels.pop()
                layers = [
                    ResBlock(
                        ch + skip_ch,
                        model_channels * self.channel_mult[level],
                        time_emb_dim,
                        dropout,
                        use_checkpoint=checkpoint_res_blocks and use_checkpoint,
                    )
                ]
                ch = model_channels * self.channel_mult[level]

                if level in attention_resolutions_set:
                    layers.append(
                        MambaAttentionBlock(
                            ch,
                            context_dim=context_dim,
                            mamba_d_state=mamba_d_state,
                            use_cross_attn=use_cross_attn,
                            num_heads=num_heads,
                            use_checkpoint=checkpoint_attn_blocks and use_checkpoint,
                        )
                    )

                # Upsample at the end of each level (except the first).
                if level != 0 and block_idx == num_res_blocks:
                    layers.append(
                        nn.Sequential(
                            nn.Upsample(scale_factor=2, mode="nearest"),
                            nn.Conv2d(ch, ch, kernel_size=3, padding=1),
                        )
                    )

                self.decoder_blocks.append(nn.ModuleList(layers))

        # --- Output ---
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

        # Zero-initialize output convolution for training stability.
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Predict noise from a noisy latent.

        Args:
            x: Noisy latent of shape ``(B, C, H, W)``.
            t: Integer timesteps of shape ``(B,)``.
            context: Optional text embeddings of shape ``(B, M, D)``.

        Returns:
            Predicted noise of shape ``(B, C, H, W)``.
        """
        # Time embedding.
        t_emb = self.time_mlp(self.time_embed(t))

        # Input projection.
        h = self.input_conv(x)

        # Encoder with skip connections.
        hs: list[torch.Tensor] = [h]
        for level in range(self.num_resolutions):
            for block_idx in range(self.num_res_blocks):
                layers = self.encoder_blocks[level * self.num_res_blocks + block_idx]
                for layer in layers:
                    if isinstance(layer, ResBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, MambaAttentionBlock):
                        h = layer(h, context)
                    else:
                        h = layer(h)
                hs.append(h)

            if self.encoder_downs[level] is not None:
                h = self.encoder_downs[level](h)
                hs.append(h)

        # Bottleneck.
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, MambaAttentionBlock):
                h = layer(h, context)
            else:
                h = layer(h)

        # Decoder with skip connections.
        for layers in self.decoder_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, MambaAttentionBlock):
                    h = layer(h, context)
                else:
                    h = layer(h)

        # Output projection.
        h = self.out_act(self.out_norm(h))
        return self.out_conv(h)

    def enable_gradient_checkpointing(self) -> None:
        """Enable gradient checkpointing for all residual and attention blocks."""
        self.use_checkpoint = True
        for module in self.modules():
            if isinstance(module, (ResBlock, MambaAttentionBlock)):
                module.use_checkpoint = True

    def disable_gradient_checkpointing(self) -> None:
        """Disable gradient checkpointing for all residual and attention blocks."""
        self.use_checkpoint = False
        for module in self.modules():
            if isinstance(module, (ResBlock, MambaAttentionBlock)):
                module.use_checkpoint = False
