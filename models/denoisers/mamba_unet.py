"""
Mamba-based U-Net Denoiser for Latent Diffusion Model
Uses Mamba SSM blocks instead of Transformer for linear complexity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

# Import existing modules
from models.bottlenecks.mamba_block import MambaVisionBlock
from models.modules.embedding import SinusoidalTimeEmbedding
from models.modules.cross_attention import SpatialCrossAttention


class ResBlock(nn.Module):
    """Residual Block with time embedding injection"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding projection (optional)
        self.time_emb_proj = None
        if time_emb_dim is not None:
            self.time_emb_proj = nn.Linear(time_emb_dim, out_channels)
        
        # Second conv
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        # Shortcut
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x, time_emb=None):
        """
        Args:
            x: (B, C, H, W)
            time_emb: (B, time_emb_dim) - Optional time embedding
        Returns:
            out: (B, C, H, W)
        """
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        # Add time embedding
        if time_emb is not None and self.time_emb_proj is not None:
            time_emb = self.time_emb_proj(self.act(time_emb))
            h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Downsample(nn.Module):
    """Downsampling layer (2x)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)


class Upsample(nn.Module):
    """Upsampling layer (2x)"""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        return self.conv(x)


class MambaAttentionBlock(nn.Module):
    """
    Combined Mamba + Cross-Attention block
    Novel architecture: Mamba for spatial modeling + Cross-Attn for text conditioning
    """
    def __init__(
        self,
        channels,
        context_dim=768,
        mamba_d_state=16,
        use_cross_attn=True,
        num_heads=8
    ):
        super().__init__()
        
        self.use_cross_attn = use_cross_attn
        
        # Mamba block for spatial processing (linear complexity!)
        try:
            self.mamba = MambaVisionBlock(
                d_model=channels,
                d_state=mamba_d_state
            )
        except ImportError:
            print("⚠️ Mamba not available, using ResBlock fallback")
            self.mamba = ResBlock(channels, channels)
        
        # Cross-attention for text conditioning
        if use_cross_attn:
            self.cross_attn = SpatialCrossAttention(
                channels=channels,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=channels // num_heads
            )
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, C, H, W) - Spatial features
            context: (B, M, context_dim) - Text embeddings
        Returns:
            out: (B, C, H, W)
        """
        # Mamba (spatial processing)
        x = self.mamba(x)
        
        # Cross-attention (text conditioning)
        if self.use_cross_attn and context is not None:
            x = self.cross_attn(x, context)
        
        return x


class MambaUNet(nn.Module):
    """
    U-Net with Mamba blocks + Cross-Attention for text-conditional diffusion
    
    Architecture:
        - Input: Noisy latent (B, 4, 32, 32) + timestep + text context
        - Encoder: 4 levels with Mamba + Cross-Attention
        - Bottleneck: Pure Mamba blocks
        - Decoder: 4 levels with skip connections
        - Output: Predicted noise or denoised latent
    
    Novel Features:
        - Mamba SSM instead of Transformer (O(N) vs O(N²))
        - Text conditioning via cross-attention
        - Hierarchical skip connections
    """
    def __init__(
        self,
        in_channels=4,           # Latent channels
        out_channels=4,
        model_channels=320,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],  # Which levels get Mamba+Attn
        context_dim=768,         # CLIP embedding dim
        num_heads=8,
        mamba_d_state=16,
        dropout=0.0,
        use_cross_attn=True
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_resolutions = len(channel_mult)
        
        # Time embedding
        time_emb_dim = model_channels * 4
        self.time_embed = SinusoidalTimeEmbedding(model_channels)
        self.time_mlp = nn.Sequential(
            nn.Linear(model_channels, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )
        
        # Input conv
        self.input_conv = nn.Conv2d(in_channels, model_channels, kernel_size=3, padding=1)
        
        # ========== ENCODER ==========
        self.encoder_blocks = nn.ModuleList()
        self.encoder_mamba_attn = nn.ModuleList()
        self.encoder_downs = nn.ModuleList()
        
        ch = model_channels
        input_block_channels = [ch]
        
        for level in range(self.num_resolutions):
            for block_idx in range(num_res_blocks):
                # ResBlock
                layers = [ResBlock(ch, model_channels * channel_mult[level], time_emb_dim, dropout)]
                ch = model_channels * channel_mult[level]
                
                # Mamba + Cross-Attention
                if level in attention_resolutions:
                    layers.append(
                        MambaAttentionBlock(
                            ch,
                            context_dim=context_dim,
                            mamba_d_state=mamba_d_state,
                            use_cross_attn=use_cross_attn,
                            num_heads=num_heads
                        )
                    )
                
                self.encoder_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)
            
            # Downsample (except last level)
            if level != self.num_resolutions - 1:
                self.encoder_downs.append(Downsample(ch))
                input_block_channels.append(ch)
            else:
                self.encoder_downs.append(None)
        
        # ========== BOTTLENECK ==========
        # Pure Mamba blocks for global context
        self.bottleneck = nn.ModuleList([
            ResBlock(ch, ch, time_emb_dim, dropout),
            MambaAttentionBlock(
                ch,
                context_dim=context_dim,
                mamba_d_state=mamba_d_state,
                use_cross_attn=use_cross_attn,
                num_heads=num_heads
            ),
            ResBlock(ch, ch, time_emb_dim, dropout)
        ])
        
        # ========== DECODER ==========
        self.decoder_blocks = nn.ModuleList()
        self.decoder_ups = nn.ModuleList()
        
        for level in reversed(range(self.num_resolutions)):
            for block_idx in range(num_res_blocks + 1):
                # Concatenate with skip connection
                skip_ch = input_block_channels.pop()
                layers = [ResBlock(ch + skip_ch, model_channels * channel_mult[level], time_emb_dim, dropout)]
                ch = model_channels * channel_mult[level]
                
                # Mamba + Cross-Attention
                if level in attention_resolutions:
                    layers.append(
                        MambaAttentionBlock(
                            ch,
                            context_dim=context_dim,
                            mamba_d_state=mamba_d_state,
                            use_cross_attn=use_cross_attn,
                            num_heads=num_heads
                        )
                    )
                
                # Upsample (except last)
                if level != 0 and block_idx == num_res_blocks:
                    layers.append(Upsample(ch))
                
                self.decoder_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x, t, context=None):
        """
        Args:
            x: (B, 4, H, W) - Noisy latent
            t: (B,) - Timestep
            context: (B, 77, 768) - Text embeddings from CLIP
        Returns:
            noise_pred: (B, 4, H, W) - Predicted noise
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder
        hs = [h]
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
            
            # Downsample
            if self.encoder_downs[level] is not None:
                h = self.encoder_downs[level](h)
                hs.append(h)
        
        # Bottleneck
        for layer in self.bottleneck:
            if isinstance(layer, ResBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, MambaAttentionBlock):
                h = layer(h, context)
            else:
                h = layer(h)
        
        # Decoder
        for layers in self.decoder_blocks:
            # Skip connection
            h = torch.cat([h, hs.pop()], dim=1)
            
            for layer in layers:
                if isinstance(layer, ResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, MambaAttentionBlock):
                    h = layer(h, context)
                elif isinstance(layer, (Upsample, Downsample)):
                    h = layer(h)
                else:
                    h = layer(h)
        
        # Output
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        
        return h


if __name__ == "__main__":
    # Test Mamba U-Net
    print("\n🧪 Testing Mamba U-Net Denoiser...\n")
    
    model = MambaUNet(
        in_channels=4,
        out_channels=4,
        model_channels=320,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[1, 2, 3],
        context_dim=768,
        use_cross_attn=True
    )
    
    # Test inputs
    x = torch.randn(2, 4, 32, 32)  # Noisy latent
    t = torch.randint(0, 1000, (2,))  # Timestep
    context = torch.randn(2, 77, 768)  # Text embeddings
    
    print(f"Input latent: {x.shape}")
    print(f"Timestep: {t.shape}")
    print(f"Text context: {context.shape}")
    
    # Forward pass
    noise_pred = model(x, t, context)
    
    print(f"Output noise prediction: {noise_pred.shape}")
    assert noise_pred.shape == x.shape
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    print("\n✅ Mamba U-Net tests passed!")
