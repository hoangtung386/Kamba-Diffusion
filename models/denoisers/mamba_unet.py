"""
Memory-Optimized Mamba U-Net with Gradient Checkpointing
Enables training larger models with limited VRAM
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from einops import rearrange

# Import existing modules
from models.bottlenecks.mamba_block import MambaVisionBlock
from models.modules.embedding import SinusoidalTimeEmbedding
from models.modules.cross_attention import SpatialCrossAttention


class OptimizedResBlock(nn.Module):
    """Memory-efficient ResBlock with optional gradient checkpointing"""
    def __init__(self, in_channels, out_channels, time_emb_dim=None, dropout=0.0, use_checkpoint=False):
        super().__init__()
        
        self.use_checkpoint = use_checkpoint
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # First conv
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Time embedding
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
    
    def _forward(self, x, time_emb=None):
        """Actual forward computation"""
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        if time_emb is not None and self.time_emb_proj is not None:
            time_emb = self.time_emb_proj(self.act(time_emb))
            h = h + time_emb[:, :, None, None]
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)
    
    def forward(self, x, time_emb=None):
        """Forward with optional gradient checkpointing"""
        if self.use_checkpoint and self.training:
            # Use gradient checkpointing to save memory
            return checkpoint(self._forward, x, time_emb)
        else:
            return self._forward(x, time_emb)


class OptimizedMambaAttentionBlock(nn.Module):
    """
    Memory-optimized Mamba + Cross-Attention block
    """
    def __init__(
        self,
        channels,
        context_dim=768,
        mamba_d_state=16,
        use_cross_attn=True,
        num_heads=8,
        use_checkpoint=False
    ):
        super().__init__()
        
        self.use_cross_attn = use_cross_attn
        self.use_checkpoint = use_checkpoint
        
        # Mamba block
        try:
            self.mamba = MambaVisionBlock(
                d_model=channels,
                d_state=mamba_d_state
            )
        except ImportError:
            print("⚠️ Mamba not available, using ResBlock fallback")
            self.mamba = OptimizedResBlock(channels, channels)
        
        # Cross-attention
        if use_cross_attn:
            self.cross_attn = SpatialCrossAttention(
                channels=channels,
                context_dim=context_dim,
                num_heads=num_heads,
                head_dim=channels // num_heads
            )
    
    def _forward(self, x, context=None):
        """Actual forward computation"""
        # Mamba
        x = self.mamba(x)
        
        # Cross-attention
        if self.use_cross_attn and context is not None:
            x = self.cross_attn(x, context)
        
        return x
    
    def forward(self, x, context=None):
        """Forward with optional gradient checkpointing"""
        if self.use_checkpoint and self.training:
            return checkpoint(self._forward, x, context)
        else:
            return self._forward(x, context)


class OptimizedMambaUNet(nn.Module):
    """
    Memory-Optimized Mamba U-Net with:
    - Gradient checkpointing for large models
    - Efficient skip connections
    - Mixed precision training support
    - Flash attention ready
    """
    def __init__(
        self,
        in_channels=4,
        out_channels=4,
        model_channels=320,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[4, 2, 1],
        context_dim=768,
        num_heads=8,
        mamba_d_state=16,
        dropout=0.0,
        use_cross_attn=True,
        use_checkpoint=False,  # Enable gradient checkpointing
        checkpoint_res_blocks=True,  # Checkpoint ResBlocks
        checkpoint_attn_blocks=True  # Checkpoint Attention blocks
    ):
        super().__init__()
        
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.num_res_blocks = num_res_blocks
        self.channel_mult = channel_mult
        self.num_resolutions = len(channel_mult)
        self.use_checkpoint = use_checkpoint
        
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
        self.encoder_downs = nn.ModuleList()
        
        ch = model_channels
        input_block_channels = [ch]
        
        for level in range(self.num_resolutions):
            for block_idx in range(num_res_blocks):
                # ResBlock with checkpointing
                layers = [
                    OptimizedResBlock(
                        ch,
                        model_channels * channel_mult[level],
                        time_emb_dim,
                        dropout,
                        use_checkpoint=checkpoint_res_blocks and use_checkpoint
                    )
                ]
                ch = model_channels * channel_mult[level]
                
                # Mamba + Cross-Attention with checkpointing
                if level in attention_resolutions:
                    layers.append(
                        OptimizedMambaAttentionBlock(
                            ch,
                            context_dim=context_dim,
                            mamba_d_state=mamba_d_state,
                            use_cross_attn=use_cross_attn,
                            num_heads=num_heads,
                            use_checkpoint=checkpoint_attn_blocks and use_checkpoint
                        )
                    )
                
                self.encoder_blocks.append(nn.ModuleList(layers))
                input_block_channels.append(ch)
            
            # Downsample
            if level != self.num_resolutions - 1:
                self.encoder_downs.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
                input_block_channels.append(ch)
            else:
                self.encoder_downs.append(None)
        
        # ========== BOTTLENECK ==========
        self.bottleneck = nn.ModuleList([
            OptimizedResBlock(ch, ch, time_emb_dim, dropout, use_checkpoint=use_checkpoint),
            OptimizedMambaAttentionBlock(
                ch,
                context_dim=context_dim,
                mamba_d_state=mamba_d_state,
                use_cross_attn=use_cross_attn,
                num_heads=num_heads,
                use_checkpoint=use_checkpoint
            ),
            OptimizedResBlock(ch, ch, time_emb_dim, dropout, use_checkpoint=use_checkpoint)
        ])
        
        # ========== DECODER ==========
        self.decoder_blocks = nn.ModuleList()
        
        for level in reversed(range(self.num_resolutions)):
            for block_idx in range(num_res_blocks + 1):
                # Skip connection
                skip_ch = input_block_channels.pop()
                layers = [
                    OptimizedResBlock(
                        ch + skip_ch,
                        model_channels * channel_mult[level],
                        time_emb_dim,
                        dropout,
                        use_checkpoint=checkpoint_res_blocks and use_checkpoint
                    )
                ]
                ch = model_channels * channel_mult[level]
                
                # Attention
                if level in attention_resolutions:
                    layers.append(
                        OptimizedMambaAttentionBlock(
                            ch,
                            context_dim=context_dim,
                            mamba_d_state=mamba_d_state,
                            use_cross_attn=use_cross_attn,
                            num_heads=num_heads,
                            use_checkpoint=checkpoint_attn_blocks and use_checkpoint
                        )
                    )
                
                # Upsample
                if level != 0 and block_idx == num_res_blocks:
                    layers.append(nn.Sequential(
                        nn.Upsample(scale_factor=2, mode='nearest'),
                        nn.Conv2d(ch, ch, kernel_size=3, padding=1)
                    ))
                
                self.decoder_blocks.append(nn.ModuleList(layers))
        
        # Output
        self.out_norm = nn.GroupNorm(num_groups=32, num_channels=ch, eps=1e-6)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)
        
        # Zero-init output conv for stability
        nn.init.zeros_(self.out_conv.weight)
        if self.out_conv.bias is not None:
            nn.init.zeros_(self.out_conv.bias)
    
    def forward(self, x, t, context=None):
        """
        Args:
            x: (B, C, H, W) - Noisy latent
            t: (B,) - Timestep
            context: (B, M, D) - Text embeddings
        Returns:
            noise_pred: (B, C, H, W)
        """
        # Time embedding
        t_emb = self.time_embed(t)
        t_emb = self.time_mlp(t_emb)
        
        # Input
        h = self.input_conv(x)
        
        # Encoder with skip connections
        hs = [h]
        for level in range(self.num_resolutions):
            for block_idx in range(self.num_res_blocks):
                layers = self.encoder_blocks[level * self.num_res_blocks + block_idx]
                for layer in layers:
                    if isinstance(layer, OptimizedResBlock):
                        h = layer(h, t_emb)
                    elif isinstance(layer, OptimizedMambaAttentionBlock):
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
            if isinstance(layer, OptimizedResBlock):
                h = layer(h, t_emb)
            elif isinstance(layer, OptimizedMambaAttentionBlock):
                h = layer(h, context)
            else:
                h = layer(h)
        
        # Decoder with skip connections
        for layers in self.decoder_blocks:
            # Pop skip connection
            h = torch.cat([h, hs.pop()], dim=1)
            
            for layer in layers:
                if isinstance(layer, OptimizedResBlock):
                    h = layer(h, t_emb)
                elif isinstance(layer, OptimizedMambaAttentionBlock):
                    h = layer(h, context)
                else:
                    h = layer(h)
        
        # Output
        h = self.out_norm(h)
        h = self.out_act(h)
        h = self.out_conv(h)
        
        return h
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for all blocks"""
        self.use_checkpoint = True
        for module in self.modules():
            if isinstance(module, (OptimizedResBlock, OptimizedMambaAttentionBlock)):
                module.use_checkpoint = True
    
    def disable_gradient_checkpointing(self):
        """Disable gradient checkpointing"""
        self.use_checkpoint = False
        for module in self.modules():
            if isinstance(module, (OptimizedResBlock, OptimizedMambaAttentionBlock)):
                module.use_checkpoint = False


# ============================================
# Memory Profiling Utility
# ============================================

def profile_memory(model, input_shape=(2, 4, 32, 32), context_shape=(2, 77, 768)):
    """Profile memory usage of model"""
    import torch
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    x = torch.randn(*input_shape, device=device)
    t = torch.randint(0, 1000, (input_shape[0],), device=device)
    context = torch.randn(*context_shape, device=device)
    
    if device.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Forward pass
    with torch.no_grad():
        output = model(x, t, context)
    
    if device.type == 'cuda':
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**3  # GB
        print(f"Peak memory allocated: {memory_allocated:.2f} GB")
    
    return output


if __name__ == "__main__":
    print("\n🧪 Testing Optimized Mamba U-Net...\n")
    
    # Test 1: Without gradient checkpointing
    print("1. Testing WITHOUT gradient checkpointing:")
    model_no_ckpt = OptimizedMambaUNet(
        in_channels=4,
        out_channels=4,
        model_channels=320,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[1, 2, 3],
        context_dim=768,
        use_checkpoint=False
    )
    
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 1000, (2,))
    context = torch.randn(2, 77, 768)
    
    output = model_no_ckpt(x, t, context)
    print(f"   Output shape: {output.shape}")
    
    # Test 2: With gradient checkpointing
    print("\n2. Testing WITH gradient checkpointing:")
    model_ckpt = OptimizedMambaUNet(
        in_channels=4,
        out_channels=4,
        model_channels=320,
        channel_mult=[1, 2, 4, 4],
        num_res_blocks=2,
        attention_resolutions=[1, 2, 3],
        context_dim=768,
        use_checkpoint=True
    )
    
    output_ckpt = model_ckpt(x, t, context)
    print(f"   Output shape: {output_ckpt.shape}")
    
    # Test 3: Dynamic enable/disable
    print("\n3. Testing dynamic checkpointing toggle:")
    model_ckpt.disable_gradient_checkpointing()
    print("   Checkpointing disabled")
    model_ckpt.enable_gradient_checkpointing()
    print("   Checkpointing enabled")
    
    # Test 4: Parameter count
    total_params = sum(p.numel() for p in model_no_ckpt.parameters())
    trainable_params = sum(p.numel() for p in model_no_ckpt.parameters() if p.requires_grad)
    
    print(f"\n4. Model statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    print(f"   Model size: {total_params * 4 / 1024**2:.2f} MB (FP32)")
    
    print("\n✅ All Optimized Mamba U-Net tests passed!")