import torch
import torch.nn as nn
from einops import rearrange
try:
    from mamba_ssm import Mamba
except ImportError:
    Mamba = None

from utils.registry import BACKBONE_REGISTRY

class LayerNorm2d(nn.LayerNorm):
    """ LayerNorm for Channel First mode (B, C, H, W) """
    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c')
        x = super().forward(x)
        x = rearrange(x, 'b h w c -> b c h w')
        return x

@BACKBONE_REGISTRY.register('mamba_block')
class MambaVisionBlock(nn.Module):
    """
    Vision Mamba Block for Bottleneck or Encoder/Decoder
    Adapts 2D images for 1D Mamba processing (Flatten -> Mamba -> Unflatten)
    """
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        if Mamba is None:
            raise ImportError("mamba_ssm is not installed. Please install it to use MambaVisionBlock.")
            
        self.d_model = d_model
        
        # Norm before Mamba
        self.norm = LayerNorm2d(d_model)
        
        # Mamba SSM
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand
        )
        
        # Optional: Depthwise conv for local features before/after? 
        # For pure Mamba block, we often just rely on Mamba's internal conv.
        
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        B, C, H, W = x.shape
        assert C == self.d_model, f"Input channels {C} != d_model {self.d_model}"
        
        residual = x
        
        # 1. Norm
        x_norm = self.norm(x)
        
        # 2. Flatten (B, C, H, W) -> (B, L, C) where L = H*W
        # Mamba expects (B, L, D)
        x_flat = rearrange(x_norm, 'b c h w -> b (h w) c')
        
        # 3. Mamba Forward
        x_mamba = self.mamba(x_flat)
        
        # 4. Unflatten (B, L, C) -> (B, C, H, W)
        out = rearrange(x_mamba, 'b (h w) c -> b c h w', h=H, w=W)
        
        # Residual connection
        return out + residual

# Example of a stage containing multiple blocks
class MambaStage(nn.Module):
    def __init__(self, dim, depth, **kwargs):
        super().__init__()
        self.blocks = nn.ModuleList([
            MambaVisionBlock(d_model=dim, **kwargs) for _ in range(depth)
        ])
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x
