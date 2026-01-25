import torch
import torch.nn as nn
from utils.registry import BACKBONE_REGISTRY, DECODER_REGISTRY

# Note: Registry for "bottleneck" is not defined separately, usually reused 
# or handled in model logic. But we'll adding a class here.

class TransformerBlock(nn.Module):
    def __init__(self, channels=1024, num_heads=8, num_layers=4):
        super().__init__()
        # Flatten -> Transformer -> Reshape
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=channels, 
            nhead=num_heads,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=num_layers)
        
    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_flat = x.flatten(2).transpose(1, 2) # (B, H*W, C)
        
        x_out = self.transformer_encoder(x_flat)
        
        x_out = x_out.transpose(1, 2).reshape(B, C, H, W)
        return x_out

# Register alias if desired, or just import
