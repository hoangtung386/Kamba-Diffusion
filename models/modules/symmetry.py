import torch
import torch.nn as nn

class SymmetryFusion(nn.Module):
    """
    Exploits bilateral symmetry of the brain.
    Fuses features with their horizontally flipped counterpart.
    """
    def __init__(self, channels, axis=3): # axis 3 is Width (B, C, H, W)
        super().__init__()
        self.axis = axis
        self.fusion = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        # Flip along width
        x_flip = torch.flip(x, dims=[self.axis])
        
        # Concatenate
        x_cat = torch.cat([x, x_flip], dim=1)
        
        # Fuse
        return self.fusion(x_cat)
