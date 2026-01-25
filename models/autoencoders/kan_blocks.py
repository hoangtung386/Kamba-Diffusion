"""
KAN (Kolmogorov-Arnold Network) blocks for VAE
Proper implementation with B-spline basis functions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class BSpline(nn.Module):
    """
    B-spline basis function computation
    Implements proper B-spline interpolation for KAN layers
    """
    def __init__(self, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        super().__init__()
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # Create uniform grid
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)
    
    def forward(self, x):
        """
        Compute B-spline basis values
        
        Args:
            x: (batch, in_features) - Input values
        Returns:
            basis: (batch, in_features, num_basis) - B-spline basis values
        """
        # Ensure x is in grid range
        x = x.clamp(self.grid_range[0], self.grid_range[1])
        
        # Initialize with zeroth-order (piecewise constant)
        grid = self.grid
        num_basis = len(grid) - 1
        
        # Find which interval each x falls into
        # basis will be (batch, in_features, num_basis)
        x_expanded = x.unsqueeze(-1)  # (batch, in_features, 1)
        grid_expanded = grid.unsqueeze(0).unsqueeze(0)  # (1, 1, num_grid)
        
        # Order 0: indicator functions
        basis = ((x_expanded >= grid_expanded[:, :, :-1]) & 
                 (x_expanded < grid_expanded[:, :, 1:])).float()
        
        # Recursively compute higher orders using Cox-de Boor formula
        for k in range(1, self.spline_order + 1):
            # Left term
            left_num = x_expanded - grid_expanded[:, :, :-(k+1)]
            left_den = grid_expanded[:, :, k:-1] - grid_expanded[:, :, :-(k+1)]
            left_den = torch.where(left_den == 0, torch.ones_like(left_den), left_den)
            left = (left_num / left_den) * basis[:, :, :-1]
            
            # Right term  
            right_num = grid_expanded[:, :, k+1:] - x_expanded
            right_den = grid_expanded[:, :, k+1:] - grid_expanded[:, :, 1:-k]
            right_den = torch.where(right_den == 0, torch.ones_like(right_den), right_den)
            right = (right_num / right_den) * basis[:, :, 1:]
            
            # Combine
            basis = left + right
        
        return basis


class KANLinear(nn.Module):
    """
    KAN Linear layer with learnable B-spline activations
    Replaces traditional Linear + Activation with interpretable spline functions
    
    y = w_base * SiLU(x) + w_spline * Σ c_i * B_i(x)
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_base=1.0,
        scale_spline=1.0,
        grid_range=[-1, 1]
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        
        # Base activation (standard MLP path)
        self.base_weight = nn.Parameter(torch.randn(out_features, in_features))
        self.base_activation = nn.SiLU()
        self.scale_base = scale_base
        
        # Spline activation (KAN path)
        self.bspline = BSpline(grid_size, spline_order, grid_range)
        num_basis = grid_size + spline_order
        self.spline_weight = nn.Parameter(torch.randn(out_features, in_features, num_basis))
        self.scale_spline = scale_spline
        
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        nn.init.xavier_uniform_(self.spline_weight)
    
    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            out: (..., out_features)
        """
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        batch_size = x.shape[0]
        
        # Base path: standard MLP
        base_output = F.linear(self.base_activation(x), self.base_weight)
        base_output = base_output * self.scale_base
        
        # Spline path: learnable activations
        # Compute B-spline basis for each input
        basis = self.bspline(x)  # (batch, in_features, num_basis)
        
        # Apply spline weights
        # basis: (batch, in_features, num_basis)
        # spline_weight: (out_features, in_features, num_basis)
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        spline_output = spline_output * self.scale_spline
        
        # Combine both paths
        output = base_output + spline_output
        
        return output.reshape(*original_shape[:-1], self.out_features)


class KANBlock2d(nn.Module):
    """
    2D Convolutional block with KAN activation
    For use in VAE decoder upsampling
    """
    def __init__(self, channels, expansion=2, grid_size=5):
        super().__init__()
        
        # Spatial mixing: Depthwise convolution
        self.dwconv = nn.Conv2d(channels, channels, kernel_size=7, padding=3, groups=channels)
        self.norm = nn.GroupNorm(num_groups=min(32, channels), num_channels=channels, eps=1e-6)
        
        # Channel mixing: KAN instead of MLP
        hidden_dim = int(channels * expansion)
        self.kan1 = KANLinear(channels, hidden_dim, grid_size=grid_size)
        self.kan2 = KANLinear(hidden_dim, channels, grid_size=grid_size)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        residual = x
        
        # Depthwise spatial conv
        x = self.dwconv(x)
        
        # Reshape for KAN: (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        
        # KAN channel mixing
        x = self.kan1(x)
        x = self.kan2(x)
        
        # Restore shape: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        return x + residual


if __name__ == "__main__":
    # Test KAN blocks
    print("Testing KAN blocks...\n")
    
    # Test BSpline
    print("1. Testing B-Spline basis:")
    bspline = BSpline(grid_size=5, spline_order=3)
    x = torch.randn(2, 10)
    basis = bspline(x)
    print(f"   Input: {x.shape}, Basis: {basis.shape}")
    
    # Test KANLinear
    print("\n2. Testing KAN Linear:")
    kan_linear = KANLinear(10, 20, grid_size=5)
    out = kan_linear(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    
    # Test KANBlock2d
    print("\n3. Testing KAN Block 2D:")
    kan_block = KANBlock2d(channels=64)
    x_2d = torch.randn(2, 64, 16, 16)
    out_2d = kan_block(x_2d)
    print(f"   Input: {x_2d.shape}, Output: {out_2d.shape}")
    
    print("\n✅ All KAN tests passed!")
