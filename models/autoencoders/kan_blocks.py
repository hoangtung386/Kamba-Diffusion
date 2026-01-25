"""
KAN (Kolmogorov-Arnold Network) blocks for VAE
Fixed B-spline implementation with proper gradient flow
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ImprovedBSpline(nn.Module):
    """
    Efficient B-spline basis with proper gradient handling
    Uses grid extension instead of clamping
    """
    def __init__(self, grid_size=5, spline_order=3, grid_range=[-1, 1]):
        super().__init__()
        
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range
        
        # Extended grid for boundary handling
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1
        )
        self.register_buffer('grid', grid)
        
        # Pre-compute constants
        self.num_basis = len(grid) - spline_order - 1
    
    def forward(self, x):
        """
        Vectorized B-spline computation using Cox-de Boor recursion
        
        Args:
            x: (B, in_features) - Input in range [grid_range[0], grid_range[1]]
        Returns:
            basis: (B, in_features, num_basis) - B-spline basis values
        """
        # Soft boundary handling instead of hard clamp
        # Use tanh to smoothly map outside values to boundary
        range_size = self.grid_range[1] - self.grid_range[0]
        x_normalized = (x - self.grid_range[0]) / range_size  # [0, 1]
        x_normalized = torch.tanh(x_normalized * 2 - 1) * 0.5 + 0.5  # Soft clamp
        x = x_normalized * range_size + self.grid_range[0]
        
        batch_size, in_features = x.shape
        x = x.unsqueeze(-1)  # (B, in_features, 1)
        grid = self.grid.unsqueeze(0).unsqueeze(0)  # (1, 1, num_grid)
        
        # Order 0: indicator functions (piecewise constant)
        # basis[i] = 1 if grid[i] <= x < grid[i+1], else 0
        basis = ((x >= grid[:, :, :-1]) & (x < grid[:, :, 1:])).float()
        
        # Handle right boundary (include last point)
        basis[:, :, -1] = basis[:, :, -1] + (x[:, :, 0] == grid[0, 0, -1]).float()
        
        # Cox-de Boor recursion for higher orders
        for k in range(1, self.spline_order + 1):
            # Left term coefficients
            left_num = x - grid[:, :, :-(k+1)]
            left_den = grid[:, :, k:-1] - grid[:, :, :-(k+1)]
            left_den = torch.where(
                torch.abs(left_den) < 1e-8,
                torch.ones_like(left_den),
                left_den
            )
            left_coef = left_num / left_den
            
            # Right term coefficients
            right_num = grid[:, :, k+1:] - x
            right_den = grid[:, :, k+1:] - grid[:, :, 1:-k]
            right_den = torch.where(
                torch.abs(right_den) < 1e-8,
                torch.ones_like(right_den),
                right_den
            )
            right_coef = right_num / right_den
            
            # Compute new basis
            new_basis = left_coef * basis[:, :, :-1] + right_coef * basis[:, :, 1:]
            basis = new_basis
        
        return basis


class ImprovedKANLinear(nn.Module):
    """
    Enhanced KAN Linear with better initialization and regularization
    
    Key improvements:
    - Proper weight initialization based on input/output dims
    - Layer normalization for stability
    - Optional dropout for regularization
    """
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_base=1.0,
        scale_spline=1.0,
        grid_range=[-1, 1],
        dropout=0.0,
        use_layernorm=False
    ):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.use_layernorm = use_layernorm
        
        # Optional layer normalization
        if use_layernorm:
            self.norm = nn.LayerNorm(in_features)
        
        # Base path (standard MLP)
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.base_activation = nn.SiLU()
        self.scale_base = scale_base
        
        # Spline path (KAN)
        self.bspline = ImprovedBSpline(grid_size, spline_order, grid_range)
        num_basis = self.bspline.num_basis
        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features, num_basis))
        self.scale_spline = scale_spline
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Proper initialization for KAN layers"""
        # Base weight: Kaiming init
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))
        
        # Spline weight: Small random init to start from base function
        # We want spline path to initially contribute less
        with torch.no_grad():
            std = 1.0 / math.sqrt(self.in_features * self.grid_size)
            nn.init.normal_(self.spline_weight, mean=0.0, std=std)
    
    def forward(self, x):
        """
        Args:
            x: (..., in_features)
        Returns:
            out: (..., out_features)
        """
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)
        
        # Optional normalization
        if self.use_layernorm:
            x_norm = self.norm(x)
        else:
            x_norm = x
        
        # Base path: y_base = W_base * activation(x)
        base_output = F.linear(self.base_activation(x_norm), self.base_weight)
        base_output = base_output * self.scale_base
        
        # Spline path: y_spline = sum_i c_i * B_i(x)
        basis = self.bspline(x_norm)  # (batch, in_features, num_basis)
        
        # Einstein summation for efficiency
        # basis: (B, I, K), spline_weight: (O, I, K) -> output: (B, O)
        spline_output = torch.einsum('bik,oik->bo', basis, self.spline_weight)
        spline_output = spline_output * self.scale_spline
        
        # Combine paths
        output = base_output + spline_output
        output = self.dropout(output)
        
        return output.reshape(*original_shape[:-1], self.out_features)


class ImprovedKANBlock2d(nn.Module):
    """
    Enhanced 2D KAN block for VAE decoder with better architecture
    """
    def __init__(
        self,
        channels,
        expansion=2,
        grid_size=5,
        dropout=0.0,
        use_residual=True
    ):
        super().__init__()
        
        self.use_residual = use_residual
        
        # 1. Depthwise convolution for spatial mixing
        self.dwconv = nn.Conv2d(
            channels, channels,
            kernel_size=7, padding=3,
            groups=channels,
            bias=False
        )
        
        # 2. GroupNorm for stability (better than LayerNorm for images)
        self.norm1 = nn.GroupNorm(
            num_groups=min(32, channels),
            num_channels=channels,
            eps=1e-6
        )
        
        # 3. KAN channel mixing (replaces standard MLP)
        hidden_dim = int(channels * expansion)
        self.kan1 = ImprovedKANLinear(
            channels, hidden_dim,
            grid_size=grid_size,
            dropout=dropout,
            use_layernorm=False  # Already have GroupNorm
        )
        
        self.norm2 = nn.GroupNorm(
            num_groups=min(32, hidden_dim),
            num_channels=hidden_dim,
            eps=1e-6
        )
        
        self.kan2 = ImprovedKANLinear(
            hidden_dim, channels,
            grid_size=grid_size,
            dropout=dropout,
            use_layernorm=False
        )
        
        # Layer scale for better training stability
        self.layer_scale = nn.Parameter(
            torch.ones(channels, 1, 1) * 1e-6
        )
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W)
        Returns:
            out: (B, C, H, W)
        """
        if self.use_residual:
            residual = x
        
        # Spatial mixing with depthwise conv
        x = self.dwconv(x)
        x = self.norm1(x)
        
        # Channel mixing with KAN
        # (B, C, H, W) -> (B, H, W, C)
        x = x.permute(0, 2, 3, 1)
        
        # First KAN layer
        x = self.kan1(x)
        
        # Reshape for GroupNorm: (B, H, W, C) -> (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        x = self.norm2(x)
        x = x.permute(0, 2, 3, 1)
        
        # Second KAN layer
        x = self.kan2(x)
        
        # Back to (B, C, H, W)
        x = x.permute(0, 3, 1, 2)
        
        # Layer scale and residual
        x = x * self.layer_scale
        
        if self.use_residual:
            x = x + residual
        
        return x


# ============================================
# Testing Code
# ============================================

if __name__ == "__main__":
    print("Testing Improved KAN Blocks...\n")
    
    # Test 1: B-spline with gradient flow
    print("1. Testing Improved B-Spline:")
    bspline = ImprovedBSpline(grid_size=5, spline_order=3)
    x = torch.randn(2, 10, requires_grad=True)
    
    # Test forward
    basis = bspline(x)
    print(f"   Input: {x.shape}, Basis: {basis.shape}")
    
    # Test gradient flow
    loss = basis.sum()
    loss.backward()
    print(f"   Gradient exists: {x.grad is not None}")
    print(f"   Gradient mean: {x.grad.abs().mean():.6f}")
    
    # Test 2: KAN Linear
    print("\n2. Testing Improved KAN Linear:")
    kan_linear = ImprovedKANLinear(
        10, 20,
        grid_size=5,
        dropout=0.1,
        use_layernorm=True
    )
    x = torch.randn(2, 10)
    out = kan_linear(x)
    print(f"   Input: {x.shape}, Output: {out.shape}")
    
    # Test 3: KAN Block 2D
    print("\n3. Testing Improved KAN Block 2D:")
    kan_block = ImprovedKANBlock2d(
        channels=64,
        expansion=2,
        dropout=0.1
    )
    x_2d = torch.randn(2, 64, 16, 16)
    out_2d = kan_block(x_2d)
    print(f"   Input: {x_2d.shape}, Output: {out_2d.shape}")
    
    # Test residual connection
    assert torch.allclose(out_2d.shape, x_2d.shape)
    print(f"   Residual working: {not torch.allclose(out_2d, x_2d)}")
    
    # Test 4: Parameter count
    print("\n4. Parameter Comparison:")
    
    # Standard Conv block
    conv_block = nn.Sequential(
        nn.Conv2d(64, 128, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(128, 64, 3, padding=1)
    )
    conv_params = sum(p.numel() for p in conv_block.parameters())
    
    kan_params = sum(p.numel() for p in kan_block.parameters())
    
    print(f"   Conv block params: {conv_params:,}")
    print(f"   KAN block params: {kan_params:,}")
    print(f"   Ratio: {kan_params/conv_params:.2f}x")
    
    print("\n✅ All improved KAN tests passed!")