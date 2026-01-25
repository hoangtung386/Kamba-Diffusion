"""
Cross-Attention layer for text-to-image conditioning
Allows image features to attend to text embeddings
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention layer
    
    Query: from image features
    Key, Value: from text embeddings
    """
    def __init__(
        self,
        query_dim,       # Dimension of image features
        context_dim,     # Dimension of text embeddings (768 for CLIP)
        num_heads=8,
        head_dim=64,
        dropout=0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = head_dim * num_heads
        
        # Query projection (from image)
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        
        # Key, Value projection (from text)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context, mask=None):
        """
        Args:
            x: (B, N, query_dim) - Image features (query)
            context: (B, M, context_dim) - Text embeddings (key, value)
            mask: Optional attention mask
        Returns:
            out: (B, N, query_dim) - Attended features
        """
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        q = self.to_q(x)  # (B, N, inner_dim)
        k = self.to_k(context)  # (B, M, inner_dim)
        v = self.to_v(context)  # (B, M, inner_dim)
        
        # Reshape for multi-head attention
        # (B, N, inner_dim) -> (B, num_heads, N, head_dim)
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Attention: (B, num_heads, N, M)
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention to values
        # (B, num_heads, N, M) x (B, num_heads, M, head_dim) -> (B, num_heads, N, head_dim)
        out = torch.matmul(attn, v)
        
        # Reshape back
        # (B, num_heads, N, head_dim) -> (B, N, num_heads * head_dim)
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Output projection
        out = self.to_out(out)
        
        return out


class SelfAttention(nn.Module):
    """
    Multi-head self-attention layer
    Query, Key, Value all come from the same input
    """
    def __init__(
        self,
        dim,
        num_heads=8,
        head_dim=64,
        dropout=0.0
    ):
        super().__init__()
        
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        
        inner_dim = head_dim * num_heads
        
        # QKV projection
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        
        # Output projection
        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, mask=None):
        """
        Args:
            x: (B, N, dim) - Input features
            mask: Optional attention mask
        Returns:
            out: (B, N, dim) - Attended features
        """
        batch_size = x.shape[0]
        
        # Project to Q, K, V
        qkv = self.to_qkv(x)  # (B, N, inner_dim * 3)
        qkv = qkv.view(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
        
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Attention
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))
        
        attn = F.softmax(attn, dim=-1)
        
        # Apply to values
        out = torch.matmul(attn, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)
        
        # Output projection
        out = self.to_out(out)
        
        return out


class AttentionBlock(nn.Module):
    """
    Combined attention block with LayerNorm and residual
    Includes both self-attention and cross-attention
    """
    def __init__(
        self,
        dim,
        context_dim=None,
        num_heads=8,
        head_dim=64,
        dropout=0.0,
        use_cross_attn=True
    ):
        super().__init__()
        
        self.use_cross_attn = use_cross_attn
        
        # Self-attention
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, head_dim, dropout)
        
        # Cross-attention (optional)
        if use_cross_attn and context_dim is not None:
            self.norm2 = nn.LayerNorm(dim)
            self.cross_attn = CrossAttention(dim, context_dim, num_heads, head_dim, dropout)
        else:
            self.norm2 = None
            self.cross_attn = None
        
        # Feedforward
        self.norm3 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, context=None):
        """
        Args:
            x: (B, N, dim) - Input features
            context: (B, M, context_dim) - Optional text context
        Returns:
            out: (B, N, dim) - Output features
        """
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention
        if self.cross_attn is not None and context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        
        # Feedforward
        x = x + self.ff(self.norm3(x))
        
        return x


class SpatialCrossAttention(nn.Module):
    """
    Cross-attention for 2D spatial features (used in U-Net)
    Converts (B, C, H, W) -> (B, H*W, C) for attention -> (B, C, H, W)
    """
    def __init__(
        self,
        channels,
        context_dim=768,
        num_heads=8,
        head_dim=64,
        dropout=0.0
    ):
        super().__init__()
        
        self.channels = channels
        
        # Group norm
        self.norm = nn.GroupNorm(num_groups=32, num_channels=channels, eps=1e-6)
        
        # Cross-attention
        self.attn = CrossAttention(channels, context_dim, num_heads, head_dim, dropout)
    
    def forward(self, x, context):
        """
        Args:
            x: (B, C, H, W) - Spatial features
            context: (B, M, context_dim) - Text embeddings
        Returns:
            out: (B, C, H, W) - Attended features
        """
        b, c, h, w = x.shape
        residual = x
        
        # Normalize
        x = self.norm(x)
        
        # Reshape to sequence: (B, C, H, W) -> (B, H*W, C)
        x = x.view(b, c, h * w).transpose(1, 2)
        
        # Cross-attention
        x = self.attn(x, context)
        
        # Reshape back: (B, H*W, C) -> (B, C, H, W)
        x = x.transpose(1, 2).view(b, c, h, w)
        
        return x + residual


if __name__ == "__main__":
    # Test attention modules
    print("\n🧪 Testing Attention Modules...\n")
    
    # Test CrossAttention
    print("Testing CrossAttention...")
    cross_attn = CrossAttention(query_dim=512, context_dim=768, num_heads=8, head_dim=64)
    
    x = torch.randn(2, 256, 512)  # Image features (batch, spatial, channels)
    context = torch.randn(2, 77, 768)  # Text embeddings (batch, tokens, dim)
    
    out = cross_attn(x, context)
    print(f"Input: {x.shape}, Context: {context.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    
    # Test SelfAttention
    print("\nTesting SelfAttention...")
    self_attn = SelfAttention(dim=512, num_heads=8, head_dim=64)
    
    out = self_attn(x)
    print(f"Input: {x.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    
    # Test AttentionBlock
    print("\nTesting AttentionBlock...")
    attn_block = AttentionBlock(dim=512, context_dim=768, num_heads=8, use_cross_attn=True)
    
    out = attn_block(x, context)
    print(f"Input: {x.shape}, Context: {context.shape} -> Output: {out.shape}")
    assert out.shape == x.shape
    
    # Test SpatialCrossAttention
    print("\nTesting SpatialCrossAttention...")
    spatial_attn = SpatialCrossAttention(channels=512, context_dim=768, num_heads=8)
    
    x_spatial = torch.randn(2, 512, 16, 16)  # (B, C, H, W)
    out = spatial_attn(x_spatial, context)
    print(f"Input: {x_spatial.shape}, Context: {context.shape} -> Output: {out.shape}")
    assert out.shape == x_spatial.shape
    
    print("\n✅ All attention tests passed!")
