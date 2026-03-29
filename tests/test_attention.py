"""Tests for attention modules."""

import pytest
import torch

from kamba.models.blocks.attention import (
    AttentionBlock,
    CrossAttention,
    SelfAttention,
    SpatialCrossAttention,
)


class TestCrossAttention:
    def test_output_shape(self):
        attn = CrossAttention(query_dim=64, context_dim=128, num_heads=4, head_dim=16)
        x = torch.randn(2, 10, 64)
        ctx = torch.randn(2, 20, 128)
        out = attn(x, ctx)
        assert out.shape == x.shape

    def test_gradient_flow(self):
        attn = CrossAttention(query_dim=64, context_dim=128, num_heads=4, head_dim=16)
        x = torch.randn(2, 10, 64, requires_grad=True)
        ctx = torch.randn(2, 20, 128)
        out = attn(x, ctx)
        out.sum().backward()
        assert x.grad is not None


class TestSelfAttention:
    def test_output_shape(self):
        attn = SelfAttention(dim=64, num_heads=4, head_dim=16)
        x = torch.randn(2, 10, 64)
        out = attn(x)
        assert out.shape == x.shape


class TestAttentionBlock:
    def test_with_cross_attention(self):
        block = AttentionBlock(dim=64, context_dim=128, num_heads=4)
        x = torch.randn(2, 10, 64)
        ctx = torch.randn(2, 20, 128)
        out = block(x, ctx)
        assert out.shape == x.shape

    def test_without_cross_attention(self):
        block = AttentionBlock(dim=64, use_cross_attn=False)
        x = torch.randn(2, 10, 64)
        out = block(x)
        assert out.shape == x.shape


class TestSpatialCrossAttention:
    def test_output_shape(self):
        attn = SpatialCrossAttention(channels=64, context_dim=128, num_heads=4)
        x = torch.randn(2, 64, 8, 8)
        ctx = torch.randn(2, 10, 128)
        out = attn(x, ctx)
        assert out.shape == x.shape

    def test_residual_connection(self):
        attn = SpatialCrossAttention(channels=64, context_dim=128, num_heads=4)
        x = torch.randn(2, 64, 8, 8)
        ctx = torch.randn(2, 10, 128)
        out = attn(x, ctx)
        # Output should differ from input (attention modifies it)
        assert not torch.allclose(out, x)
