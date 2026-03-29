"""Tests for KAN blocks."""

import pytest
import torch

from kamba.models.blocks.kan_blocks import BSplineBasis, KANBlock2d, KANLinear


class TestBSplineBasis:
    def test_output_shape(self):
        bspline = BSplineBasis(grid_size=5, spline_order=3)
        x = torch.randn(4, 10)
        basis = bspline(x)
        assert basis.shape[0] == 4
        assert basis.shape[1] == 10
        assert basis.shape[2] == bspline.num_basis

    def test_gradient_flow(self):
        bspline = BSplineBasis(grid_size=5, spline_order=3)
        x = torch.randn(4, 10, requires_grad=True)
        basis = bspline(x)
        basis.sum().backward()
        assert x.grad is not None
        assert x.grad.abs().mean() > 0


class TestKANLinear:
    def test_output_shape(self):
        layer = KANLinear(10, 20, grid_size=5)
        x = torch.randn(4, 10)
        out = layer(x)
        assert out.shape == (4, 20)

    def test_batched_input(self):
        layer = KANLinear(10, 20)
        x = torch.randn(4, 8, 10)
        out = layer(x)
        assert out.shape == (4, 8, 20)

    def test_gradient_flow(self):
        layer = KANLinear(10, 20)
        x = torch.randn(4, 10, requires_grad=True)
        out = layer(x)
        out.sum().backward()
        assert x.grad is not None


class TestKANBlock2d:
    def test_output_shape(self):
        block = KANBlock2d(channels=64)
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape

    def test_residual(self):
        block = KANBlock2d(channels=64, use_residual=True)
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        # Should not be identical (KAN modifies)
        assert not torch.allclose(out, x)

    def test_no_residual(self):
        block = KANBlock2d(channels=64, use_residual=False)
        x = torch.randn(2, 64, 8, 8)
        out = block(x)
        assert out.shape == x.shape
