"""Tests for Mamba U-Net denoiser."""

import pytest
import torch

from kamba.models.denoiser.mamba_unet import MambaAttentionBlock, MambaUNet, ResBlock


class TestResBlock:
    """Tests for ResBlock with time embedding."""

    def test_same_channels(self):
        block = ResBlock(64, 64)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == x.shape

    def test_different_channels(self):
        block = ResBlock(64, 128)
        x = torch.randn(2, 64, 16, 16)
        out = block(x)
        assert out.shape == (2, 128, 16, 16)

    def test_with_time_embedding(self):
        block = ResBlock(64, 64, time_emb_dim=256)
        x = torch.randn(2, 64, 16, 16)
        t_emb = torch.randn(2, 256)
        out = block(x, t_emb)
        assert out.shape == x.shape

    def test_gradient_checkpointing(self):
        block = ResBlock(64, 64, use_checkpoint=True)
        block.train()
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestMambaUNet:
    """Tests for MambaUNet."""

    @pytest.fixture
    def small_unet(self):
        return MambaUNet(
            in_channels=4,
            out_channels=4,
            model_channels=32,
            channel_mult=(1, 2),
            num_res_blocks=1,
            attention_resolutions=(1,),
            context_dim=64,
            num_heads=4,
            use_cross_attn=False,  # Skip Mamba dependency
        )

    def test_output_shape(self, small_unet):
        x = torch.randn(2, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        out = small_unet(x, t)
        assert out.shape == x.shape

    def test_with_context(self, small_unet):
        x = torch.randn(2, 4, 16, 16)
        t = torch.randint(0, 1000, (2,))
        ctx = torch.randn(2, 10, 64)
        out = small_unet(x, t, ctx)
        assert out.shape == x.shape

    def test_gradient_flow(self, small_unet):
        x = torch.randn(2, 4, 16, 16, requires_grad=True)
        t = torch.randint(0, 1000, (2,))
        out = small_unet(x, t)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None

    def test_parameter_count(self, small_unet):
        total = sum(p.numel() for p in small_unet.parameters())
        assert total > 0

    def test_enable_disable_checkpointing(self, small_unet):
        small_unet.enable_gradient_checkpointing()
        assert small_unet.use_checkpoint is True
        small_unet.disable_gradient_checkpointing()
        assert small_unet.use_checkpoint is False

    def test_zero_init_output(self, small_unet):
        """Output conv should be zero-initialized."""
        assert torch.all(small_unet.out_conv.weight == 0)
        assert torch.all(small_unet.out_conv.bias == 0)
