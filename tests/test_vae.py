"""Tests for VAE model."""

import pytest
import torch

from kamba.models.vae import VAE
from kamba.models.vae.encoder import Encoder, ResBlock
from kamba.models.vae.decoder import KANDecoder


class TestResBlock:
    """Tests for ResBlock."""

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

    def test_gradient_flow(self):
        block = ResBlock(64, 64)
        x = torch.randn(2, 64, 16, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None


class TestEncoder:
    """Tests for VAE Encoder."""

    def test_output_shape(self):
        encoder = Encoder(
            in_channels=3,
            hidden_dims=(64, 128),
            latent_channels=4,
        )
        x = torch.randn(2, 3, 64, 64)
        mean, logvar = encoder(x)
        # 2 downsamples: 64 -> 32 -> 16
        assert mean.shape == (2, 4, 16, 16)
        assert logvar.shape == (2, 4, 16, 16)


class TestKANDecoder:
    """Tests for KAN-based Decoder."""

    def test_output_shape_no_kan(self):
        decoder = KANDecoder(
            latent_channels=4,
            hidden_dims=(128, 64),
            out_channels=3,
            use_kan=False,
        )
        z = torch.randn(2, 4, 16, 16)
        out = decoder(z)
        # 1 upsample: 16 -> 32
        assert out.shape == (2, 3, 32, 32)


class TestVAE:
    """Tests for full VAE."""

    @pytest.fixture
    def vae(self):
        return VAE(
            in_channels=3,
            latent_channels=4,
            hidden_dims=(64, 128),
            image_size=64,
            use_kan_decoder=False,
        )

    def test_encode(self, vae):
        x = torch.randn(2, 3, 64, 64)
        mean, logvar = vae.encode(x)
        assert mean.shape[1] == 4
        assert logvar.shape[1] == 4

    def test_decode(self, vae):
        z = torch.randn(2, 4, 8, 8)
        img = vae.decode(z)
        assert img.shape[1] == 3

    def test_forward(self, vae):
        x = torch.randn(2, 3, 64, 64)
        recon, mean, logvar = vae(x)
        assert recon.shape == x.shape
        assert mean.shape[1] == 4
        assert logvar.shape[1] == 4

    def test_reparameterize(self, vae):
        mean = torch.zeros(2, 4, 8, 8)
        logvar = torch.zeros(2, 4, 8, 8)
        z = vae.reparameterize(mean, logvar)
        assert z.shape == mean.shape

    def test_sample(self, vae):
        samples = vae.sample(2, torch.device("cpu"))
        assert samples.shape[0] == 2
        assert samples.shape[1] == 3

    def test_gradient_flow(self, vae):
        x = torch.randn(2, 3, 64, 64, requires_grad=True)
        recon, mean, logvar = vae(x)
        loss = recon.sum() + mean.sum() + logvar.sum()
        loss.backward()
        assert x.grad is not None
