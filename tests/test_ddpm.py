"""Tests for DDPM diffusion model."""

import pytest
import torch
import torch.nn as nn

from kamba.models.diffusion.ddpm import DDPM, cosine_beta_schedule, linear_beta_schedule


class DummyDenoiser(nn.Module):
    """Simple denoiser for testing."""

    def __init__(self, channels: int = 4):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.conv(x)


class TestBetaSchedules:
    """Tests for noise schedules."""

    def test_linear_schedule_shape(self):
        betas = linear_beta_schedule(100)
        assert betas.shape == (100,)

    def test_linear_schedule_range(self):
        betas = linear_beta_schedule(1000)
        assert betas.min() > 0
        assert betas.max() < 1

    def test_cosine_schedule_shape(self):
        betas = cosine_beta_schedule(100)
        assert betas.shape == (100,)

    def test_cosine_schedule_range(self):
        betas = cosine_beta_schedule(1000)
        assert betas.min() >= 0.0001
        assert betas.max() <= 0.9999


class TestDDPM:
    """Tests for DDPM."""

    @pytest.fixture
    def ddpm(self):
        model = DummyDenoiser(4)
        return DDPM(model, timesteps=100, beta_schedule="linear")

    def test_q_sample_shape(self, ddpm):
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 100, (2,))
        noisy = ddpm.q_sample(x, t)
        assert noisy.shape == x.shape

    def test_q_sample_adds_noise(self, ddpm):
        x = torch.randn(2, 4, 8, 8)
        t = torch.tensor([50, 50])
        noisy = ddpm.q_sample(x, t)
        assert not torch.allclose(noisy, x)

    def test_p_losses_returns_scalar(self, ddpm):
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 100, (2,))
        loss = ddpm.p_losses(x, t)
        assert loss.dim() == 0
        assert loss.requires_grad

    def test_p_losses_epsilon_prediction(self):
        model = DummyDenoiser(4)
        ddpm = DDPM(model, timesteps=100, prediction_type="epsilon")
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 100, (2,))
        loss = ddpm.p_losses(x, t)
        assert loss.item() > 0

    def test_p_losses_v_prediction(self):
        model = DummyDenoiser(4)
        ddpm = DDPM(model, timesteps=100, prediction_type="v")
        x = torch.randn(2, 4, 8, 8)
        t = torch.randint(0, 100, (2,))
        loss = ddpm.p_losses(x, t)
        assert loss.item() > 0

    def test_snr_weights(self, ddpm):
        t_early = torch.tensor([5])
        t_late = torch.tensor([90])
        w_early = ddpm.get_loss_weights(t_early)
        w_late = ddpm.get_loss_weights(t_late)
        assert w_early.shape == (1, 1, 1, 1)
        assert w_late.shape == (1, 1, 1, 1)

    def test_sample(self, ddpm):
        samples = ddpm.sample((1, 4, 8, 8))
        assert samples.shape == (1, 4, 8, 8)

    def test_extract(self, ddpm):
        a = torch.randn(100)
        t = torch.tensor([10, 50])
        out = ddpm._extract(a, t, (2, 4, 8, 8))
        assert out.shape == (2, 1, 1, 1)
