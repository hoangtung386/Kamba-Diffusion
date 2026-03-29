"""Shared test fixtures for Kamba Diffusion."""

import pytest
import torch


@pytest.fixture
def device() -> torch.device:
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def batch_size() -> int:
    return 2


@pytest.fixture
def image_size() -> int:
    return 64


@pytest.fixture
def latent_channels() -> int:
    return 4


@pytest.fixture
def latent_size(image_size: int) -> int:
    return image_size // 8


@pytest.fixture
def dummy_images(batch_size: int, image_size: int) -> torch.Tensor:
    """Create dummy image batch."""
    return torch.randn(batch_size, 3, image_size, image_size)


@pytest.fixture
def dummy_latents(batch_size: int, latent_channels: int, latent_size: int) -> torch.Tensor:
    """Create dummy latent batch."""
    return torch.randn(batch_size, latent_channels, latent_size, latent_size)


@pytest.fixture
def dummy_timesteps(batch_size: int) -> torch.Tensor:
    """Create dummy timestep batch."""
    return torch.randint(0, 100, (batch_size,))


@pytest.fixture
def dummy_context(batch_size: int) -> torch.Tensor:
    """Create dummy text context embeddings."""
    return torch.randn(batch_size, 77, 768)
