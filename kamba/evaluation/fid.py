"""Frechet Inception Distance (FID) computation.

Provides an InceptionV3 feature extractor and the FID calculation.
"""

import logging
from typing import Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import linalg

logger = logging.getLogger(__name__)

# Optional dependency flags.
try:
    from torchvision.models import inception_v3
    from torchvision.models.inception import Inception_V3_Weights

    INCEPTION_AVAILABLE = True
except ImportError:
    INCEPTION_AVAILABLE = False


class InceptionFeatureExtractor(nn.Module):
    """Pretrained InceptionV3 feature extractor for FID and IS computation.

    Args:
        device: Device on which to run the model.

    Raises:
        ImportError: If torchvision is not installed.
    """

    def __init__(self, device: str = "cuda") -> None:
        super().__init__()

        if not INCEPTION_AVAILABLE:
            raise ImportError(
                "torchvision is required for InceptionFeatureExtractor."
            )

        self.device = device

        self.inception = inception_v3(
            weights=Inception_V3_Weights.IMAGENET1K_V1,
            transform_input=False,
        )
        self.inception.fc = nn.Identity()  # Remove classifier head.
        self.inception.eval()
        self.inception.to(device)

        for param in self.inception.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract 2048-d feature vectors from images.

        Args:
            x: Batch of images with shape ``(B, 3, H, W)`` in range
                ``[0, 1]``.

        Returns:
            Feature tensor of shape ``(B, 2048)``.
        """
        if x.shape[2] != 299 or x.shape[3] != 299:
            x = F.interpolate(
                x, size=(299, 299), mode="bilinear", align_corners=False
            )

        # Normalize from [0, 1] to [-1, 1].
        x = 2.0 * x - 1.0

        features: torch.Tensor = self.inception(x)
        return features


def calculate_fid(
    real_features: Union[torch.Tensor, np.ndarray],
    fake_features: Union[torch.Tensor, np.ndarray],
) -> float:
    """Calculate the Frechet Inception Distance (FID).

    FID = ||mu_r - mu_f||^2
          + Tr(Sigma_r + Sigma_f - 2 * sqrt(Sigma_r * Sigma_f))

    Args:
        real_features: Feature matrix of shape ``(N, D)`` from real images.
        fake_features: Feature matrix of shape ``(M, D)`` from generated
            images.

    Returns:
        The FID score (lower is better).
    """
    if isinstance(real_features, torch.Tensor):
        real_features = real_features.cpu().numpy()
    if isinstance(fake_features, torch.Tensor):
        fake_features = fake_features.cpu().numpy()

    mu_real = np.mean(real_features, axis=0)
    mu_fake = np.mean(fake_features, axis=0)

    sigma_real = np.cov(real_features, rowvar=False)
    sigma_fake = np.cov(fake_features, rowvar=False)

    diff = mu_real - mu_fake

    covmean, _ = linalg.sqrtm(sigma_real.dot(sigma_fake), disp=False)

    # Discard tiny imaginary components from numerical error.
    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = float(
        diff.dot(diff) + np.trace(sigma_real + sigma_fake - 2.0 * covmean)
    )
    return fid
