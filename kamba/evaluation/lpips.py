"""Learned Perceptual Image Patch Similarity (LPIPS) metric."""

import torch
import torch.nn as nn


class LPIPSMetric(nn.Module):
    """Learned Perceptual Image Patch Similarity (LPIPS).

    Measures perceptual distance between pairs of images.  Lower values
    indicate higher perceptual similarity.

    Args:
        net: Backbone network for LPIPS (``"alex"`` or ``"vgg"``).
        device: Device on which to run the model.

    Raises:
        ImportError: If the ``lpips`` library is not installed.
    """

    def __init__(self, net: str = "alex", device: str = "cuda") -> None:
        super().__init__()

        try:
            import lpips  # noqa: F811
        except ImportError as exc:
            raise ImportError(
                "The lpips library is required. "
                "Install with: pip install lpips"
            ) from exc

        self.device = device
        self.lpips_fn = lpips.LPIPS(net=net).to(device)

    @torch.no_grad()
    def forward(
        self, img1: torch.Tensor, img2: torch.Tensor
    ) -> float:
        """Calculate mean LPIPS distance between two batches of images.

        Args:
            img1: First batch of images ``(B, 3, H, W)`` in ``[-1, 1]``
                or ``[0, 1]``.
            img2: Second batch of images ``(B, 3, H, W)`` in ``[-1, 1]``
                or ``[0, 1]``.

        Returns:
            Mean LPIPS distance (lower is better).
        """
        # LPIPS expects [-1, 1].
        if img1.min() >= 0:
            img1 = 2.0 * img1 - 1.0
        if img2.min() >= 0:
            img2 = 2.0 * img2 - 1.0

        dist = self.lpips_fn(img1, img2)
        return float(dist.mean().item())
