"""Convenience wrapper for computing multiple evaluation metrics."""

import logging
from typing import Dict, List, Optional

import numpy as np
import torch
from tqdm import tqdm

from kamba.evaluation.clip_score import CLIP_AVAILABLE, CLIPScore
from kamba.evaluation.fid import INCEPTION_AVAILABLE, InceptionFeatureExtractor, calculate_fid
from kamba.evaluation.lpips import LPIPSMetric

logger = logging.getLogger(__name__)


class EvaluationSuite:
    """Convenience wrapper for computing multiple evaluation metrics.

    Args:
        device: Device on which to run all metric computations.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device

        self.inception: Optional[InceptionFeatureExtractor] = None
        if INCEPTION_AVAILABLE:
            self.inception = InceptionFeatureExtractor(device)

        self.clip_scorer: Optional[CLIPScore] = None
        if CLIP_AVAILABLE:
            self.clip_scorer = CLIPScore(device=device)

        self.lpips: Optional[LPIPSMetric] = None
        try:
            self.lpips = LPIPSMetric(device=device)
        except ImportError:
            logger.info("LPIPS not available; skipping LPIPS metric.")

    @torch.no_grad()
    def compute_all_metrics(
        self,
        real_images: torch.Tensor,
        fake_images: torch.Tensor,
        captions: Optional[List[str]] = None,
        batch_size: int = 32,
        compute_fid: bool = True,
        compute_is: bool = True,
        compute_clip: bool = True,
        compute_lpips: bool = False,
    ) -> Dict[str, float]:
        """Compute all requested evaluation metrics.

        Args:
            real_images: Real images tensor ``(N, 3, H, W)`` in ``[0, 1]``.
            fake_images: Generated images tensor ``(M, 3, H, W)`` in
                ``[0, 1]``.
            captions: Optional list of text captions for CLIP score.
            batch_size: Batch size for processing images through models.
            compute_fid: Whether to compute FID.
            compute_is: Whether to compute Inception Score.
            compute_clip: Whether to compute CLIP Score.
            compute_lpips: Whether to compute LPIPS.

        Returns:
            Dictionary mapping metric names to their scalar values.
        """
        metrics: Dict[str, float] = {}

        if (compute_fid or compute_is) and self.inception is not None:
            logger.info("Extracting Inception features...")

            real_features_list: List[torch.Tensor] = []
            fake_features_list: List[torch.Tensor] = []

            for i in tqdm(
                range(0, len(real_images), batch_size),
                desc="Inception features",
            ):
                batch_real = real_images[i : i + batch_size].to(self.device)
                batch_fake = fake_images[i : i + batch_size].to(self.device)

                real_features_list.append(
                    self.inception(batch_real).cpu()
                )
                fake_features_list.append(
                    self.inception(batch_fake).cpu()
                )

            real_features = torch.cat(real_features_list, dim=0)
            fake_features = torch.cat(fake_features_list, dim=0)

            if compute_fid:
                logger.info("Computing FID...")
                fid = calculate_fid(real_features, fake_features)
                metrics["fid"] = fid
                logger.info("FID: %.2f", fid)

        if (
            compute_clip
            and captions is not None
            and self.clip_scorer is not None
        ):
            logger.info("Computing CLIP Score...")
            clip_scores: List[float] = []

            for i in tqdm(
                range(0, len(fake_images), batch_size),
                desc="CLIP Score",
            ):
                batch_fake = fake_images[i : i + batch_size].to(self.device)
                batch_captions = captions[i : i + batch_size]
                score = self.clip_scorer(batch_fake, batch_captions)
                clip_scores.append(score)

            clip_mean = float(np.mean(clip_scores))
            metrics["clip_score"] = clip_mean
            logger.info("CLIP Score: %.4f", clip_mean)

        if compute_lpips and self.lpips is not None:
            logger.info("Computing LPIPS...")
            lpips_scores: List[float] = []

            num_samples = min(len(real_images), len(fake_images))
            for i in tqdm(
                range(0, num_samples, batch_size), desc="LPIPS"
            ):
                batch_real = real_images[i : i + batch_size].to(self.device)
                batch_fake = fake_images[i : i + batch_size].to(self.device)
                score = self.lpips(batch_real, batch_fake)
                lpips_scores.append(score)

            lpips_mean = float(np.mean(lpips_scores))
            metrics["lpips"] = lpips_mean
            logger.info("LPIPS: %.4f", lpips_mean)

        return metrics
