"""Inception Score (IS) computation."""

import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def calculate_inception_score(
    logits: torch.Tensor,
    splits: int = 10,
) -> Tuple[float, float]:
    """Calculate the Inception Score (IS).

    IS = exp(E_x[KL(p(y|x) || p(y))])

    Args:
        logits: Class logits of shape ``(N, num_classes)`` from the
            Inception classifier.
        splits: Number of splits for computing mean and standard deviation.

    Returns:
        A tuple ``(is_mean, is_std)``.
    """
    probs = F.softmax(logits, dim=1).cpu().numpy()

    split_scores: List[float] = []
    split_size = len(probs) // splits

    for i in range(splits):
        part = probs[i * split_size : (i + 1) * split_size]
        if len(part) == 0:
            continue

        py = np.mean(part, axis=0)
        kl = part * (np.log(part + 1e-10) - np.log(py + 1e-10))
        kl_sum = np.sum(kl, axis=1)
        split_scores.append(float(np.exp(np.mean(kl_sum))))

    return float(np.mean(split_scores)), float(np.std(split_scores))
