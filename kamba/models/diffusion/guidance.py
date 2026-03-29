"""Classifier-free guidance utility for text-conditional diffusion."""

import torch


def classifier_free_guidance(
    noise_pred_cond: torch.Tensor,
    noise_pred_uncond: torch.Tensor,
    guidance_scale: float = 7.5,
) -> torch.Tensor:
    """Apply classifier-free guidance to noise predictions.

    Computes the guided prediction as:
    ``output = uncond + guidance_scale * (cond - uncond)``

    Args:
        noise_pred_cond: Conditional noise prediction of shape
            ``(B, C, H, W)``.
        noise_pred_uncond: Unconditional noise prediction of shape
            ``(B, C, H, W)``.
        guidance_scale: Guidance strength.  A value of ``1.0`` disables
            guidance.  Typical values range from ``3.0`` to ``15.0``.

    Returns:
        Guided noise prediction of shape ``(B, C, H, W)``.
    """
    return noise_pred_uncond + guidance_scale * (
        noise_pred_cond - noise_pred_uncond
    )
