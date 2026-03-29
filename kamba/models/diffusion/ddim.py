"""DDIM sampler for accelerated deterministic diffusion sampling.

Implements the Denoising Diffusion Implicit Models sampling procedure
(Song et al., "Denoising Diffusion Implicit Models", arXiv:2010.02502),
which allows skipping timesteps for faster generation and supports an
``eta`` parameter to interpolate between deterministic (``eta=0``) and
stochastic DDPM-like (``eta=1``) sampling.
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from kamba.models.diffusion.ddpm import DDPM


class DDIMSampler(nn.Module):
    """DDIM sampler that wraps a trained DDPM instance.

    Performs accelerated sampling by sub-selecting a subset of the
    original diffusion timesteps and applying the DDIM update rule.

    Args:
        ddpm: A trained ``DDPM`` instance providing the denoising model
            and pre-computed diffusion coefficients.
    """

    def __init__(self, ddpm: DDPM) -> None:
        super().__init__()
        self.ddpm = ddpm

    @staticmethod
    def _make_ddim_timesteps(
        num_ddpm_timesteps: int, num_ddim_steps: int
    ) -> torch.Tensor:
        """Create a sub-sampled sequence of DDPM timesteps for DDIM.

        The timesteps are evenly spaced across the full DDPM schedule and
        returned in descending order (from noisiest to cleanest).

        Args:
            num_ddpm_timesteps: Total number of DDPM training timesteps.
            num_ddim_steps: Number of DDIM sampling steps.

        Returns:
            Long tensor of shape ``(num_ddim_steps,)`` with descending
            timestep indices.
        """
        step_ratio = num_ddpm_timesteps / num_ddim_steps
        # Use float linspace then round to get evenly spaced integer steps.
        timesteps = (
            torch.linspace(0, num_ddpm_timesteps - 1, num_ddim_steps)
            .flip(0)
            .long()
        )
        return timesteps

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        eta: float = 0.0,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Generate samples using the DDIM update rule.

        Args:
            shape: Desired output shape ``(B, C, H, W)``.
            context: Optional conditioning tensor (e.g. text embeddings).
            num_steps: Number of DDIM sampling steps.  Fewer steps yield
                faster but potentially lower-quality results.
            eta: Stochasticity parameter.  ``eta=0`` gives fully
                deterministic sampling; ``eta=1`` recovers DDPM sampling.
            return_intermediates: If ``True``, also return a list of
                intermediate samples at each step.

        Returns:
            Generated samples tensor, or a tuple of
            ``(samples, intermediates)`` when ``return_intermediates``
            is ``True``.
        """
        device = self.ddpm.betas.device
        img = torch.randn(shape, device=device)

        timesteps = self._make_ddim_timesteps(
            self.ddpm.timesteps, num_steps
        ).to(device)

        intermediates: List[torch.Tensor] = []

        for i in tqdm(range(len(timesteps)), desc="DDIM Sampling", total=num_steps):
            t = timesteps[i]
            t_batch = torch.full(
                (shape[0],), t, device=device, dtype=torch.long
            )

            # Determine the previous alpha_cumprod.
            if i + 1 < len(timesteps):
                t_prev = timesteps[i + 1]
                alpha_cumprod_prev = self.ddpm.alphas_cumprod[t_prev]
            else:
                alpha_cumprod_prev = torch.tensor(1.0, device=device)

            alpha_cumprod_t = self.ddpm.alphas_cumprod[t]

            # Model prediction.
            model_output = self.ddpm.model(img, t_batch, context)

            # Convert to predicted x_0 regardless of prediction type.
            predicted_x0 = self._predict_x0(
                model_output, img, t_batch, alpha_cumprod_t
            )

            # Compute the noise component of x_t given predicted x_0.
            # eps = (x_t - sqrt(alpha_t) * x_0) / sqrt(1 - alpha_t)
            eps = (
                img - torch.sqrt(alpha_cumprod_t) * predicted_x0
            ) / torch.sqrt(1.0 - alpha_cumprod_t)

            # DDIM variance.
            sigma = self._compute_sigma(
                alpha_cumprod_t, alpha_cumprod_prev, eta
            )

            # Direction pointing to x_t.
            dir_xt = torch.sqrt(
                torch.clamp(1.0 - alpha_cumprod_prev - sigma ** 2, min=0.0)
            ) * eps

            # Stochastic noise term.
            if sigma > 0:
                noise = torch.randn_like(img)
            else:
                noise = torch.zeros_like(img)

            img = (
                torch.sqrt(alpha_cumprod_prev) * predicted_x0
                + dir_xt
                + sigma * noise
            )

            if return_intermediates:
                intermediates.append(img.cpu())

        if return_intermediates:
            return img, intermediates
        return img

    def _predict_x0(
        self,
        model_output: torch.Tensor,
        x_t: torch.Tensor,
        t_batch: torch.Tensor,
        alpha_cumprod_t: torch.Tensor,
    ) -> torch.Tensor:
        """Convert the model output to a predicted clean sample ``x_0``.

        Handles epsilon, x0, and v-prediction parameterisations.

        Args:
            model_output: Raw output from the denoising network.
            x_t: Current noisy sample.
            t_batch: Timestep tensor of shape ``(B,)``.
            alpha_cumprod_t: Cumulative alpha at the current timestep.

        Returns:
            Predicted ``x_0``.
        """
        prediction_type = self.ddpm.prediction_type

        if prediction_type == "epsilon":
            return (
                x_t - torch.sqrt(1.0 - alpha_cumprod_t) * model_output
            ) / torch.sqrt(alpha_cumprod_t)

        if prediction_type == "x0":
            return model_output

        if prediction_type == "v":
            sqrt_alpha = self.ddpm._extract(
                self.ddpm.sqrt_alphas_cumprod, t_batch, x_t.shape
            )
            sqrt_one_minus_alpha = self.ddpm._extract(
                self.ddpm.sqrt_one_minus_alphas_cumprod, t_batch, x_t.shape
            )
            return sqrt_alpha * x_t - sqrt_one_minus_alpha * model_output

        raise ValueError(
            f"Unknown prediction type: {prediction_type!r}"
        )

    @staticmethod
    def _compute_sigma(
        alpha_cumprod_t: torch.Tensor,
        alpha_cumprod_prev: torch.Tensor,
        eta: float,
    ) -> torch.Tensor:
        """Compute the DDIM noise standard deviation sigma.

        ``sigma = eta * sqrt((1 - alpha_prev) / (1 - alpha_t)) * sqrt(1 - alpha_t / alpha_prev)``

        Args:
            alpha_cumprod_t: Cumulative alpha at timestep ``t``.
            alpha_cumprod_prev: Cumulative alpha at the previous timestep.
            eta: Stochasticity parameter.

        Returns:
            Scalar sigma value.
        """
        sigma = eta * torch.sqrt(
            (1.0 - alpha_cumprod_prev)
            / torch.clamp(1.0 - alpha_cumprod_t, min=1e-8)
            * torch.clamp(1.0 - alpha_cumprod_t / alpha_cumprod_prev, min=0.0)
        )
        return sigma
