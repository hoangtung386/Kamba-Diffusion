"""Denoising Diffusion Probabilistic Model with improved training features.

Implements DDPM with the following enhancements:
- Min-SNR loss weighting (Hang et al., "Efficient Diffusion Training via
  Min-SNR Weighting Strategy", arXiv:2303.09556)
- Velocity parameterisation (v-prediction)
- Offset noise for improved dark/bright image generation
- Linear and cosine beta schedules
"""

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


def linear_beta_schedule(timesteps: int) -> torch.Tensor:
    """Compute a linear beta schedule scaled to the number of timesteps.

    Args:
        timesteps: Total number of diffusion steps.

    Returns:
        Tensor of shape ``(timesteps,)`` with float64 beta values.
    """
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """Compute a cosine beta schedule (Nichol & Dhariwal, 2021).

    Args:
        timesteps: Total number of diffusion steps.
        s: Small offset to prevent singularity near ``t = 0``.

    Returns:
        Tensor of shape ``(timesteps,)`` with float64 beta values
        clipped to ``[0.0001, 0.9999]``.
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class DDPM(nn.Module):
    """Denoising Diffusion Probabilistic Model with Min-SNR weighting.

    Supports epsilon-prediction, x0-prediction, and v-prediction
    parameterisations.  Optional offset noise improves generation of
    very dark or very bright images.

    Args:
        model: Denoising network that accepts
            ``(x_noisy, timestep, context)`` and returns a prediction of
            the same spatial shape as ``x_noisy``.
        timesteps: Total number of diffusion steps.
        beta_schedule: Schedule type, ``"linear"`` or ``"cosine"``.
        loss_type: Loss function, ``"l1"``, ``"l2"``, or ``"huber"``.
        prediction_type: Prediction target, ``"epsilon"``, ``"v"``, or
            ``"x0"``.
        min_snr_gamma: Gamma value for Min-SNR weighting.  Set to
            ``None`` to disable.
        use_offset_noise: If ``True``, add per-channel offset noise.
        offset_noise_strength: Standard deviation of the offset noise.
    """

    def __init__(
        self,
        model: nn.Module,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        loss_type: str = "l2",
        prediction_type: str = "epsilon",
        min_snr_gamma: Optional[float] = 5.0,
        use_offset_noise: bool = False,
        offset_noise_strength: float = 0.1,
    ) -> None:
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.min_snr_gamma = min_snr_gamma
        self.use_offset_noise = use_offset_noise
        self.offset_noise_strength = offset_noise_strength

        if beta_schedule == "linear":
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == "cosine":
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule!r}")

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        self.register_buffer("betas", betas.to(torch.float32))
        self.register_buffer("alphas", alphas.to(torch.float32))
        self.register_buffer("alphas_cumprod", alphas_cumprod.to(torch.float32))
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev.to(torch.float32))

        self.register_buffer(
            "sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod).to(torch.float32)
        )
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod",
            torch.sqrt(1.0 - alphas_cumprod).to(torch.float32),
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod).to(torch.float32),
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod",
            torch.sqrt(1.0 / alphas_cumprod - 1).to(torch.float32),
        )

        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        self.register_buffer(
            "posterior_variance", posterior_variance.to(torch.float32)
        )

        snr = alphas_cumprod / (1 - alphas_cumprod)
        self.register_buffer("snr", snr.to(torch.float32))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract(
        a: torch.Tensor, t: torch.Tensor, x_shape: Tuple[int, ...]
    ) -> torch.Tensor:
        """Gather values from ``a`` at indices ``t`` and reshape for broadcasting.

        Args:
            a: 1-D coefficient tensor.
            t: Timestep indices of shape ``(B,)``.
            x_shape: Target tensor shape for broadcasting alignment.

        Returns:
            Gathered values reshaped to ``(B, 1, ..., 1)``.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

    def _make_noise(self, x: torch.Tensor) -> torch.Tensor:
        """Generate noise, optionally with per-channel offset.

        Args:
            x: Reference tensor whose shape and device are used.

        Returns:
            Noise tensor with the same shape as ``x``.
        """
        noise = torch.randn_like(x)
        if self.use_offset_noise:
            offset = torch.randn(
                x.shape[0], x.shape[1], 1, 1, device=x.device
            )
            noise = noise + self.offset_noise_strength * offset
        return noise

    # ------------------------------------------------------------------
    # Forward diffusion
    # ------------------------------------------------------------------

    def q_sample(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Diffuse clean data to timestep ``t``.

        Args:
            x_start: Clean data of shape ``(B, C, H, W)``.
            t: Timestep indices of shape ``(B,)``.
            noise: Optional pre-generated noise.

        Returns:
            Noisy data at timestep ``t``.
        """
        if noise is None:
            noise = self._make_noise(x_start)

        sqrt_alpha_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_t = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )
        return sqrt_alpha_t * x_start + sqrt_one_minus_alpha_t * noise

    # ------------------------------------------------------------------
    # Predictions
    # ------------------------------------------------------------------

    def predict_start_from_noise(
        self, x_t: torch.Tensor, t: torch.Tensor, noise: torch.Tensor
    ) -> torch.Tensor:
        """Predict ``x_0`` from ``x_t`` and predicted noise.

        Args:
            x_t: Noisy data at timestep ``t``.
            t: Timestep indices.
            noise: Predicted noise.

        Returns:
            Predicted clean data ``x_0``.
        """
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise

    def predict_noise_from_start(
        self, x_t: torch.Tensor, t: torch.Tensor, x0: torch.Tensor
    ) -> torch.Tensor:
        """Predict noise from ``x_t`` and predicted ``x_0``.

        Args:
            x_t: Noisy data at timestep ``t``.
            t: Timestep indices.
            x0: Predicted clean data.

        Returns:
            Predicted noise.
        """
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip * x_t - x0) / sqrt_recipm1

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------

    def get_loss_weights(self, t: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep Min-SNR loss weights.

        ``weight(t) = min(SNR(t), gamma) / SNR(t)``

        Args:
            t: Timestep indices of shape ``(B,)``.

        Returns:
            Weight tensor broadcastable to ``(B, 1, 1, 1)``.
        """
        snr = self._extract(self.snr, t, (t.shape[0], 1, 1, 1))

        if self.min_snr_gamma is not None:
            weight = torch.minimum(
                snr, torch.ones_like(snr) * self.min_snr_gamma
            ) / snr
        else:
            weight = torch.ones_like(snr)

        return weight

    def p_losses(
        self,
        x_start: torch.Tensor,
        t: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the weighted diffusion training loss.

        Args:
            x_start: Clean latent of shape ``(B, C, H, W)``.
            t: Timestep indices of shape ``(B,)``.
            context: Optional conditioning tensor, e.g. text embeddings
                of shape ``(B, M, D)``.
            noise: Optional pre-generated noise.

        Returns:
            Scalar weighted loss.
        """
        if noise is None:
            noise = self._make_noise(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.model(x_noisy, t, context)

        # Determine the target based on prediction type.
        if self.prediction_type == "epsilon":
            target = noise
        elif self.prediction_type == "x0":
            target = x_start
        elif self.prediction_type == "v":
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
            )
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type!r}")

        # Compute per-element loss.
        if self.loss_type == "l1":
            loss = F.l1_loss(model_output, target, reduction="none")
        elif self.loss_type == "l2":
            loss = F.mse_loss(model_output, target, reduction="none")
        elif self.loss_type == "huber":
            loss = F.smooth_l1_loss(model_output, target, reduction="none")
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type!r}")

        loss_weights = self.get_loss_weights(t)
        return (loss * loss_weights).mean()

    # ------------------------------------------------------------------
    # Sampling
    # ------------------------------------------------------------------

    def _to_noise_prediction(
        self, model_output: torch.Tensor, x: torch.Tensor, t_batch: torch.Tensor
    ) -> torch.Tensor:
        """Convert any prediction type to a noise prediction.

        Args:
            model_output: Raw model output.
            x: Current noisy sample.
            t_batch: Timestep tensor of shape ``(B,)``.

        Returns:
            Predicted noise tensor.
        """
        if self.prediction_type == "epsilon":
            return model_output
        if self.prediction_type == "x0":
            return self.predict_noise_from_start(x, t_batch, model_output)
        if self.prediction_type == "v":
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t_batch, x.shape)
            sqrt_one_minus_alpha = self._extract(
                self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
            )
            return sqrt_alpha * model_output + sqrt_one_minus_alpha * x
        raise ValueError(f"Unknown prediction type: {self.prediction_type!r}")

    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: int,
        context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Sample ``x_{t-1}`` from ``x_t`` (single reverse step).

        Args:
            x: Noisy sample of shape ``(B, C, H, W)`` at timestep ``t``.
            t: Current (scalar) timestep.
            context: Optional conditioning tensor.

        Returns:
            Denoised sample at timestep ``t - 1``.
        """
        t_batch = torch.full(
            (x.shape[0],), t, device=x.device, dtype=torch.long
        )
        model_output = self.model(x, t_batch, context)
        predicted_noise = self._to_noise_prediction(model_output, x, t_batch)

        sqrt_recip_alpha = torch.sqrt(1.0 / self.alphas[t])
        sqrt_one_minus_alpha_cumprod = self._extract(
            self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape
        )

        model_mean = sqrt_recip_alpha * (
            x - self.betas[t] * predicted_noise / sqrt_one_minus_alpha_cumprod
        )

        if t > 0:
            posterior_var = self._extract(
                self.posterior_variance, t_batch, x.shape
            )
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_var) * noise

        return model_mean

    @torch.no_grad()
    def sample(
        self,
        shape: Tuple[int, ...],
        context: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        """Generate samples by iteratively denoising from pure noise.

        Args:
            shape: Desired output shape ``(B, C, H, W)``.
            context: Optional conditioning tensor.
            return_intermediates: If ``True``, also return a list of
                intermediate samples (every 50 steps).

        Returns:
            Generated samples, or a tuple of ``(samples, intermediates)``
            when ``return_intermediates`` is ``True``.
        """
        device = self.betas.device
        img = torch.randn(shape, device=device)

        intermediates: List[torch.Tensor] = []

        for i in tqdm(
            reversed(range(self.timesteps)), desc="Sampling", total=self.timesteps
        ):
            img = self.p_sample(img, i, context)
            if return_intermediates and i % 50 == 0:
                intermediates.append(img.cpu())

        if return_intermediates:
            return img, intermediates
        return img
