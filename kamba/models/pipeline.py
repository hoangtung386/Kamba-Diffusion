"""Latent Diffusion Model pipeline.

Integrates a VAE, CLIP text encoder, Mamba U-Net denoiser, and DDPM/DDIM
schedulers into a single module for training and text-to-image generation.
"""

import logging
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn
from tqdm import tqdm

from kamba.models.diffusion.ddim import DDIMSampler
from kamba.models.diffusion.ddpm import DDPM
from kamba.models.diffusion.guidance import classifier_free_guidance
from kamba.models.denoiser.mamba_unet import MambaUNet
from kamba.models.text_encoder.clip_encoder import CLIPTextEncoder
from kamba.models.vae.model import VAE

logger = logging.getLogger(__name__)

# Default VAE configuration used when none is supplied.
_DEFAULT_VAE_CONFIG: Dict[str, Any] = {
    "in_channels": 3,
    "latent_channels": 4,
    "hidden_dims": [128, 256, 512, 512],
    "image_size": 256,
    "use_kan_decoder": True,
}

# Default denoiser configuration factory (requires ``context_dim`` and
# ``latent_channels`` to be filled in at runtime).
_DEFAULT_DENOISER_CONFIG: Dict[str, Any] = {
    "model_channels": 320,
    "channel_mult": [1, 2, 4, 4],
    "num_res_blocks": 2,
    "attention_resolutions": [1, 2, 3],
    "num_heads": 8,
    "use_cross_attn": True,
}


class LatentDiffusionModel(nn.Module):
    """Complete Latent Diffusion Model for text-to-image generation.

    Components:
        1. **VAE** -- compresses images to latent space (frozen after
           pretraining).
        2. **CLIP Text Encoder** -- produces text embeddings (frozen).
        3. **Mamba U-Net** -- denoiser network (trainable).
        4. **DDPM** -- forward diffusion process with Min-SNR weighting.
        5. **DDIMSampler** -- accelerated deterministic sampler for inference.

    Args:
        vae_config: Keyword arguments forwarded to :class:`VAE`. Uses a
            sensible default when ``None``.
        vae_checkpoint: Optional path to a pretrained VAE state dict.
        text_encoder_model: Hugging Face model identifier for CLIP.
        denoiser_config: Keyword arguments forwarded to :class:`MambaUNet`.
            Uses a sensible default when ``None``.
        timesteps: Number of diffusion timesteps.
        beta_schedule: Beta schedule type (``'linear'`` or ``'cosine'``).
        unconditional_prob: Probability of dropping text conditioning during
            training for classifier-free guidance.
        device: Device string.
    """

    def __init__(
        self,
        vae_config: Optional[Dict[str, Any]] = None,
        vae_checkpoint: Optional[str] = None,
        text_encoder_model: str = "openai/clip-vit-large-patch14",
        denoiser_config: Optional[Dict[str, Any]] = None,
        timesteps: int = 1000,
        beta_schedule: str = "linear",
        prediction_type: str = "epsilon",
        min_snr_gamma: Optional[float] = 5.0,
        unconditional_prob: float = 0.1,
        device: str = "cuda",
    ) -> None:
        super().__init__()

        self.device = device
        self.unconditional_prob = unconditional_prob

        # -- VAE --
        if vae_config is None:
            vae_config = dict(_DEFAULT_VAE_CONFIG)

        logger.info("Initializing VAE.")
        self.vae = VAE(**vae_config)

        if vae_checkpoint is not None:
            logger.info("Loading VAE checkpoint from %s", vae_checkpoint)
            state_dict = torch.load(
                vae_checkpoint, map_location="cpu", weights_only=True
            )
            self.vae.load_state_dict(state_dict)

        # Freeze the VAE.
        self.vae.requires_grad_(False)
        self.vae.eval()

        # -- Text Encoder --
        logger.info("Initializing CLIP text encoder.")
        self.text_encoder = CLIPTextEncoder(
            model_name=text_encoder_model,
            device=device,
        )
        self.context_dim: int = self.text_encoder.embed_dim

        # -- Denoiser (Mamba U-Net) --
        logger.info("Initializing Mamba U-Net denoiser.")
        if denoiser_config is None:
            denoiser_config = {
                **_DEFAULT_DENOISER_CONFIG,
                "in_channels": vae_config["latent_channels"],
                "out_channels": vae_config["latent_channels"],
                "context_dim": self.context_dim,
            }

        self.denoiser = MambaUNet(**denoiser_config)

        # -- DDPM --
        logger.info("Initializing DDPM scheduler.")
        self.ddpm = DDPM(
            model=self.denoiser,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            loss_type="l2",
            prediction_type=prediction_type,
            min_snr_gamma=min_snr_gamma,
        )

        # -- DDIM Sampler --
        self.ddim_sampler = DDIMSampler(ddpm=self.ddpm)

        # Cache a null (unconditional) context buffer.
        self.register_buffer(
            "null_context", torch.zeros(1, 77, self.context_dim)
        )

        logger.info(
            "LDM initialized on %s -- VAE latent channels: %d, "
            "context dim: %d, timesteps: %d",
            device,
            vae_config["latent_channels"],
            self.context_dim,
            timesteps,
        )

    # ------------------------------------------------------------------
    # Encoding helpers
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_images(self, images: torch.Tensor) -> torch.Tensor:
        """Encode images to latent space using the frozen VAE encoder.

        Args:
            images: Input images of shape ``(B, 3, H, W)``.

        Returns:
            Latent codes of shape ``(B, latent_channels, H/8, W/8)``
            (deterministic -- uses the mean, not a sample).
        """
        mean, _logvar = self.vae.encode(images)
        return mean

    @torch.no_grad()
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to images using the frozen VAE decoder.

        Args:
            latents: Latent codes of shape ``(B, latent_channels, H/8, W/8)``.

        Returns:
            Decoded images of shape ``(B, 3, H, W)``.
        """
        return self.vae.decode(latents)

    @torch.no_grad()
    def encode_text(
        self, captions: Union[str, List[str]]
    ) -> torch.Tensor:
        """Encode captions to token-level text embeddings.

        Args:
            captions: A string or list of strings.

        Returns:
            Text embeddings of shape ``(B, 77, context_dim)``.
        """
        embeddings, _pooled = self.text_encoder(captions)
        return embeddings

    def get_null_context(self, batch_size: int) -> torch.Tensor:
        """Return the unconditional (empty-string) context repeated for a batch.

        Args:
            batch_size: Number of copies.

        Returns:
            Tensor of shape ``(batch_size, 77, context_dim)``.
        """
        return self.null_context.repeat(batch_size, 1, 1)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def forward(
        self, images: torch.Tensor, captions: List[str]
    ) -> torch.Tensor:
        """Training forward pass.

        Encodes images and text, optionally drops conditioning for
        classifier-free guidance, and computes the DDPM loss with Min-SNR
        weighting via :meth:`DDPM.p_losses`.

        Args:
            images: Input images of shape ``(B, 3, H, W)``.
            captions: List of text captions (length ``B``).

        Returns:
            Scalar diffusion training loss.
        """
        batch_size = images.shape[0]
        device = images.device

        # Encode images to latents (frozen VAE).
        with torch.no_grad():
            latents = self.encode_images(images)

        # Encode text (frozen CLIP).
        with torch.no_grad():
            context = self.encode_text(captions)

        # Random timesteps.
        t = torch.randint(0, self.ddpm.timesteps, (batch_size,), device=device)

        # Classifier-free guidance: randomly replace context with null.
        if self.training:
            drop_mask = (
                torch.rand(batch_size, device=device) < self.unconditional_prob
            )
            null_context = self.get_null_context(batch_size).to(device)
            context = torch.where(
                drop_mask[:, None, None], null_context, context
            )

        # Compute diffusion loss with Min-SNR weighting.
        loss: torch.Tensor = self.ddpm.p_losses(latents, t, context=context)
        return loss

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate(
        self,
        captions: List[str],
        num_steps: int = 50,
        guidance_scale: float = 7.5,
        height: int = 256,
        width: int = 256,
        return_latents: bool = False,
    ) -> torch.Tensor:
        """Generate images from text captions using DDIM with CFG.

        Args:
            captions: List of text prompts.
            num_steps: Number of DDIM sampling steps.
            guidance_scale: Classifier-free guidance strength.
            height: Desired output image height in pixels.
            width: Desired output image width in pixels.
            return_latents: If ``True``, return raw latents instead of
                decoded images.

        Returns:
            Generated images of shape ``(B, 3, H, W)`` (or latents if
            *return_latents* is ``True``).
        """
        batch_size = len(captions)
        latent_h = height // 8
        latent_w = width // 8

        # Encode text conditioning.
        context = self.encode_text(captions).to(self.device)
        null_context = self.get_null_context(batch_size).to(self.device)

        # Start from pure noise.
        latents = torch.randn(
            batch_size, 4, latent_h, latent_w, device=self.device
        )

        # DDIM timestep schedule.
        timesteps = torch.linspace(
            self.ddpm.timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=self.device,
        )

        for i, t in enumerate(tqdm(timesteps, desc="Generating")):
            t_batch = torch.full(
                (batch_size,), t, device=self.device, dtype=torch.long
            )

            # Conditional and unconditional predictions.
            noise_pred_cond = self.denoiser(latents, t_batch, context)
            noise_pred_uncond = self.denoiser(latents, t_batch, null_context)

            # Apply classifier-free guidance.
            noise_pred = classifier_free_guidance(
                noise_pred_cond, noise_pred_uncond, guidance_scale
            )

            # DDIM step.
            t_next = (
                timesteps[i + 1]
                if i < len(timesteps) - 1
                else torch.tensor(0, device=self.device)
            )
            latents = self._ddim_step(latents, noise_pred, t, t_next)

        if return_latents:
            return latents

        return self.decode_latents(latents)

    def _ddim_step(
        self,
        x: torch.Tensor,
        noise_pred: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor:
        """Perform a single deterministic DDIM sampling step.

        Args:
            x: Current noisy latent of shape ``(B, C, H, W)``.
            noise_pred: Predicted noise of shape ``(B, C, H, W)``.
            t: Current timestep (scalar tensor).
            t_next: Next timestep (scalar tensor).

        Returns:
            Denoised latent for the next step.
        """
        if isinstance(t, int):
            t = torch.tensor(t, device=x.device)
        if isinstance(t_next, int):
            t_next = torch.tensor(t_next, device=x.device)

        t_idx = t.unsqueeze(0) if t.dim() == 0 else t
        t_next_idx = t_next.unsqueeze(0) if t_next.dim() == 0 else t_next

        alpha_t = self.ddpm._extract(
            self.ddpm.alphas_cumprod, t_idx, x.shape
        )
        alpha_next = self.ddpm._extract(
            self.ddpm.alphas_cumprod, t_next_idx, x.shape
        )

        # Predict x_0.
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(
            alpha_t
        )

        # Deterministic DDIM update.
        if t_next > 0:
            x_next = (
                torch.sqrt(alpha_next) * x0_pred
                + torch.sqrt(1 - alpha_next) * noise_pred
            )
        else:
            x_next = x0_pred

        return x_next
