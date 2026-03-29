"""Configuration dataclasses for Kamba Diffusion."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VAEConfig:
    """Configuration for VAE model."""

    in_channels: int = 3
    latent_channels: int = 4
    hidden_dims: tuple[int, ...] = (128, 256, 512, 512)
    image_size: int = 256
    num_res_blocks: int = 2
    dropout: float = 0.0
    use_kan_decoder: bool = True


@dataclass
class DenoiserConfig:
    """Configuration for Mamba U-Net denoiser."""

    in_channels: int = 4
    out_channels: int = 4
    model_channels: int = 320
    channel_mult: tuple[int, ...] = (1, 2, 4, 4)
    num_res_blocks: int = 2
    attention_resolutions: tuple[int, ...] = (1, 2, 3)
    context_dim: int = 768
    num_heads: int = 8
    mamba_d_state: int = 16
    dropout: float = 0.0
    use_cross_attn: bool = True
    use_checkpoint: bool = False


@dataclass
class DiffusionConfig:
    """Configuration for diffusion process."""

    timesteps: int = 1000
    beta_schedule: str = "linear"
    loss_type: str = "l2"
    prediction_type: str = "epsilon"
    min_snr_gamma: float = 5.0
    use_offset_noise: bool = False
    offset_noise_strength: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""

    batch_size: int = 32
    epochs: int = 100
    lr: float = 1e-4
    weight_decay: float = 0.01
    grad_clip_norm: float = 1.0
    num_workers: int = 4
    use_amp: bool = True
    use_ema: bool = True
    ema_decay: float = 0.9999
    gradient_accumulation_steps: int = 1
    seed: int = 42

    # Logging
    log_every: int = 100
    val_every: int = 5
    sample_every: int = 10
    save_every: int = 10

    # Paths
    exp_name: str = "default"
    data_root: str = ""
    output_dir: str = "experiments"
    resume_from: Optional[str] = None


@dataclass
class VAETrainingConfig(TrainingConfig):
    """Configuration for VAE training."""

    # Loss weights
    kl_weight: float = 1e-6
    perceptual_weight: float = 1.0
    use_perceptual: bool = True
    use_gan: bool = True
    gan_weight: float = 0.5
    disc_start_epoch: int = 10
    disc_lr_factor: float = 0.5


@dataclass
class LDMTrainingConfig(TrainingConfig):
    """Configuration for LDM training."""

    vae_checkpoint: Optional[str] = None
    unconditional_prob: float = 0.1
    text_encoder_model: str = "openai/clip-vit-large-patch14"


@dataclass
class GenerationConfig:
    """Configuration for image generation."""

    num_steps: int = 50
    guidance_scale: float = 7.5
    eta: float = 0.0
    num_samples: int = 1
    image_size: int = 256
    output_dir: str = "outputs"
