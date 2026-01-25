from dataclasses import dataclass, field
from typing import List, Tuple, Optional
from abc import ABC, abstractmethod


@dataclass
class BaseConfig(ABC):
    """Abstract base configuration"""
    
    # Dataset
    dataset_name: str = "base"
    data_root: str = "./data"
    num_classes: int = 2
    input_channels: int = 1
    image_size: Tuple[int, int] = (512, 512)
    
    # Model Architecture
    backbone: str = "convnext_v2"        # convnext_v2, resnet50, efficientnet
    bottleneck: str = "mamba"            # mamba, transformer, conv
    decoder: str = "kan"                 # kan, unet, fpn
    
    # Mamba Config
    mamba_d_state: int = 16
    mamba_d_conv: int = 4
    mamba_expand: int = 2
    
    # KAN Config
    kan_grid_size: int = 5
    kan_spline_order: int = 3
    
    # Diffusion Config
    use_diffusion: bool = True
    diffusion_steps: int = 1000
    noise_schedule: str = "linear"       # linear, cosine
    ddim_steps: int = 50                 # For fast sampling
    
    # Training
    batch_size: int = 4
    num_epochs: int = 300
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    
    # Optimizer
    optimizer: str = "adamw"
    scheduler: str = "cosine"
    
    # Loss
    loss_type: str = "dice_ce"           # dice_ce, focal, boundary
    
    # Augmentation
    use_augmentation: bool = True
    augmentation_strength: float = 0.5
    
    # Hardware
    device: str = "cuda"
    num_workers: int = 4
    mixed_precision: bool = True
    
    # Logging
    log_interval: int = 10
    eval_interval: int = 1
    save_interval: int = 10
    
    # Wandb
    use_wandb: bool = True
    wandb_project: str = "DMK-Stroke"
    wandb_entity: Optional[str] = None
    
    # Paths
    output_dir: str = "./experiments"
    checkpoint_dir: str = "./checkpoints"
    
    @abstractmethod
    def get_dataset_config(self) -> dict:
        """Return dataset-specific configuration"""
        pass
    
    @abstractmethod
    def get_transform_config(self) -> dict:
        """Return transformation configuration"""
        pass
    