from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class StrokeConfig(BaseConfig):
    # Dataset info
    dataset_name: str = "stroke"
    num_classes: int = 2  # Background + Lesion
    # For Diffusion + CT, input might be: 
    # - 1 channel (CT slice) + 1 channel (Noisy Mask) = 2 channels
    # But usually we keep input_channels as Image Channels (1), and Model handles the extra mask channel internally or we sum them here.
    # Let's set input_channels = 1 (just the image) and let the Model/Trainer handle the concatenation logic.
    # Wait, in DDPM I wrote: input = cat(condition, x_noisy).
    # So network sees C_cond + C_mask.
    # If mask is 1 channel (grayscale) and image is 1 channel. Total 2.
    input_channels: int = 2 
    
    # Model
    backbone: str = "convnext_v2"
    bottleneck: str = "mamba"
    decoder: str = "kan"
    
    # Diffusion
    use_diffusion: bool = True
    diffusion_steps: int = 1000
    
    # Training
    batch_size: int = 8
    learning_rate: float = 1e-4
    num_epochs: int = 300
    
    def get_dataset_config(self):
        return {
            'data_root': './data/stroke',
            'train_dir': 'train',
            'val_dir': 'val',
            'image_ext': '.png',
            'mask_ext': '.png'
        }
    
    def get_transform_config(self):
        return {
            'resize': (256, 256),
            'normalize': {'mean': [0.5], 'std': [0.5]}
        }