"""
Denoisers package - U-Net architectures for diffusion models
Replaces previous 'decoders' package
"""

from .mamba_unet import MambaUNet

__all__ = [
    'MambaUNet',
]
