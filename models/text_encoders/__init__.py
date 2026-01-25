"""
Text Encoders package for text-conditional image generation
Includes CLIP, T5, and cross-attention mechanisms
"""

from .clip_encoder import CLIPTextEncoder
# from .t5_encoder import T5TextEncoder  # TODO: Implement later

__all__ = [
    'CLIPTextEncoder',
    # 'T5TextEncoder'
]
