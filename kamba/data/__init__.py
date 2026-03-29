"""Data loading modules for Kamba Diffusion."""

from kamba.data.coco import COCODataset
from kamba.data.imagenet import ImageNetDataset

__all__ = [
    "COCODataset",
    "ImageNetDataset",
]
