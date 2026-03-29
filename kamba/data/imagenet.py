"""ImageNet dataset loader for VAE pretraining."""

import glob
import logging
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class ImageNetDataset(Dataset):
    """ImageNet dataset for VAE pretraining.

    Expected directory structure::

        imagenet/
            train/
                n01440764/
                    n01440764_10026.JPEG
                    ...
                n01443537/
                    ...
            val/
                ...

    Args:
        data_root: Path to the ImageNet root directory.
        split: Dataset split, either ``"train"`` or ``"val"``.
        image_size: Target image size after resizing and cropping.
        center_crop: If True, use deterministic center crop (validation).
            If False, use random crop with augmentation (training).
    """

    _IMAGE_EXTENSIONS = ("*.JPEG", "*.jpg", "*.png")

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: int = 256,
        center_crop: bool = True,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        split_dir = os.path.join(data_root, split)
        self.image_paths: List[str] = self._discover_images(split_dir)

        logger.info(
            "ImageNet %s: found %d images.", split, len(self.image_paths)
        )

        self.transform = self._build_transform(image_size, center_crop)

    @classmethod
    def _discover_images(cls, split_dir: str) -> List[str]:
        """Discover all image files within class subdirectories.

        Args:
            split_dir: Path to the split directory (e.g. ``imagenet/train``).

        Returns:
            Sorted list of absolute paths to image files.
        """
        image_paths: List[str] = []
        if not os.path.exists(split_dir):
            logger.warning("Split directory does not exist: %s", split_dir)
            return image_paths

        for class_dir in sorted(os.listdir(split_dir)):
            class_path = os.path.join(split_dir, class_dir)
            if not os.path.isdir(class_path):
                continue
            for ext in cls._IMAGE_EXTENSIONS:
                image_paths.extend(
                    glob.glob(os.path.join(class_path, ext))
                )

        image_paths.sort()
        return image_paths

    @staticmethod
    def _build_transform(
        image_size: int, center_crop: bool
    ) -> transforms.Compose:
        """Build the image transform pipeline.

        Args:
            image_size: Target spatial resolution.
            center_crop: Whether to use deterministic center crop.

        Returns:
            A composed torchvision transform.
        """
        normalize = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

        if center_crop:
            return transforms.Compose(
                [
                    transforms.Resize(
                        image_size,
                        interpolation=transforms.InterpolationMode.BICUBIC,
                    ),
                    transforms.CenterCrop(image_size),
                    transforms.ToTensor(),
                    normalize,
                ]
            )

        return transforms.Compose(
            [
                transforms.Resize(
                    int(image_size * 1.1),
                    interpolation=transforms.InterpolationMode.BICUBIC,
                ),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Load a single image sample.

        Args:
            idx: Index into the dataset.

        Returns:
            A dictionary with ``"image"`` (tensor of shape ``(3, H, W)``
            normalized to ``[-1, 1]``), or ``None`` if the image could not
            be loaded.  Use :func:`collate_fn` to filter ``None`` entries.
        """
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            logger.warning(
                "Failed to load image %s, returning None.", img_path,
                exc_info=True,
            )
            return None

        return {"image": image}

    @staticmethod
    def collate_fn(
        batch: List[Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Custom collate function that filters out ``None`` samples.

        Args:
            batch: List of samples, some of which may be ``None``.

        Returns:
            A collated batch dictionary with stacked images.
        """
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return {"image": torch.empty(0)}

        images = torch.stack([sample["image"] for sample in batch], dim=0)
        return {"image": images}
