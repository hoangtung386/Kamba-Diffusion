"""COCO Captions dataset loader for text-to-image training."""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

logger = logging.getLogger(__name__)


class COCODataset(Dataset):
    """COCO Captions dataset for text-to-image generation.

    Expected directory structure::

        coco/
            train2017/
                000000000009.jpg
                ...
            val2017/
                ...
            annotations/
                captions_train2017.json
                captions_val2017.json

    Args:
        data_root: Path to the COCO dataset root directory.
        split: Dataset split, either ``"train"`` or ``"val"``.
        year: COCO dataset year.
        image_size: Target image size after resizing and cropping.
        center_crop: If True, use deterministic center crop (validation).
            If False, use random crop with augmentation (training).
    """

    def __init__(
        self,
        data_root: str,
        split: str = "train",
        year: str = "2017",
        image_size: int = 256,
        center_crop: bool = True,
    ) -> None:
        super().__init__()

        self.data_root = data_root
        self.split = split
        self.image_size = image_size

        self.image_dir = os.path.join(data_root, f"{split}{year}")
        ann_file = os.path.join(
            data_root, "annotations", f"captions_{split}{year}.json"
        )

        logger.info("Loading COCO %s annotations from %s...", split, ann_file)
        with open(ann_file, "r") as f:
            coco_data = json.load(f)

        # Build image_id -> captions mapping.
        image_id_to_captions: Dict[int, List[str]] = {}
        for ann in coco_data["annotations"]:
            img_id: int = ann["image_id"]
            if img_id not in image_id_to_captions:
                image_id_to_captions[img_id] = []
            image_id_to_captions[img_id].append(ann["caption"])

        # Build list of image entries that have captions.
        self.images: List[Dict[str, Any]] = []
        for img_info in coco_data["images"]:
            img_id = img_info["id"]
            if img_id in image_id_to_captions:
                self.images.append(
                    {
                        "image_id": img_id,
                        "file_name": img_info["file_name"],
                        "captions": image_id_to_captions[img_id],
                    }
                )

        logger.info(
            "COCO %s: found %d images with captions.", split, len(self.images)
        )

        self.transform = self._build_transform(image_size, center_crop)

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
        return len(self.images)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        """Load a single sample.

        Args:
            idx: Index into the dataset.

        Returns:
            A dictionary with ``"image"`` (tensor of shape ``(3, H, W)``
            normalized to ``[-1, 1]``) and ``"caption"`` (string), or
            ``None`` if the image could not be loaded.  Use
            :func:`collate_fn` to filter out ``None`` entries.
        """
        img_info = self.images[idx]
        img_path = os.path.join(self.image_dir, img_info["file_name"])

        try:
            image = Image.open(img_path).convert("RGB")
            image = self.transform(image)
        except Exception:
            logger.warning(
                "Failed to load image %s, returning None.", img_path,
                exc_info=True,
            )
            return None

        captions: List[str] = img_info["captions"]
        caption = captions[random.randrange(len(captions))]

        return {"image": image, "caption": caption}

    @staticmethod
    def collate_fn(
        batch: List[Optional[Dict[str, Any]]],
    ) -> Dict[str, Any]:
        """Custom collate function that filters out ``None`` samples.

        Args:
            batch: List of samples, some of which may be ``None``.

        Returns:
            A collated batch dictionary with stacked images and a list of
            caption strings.
        """
        batch = [sample for sample in batch if sample is not None]
        if not batch:
            return {"image": torch.empty(0), "caption": []}

        images = torch.stack([sample["image"] for sample in batch], dim=0)
        captions = [sample["caption"] for sample in batch]
        return {"image": images, "caption": captions}
