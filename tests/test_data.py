"""Tests for dataset classes."""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image

from kamba.data.coco import COCODataset
from kamba.data.imagenet import ImageNetDataset


@pytest.fixture
def coco_tmpdir():
    """Create a temporary COCO-like dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img_dir = os.path.join(tmpdir, "train2017")
        ann_dir = os.path.join(tmpdir, "annotations")
        os.makedirs(img_dir)
        os.makedirs(ann_dir)

        images = []
        annotations = []
        for i in range(3):
            filename = f"{i:012d}.jpg"
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(os.path.join(img_dir, filename))
            images.append({"id": i, "file_name": filename})
            for j in range(5):
                annotations.append({
                    "image_id": i,
                    "id": i * 5 + j,
                    "caption": f"Caption {j} for image {i}",
                })

        with open(os.path.join(ann_dir, "captions_train2017.json"), "w") as f:
            json.dump({"images": images, "annotations": annotations}, f)

        yield tmpdir


@pytest.fixture
def imagenet_tmpdir():
    """Create a temporary ImageNet-like dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        class_dir = os.path.join(tmpdir, "train", "n01440764")
        os.makedirs(class_dir)

        for i in range(3):
            img = Image.fromarray(
                np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            )
            img.save(os.path.join(class_dir, f"img_{i}.JPEG"))

        yield tmpdir


class TestCOCODataset:
    def test_len(self, coco_tmpdir):
        ds = COCODataset(coco_tmpdir, split="train", image_size=64)
        assert len(ds) == 3

    def test_getitem(self, coco_tmpdir):
        ds = COCODataset(coco_tmpdir, split="train", image_size=64)
        sample = ds[0]
        assert "image" in sample
        assert "caption" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, 64, 64)

    def test_image_range(self, coco_tmpdir):
        ds = COCODataset(coco_tmpdir, split="train", image_size=64)
        sample = ds[0]
        assert sample["image"].min() >= -1.1
        assert sample["image"].max() <= 1.1


class TestImageNetDataset:
    def test_len(self, imagenet_tmpdir):
        ds = ImageNetDataset(imagenet_tmpdir, split="train", image_size=64)
        assert len(ds) == 3

    def test_getitem(self, imagenet_tmpdir):
        ds = ImageNetDataset(imagenet_tmpdir, split="train", image_size=64)
        sample = ds[0]
        assert "image" in sample
        assert isinstance(sample["image"], torch.Tensor)
        assert sample["image"].shape == (3, 64, 64)
