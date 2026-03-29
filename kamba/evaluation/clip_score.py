"""CLIP-based text-image alignment scoring."""

import logging
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from transformers import CLIPModel, CLIPProcessor

    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False


class CLIPScore(nn.Module):
    """CLIP-based text-image alignment scorer.

    Higher scores indicate better alignment between generated images and
    their corresponding text prompts.

    Args:
        model_name: HuggingFace identifier for the CLIP model.
        device: Device on which to run the model.

    Raises:
        ImportError: If the ``transformers`` library is not installed.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-base-patch32",
        device: str = "cuda",
    ) -> None:
        super().__init__()

        if not CLIP_AVAILABLE:
            raise ImportError(
                "The transformers library is required for CLIPScore."
            )

        self.device = device

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

    @torch.no_grad()
    def forward(
        self, images: torch.Tensor, captions: List[str]
    ) -> float:
        """Calculate the mean CLIP similarity score.

        Args:
            images: Batch of images with shape ``(B, 3, H, W)`` in range
                ``[0, 1]``.
            captions: List of text captions, one per image.

        Returns:
            Mean cosine similarity between image and text embeddings.
        """
        inputs = self.processor(
            text=captions,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        outputs = self.model(**inputs)

        image_features = F.normalize(outputs.image_embeds, dim=-1)
        text_features = F.normalize(outputs.text_embeds, dim=-1)

        similarity = (image_features * text_features).sum(dim=-1)
        return float(similarity.mean().item())
