"""CLIP text encoder for text-conditional image generation.

Wraps a frozen pretrained CLIP model from OpenAI / Hugging Face to produce
token-level and pooled text embeddings used as conditioning context for the
diffusion denoiser.
"""

import logging
from typing import List, Tuple, Union

import torch
import torch.nn as nn

try:
    from transformers import CLIPTextModel, CLIPTokenizer

    _TRANSFORMERS_AVAILABLE = True
except ImportError:
    _TRANSFORMERS_AVAILABLE = False

logger = logging.getLogger(__name__)


class CLIPTextEncoder(nn.Module):
    """Frozen CLIP text encoder for text conditioning.

    Loads a pretrained CLIP text model and freezes all parameters so that
    only the denoiser is trained. Produces token-level embeddings suitable
    for cross-attention and a pooled sentence embedding.

    Args:
        model_name: Hugging Face model identifier for the CLIP checkpoint.
        max_length: Maximum token sequence length (typically 77 for CLIP).
        device: Device to place the model on.

    Raises:
        ImportError: If the ``transformers`` library is not installed.
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        max_length: int = 77,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        if not _TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "The 'transformers' library is required for CLIPTextEncoder. "
                "Install it with: pip install transformers"
            )

        self.device = device
        self.max_length = max_length

        logger.info("Loading CLIP model: %s", model_name)
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)

        # Move to the requested device.
        self.model = self.model.to(device)

        # Freeze all parameters.
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

        self.embed_dim: int = self.model.config.hidden_size  # 768 for ViT-L/14
        logger.info(
            "CLIP loaded successfully -- embedding dim: %d", self.embed_dim
        )

    @torch.no_grad()
    def forward(
        self, text: Union[str, List[str]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode text to embeddings.

        Args:
            text: A single string or a list of strings to encode.

        Returns:
            A tuple of ``(embeddings, pooled)`` where *embeddings* has shape
            ``(B, max_length, embed_dim)`` (token-wise) and *pooled* has
            shape ``(B, embed_dim)`` (CLS token).
        """
        if isinstance(text, str):
            text = [text]

        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.model(**tokens)

        embeddings: torch.Tensor = outputs.last_hidden_state  # (B, 77, 768)
        pooled: torch.Tensor = outputs.pooler_output  # (B, 768)

        return embeddings, pooled

    @torch.no_grad()
    def encode_batch(
        self, texts: List[str], batch_size: int = 32
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode a large list of texts in batches.

        Args:
            texts: List of strings to encode.
            batch_size: Number of texts per forward pass.

        Returns:
            A tuple of ``(all_embeddings, all_pooled)`` concatenated across
            all batches.
        """
        all_embeddings: list[torch.Tensor] = []
        all_pooled: list[torch.Tensor] = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings, pooled = self.forward(batch)
            all_embeddings.append(embeddings)
            all_pooled.append(pooled)

        return torch.cat(all_embeddings, dim=0), torch.cat(all_pooled, dim=0)

    @torch.no_grad()
    def get_unconditional_embedding(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return embeddings for empty strings (unconditional context).

        Used during classifier-free guidance to obtain the null conditioning
        signal.

        Args:
            batch_size: Number of unconditional embeddings to produce.

        Returns:
            A tuple of ``(embeddings, pooled)`` for empty-string inputs.
        """
        empty_text = [""] * batch_size
        return self.forward(empty_text)
