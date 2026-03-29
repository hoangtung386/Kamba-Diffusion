"""Sinusoidal positional embeddings for diffusion timesteps."""

import math

import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps.

    Encodes scalar timesteps into dense vectors using sinusoidal
    functions at geometrically spaced frequencies, following the
    formulation from "Attention Is All You Need" (Vaswani et al., 2017).

    Args:
        embed_dim: Dimensionality of the output embedding.
    """

    def __init__(self, embed_dim: int) -> None:
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        """Compute sinusoidal embeddings for the given timesteps.

        Args:
            timesteps: Tensor of shape ``(B,)`` containing integer or
                float timestep values.

        Returns:
            Tensor of shape ``(B, embed_dim)`` with positional
            embeddings.
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2

        log_timescale = math.log(10000.0) / (half_dim - 1)
        freqs = torch.exp(torch.arange(half_dim, device=device) * -log_timescale)

        args = timesteps[:, None].float() * freqs[None, :]
        embedding = torch.cat([args.sin(), args.cos()], dim=-1)

        if self.embed_dim % 2 == 1:
            embedding = nn.functional.pad(embedding, (0, 1, 0, 0))

        return embedding
