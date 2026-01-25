import torch
import torch.nn as nn
import math

class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, timesteps):
        """
        Args:
            timesteps: (B,)
        Returns:
            embeddings: (B, embed_dim)
        """
        device = timesteps.device
        half_dim = self.embed_dim // 2
        
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        
        if self.embed_dim % 2 == 1:
            emb = torch.nn.functional.pad(emb, (0, 1, 0, 0))
            
        return emb
