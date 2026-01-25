"""
CLIP Text Encoder for text-conditional image generation
Uses frozen pretrained CLIP model from OpenAI
"""

import torch
import torch.nn as nn

try:
    from transformers import CLIPTextModel, CLIPTokenizer
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("Warning: transformers library not installed. Install with: pip install transformers")


class CLIPTextEncoder(nn.Module):
    """
    Frozen CLIP text encoder for text conditioning
    
    Uses pretrained CLIP ViT-L/14 model
    Outputs: (B, 77, 768) token embeddings
    """
    def __init__(
        self,
        model_name="openai/clip-vit-large-patch14",
        max_length=77,
        device="cuda" if torch.cuda.is_available() else "cpu"
    ):
        super().__init__()
        
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "transformers library required for CLIP encoder. "
                "Install with: pip install transformers"
            )
        
        self.device = device
        self.max_length = max_length
        
        # Load pretrained CLIP
        print(f"Loading CLIP model: {model_name}...")
        self.tokenizer = CLIPTokenizer.from_pretrained(model_name)
        self.model = CLIPTextModel.from_pretrained(model_name)
        
        # Move to device
        self.model = self.model.to(device)
        
        # Freeze all parameters
        for param in self.model.parameters():
            param.requires_grad = False
        
        self.model.eval()
        
        # Embedding dimension
        self.embed_dim = self.model.config.hidden_size  # 768 for ViT-L/14
        
        print(f"✅ CLIP loaded - embedding dim: {self.embed_dim}")
    
    @torch.no_grad()
    def forward(self, text):
        """
        Encode text to embeddings
        
        Args:
            text: List of strings or single string
        Returns:
            embeddings: (B, 77, 768) - Token-wise embeddings
            pooled: (B, 768) - Pooled sentence embedding [CLS] token
        """
        # Handle single string
        if isinstance(text, str):
            text = [text]
        
        # Tokenize
        tokens = self.tokenizer(
            text,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        # Encode
        outputs = self.model(**tokens)
        
        # Token embeddings (for cross-attention)
        embeddings = outputs.last_hidden_state  # (B, 77, 768)
        
        # Pooled embedding (optional, for global conditioning)
        pooled = outputs.pooler_output  # (B, 768)
        
        return embeddings, pooled
    
    def encode_batch(self, texts, batch_size=32):
        """
        Encode large list of texts in batches
        
        Args:
            texts: List of strings
            batch_size: Batch size for encoding
        Returns:
            all_embeddings: (N, 77, 768)
            all_pooled: (N, 768)
        """
        all_embeddings = []
        all_pooled = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            embeddings, pooled = self.forward(batch)
            all_embeddings.append(embeddings)
            all_pooled.append(pooled)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        all_pooled = torch.cat(all_pooled, dim=0)
        
        return all_embeddings, all_pooled
    
    def get_unconditional_embedding(self, batch_size):
        """
        Get embedding for empty string (unconditional)
        Used for classifier-free guidance
        
        Args:
            batch_size: Number of unconditional embeddings
        Returns:
            embeddings: (batch_size, 77, 768)
            pooled: (batch_size, 768)
        """
        empty_text = [""] * batch_size
        return self.forward(empty_text)


class SimpleCLIPEncoder(nn.Module):
    """
    Fallback: Simple learned text encoder if CLIP not available
    NOT AS GOOD - use only for testing without transformers
    """
    def __init__(self, vocab_size=10000, embed_dim=768, max_length=77):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.max_length = max_length
        
        # Simple embedding + transformer
        self.token_embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.randn(1, max_length, embed_dim))
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=8,
            dim_feedforward=2048,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=6)
        
        print("⚠️ Using SimpleCLIPEncoder - install transformers for real CLIP")
    
    def forward(self, token_ids):
        """
        Args:
            token_ids: (B, max_length) - Token IDs
        Returns:
            embeddings: (B, max_length, embed_dim)
            pooled: (B, embed_dim)
        """
        # Embed tokens
        x = self.token_embed(token_ids)
        
        # Add positional embedding
        x = x + self.pos_embed[:, :token_ids.shape[1], :]
        
        # Transform
        embeddings = self.transformer(x)
        
        # Pool (use first token)
        pooled = embeddings[:, 0, :]
        
        return embeddings, pooled


if __name__ == "__main__":
    # Test CLIP encoder
    print("\n🧪 Testing CLIP Text Encoder...\n")
    
    if TRANSFORMERS_AVAILABLE:
        # Test real CLIP
        encoder = CLIPTextEncoder(device='cpu')
        
        # Test single caption
        caption = "A beautiful sunset over mountains"
        embeddings, pooled = encoder(caption)
        
        print(f"Input: '{caption}'")
        print(f"Embeddings shape: {embeddings.shape}")  # (1, 77, 768)
        print(f"Pooled shape: {pooled.shape}")  # (1, 768)
        
        # Test batch
        captions = [
            "A cat sitting on a table",
            "A dog playing in the park",
            "A bird flying in the sky"
        ]
        embeddings, pooled = encoder(captions)
        
        print(f"\nBatch input: {len(captions)} captions")
        print(f"Embeddings shape: {embeddings.shape}")  # (3, 77, 768)
        print(f"Pooled shape: {pooled.shape}")  # (3, 768)
        
        # Test unconditional
        uncond_emb, uncond_pooled = encoder.get_unconditional_embedding(2)
        print(f"\nUnconditional embeddings shape: {uncond_emb.shape}")  # (2, 77, 768)
        
        print("\n✅ CLIP encoder tests passed!")
    else:
        print("❌ transformers not available, testing SimpleCLIPEncoder...")
        
        encoder = SimpleCLIPEncoder()
        token_ids = torch.randint(0, 10000, (2, 77))
        embeddings, pooled = encoder(token_ids)
        
        print(f"Embeddings shape: {embeddings.shape}")
        print(f"Pooled shape: {pooled.shape}")
        print("\n✅ SimpleCLIPEncoder tests passed!")
