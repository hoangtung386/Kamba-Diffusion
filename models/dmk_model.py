import torch
import torch.nn as nn
from utils.registry import BACKBONE_REGISTRY, DECODER_REGISTRY

class UniversalDMK(nn.Module):
    """
    Universal Diffusion-Mamba-KAN Model
    """
    
    def __init__(self, config):
        # Note: Type hint 'BaseConfig' removed to avoid circular import if not handled carefuly, 
        # or we can import it inside.
        super().__init__()
        self.config = config
        
        # 1. Input Projection
        # Input channels = image_channels + mask_channels (if diffusion)
        # We assume config.input_channels is set correctly for the task
        self.input_proj = nn.Conv2d(
            config.input_channels, 
            64, 
            kernel_size=3, 
            padding=1
        )
        
        # 2. Encoder (Backbone)
        # Using a simple ResNet-like or loading from registry
        # For now, let's use a placeholder Identity if backbone not found, or expect it to be registered.
        # Ideally, we should have a `models/backbones/convnext.py`. 
        # For Phase 1, we will simulate a basic feature extractor if generic registry fails.
        if config.backbone in BACKBONE_REGISTRY.list():
             self.encoder = BACKBONE_REGISTRY.get(config.backbone)(
                in_channels=64,
                features=[64, 128, 256, 512, 1024]
            )
        else:
            # Fallback for prototype: Simple Conv Encoder
            self.encoder = nn.Identity() # Placeholder

        # 3. Bottleneck (Mamba/Transformer)
        if config.bottleneck == 'mamba':
            from models.bottlenecks.mamba_block import MambaVisionBlock
            # Stack a few blocks
            self.bottleneck = nn.Sequential(
                MambaVisionBlock(d_model=1024, d_state=config.mamba_d_state),
                MambaVisionBlock(d_model=1024, d_state=config.mamba_d_state)
            )
        else:
            self.bottleneck = nn.Identity()
        
        # 4. Time Embedding (Diffusion)
        if config.use_diffusion:
            from models.modules.embedding import SinusoidalTimeEmbedding
            self.time_embed = SinusoidalTimeEmbedding(embed_dim=1024)
            self.time_proj = nn.Linear(1024, 1024)
            self.act = nn.SiLU()
        else:
            self.time_embed = None
            
        # 5. Decoder (KAN/UNet)
        # We need to ensure Decoder takes [bottleneck, skip_connections]
        if config.decoder == 'kan':
            from models.decoders.kan_decoder import KANDecoder
            self.decoder = KANDecoder(
                encoder_channels=[64, 128, 256, 512, 1024], # Placeholder dims
                decoder_channels=[512, 256, 128, 64],
                num_classes=config.num_classes
            )
        else:
             self.decoder = nn.Identity() # Placeholder

        # 6. Final Projection
        self.final_conv = nn.Conv2d(64, config.num_classes, kernel_size=1)

    def forward(self, x, t=None):
        """
        x: (B, Cin, H, W) - Input tensor (Image + NoisyMask if diffusion)
        t: (B,) - Timestep
        """
        # 1. Input Proj
        x = self.input_proj(x)
        
        # 2. Encoder
        # Placeholder logic: usually encoder returns list of features
        # If Identity, we just fake features for now
        if isinstance(self.encoder, nn.Identity):
            feats = [x, x, x, x, x] # Fake multiscale
            enc_out = x
            # Fake channel expansion to 1024 for bottleneck
            if x.shape[1] != 1024:
                # We can't easily expand without layers.
                # In real impl, Encoder does this.
                pass 
        else:
            feats = self.encoder(x)
            enc_out = feats[-1]

        # 3. Time Embedding
        if self.time_embed is not None and t is not None:
            t_emb = self.time_embed(t)
            t_emb = self.time_proj(self.act(t_emb))
            # Broadcast to spatial
            t_emb = t_emb[:, :, None, None]
            # Add to bottleneck input (assuming channels match 1024)
            # If channels don't match (due to missing encoder), this will fail.
            # We assume Encoder outputs 1024 channels at top level.
            if enc_out.shape[1] == 1024:
                enc_out = enc_out + t_emb

        # 4. Bottleneck
        bottleneck_out = self.bottleneck(enc_out)
        
        # 5. Decoder
        # Requires (bottleneck, skips)
        # Decoder logic inside KANDecoder needs to be robust
        # For now, returns same spatial as input
        out = self.decoder([bottleneck_out, feats[:-1]])
        
        # 6. Final
        return self.final_conv(out)

        