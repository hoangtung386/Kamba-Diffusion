"""
Variational Autoencoder (VAE) with KAN Decoder
Compresses images to latent space for Latent Diffusion Model
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    """Residual Block for Encoder/Decoder"""
    def __init__(self, in_channels, out_channels, dropout=0.0):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        self.norm1 = nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        
        self.norm2 = nn.GroupNorm(num_groups=32, num_channels=out_channels, eps=1e-6)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.shortcut = nn.Identity()
        
        self.act = nn.SiLU()
    
    def forward(self, x):
        h = x
        h = self.norm1(h)
        h = self.act(h)
        h = self.conv1(h)
        
        h = self.norm2(h)
        h = self.act(h)
        h = self.dropout(h)
        h = self.conv2(h)
        
        return h + self.shortcut(x)


class Encoder(nn.Module):
    """
    ResNet-style Encoder for VAE
    Downsamples image and outputs mean/logvar for latent distribution
    
    256x256 -> 128x128 -> 64x64 -> 32x32 (8x downsampling)
    """
    def __init__(
        self,
        in_channels=3,
        hidden_dims=[128, 256, 512, 512],
        latent_channels=4,
        num_res_blocks=2,
        dropout=0.0
    ):
        super().__init__()
        
        # Input conv
        self.input_conv = nn.Conv2d(in_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        # Downsampling blocks
        self.down_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i-1] if i > 0 else hidden_dims[0]
            out_ch = hidden_dims[i]
            
            # ResBlocks at this level
            blocks = []
            for _ in range(num_res_blocks):
                blocks.append(ResBlock(in_ch, out_ch, dropout=dropout))
                in_ch = out_ch
            
            # Downsampling (except last level)
            if i < len(hidden_dims) - 1:
                blocks.append(nn.Conv2d(out_ch, out_ch, kernel_size=3, stride=2, padding=1))
            
            self.down_blocks.append(nn.Sequential(*blocks))
        
        # Output projection to latent distribution parameters
        final_ch = hidden_dims[-1]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_ch, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(final_ch, 2 * latent_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # Input
        h = self.input_conv(x)
        
        # Downsample
        for down_block in self.down_blocks:
            h = down_block(h)
        
        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        moments = self.conv_out(h)
        
        # Split into mean and logvar
        mean, logvar = torch.chunk(moments, 2, dim=1)
        
        return mean, logvar


class KANDecoder(nn.Module):
    """
    KAN-based Decoder for VAE (NOVEL!)
    Uses KAN blocks from existing codebase for interpretable upsampling
    
    32x32 -> 64x64 -> 128x128 -> 256x256
    """
    def __init__(
        self,
        latent_channels=4,
        hidden_dims=[512, 512, 256, 128],
        out_channels=3,
        num_res_blocks=2,
        use_kan=True
    ):
        super().__init__()
        
        # Import KAN block from existing decoder
        if use_kan:
            from models.decoders.kan_decoder import KANBlock2d
            self.BlockType = KANBlock2d
        else:
            self.BlockType = ResBlock
        
        # Input projection from latent
        self.input_conv = nn.Conv2d(latent_channels, hidden_dims[0], kernel_size=3, padding=1)
        
        # Upsampling blocks
        self.up_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims)):
            in_ch = hidden_dims[i]
            out_ch = hidden_dims[i+1] if i < len(hidden_dims)-1 else hidden_dims[-1]
            
            blocks = []
            
            # ResBlocks or KAN blocks
            for _ in range(num_res_blocks):
                if use_kan:
                    blocks.append(KANBlock2d(in_ch))
                else:
                    blocks.append(ResBlock(in_ch, in_ch))
            
            # Upsampling (except last level)
            if i < len(hidden_dims) - 1:
                blocks.append(nn.Sequential(
                    nn.Upsample(scale_factor=2, mode='nearest'),
                    nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
                ))
            
            self.up_blocks.append(nn.Sequential(*blocks))
        
        # Output conv
        final_ch = hidden_dims[-1]
        self.norm_out = nn.GroupNorm(num_groups=32, num_channels=final_ch, eps=1e-6)
        self.act_out = nn.SiLU()
        self.conv_out = nn.Conv2d(final_ch, out_channels, kernel_size=3, padding=1)
    
    def forward(self, z):
        # Input
        h = self.input_conv(z)
        
        # Upsample
        for up_block in self.up_blocks:
            h = up_block(h)
        
        # Output
        h = self.norm_out(h)
        h = self.act_out(h)
        img = self.conv_out(h)
        
        return img


class VAE(nn.Module):
    """
    Variational Autoencoder with KL-regularization
    
    Architecture:
        - Encoder: ResNet-style downsampling
        - Latent: Gaussian distribution (mean, logvar) 
        - Decoder: KAN-based upsampling (NOVEL!)
    
    Compression: 8x spatial (256x256 -> 32x32), 4 channels
    """
    def __init__(
        self,
        in_channels=3,
        latent_channels=4,
        hidden_dims=[128, 256, 512, 512],
        image_size=256,
        num_res_blocks=2,
        dropout=0.0,
        use_kan_decoder=True
    ):
        super().__init__()
        
        self.latent_channels = latent_channels
        self.image_size = image_size
        self.latent_size = image_size // 8  # 8x downsampling
        
        # Encoder
        self.encoder = Encoder(
            in_channels=in_channels,
            hidden_dims=hidden_dims,
            latent_channels=latent_channels,
            num_res_blocks=num_res_blocks,
            dropout=dropout
        )
        
        # Decoder
        self.decoder = KANDecoder(
            latent_channels=latent_channels,
            hidden_dims=list(reversed(hidden_dims)),
            out_channels=in_channels,
            num_res_blocks=num_res_blocks,
            use_kan=use_kan_decoder
        )
        
        # Quant/Post-quant convs (from Stable Diffusion)
        self.quant_conv = nn.Conv2d(latent_channels * 2, latent_channels * 2, kernel_size=1)
        self.post_quant_conv = nn.Conv2d(latent_channels, latent_channels, kernel_size=1)
    
    def encode(self, x):
        """
        Encode image to latent distribution
        
        Args:
            x: (B, C, H, W) - Input image
        Returns:
            mean: (B, latent_channels, H//8, W//8)
            logvar: (B, latent_channels, H//8, W//8)
        """
        mean, logvar = self.encoder(x)
        
        # Apply quant conv
        moments = torch.cat([mean, logvar], dim=1)
        moments = self.quant_conv(moments)
        mean, logvar = torch.chunk(moments, 2, dim=1)
        
        return mean, logvar
    
    def decode(self, z):
        """
        Decode latent to image
        
        Args:
            z: (B, latent_channels, H//8, W//8) - Latent code
        Returns:
            img: (B, C, H, W) - Reconstructed image
        """
        z = self.post_quant_conv(z)
        img = self.decoder(z)
        return img
    
    def reparameterize(self, mean, logvar):
        """
        Reparameterization trick: z = mean + std * epsilon
        
        Args:
            mean: (B, latent_channels, H, W)
            logvar: (B, latent_channels, H, W)
        Returns:
            z: (B, latent_channels, H, W) - Sampled latent
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mean + eps * std
        return z
    
    def forward(self, x, sample=True):
        """
        Full forward pass: encode -> sample -> decode
        
        Args:
            x: (B, C, H, W) - Input image
            sample: If True, sample from distribution. If False, use mean (deterministic)
        Returns:
            recon: (B, C, H, W) - Reconstructed image
            mean: (B, latent_channels, H//8, W//8)
            logvar: (B, latent_channels, H//8, W//8)
        """
        # Encode
        mean, logvar = self.encode(x)
        
        # Sample latent
        if sample:
            z = self.reparameterize(mean, logvar)
        else:
            z = mean
        
        # Decode
        recon = self.decode(z)
        
        return recon, mean, logvar
    
    def sample(self, num_samples, device):
        """
        Sample from prior p(z) = N(0, I)
        
        Args:
            num_samples: Number of samples to generate
            device: torch device
        Returns:
            samples: (num_samples, C, H, W) - Generated images
        """
        z = torch.randn(
            num_samples, 
            self.latent_channels, 
            self.latent_size, 
            self.latent_size
        ).to(device)
        
        samples = self.decode(z)
        return samples


if __name__ == "__main__":
    # Test VAE
    print("Testing VAE...")
    
    vae = VAE(
        in_channels=3,
        latent_channels=4,
        hidden_dims=[128, 256, 512, 512],
        image_size=256,
        use_kan_decoder=True
    )
    
    # Test forward pass
    x = torch.randn(2, 3, 256, 256)
    recon, mean, logvar = vae(x)
    
    print(f"Input shape: {x.shape}")
    print(f"Latent mean shape: {mean.shape}")
    print(f"Latent logvar shape: {logvar.shape}")
    print(f"Reconstruction shape: {recon.shape}")
    
    # Test encoding only
    mean, logvar = vae.encode(x)
    print(f"\nEncode only - mean: {mean.shape}, logvar: {logvar.shape}")
    
    # Test decoding only
    z = torch.randn(2, 4, 32, 32)
    img = vae.decode(z)
    print(f"Decode only - output: {img.shape}")
    
    # Test sampling
    samples = vae.sample(4, 'cpu')
    print(f"Sampled images: {samples.shape}")
    
    print("\n✅ VAE tests passed!")
