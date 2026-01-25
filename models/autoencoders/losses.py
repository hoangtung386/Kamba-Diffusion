"""
Loss functions for VAE training
Includes reconstruction, perceptual, and KL divergence losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG16 features
    Measures high-level feature similarity instead of pixel-wise
    """
    def __init__(self, layers=[3, 8, 15, 22], weights=[1.0, 1.0, 1.0, 1.0]):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Extract specific layers
        self.blocks = nn.ModuleList()
        prev_idx = 0
        for idx in layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev_idx, idx + 1)])
            self.blocks.append(block)
            prev_idx = idx + 1
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = weights
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    
    def normalize(self, x):
        """Normalize to ImageNet stats"""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 3, H, W) - Predicted image, range [-1, 1] or [0, 1]
            target: (B, 3, H, W) - Target image
        Returns:
            loss: Perceptual loss
        """
        # Normalize inputs to [0, 1] if needed
        if pred.min() < 0:
            pred = (pred + 1) / 2
        if target.min() < 0:
            target = (target + 1) / 2
        
        # Apply ImageNet normalization
        pred = self.normalize(pred)
        target = self.normalize(target)
        
        # Extract features
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, block in enumerate(self.blocks):
            x_pred = block(x_pred)
            x_target = block(x_target)
            
            # L1 loss on features
            loss += self.weights[i] * F.l1_loss(x_pred, x_target)
        
        return loss


class VAELoss(nn.Module):
    """
    Combined loss for VAE training:
        L = L_recon + kl_weight * L_kl + perceptual_weight * L_perceptual
    """
    def __init__(
        self,
        kl_weight=1e-6,
        perceptual_weight=1.0,
        recon_loss_type='l1',  # 'l1' or 'l2'
        use_perceptual=True
    ):
        super().__init__()
        
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.recon_loss_type = recon_loss_type
        self.use_perceptual = use_perceptual
        
        # Perceptual loss
        if use_perceptual:
            self.perceptual_loss = PerceptualLoss()
    
    def kl_divergence(self, mean, logvar):
        """
        KL divergence: KL(q(z|x) || p(z)) where p(z) = N(0, I)
        
        KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
        
        Args:
            mean: (B, C, H, W)
            logvar: (B, C, H, W)
        Returns:
            kl: scalar
        """
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        # Normalize by batch size
        kl = kl / mean.shape[0]
        return kl
    
    def reconstruction_loss(self, pred, target):
        """
        Reconstruction loss (L1 or L2)
        
        Args:
            pred: (B, C, H, W)
            target: (B, C, H, W)
        Returns:
            loss: scalar
        """
        if self.recon_loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.recon_loss_type == 'l2':
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")
    
    def forward(self, pred, target, mean, logvar):
        """
        Compute total VAE loss
        
        Args:
            pred: (B, C, H, W) - Reconstructed image
            target: (B, C, H, W) - Original image
            mean: (B, latent_ch, H//8, W//8) - Latent mean
            logvar: (B, latent_ch, H//8, W//8) - Latent logvar
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        # Reconstruction loss
        recon_loss = self.reconstruction_loss(pred, target)
        
        # Perceptual loss
        if self.use_perceptual:
            perceptual_loss = self.perceptual_loss(pred, target)
        else:
            perceptual_loss = torch.tensor(0.0).to(pred.device)
        
        # KL divergence
        kl_loss = self.kl_divergence(mean, logvar)
        
        # Total loss
        total_loss = (
            recon_loss + 
            self.perceptual_weight * perceptual_loss + 
            self.kl_weight * kl_loss
        )
        
        # Loss dictionary for logging
        loss_dict = {
            'total_loss': total_loss.item(),
            'recon_loss': recon_loss.item(),
            'perceptual_loss': perceptual_loss.item() if self.use_perceptual else 0.0,
            'kl_loss': kl_loss.item()
        }
        
        return total_loss, loss_dict


class AdversarialLoss(nn.Module):
    """
    Optional: GAN-based adversarial loss for higher quality
    TODO: Implement discriminator and GAN training
    """
    def __init__(self):
        super().__init__()
        raise NotImplementedError("GAN loss not yet implemented")


if __name__ == "__main__":
    # Test losses
    print("Testing VAE losses...")
    
    # Create dummy data
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    mean = torch.randn(2, 4, 32, 32)
    logvar = torch.randn(2, 4, 32, 32)
    
    # Test VAE loss
    vae_loss = VAELoss(
        kl_weight=1e-6,
        perceptual_weight=1.0,
        use_perceptual=True
    )
    
    total_loss, loss_dict = vae_loss(pred, target, mean, logvar)
    
    print(f"Total loss: {total_loss.item():.4f}")
    print(f"Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"  {key}: {value:.4f}")
    
    # Test perceptual loss alone
    perceptual = PerceptualLoss()
    p_loss = perceptual(pred, target)
    print(f"\nPerceptual loss only: {p_loss.item():.4f}")
    
    print("\n✅ Loss tests passed!")
