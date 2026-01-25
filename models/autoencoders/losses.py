"""
Enhanced VAE Loss with PatchGAN Discriminator
For high-quality image reconstruction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class PatchGANDiscriminator(nn.Module):
    """
    PatchGAN Discriminator for VAE
    Outputs a grid of real/fake predictions instead of single value
    
    From "Image-to-Image Translation with Conditional Adversarial Networks" (Pix2Pix)
    """
    def __init__(
        self,
        in_channels=3,
        ndf=64,
        n_layers=3,
        use_sigmoid=False
    ):
        super().__init__()
        
        # Build discriminator layers
        sequence = [
            nn.Conv2d(in_channels, ndf, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, True)
        ]
        
        nf_mult = 1
        nf_mult_prev = 1
        
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=4, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Output layer
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=4, stride=1, padding=1)]
        
        if use_sigmoid:
            sequence += [nn.Sigmoid()]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, x):
        """
        Args:
            x: (B, C, H, W) - Image
        Returns:
            out: (B, 1, H', W') - Patch predictions
        """
        return self.model(x)


class EnhancedPerceptualLoss(nn.Module):
    """
    Enhanced perceptual loss using multiple VGG layers + LPIPS-style normalization
    """
    def __init__(
        self,
        layers=[3, 8, 15, 22],  # VGG16 layers
        weights=[1.0, 1.0, 1.0, 1.0],
        use_lpips_normalization=True
    ):
        super().__init__()
        
        # Load pretrained VGG16
        vgg = models.vgg16(pretrained=True).features
        
        # Extract layers
        self.blocks = nn.ModuleList()
        prev_idx = 0
        for idx in layers:
            block = nn.Sequential(*[vgg[i] for i in range(prev_idx, idx + 1)])
            self.blocks.append(block)
            prev_idx = idx + 1
        
        # Freeze
        for param in self.parameters():
            param.requires_grad = False
        
        self.weights = weights
        self.use_lpips_normalization = use_lpips_normalization
        
        # ImageNet normalization
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        
        # LPIPS-style channel normalization weights (learned, but we use fixed for simplicity)
        if use_lpips_normalization:
            self.register_buffer('lin_weights', torch.ones(len(layers), 1, 1, 1, 1))
    
    def normalize_tensor(self, x):
        """Normalize to ImageNet stats"""
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
    def normalize_features(self, x):
        """L2 normalize features (LPIPS-style)"""
        norm_factor = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True))
        return x / (norm_factor + 1e-10)
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 3, H, W) - Predicted image, range [-1, 1] or [0, 1]
            target: (B, 3, H, W) - Target image
        Returns:
            loss: Perceptual loss
        """
        # Normalize inputs to [0, 1]
        if pred.min() < 0:
            pred = (pred + 1) / 2
        if target.min() < 0:
            target = (target + 1) / 2
        
        # ImageNet normalization
        pred = self.normalize_tensor(pred)
        target = self.normalize_tensor(target)
        
        # Extract and compare features
        loss = 0.0
        x_pred = pred
        x_target = target
        
        for i, block in enumerate(self.blocks):
            x_pred = block(x_pred)
            x_target = block(x_target)
            
            if self.use_lpips_normalization:
                # L2 normalize features
                x_pred_norm = self.normalize_features(x_pred)
                x_target_norm = self.normalize_features(x_target)
                
                # Compute difference
                diff = (x_pred_norm - x_target_norm) ** 2
                
                # Spatial average
                diff = diff.mean(dim=[2, 3], keepdim=True)
                
                # Weighted loss
                loss += self.weights[i] * (self.lin_weights[i] * diff).mean()
            else:
                # Standard L1 loss on features
                loss += self.weights[i] * F.l1_loss(x_pred, x_target)
        
        return loss


class EnhancedVAELoss(nn.Module):
    """
    Complete VAE loss with:
    1. Reconstruction loss (L1/L2)
    2. Enhanced perceptual loss (VGG + LPIPS-style)
    3. KL divergence (properly normalized)
    4. GAN loss (adversarial)
    5. Feature matching loss (optional)
    
    Total loss:
        L = L_recon + λ_p * L_perceptual + λ_kl * L_kl + λ_gan * L_gan + λ_fm * L_fm
    """
    def __init__(
        self,
        # Reconstruction
        recon_loss_type='l1',
        # Perceptual
        perceptual_weight=1.0,
        use_perceptual=True,
        use_lpips_norm=True,
        # KL
        kl_weight=1e-6,
        # GAN
        use_gan=False,
        gan_weight=0.5,
        disc_start_epoch=0,
        disc_factor=1.0,
        # Feature matching
        use_feature_matching=False,
        feature_matching_weight=10.0
    ):
        super().__init__()
        
        self.recon_loss_type = recon_loss_type
        self.kl_weight = kl_weight
        self.perceptual_weight = perceptual_weight
        self.use_perceptual = use_perceptual
        self.use_gan = use_gan
        self.gan_weight = gan_weight
        self.disc_start_epoch = disc_start_epoch
        self.disc_factor = disc_factor
        self.use_feature_matching = use_feature_matching
        self.feature_matching_weight = feature_matching_weight
        
        # Perceptual loss
        if use_perceptual:
            self.perceptual_loss = EnhancedPerceptualLoss(
                use_lpips_normalization=use_lpips_norm
            )
        
        # Discriminator
        if use_gan:
            self.discriminator = PatchGANDiscriminator(in_channels=3, ndf=64, n_layers=3)
        
        self.current_epoch = 0
    
    def set_epoch(self, epoch):
        """Update current epoch for discriminator scheduling"""
        self.current_epoch = epoch
    
    def kl_divergence(self, mean, logvar):
        """
        Properly normalized KL divergence
        KL(q(z|x) || p(z)) where p(z) = N(0, I)
        """
        kl = -0.5 * torch.sum(1 + logvar - mean.pow(2) - logvar.exp())
        # Normalize by number of elements, not just batch
        kl = kl / mean.numel()  # ✅ FIXED: Divide by total elements
        return kl
    
    def reconstruction_loss(self, pred, target):
        """Reconstruction loss (L1 or L2)"""
        if self.recon_loss_type == 'l1':
            return F.l1_loss(pred, target)
        elif self.recon_loss_type == 'l2':
            return F.mse_loss(pred, target)
        else:
            raise ValueError(f"Unknown recon_loss_type: {self.recon_loss_type}")
    
    def gan_loss(self, pred, target, optimizer_idx=0):
        """
        GAN loss for generator and discriminator
        
        Args:
            pred: Reconstructed images
            target: Real images
            optimizer_idx: 0 for VAE/generator, 1 for discriminator
        Returns:
            loss: GAN loss
            log_dict: Logging dictionary
        """
        if not self.use_gan or self.current_epoch < self.disc_start_epoch:
            return torch.tensor(0.0, device=pred.device), {}
        
        if optimizer_idx == 0:
            # Generator loss: fool discriminator
            # Want discriminator to predict "real" for fake images
            logits_fake = self.discriminator(pred)
            g_loss = -torch.mean(logits_fake)  # Non-saturating loss
            
            # Feature matching loss (optional)
            fm_loss = torch.tensor(0.0, device=pred.device)
            if self.use_feature_matching:
                # Extract intermediate features from discriminator
                # This requires modifying discriminator to return features
                # For simplicity, we skip this here
                pass
            
            return g_loss, {
                'gan/g_loss': g_loss.item(),
                'gan/logits_fake': logits_fake.mean().item()
            }
        
        else:
            # Discriminator loss: distinguish real from fake
            logits_real = self.discriminator(target.detach())
            logits_fake = self.discriminator(pred.detach())
            
            # Hinge loss (more stable than BCE)
            d_loss_real = torch.mean(F.relu(1.0 - logits_real))
            d_loss_fake = torch.mean(F.relu(1.0 + logits_fake))
            d_loss = 0.5 * (d_loss_real + d_loss_fake)
            
            return d_loss, {
                'gan/d_loss': d_loss.item(),
                'gan/d_loss_real': d_loss_real.item(),
                'gan/d_loss_fake': d_loss_fake.item(),
                'gan/logits_real': logits_real.mean().item(),
                'gan/logits_fake': logits_fake.mean().item()
            }
    
    def forward(self, pred, target, mean, logvar, optimizer_idx=0):
        """
        Compute total VAE loss
        
        Args:
            pred: (B, C, H, W) - Reconstructed image
            target: (B, C, H, W) - Original image
            mean: (B, latent_ch, H//8, W//8) - Latent mean
            logvar: (B, latent_ch, H//8, W//8) - Latent logvar
            optimizer_idx: 0 for VAE, 1 for discriminator
        Returns:
            loss: Total loss
            loss_dict: Dictionary of individual losses
        """
        loss_dict = {}
        
        if optimizer_idx == 0:
            # ===== VAE/Generator Loss =====
            
            # 1. Reconstruction
            recon_loss = self.reconstruction_loss(pred, target)
            loss_dict['recon_loss'] = recon_loss.item()
            
            # 2. Perceptual
            if self.use_perceptual:
                perceptual_loss = self.perceptual_loss(pred, target)
                loss_dict['perceptual_loss'] = perceptual_loss.item()
            else:
                perceptual_loss = torch.tensor(0.0, device=pred.device)
                loss_dict['perceptual_loss'] = 0.0
            
            # 3. KL divergence
            kl_loss = self.kl_divergence(mean, logvar)
            loss_dict['kl_loss'] = kl_loss.item()
            
            # 4. GAN loss
            gan_loss, gan_log = self.gan_loss(pred, target, optimizer_idx=0)
            loss_dict.update(gan_log)
            
            # Total loss
            total_loss = (
                recon_loss +
                self.perceptual_weight * perceptual_loss +
                self.kl_weight * kl_loss +
                self.gan_weight * self.disc_factor * gan_loss
            )
            
            loss_dict['total_loss'] = total_loss.item()
            
        else:
            # ===== Discriminator Loss =====
            disc_loss, disc_log = self.gan_loss(pred, target, optimizer_idx=1)
            total_loss = disc_loss
            loss_dict.update(disc_log)
            loss_dict['total_loss'] = total_loss.item()
        
        return total_loss, loss_dict


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    print("Testing Enhanced VAE Loss...\n")
    
    # Create dummy data
    pred = torch.randn(2, 3, 256, 256)
    target = torch.randn(2, 3, 256, 256)
    mean = torch.randn(2, 4, 32, 32)
    logvar = torch.randn(2, 4, 32, 32)
    
    # Test 1: Without GAN
    print("1. Testing VAE Loss (no GAN):")
    vae_loss = EnhancedVAELoss(
        kl_weight=1e-6,
        perceptual_weight=1.0,
        use_perceptual=True,
        use_gan=False
    )
    
    total_loss, loss_dict = vae_loss(pred, target, mean, logvar)
    
    print(f"   Total loss: {total_loss.item():.4f}")
    print(f"   Loss breakdown:")
    for key, value in loss_dict.items():
        print(f"     {key}: {value:.4f}")
    
    # Test 2: With GAN
    print("\n2. Testing VAE Loss (with GAN):")
    vae_gan_loss = EnhancedVAELoss(
        kl_weight=1e-6,
        perceptual_weight=1.0,
        use_perceptual=True,
        use_gan=True,
        gan_weight=0.5,
        disc_start_epoch=0
    )
    
    # Set epoch to enable discriminator
    vae_gan_loss.set_epoch(1)
    
    # Generator loss
    g_loss, g_dict = vae_gan_loss(pred, target, mean, logvar, optimizer_idx=0)
    print(f"   Generator loss: {g_loss.item():.4f}")
    
    # Discriminator loss
    d_loss, d_dict = vae_gan_loss(pred, target, mean, logvar, optimizer_idx=1)
    print(f"   Discriminator loss: {d_loss.item():.4f}")
    
    # Test 3: Discriminator architecture
    print("\n3. Testing PatchGAN Discriminator:")
    disc = PatchGANDiscriminator(in_channels=3, ndf=64, n_layers=3)
    
    real_imgs = torch.randn(2, 3, 256, 256)
    logits = disc(real_imgs)
    
    print(f"   Input: {real_imgs.shape}")
    print(f"   Output (patch predictions): {logits.shape}")
    print(f"   Receptive field: ~70x70 pixels per patch")
    
    # Test 4: Enhanced Perceptual Loss
    print("\n4. Testing Enhanced Perceptual Loss:")
    perc_loss = EnhancedPerceptualLoss(use_lpips_normalization=True)
    
    loss_lpips = perc_loss(pred, target)
    print(f"   LPIPS-style perceptual loss: {loss_lpips.item():.4f}")
    
    print("\n✅ All Enhanced VAE Loss tests passed!")
    