"""
Improved VAE Training Script with:
- GAN discriminator
- EMA
- Mixed precision
- Better logging
- Proper validation
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

# Import models
from models.autoencoders.vae import VAE
from models.autoencoders.improved_losses import EnhancedVAELoss
from datasets.imagenet_dataset import ImageNetDataset
from utils.ema import EMA


def train_epoch_with_gan(
    model,
    dataloader,
    criterion,
    optimizer_vae,
    optimizer_disc,
    scaler,
    ema,
    device,
    epoch,
    writer,
    use_amp=True
):
    """
    Train for one epoch with GAN discriminator
    
    Training alternates between:
    1. Update discriminator
    2. Update VAE/generator
    """
    model.train()
    
    # Statistics
    stats = {
        'vae_loss': 0,
        'disc_loss': 0,
        'recon_loss': 0,
        'kl_loss': 0,
        'perc_loss': 0
    }
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    global_step = epoch * len(dataloader)
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        current_step = global_step + batch_idx
        
        # ========== Update Discriminator ==========
        optimizer_disc.zero_grad()
        
        with autocast(enabled=use_amp):
            # Forward VAE
            with torch.no_grad():
                recon, mean, logvar = model(images)
            
            # Discriminator loss
            disc_loss, disc_log = criterion(
                recon, images, mean, logvar,
                optimizer_idx=1  # Discriminator
            )
        
        # Backward discriminator
        scaler.scale(disc_loss).backward()
        scaler.unscale_(optimizer_disc)
        torch.nn.utils.clip_grad_norm_(criterion.discriminator.parameters(), max_norm=1.0)
        scaler.step(optimizer_disc)
        
        # ========== Update VAE/Generator ==========
        optimizer_vae.zero_grad()
        
        with autocast(enabled=use_amp):
            # Forward VAE
            recon, mean, logvar = model(images)
            
            # VAE loss (including GAN generator loss)
            vae_loss, vae_log = criterion(
                recon, images, mean, logvar,
                optimizer_idx=0  # Generator
            )
        
        # Backward VAE
        scaler.scale(vae_loss).backward()
        scaler.unscale_(optimizer_vae)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer_vae)
        
        # Update scaler
        scaler.update()
        
        # Update EMA
        if ema is not None:
            ema.update(model)
        
        # Logging
        stats['vae_loss'] += vae_log['total_loss']
        stats['disc_loss'] += disc_log.get('gan/d_loss', 0)
        stats['recon_loss'] += vae_log['recon_loss']
        stats['kl_loss'] += vae_log['kl_loss']
        stats['perc_loss'] += vae_log.get('perceptual_loss', 0)
        
        # TensorBoard logging (every 100 steps)
        if batch_idx % 100 == 0:
            for key, value in vae_log.items():
                writer.add_scalar(f'train/{key}', value, current_step)
            for key, value in disc_log.items():
                writer.add_scalar(f'train/{key}', value, current_step)
        
        # Progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix({
                'vae': f"{vae_log['total_loss']:.4f}",
                'disc': f"{disc_log.get('gan/d_loss', 0):.4f}",
                'recon': f"{vae_log['recon_loss']:.4f}",
                'kl': f"{vae_log['kl_loss']:.6f}"
            })
    
    # Average stats
    n = len(dataloader)
    return {k: v / n for k, v in stats.items()}


@torch.no_grad()
def validate(model, dataloader, criterion, device, ema=None):
    """Validate model with optional EMA"""
    
    # Use EMA weights if available
    if ema is not None:
        ema.apply_shadow(model)
    
    model.eval()
    
    stats = {
        'total_loss': 0,
        'recon_loss': 0,
        'kl_loss': 0
    }
    
    for batch in tqdm(dataloader, desc='Validating'):
        images = batch['image'].to(device)
        
        # Forward (no sampling for validation)
        recon, mean, logvar = model(images, sample=False)
        
        # Loss (VAE only, no GAN)
        loss, loss_dict = criterion(recon, images, mean, logvar, optimizer_idx=0)
        
        stats['total_loss'] += loss_dict['total_loss']
        stats['recon_loss'] += loss_dict['recon_loss']
        stats['kl_loss'] += loss_dict['kl_loss']
    
    # Restore original weights
    if ema is not None:
        ema.restore(model)
    
    n = len(dataloader)
    return {k: v / n for k, v in stats.items()}


def save_samples(model, val_loader, save_dir, epoch, device, num_samples=8):
    """Save reconstruction samples"""
    model.eval()
    
    # Get a batch
    batch = next(iter(val_loader))
    images = batch['image'][:num_samples].to(device)
    
    with torch.no_grad():
        recon, _, _ = model(images, sample=False)
    
    # Save comparison
    from torchvision.utils import save_image, make_grid
    
    comparison = torch.cat([images, recon])
    grid = make_grid(comparison, nrow=num_samples, normalize=True)
    
    save_path = os.path.join(save_dir, f'recon_epoch{epoch:03d}.png')
    save_image(grid, save_path)
    
    print(f"   Saved samples to {save_path}")


def main(args):
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    print(f"Mixed precision: {args.use_amp}")
    print(f"EMA decay: {args.ema_decay}")
    
    # Create directories
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    
    # TensorBoard
    writer = SummaryWriter(os.path.join(exp_dir, 'logs'))
    
    # Dataset
    print("\nLoading dataset...")
    train_dataset = ImageNetDataset(
        data_root=args.data_root,
        split='train',
        image_size=args.image_size,
        center_crop=False  # Use augmentation
    )
    
    val_dataset = ImageNetDataset(
        data_root=args.data_root,
        split='val',
        image_size=args.image_size,
        center_crop=True  # No augmentation
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        persistent_workers=True if args.num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print(f"Train: {len(train_dataset)} images")
    print(f"Val: {len(val_dataset)} images")
    
    # Model
    print("\nCreating VAE model...")
    model = VAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        hidden_dims=args.hidden_dims,
        image_size=args.image_size,
        use_kan_decoder=args.use_kan
    ).to(device)
    
    # Loss (with GAN)
    criterion = EnhancedVAELoss(
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        use_perceptual=args.use_perceptual,
        use_gan=args.use_gan,
        gan_weight=args.gan_weight,
        disc_start_epoch=args.disc_start_epoch
    )
    
    # Move discriminator to device
    if args.use_gan:
        criterion.discriminator.to(device)
    
    # Optimizers
    optimizer_vae = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        weight_decay=0.01
    )
    
    optimizer_disc = None
    if args.use_gan:
        optimizer_disc = AdamW(
            criterion.discriminator.parameters(),
            lr=args.lr * 0.5,  # Discriminator learns slower
            betas=(0.9, 0.999),
            weight_decay=0.01
        )
    
    # Scheduler
    scheduler_vae = CosineAnnealingLR(
        optimizer_vae,
        T_max=args.epochs,
        eta_min=args.lr * 0.01
    )
    
    scheduler_disc = None
    if args.use_gan:
        scheduler_disc = CosineAnnealingLR(
            optimizer_disc,
            T_max=args.epochs,
            eta_min=args.lr * 0.005
        )
    
    # EMA
    ema = None
    if args.use_ema:
        ema = EMA(model, decay=args.ema_decay, device=device)
        print(f"EMA enabled with decay={args.ema_decay}")
    
    # Mixed precision
    scaler = GradScaler(enabled=args.use_amp)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\n{'='*80}")
    print(f"Starting training for {args.epochs} epochs")
    print(f"{'='*80}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Update discriminator schedule
        criterion.set_epoch(epoch)
        
        # Train
        if args.use_gan:
            train_stats = train_epoch_with_gan(
                model, train_loader, criterion,
                optimizer_vae, optimizer_disc,
                scaler, ema, device, epoch, writer,
                use_amp=args.use_amp
            )
        else:
            # Fallback to simple training without GAN
            # (implement similar to train_epoch_with_gan but simpler)
            pass
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - VAE Loss: {train_stats['vae_loss']:.4f}, "
              f"Disc Loss: {train_stats['disc_loss']:.4f}")
        
        # Validate
        if epoch % args.val_every == 0:
            val_stats = validate(model, val_loader, criterion, device, ema)
            
            print(f"Val   - Loss: {val_stats['total_loss']:.4f}, "
                  f"Recon: {val_stats['recon_loss']:.4f}, "
                  f"KL: {val_stats['kl_loss']:.6f}")
            
            # TensorBoard
            for key, value in val_stats.items():
                writer.add_scalar(f'val/{key}', value, epoch)
            
            # Save best model
            if val_stats['total_loss'] < best_val_loss:
                best_val_loss = val_stats['total_loss']
                
                checkpoint = {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_vae_state_dict': optimizer_vae.state_dict(),
                    'val_loss': val_stats['total_loss']
                }
                
                if ema is not None:
                    checkpoint['ema_state_dict'] = ema.state_dict()
                
                if args.use_gan:
                    checkpoint['discriminator_state_dict'] = criterion.discriminator.state_dict()
                    checkpoint['optimizer_disc_state_dict'] = optimizer_disc.state_dict()
                
                torch.save(
                    checkpoint,
                    os.path.join(exp_dir, 'checkpoints', 'vae_best.pth')
                )
                
                print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save samples
        if epoch % args.sample_every == 0:
            save_samples(model, val_loader, os.path.join(exp_dir, 'samples'), epoch, device)
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_vae_state_dict': optimizer_vae.state_dict(),
                'scheduler_vae_state_dict': scheduler_vae.state_dict()
            }
            
            if ema is not None:
                checkpoint['ema_state_dict'] = ema.state_dict()
            
            if args.use_gan:
                checkpoint['discriminator_state_dict'] = criterion.discriminator.state_dict()
                checkpoint['optimizer_disc_state_dict'] = optimizer_disc.state_dict()
                checkpoint['scheduler_disc_state_dict'] = scheduler_disc.state_dict()
            
            torch.save(
                checkpoint,
                os.path.join(exp_dir, 'checkpoints', f'vae_epoch{epoch}.pth')
            )
        
        # Step schedulers
        scheduler_vae.step()
        if scheduler_disc is not None:
            scheduler_disc.step()
        
        # Log learning rates
        writer.add_scalar('lr/vae', optimizer_vae.param_groups[0]['lr'], epoch)
        if optimizer_disc is not None:
            writer.add_scalar('lr/disc', optimizer_disc.param_groups[0]['lr'], epoch)
    
    writer.close()
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE with GAN (Improved)')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--image_size', type=int, default=256)
    
    # Model
    parser.add_argument('--latent_channels', type=int, default=4)
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 512, 512])
    parser.add_argument('--use_kan', action='store_true')
    
    # Loss
    parser.add_argument('--kl_weight', type=float, default=1e-6)
    parser.add_argument('--perceptual_weight', type=float, default=1.0)
    parser.add_argument('--use_perceptual', action='store_true', default=True)
    parser.add_argument('--use_gan', action='store_true', default=True)
    parser.add_argument('--gan_weight', type=float, default=0.5)
    parser.add_argument('--disc_start_epoch', type=int, default=10)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=8)
    
    # Optimization
    parser.add_argument('--use_amp', action='store_true', default=True)
    parser.add_argument('--use_ema', action='store_true', default=True)
    parser.add_argument('--ema_decay', type=float, default=0.9999)
    
    # Logging
    parser.add_argument('--exp_name', type=str, default='vae_imagenet_improved')
    parser.add_argument('--val_every', type=int, default=1)
    parser.add_argument('--sample_every', type=int, default=5)
    parser.add_argument('--save_every', type=int, default=10)
    
    args = parser.parse_args()
    
    main(args)
    