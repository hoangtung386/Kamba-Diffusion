"""
Training script for VAE pretraining (Stage 1)
Train autoencoder on ImageNet or LAION-Aesthetics
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np

# Import models and datasets
from models.autoencoders.vae import VAE
from models.autoencoders.losses import VAELoss
from datasets.imagenet_dataset import ImageNetDataset


def train_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    total_perc = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        
        # Forward
        recon, mean, logvar = model(images)
        
        # Loss
        loss, loss_dict = criterion(recon, images, mean, logvar)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        total_loss += loss_dict['total_loss']
        total_recon += loss_dict['recon_loss']
        total_kl += loss_dict['kl_loss']
        total_perc += loss_dict.get('perceptual_loss', 0.0)
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({
                'loss': f"{loss_dict['total_loss']:.4f}",
                'recon': f"{loss_dict['recon_loss']:.4f}",
                'kl': f"{loss_dict['kl_loss']:.6f}"
            })
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'recon_loss': total_recon / n,
        'kl_loss': total_kl / n,
        'perceptual_loss': total_perc / n
    }


@torch.no_grad()
def validate(model, dataloader, criterion, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    total_recon = 0
    total_kl = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        images = batch['image'].to(device)
        
        # Forward
        recon, mean, logvar = model(images, sample=False)  # Use mean, no sampling
        
        # Loss
        loss, loss_dict = criterion(recon, images, mean, logvar)
        
        total_loss += loss_dict['total_loss']
        total_recon += loss_dict['recon_loss']
        total_kl += loss_dict['kl_loss']
    
    n = len(dataloader)
    return {
        'total_loss': total_loss / n,
        'recon_loss': total_recon / n,
        'kl_loss': total_kl / n
    }


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    
    # Dataset
    print("Loading dataset...")
    train_dataset = ImageNetDataset(
        data_root=args.data_root,
        split='train',
        image_size=args.image_size
    )
    
    val_dataset = ImageNetDataset(
        data_root=args.data_root,
        split='val',
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print("Creating VAE model...")
    model = VAE(
        in_channels=3,
        latent_channels=args.latent_channels,
        hidden_dims=args.hidden_dims,
        image_size=args.image_size,
        use_kan_decoder=args.use_kan
    ).to(device)
    
    # Loss
    criterion = VAELoss(
        kl_weight=args.kl_weight,
        perceptual_weight=args.perceptual_weight,
        use_perceptual=args.use_perceptual
    )
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\nStarting training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"KL weight: {args.kl_weight}")
    print(f"Use KAN decoder: {args.use_kan}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train - Loss: {train_metrics['total_loss']:.4f}, "
              f"Recon: {train_metrics['recon_loss']:.4f}, "
              f"KL: {train_metrics['kl_loss']:.6f}")
        
        # Validate
        if epoch % args.val_every == 0:
            val_metrics = validate(model, val_loader, criterion, device)
            print(f"Val   - Loss: {val_metrics['total_loss']:.4f}, "
                  f"Recon: {val_metrics['recon_loss']:.4f}, "
                  f"KL: {val_metrics['kl_loss']:.6f}")
            
            # Save best model
            if val_metrics['total_loss'] < best_val_loss:
                best_val_loss = val_metrics['total_loss']
                torch.save(
                    model.state_dict(),
                    os.path.join(exp_dir, 'checkpoints', 'vae_best.pth')
                )
                print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_metrics.get('total_loss', None)
                },
                os.path.join(exp_dir, 'checkpoints', f'vae_epoch{epoch}.pth')
            )
        
        # Step scheduler
        scheduler.step()
    
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train VAE (Stage 1)')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to ImageNet dataset')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    
    # Model
    parser.add_argument('--latent_channels', type=int, default=4, help='Latent channels')
    parser.add_argument('--hidden_dims', type=int, nargs='+', default=[128, 256, 512, 512], help='Hidden dimensions')
    parser.add_argument('--use_kan', action='store_true', help='Use KAN decoder')
    
    # Loss
    parser.add_argument('--kl_weight', type=float, default=1e-6, help='KL divergence weight')
    parser.add_argument('--perceptual_weight', type=float, default=1.0, help='Perceptual loss weight')
    parser.add_argument('--use_perceptual', action='store_true', default=True, help='Use perceptual loss')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=256, help='Batch size')
    parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    # Logging
    parser.add_argument('--exp_name', type=str, default='vae_imagenet', help='Experiment name')
    parser.add_argument('--val_every', type=int, default=1, help='Validate every N epochs')
    parser.add_argument('--save_every', type=int, default=10, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
