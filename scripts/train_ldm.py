"""
Training script for Latent Diffusion Model (Stage 2)
Train text-conditional diffusion model with frozen VAE
"""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

# Import models and datasets
from models.ldm_model import LatentDiffusionModel
from datasets.coco_dataset import COCODataset


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    # VAE and text encoder stay frozen
    model.vae.eval()
    model.text_encoder.model.eval()
    
    total_loss = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        captions = batch['caption']  # List of strings
        
        # Forward
        loss = model(images, captions)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.denoiser.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Logging
        total_loss += loss.item()
        
        if batch_idx % 100 == 0:
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
    
    return total_loss / len(dataloader)


@torch.no_grad()
def validate(model, dataloader, device):
    """Validate model"""
    model.eval()
    
    total_loss = 0
    
    for batch in tqdm(dataloader, desc='Validating'):
        images = batch['image'].to(device)
        captions = batch['caption']
        
        # Forward
        loss = model(images, captions)
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


@torch.no_grad()
def generate_samples(model, prompts, save_dir, epoch, device):
    """Generate sample images"""
    model.eval()
    
    images = model.generate(
        captions=prompts,
        num_steps=50,
        guidance_scale=7.5,
        height=256,
        width=256
    )
    
    # Save images
    from torchvision.utils import save_image
    
    for i, img in enumerate(images):
        # Denormalize from [-1, 1] to [0, 1]
        img = (img + 1) / 2
        save_image(img, os.path.join(save_dir, f'epoch{epoch}_sample{i}.png'))


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create experiment directory
    exp_dir = os.path.join('experiments', args.exp_name)
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(exp_dir, 'samples'), exist_ok=True)
    
    # Dataset
    print("Loading COCO dataset...")
    train_dataset = COCODataset(
        data_root=args.data_root,
        split='train',
        image_size=args.image_size
    )
    
    val_dataset = COCODataset(
        data_root=args.data_root,
        split='val',
        image_size=args.image_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True  # For consistent batch size with gradient accumulation
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Model
    print("Creating LDM model...")
    model = LatentDiffusionModel(
        vae_config={
            'in_channels': 3,
            'latent_channels': 4,
            'hidden_dims': [128, 256, 512, 512],
            'image_size': args.image_size,
            'use_kan_decoder': args.use_kan
        },
        vae_checkpoint=args.vae_checkpoint,
        denoiser_config={
            'in_channels': 4,
            'out_channels': 4,
            'model_channels': args.model_channels,
            'channel_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attention_resolutions': [1, 2, 3],
            'context_dim': 768,
            'num_heads': 8,
            'use_cross_attn': True
        },
        timesteps=1000,
        beta_schedule='linear',
        unconditional_prob=0.1,
        device=device
    ).to(device)
    
    # Optimizer (only denoiser parameters)
    optimizer = AdamW(model.denoiser.parameters(), lr=args.lr, weight_decay=0.01)
    
    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.1)
    
    # Sample prompts for visualization
    sample_prompts = [
        "A cat sitting on a table",
        "A beautiful sunset over mountains",
        "A dog playing in the park",
        "A bird flying in the sky"
    ]
    
    # Training loop
    best_val_loss = float('inf')
    
    print(f"\nStarting LDM training for {args.epochs} epochs...")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"VAE checkpoint: {args.vae_checkpoint}")
    print(f"Model channels: {args.model_channels}\n")
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        print(f"\nEpoch {epoch}/{args.epochs}")
        print(f"Train Loss: {train_loss:.4f}")
        
        # Validate
        if epoch % args.val_every == 0:
            val_loss = validate(model, val_loader, device)
            print(f"Val Loss: {val_loss:.4f}")
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    model.denoiser.state_dict(),
                    os.path.join(exp_dir, 'checkpoints', 'ldm_best.pth')
                )
                print(f"✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # Generate samples
        if epoch % args.sample_every == 0:
            print("Generating samples...")
            generate_samples(
                model,
                sample_prompts,
                os.path.join(exp_dir, 'samples'),
                epoch,
                device
            )
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save(
                {
                    'epoch': epoch,
                    'denoiser_state_dict': model.denoiser.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict(),
                    'val_loss': val_loss if epoch % args.val_every == 0 else None
                },
                os.path.join(exp_dir, 'checkpoints', f'ldm_epoch{epoch}.pth')
            )
        
        # Step scheduler
        scheduler.step()
    
    print(f"\n✅ Training completed! Best val loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train LDM (Stage 2)')
    
    # Data
    parser.add_argument('--data_root', type=str, required=True, help='Path to COCO dataset')
    parser.add_argument('--image_size', type=int, default=256, help='Image size')
    
    # Model
    parser.add_argument('--vae_checkpoint', type=str, required=True, help='Path to pretrained VAE')
    parser.add_argument('--model_channels', type=int, default=320, help='Base model channels')
    parser.add_argument('--use_kan', action='store_true', help='Use KAN decoder in VAE')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--epochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4, help='Dataloader workers')
    
    # Logging
    parser.add_argument('--exp_name', type=str, default='ldm_coco', help='Experiment name')
    parser.add_argument('--val_every', type=int, default=5, help='Validate every N epochs')
    parser.add_argument('--sample_every', type=int, default=10, help='Generate samples every N epochs')
    parser.add_argument('--save_every', type=int, default=50, help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    main(args)
