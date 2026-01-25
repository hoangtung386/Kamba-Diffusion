"""
Text-to-image generation script
Generate images from text prompts using trained LDM
"""

import argparse
import torch
from torchvision.utils import save_image
import os

from models.ldm_model import LatentDiffusionModel


def main(args):
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load model
    print("Loading LDM model...")
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
            'model_channels': 320,
            'channel_mult': [1, 2, 4, 4],
            'num_res_blocks': 2,
            'attention_resolutions': [1, 2, 3],
            'context_dim': 768,
            'num_heads': 8,
            'use_cross_attn': True
        },
        device=device
    ).to(device)
    
    # Load denoiser weights
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.denoiser.load_state_dict(torch.load(args.checkpoint, map_location=device))
    
    model.eval()
    
    # Prepare prompts
    if args.prompt:
        prompts = [args.prompt]
    elif args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default prompts
        prompts = [
            "A photo of a cat sitting on a table",
            "A beautiful sunset over mountains",
            "A dog playing in the park",
            "A modern cityscape at night",
            "A bird flying in the blue sky"
        ]
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {len(prompts)} images...")
    print(f"Guidance scale: {args.guidance_scale}")
    print(f"Sampling steps: {args.num_steps}")
    print(f"Number of samples per prompt: {args.num_samples}\n")
    
    # Generate images
    for idx, prompt in enumerate(prompts):
        print(f"[{idx+1}/{len(prompts)}] Prompt: {prompt}")
        
        # Generate multiple samples for this prompt
        batch_prompts = [prompt] * args.num_samples
        
        images = model.generate(
            captions=batch_prompts,
            num_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            height=args.image_size,
            width=args.image_size
        )
        
        # Save images
        for i, img in enumerate(images):
            # Denormalize from [-1, 1] to [0, 1]
            img = (img + 1) / 2
            img = torch.clamp(img, 0, 1)
            
            # Save
            filename = f"{idx:03d}_{i}.png" if args.num_samples > 1 else f"{idx:03d}.png"
            save_path = os.path.join(args.output_dir, filename)
            save_image(img, save_path)
            
            print(f"  Saved: {save_path}")
    
    print(f"\n✅ Generated {len(prompts) * args.num_samples} images in {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate images from text prompts')
    
    # Model
    parser.add_argument('--vae_checkpoint', type=str, required=True, help='Path to VAE checkpoint')
    parser.add_argument('--checkpoint', type=str, help='Path to LDM denoiser checkpoint')
    parser.add_argument('--use_kan', action='store_true', help='Use KAN decoder in VAE')
    
    # Prompts
    parser.add_argument('--prompt', type=str, help='Single text prompt')
    parser.add_argument('--prompts_file', type=str, help='File with prompts (one per line)')
    
    # Generation
    parser.add_argument('--num_steps', type=int, default=50, help='DDIM sampling steps')
    parser.add_argument('--guidance_scale', type=float, default=7.5, help='Classifier-free guidance scale')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples per prompt')
    parser.add_argument('--image_size', type=int, default=256, help='Output image size')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    
    args = parser.parse_args()
    
    main(args)
