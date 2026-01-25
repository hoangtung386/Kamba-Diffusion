"""
Latent Diffusion Model (LDM) - Main Pipeline
Integrates VAE + Text Encoder + Denoiser for text-to-image generation
"""

import torch
import torch.nn as nn
from tqdm import tqdm

# Import components
from models.autoencoders.vae import VAE
from models.text_encoders.clip_encoder import CLIPTextEncoder
from models.denoisers.mamba_unet import MambaUNet
from models.diffusion.ddpm import DDPM
from models.diffusion.guidance import classifier_free_guidance


class LatentDiffusionModel(nn.Module):
    """
    Complete Latent Diffusion Model for text-to-image generation
    
    Components:
        1. VAE: Compress images to latent space (frozen after pretraining)
        2. Text Encoder: CLIP for text embeddings (frozen)
        3. Denoiser: Mamba U-Net for diffusion (trainable)
        4. DDPM: Diffusion process
    
    Pipeline:
        Text → CLIP → Context
        Image → VAE Encoder → Latent z
        z_noisy + context + timestep → Mamba U-Net → z_pred
        z_clean → VAE Decoder → Image
    """
    def __init__(
        self,
        # VAE config
        vae_config=None,
        vae_checkpoint=None,
        # Text encoder config
        text_encoder_model="openai/clip-vit-large-patch14",
        # Denoiser config
        denoiser_config=None,
        # Diffusion config
        timesteps=1000,
        beta_schedule='linear',
        # Training config
        unconditional_prob=0.1,
        device='cuda'
    ):
        super().__init__()
        
        self.device = device
        self.unconditional_prob = unconditional_prob
        
        # ========== VAE ==========
        print("Initializing VAE...")
        if vae_config is None:
            vae_config = {
                'in_channels': 3,
                'latent_channels': 4,
                'hidden_dims': [128, 256, 512, 512],
                'image_size': 256,
                'use_kan_decoder': True
            }
        
        self.vae = VAE(**vae_config)
        
        # Load pretrained VAE if provided
        if vae_checkpoint is not None:
            print(f"Loading VAE from {vae_checkpoint}")
            self.vae.load_state_dict(torch.load(vae_checkpoint, map_location='cpu'))
        
        # Freeze VAE (use pretrained)
        self.vae.requires_grad_(False)
        self.vae.eval()
        
        # ========== Text Encoder ==========
        print("Initializing CLIP Text Encoder...")
        self.text_encoder = CLIPTextEncoder(
            model_name=text_encoder_model,
            device=device
        )
        # Already frozen in CLIPTextEncoder
        
        # Context dimension
        self.context_dim = self.text_encoder.embed_dim  # 768 for CLIP ViT-L
        
        # ========== Denoiser (Mamba U-Net) ==========
        print("Initializing Mamba U-Net Denoiser...")
        if denoiser_config is None:
            denoiser_config = {
                'in_channels': vae_config['latent_channels'],
                'out_channels': vae_config['latent_channels'],
                'model_channels': 320,
                'channel_mult': [1, 2, 4, 4],
                'num_res_blocks': 2,
                'attention_resolutions': [1, 2, 3],
                'context_dim': self.context_dim,
                'num_heads': 8,
                'use_cross_attn': True
            }
        
        self.denoiser = MambaUNet(**denoiser_config)
        
        # ========== DDPM ==========
        print("Initializing DDPM...")
        self.ddpm = DDPM(
            model=self.denoiser,
            timesteps=timesteps,
            beta_schedule=beta_schedule,
            loss_type='l2'
        )
        
        # Cache unconditional embedding
        self.register_buffer(
            'null_context',
            torch.zeros(1, 77, self.context_dim)
        )
        
        print(f"✅ LDM initialized on {device}")
        print(f"   VAE latent: {vae_config['latent_channels']} channels")
        print(f"   Text context: {self.context_dim} dim")
        print(f"   Timesteps: {timesteps}")
    
    @torch.no_grad()
    def encode_images(self, images):
        """
        Encode images to latent space
        
        Args:
            images: (B, 3, H, W) - Input images, range [0, 1] or [-1, 1]
        Returns:
            latents: (B, 4, H//8, W//8) - Latent codes (using mean, not sampled)
        """
        mean, logvar = self.vae.encode(images)
        # Use mean for deterministic encoding (no sampling)
        return mean
    
    @torch.no_grad()
    def decode_latents(self, latents):
        """
        Decode latents to images
        
        Args:
            latents: (B, 4, H//8, W//8) - Latent codes
        Returns:
            images: (B, 3, H, W) - Decoded images
        """
        return self.vae.decode(latents)
    
    @torch.no_grad()
    def encode_text(self, captions):
        """
        Encode text to embeddings
        
        Args:
            captions: List of strings
        Returns:
            context: (B, 77, 768) - Text embeddings
        """
        embeddings, _ = self.text_encoder(captions)
        return embeddings
    
    def get_null_context(self, batch_size):
        """Get unconditional embedding"""
        return self.null_context.repeat(batch_size, 1, 1)
    
    def forward(self, images, captions):
        """
        Training forward pass
        
        Args:
            images: (B, 3, H, W) - Input images
            captions: List of strings - Text captions
        Returns:
            loss: Diffusion loss
        """
        batch_size = images.shape[0]
        device = images.device
        
        # Encode images to latents
        with torch.no_grad():
            latents = self.encode_images(images)
        
        # Encode text
        with torch.no_grad():
            context = self.encode_text(captions)
        
        # Random timesteps
        t = torch.randint(0, self.ddpm.timesteps, (batch_size,), device=device)
        
        # Classifier-free guidance: randomly drop text
        if self.training:
            # Random drop mask
            drop_mask = torch.rand(batch_size, device=device) < self.unconditional_prob
            
            # Get null context
            null_context = self.get_null_context(batch_size).to(device)
            
            # Replace with null where dropped
            context = torch.where(
                drop_mask[:, None, None],
                null_context,
                context
            )
        
        # Diffusion forward
        noise = torch.randn_like(latents)
        latents_noisy = self.ddpm.q_sample(latents, t, noise)
        
        # Denoise
        noise_pred = self.denoiser(latents_noisy, t, context)
        
        # Loss
        loss = nn.functional.mse_loss(noise, noise_pred)
        
        return loss
    
    @torch.no_grad()
    def generate(
        self,
        captions,
        num_steps=50,
        guidance_scale=7.5,
        height=256,
        width=256,
        return_latents=False
    ):
        """
        Text-to-image generation with classifier-free guidance
        
        Args:
            captions: List of strings
            num_steps: Number of DDIM steps (default: 50)
            guidance_scale: CFG strength (default: 7.5)
            height: Output image height
            width: Output image width
            return_latents: If True, return latents instead of images
        Returns:
            images: (B, 3, H, W) - Generated images or latents
        """
        batch_size = len(captions)
        latent_h = height // 8
        latent_w = width // 8
        
        # Encode text
        context = self.encode_text(captions).to(self.device)
        null_context = self.get_null_context(batch_size).to(self.device)
        
        # Start from pure noise
        latents = torch.randn(
            batch_size,
            4,  # latent_channels
            latent_h,
            latent_w,
            device=self.device
        )
        
        # DDIM sampling with CFG
        timesteps = torch.linspace(
            self.ddpm.timesteps - 1,
            0,
            num_steps,
            dtype=torch.long,
            device=self.device
        )
        
        for i, t in enumerate(tqdm(timesteps, desc='Generating')):
            t_batch = torch.full((batch_size,), t, device=self.device, dtype=torch.long)
            
            # Conditional prediction
            noise_pred_cond = self.denoiser(latents, t_batch, context)
            
            # Unconditional prediction
            noise_pred_uncond = self.denoiser(latents, t_batch, null_context)
            
            # Classifier-free guidance
            noise_pred = classifier_free_guidance(
                noise_pred_cond,
                noise_pred_uncond,
                guidance_scale
            )
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                latents = self.ddim_step(latents, noise_pred, t, t_next)
            else:
                # Last step
                latents = self.ddim_step(latents, noise_pred, t, torch.tensor(0, device=self.device))
        
        if return_latents:
            return latents
        
        # Decode to images
        images = self.decode_latents(latents)
        
        return images
    
    def ddim_step(self, x, noise_pred, t, t_next):
        """
        Single DDIM sampling step
        
        Args:
            x: (B, C, H, W) - Current latent
            noise_pred: (B, C, H, W) - Predicted noise
            t: Current timestep (scalar or tensor)
            t_next: Next timestep
        Returns:
            x_next: (B, C, H, W) - Next latent
        """
        # Ensure t is tensor
        if isinstance(t, int):
            t = torch.tensor(t, device=x.device)
        if isinstance(t_next, int):
            t_next = torch.tensor(t_next, device=x.device)
        
        # Extract alpha values
        alpha_t = self.ddpm._extract(self.ddpm.alphas_cumprod, t.unsqueeze(0) if t.dim() == 0 else t, x.shape)
        alpha_next = self.ddpm._extract(self.ddpm.alphas_cumprod, t_next.unsqueeze(0) if t_next.dim() == 0 else t_next, x.shape)
        
        # Predict x0
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Direction pointing to xt
        if t_next > 0:
            x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise_pred
        else:
            x_next = x0_pred
        
        return x_next


if __name__ == "__main__":
    # Test LDM
    print("\n🧪 Testing Latent Diffusion Model...\n")
    
    # Initialize (without loading real CLIP to avoid download)
    try:
        ldm = LatentDiffusionModel(
            vae_config={
                'in_channels': 3,
                'latent_channels': 4,
                'hidden_dims': [64, 128, 256, 256],  # Smaller for testing
                'image_size': 256,
                'use_kan_decoder': False  # Use standard ResBlock for faster testing
            },
            timesteps=100,
            device='cpu'
        )
        
        # Test encoding
        images = torch.randn(2, 3, 256, 256)
        latents = ldm.encode_images(images)
        print(f"Encoded latents: {latents.shape}")
        
        # Test decoding
        decoded = ldm.decode_latents(latents)
        print(f"Decoded images: {decoded.shape}")
        
        # Test training forward
        captions = ["A cat", "A dog"]
        loss = ldm(images, captions)
       print(f"Training loss: {loss.item():.4f}")
        
        print("\n✅ LDM tests passed!")
    
    except Exception as e:
        print(f"⚠️ LDM test failed (expected if transformers not installed): {e}")
