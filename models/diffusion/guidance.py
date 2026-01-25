"""
Classifier-Free Guidance utilities for text-conditional diffusion
"""

import torch
import torch.nn as nn


def classifier_free_guidance(
    noise_pred_cond,
    noise_pred_uncond,
    guidance_scale=7.5
):
    """
    Apply classifier-free guidance
    
    output = uncond + guidance_scale  * (cond - uncond)
    
    Args:
        noise_pred_cond: (B, C, H, W) - Conditional noise prediction
        noise_pred_uncond: (B, C, H, W) - Unconditional noise prediction
        guidance_scale: Guidance strength (default: 7.5)
    Returns:
        noise_pred: (B, C, H, W) - Guided noise prediction
    """
    return noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)


class GuidedDDPM(nn.Module):
    """
    DDPM wrapper with classifier-free guidance support
    """
    def __init__(self, ddpm, unconditional_prob=0.1):
        """
        Args:
            ddpm: Base DDPM model
            unconditional_prob: Probability of dropping text conditioning during training
        """
        super().__init__()
        self.ddpm = ddpm
        self.unconditional_prob = unconditional_prob
    
    def forward(self, x_start, t, context, null_context=None):
        """
        Training forward with random text dropping
        
        Args:
            x_start: (B, C, H, W) - Clean latent
            t: (B,) - Timestep
            context: (B, M, D) - Text embeddings
            null_context: (B, M, D) - Empty text embedding (optional)
        Returns:
            loss: Diffusion loss
        """
        # Randomly drop text conditioning
        if self.training:
            # Create mask for dropping
            batch_size = context.shape[0]
            drop_mask = torch.rand(batch_size, device=context.device) < self.unconditional_prob
            
            # Replace with null context where mask is True
            if null_context is not None:
                context = torch.where(
                    drop_mask[:, None, None],
                    null_context,
                    context
                )
        
        # Standard DDPM loss
        noise = torch.randn_like(x_start)
        x_noisy = self.ddpm.q_sample(x_start, t, noise)
        
        # Predict noise
        noise_pred = self.ddpm.model(x_noisy, t, context)
        
        # Loss
        loss = torch.nn.functional.mse_loss(noise, noise_pred)
        
        return loss
    
    @torch.no_grad()
    def sample_guided(
        self,
        shape,
        context,
        null_context,
        guidance_scale=7.5,
        num_steps=50,
        device='cuda'
    ):
        """
        Sample with classifier-free guidance
        
        Args:
            shape: (B, C, H, W) - Shape of latent to generate
            context: (B, M, D) - Text embeddings
            null_context: (B, M, D) - Empty text embedding
            guidance_scale: Guidance strength
            num_steps: Number of sampling steps (uses DDIM)
            device: Device
        Returns:
            samples: (B, C, H, W) - Generated latents
        """
        b = shape[0]
        
        # Start from noise
        x = torch.randn(shape, device=device)
        
        # DDIM sampling with guidance
        timesteps = torch.linspace(self.ddpm.timesteps - 1, 0, num_steps, dtype=torch.long, device=device)
        
        for i, t in enumerate(timesteps):
            t_batch = torch.full((b,), t, device=device, dtype=torch.long)
            
            # Conditional prediction
            noise_pred_cond = self.ddpm.model(x, t_batch, context)
            
            # Unconditional prediction
            noise_pred_uncond = self.ddpm.model(x, t_batch, null_context)
            
            # Apply guidance
            noise_pred = classifier_free_guidance(
                noise_pred_cond,
                noise_pred_uncond,
                guidance_scale
            )
            
            # DDIM step
            if i < len(timesteps) - 1:
                t_next = timesteps[i + 1]
                x = self.ddpm_step(x, noise_pred, t, t_next)
            else:
                # Last step
                x = self.ddpm_step(x, noise_pred, t, torch.tensor(0, device=device))
        
        return x
    
    def ddpm_step(self, x, noise_pred, t, t_next):
        """
        Single DDIM step (deterministic sampling)
        
        Args:
            x: (B, C, H, W) - Current noisy latent
            noise_pred: (B, C, H, W) - Predicted noise
            t: Current timestep
            t_next: Next timestep
        Returns:
            x_next: (B, C, H, W) - Next latent
        """
        # Extract alpha values
        alpha_t = self.ddpm._extract(self.ddpm.alphas_cumprod, t, x.shape)
        alpha_next = self.ddpm._extract(self.ddpm.alphas_cumprod, t_next, x.shape)
        
        # Predict x0
        x0_pred = (x - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        
        # Direction pointing to xt
        if t_next > 0:
            noise = noise_pred
            x_next = torch.sqrt(alpha_next) * x0_pred + torch.sqrt(1 - alpha_next) * noise
        else:
            x_next = x0_pred
        
        return x_next


if __name__ == "__main__":
    # Test guidance
    print("\n🧪 Testing Classifier-Free Guidance...\n")
    
    # Test guidance function
    cond = torch.randn(2, 4, 32, 32)
    uncond = torch.randn(2, 4, 32, 32)
    
    guided = classifier_free_guidance(cond, uncond, guidance_scale=7.5)
    
    print(f"Conditional shape: {cond.shape}")
    print(f"Unconditional shape: {uncond.shape}")
    print(f"Guided output shape: {guided.shape}")
    
    # Verify guidance formula
    manual = uncond + 7.5 * (cond - uncond)
    assert torch.allclose(guided, manual)
    
    print("\n✅ Guidance tests passed!")
