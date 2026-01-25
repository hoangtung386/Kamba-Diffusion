"""
Improved DDPM with SNR-based loss weighting
Implements Min-SNR weighting from "Efficient Diffusion Training via Min-SNR Weighting Strategy"
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from functools import partial


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """Cosine schedule as proposed in https://arxiv.org/abs/2102.09672"""
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class ImprovedDDPM(nn.Module):
    """
    Improved DDPM with better loss weighting and noise prediction
    
    Key improvements:
    1. Min-SNR weighting for balanced training across timesteps
    2. Velocity parameterization option (v-prediction)
    3. Offset noise for better dark/bright image generation
    """
    def __init__(
        self,
        model,
        timesteps=1000,
        beta_schedule='linear',
        loss_type='l2',
        prediction_type='epsilon',  # 'epsilon', 'v', 'x0'
        min_snr_gamma=5.0,
        use_offset_noise=False,
        offset_noise_strength=0.1
    ):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type
        self.prediction_type = prediction_type
        self.min_snr_gamma = min_snr_gamma
        self.use_offset_noise = use_offset_noise
        self.offset_noise_strength = offset_noise_strength

        # Beta schedule
        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
            
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Helper function
        register_buffer = partial(self.register_buffer)

        register_buffer('betas', betas.to(torch.float32))
        register_buffer('alphas', alphas.to(torch.float32))
        register_buffer('alphas_cumprod', alphas_cumprod.to(torch.float32))
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(torch.float32))

        # Pre-computed values
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(torch.float32))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(torch.float32))
        register_buffer('sqrt_recip_alphas_cumprod', torch.sqrt(1. / alphas_cumprod).to(torch.float32))
        register_buffer('sqrt_recipm1_alphas_cumprod', torch.sqrt(1. / alphas_cumprod - 1).to(torch.float32))
        
        # Posterior variance
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance.to(torch.float32))
        
        # SNR (Signal-to-Noise Ratio) for weighting
        snr = alphas_cumprod / (1 - alphas_cumprod)
        register_buffer('snr', snr.to(torch.float32))

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse data with optional offset noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
            # Offset noise: add small constant noise to improve generation of very dark/bright regions
            if self.use_offset_noise:
                offset = torch.randn(x_start.shape[0], x_start.shape[1], 1, 1, device=x_start.device)
                noise = noise + self.offset_noise_strength * offset
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def get_loss_weights(self, t):
        """
        Compute Min-SNR loss weights
        
        From "Efficient Diffusion Training via Min-SNR Weighting Strategy"
        https://arxiv.org/abs/2303.09556
        
        weight = min(SNR(t), gamma) / SNR(t)
        """
        snr = self._extract(self.snr, t, (t.shape[0], 1, 1, 1))
        
        if self.min_snr_gamma is not None:
            # Min-SNR weighting
            weight = torch.minimum(snr, torch.ones_like(snr) * self.min_snr_gamma) / snr
        else:
            # No weighting
            weight = torch.ones_like(snr)
        
        return weight

    def predict_start_from_noise(self, x_t, t, noise):
        """Predict x_0 from x_t and noise"""
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return sqrt_recip * x_t - sqrt_recipm1 * noise
    
    def predict_noise_from_start(self, x_t, t, x0):
        """Predict noise from x_t and x_0"""
        sqrt_recip = self._extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape)
        sqrt_recipm1 = self._extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        return (sqrt_recip * x_t - x0) / sqrt_recipm1

    def p_losses(self, x_start, t, context=None, noise=None):
        """
        Improved loss with proper weighting and multiple prediction types
        
        Args:
            x_start: Clean latent (B, C, H, W)
            t: Timestep (B,)
            context: Text embeddings (B, M, D)
            noise: Optional pre-generated noise
        Returns:
            loss: Weighted diffusion loss
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            if self.use_offset_noise:
                offset = torch.randn(x_start.shape[0], x_start.shape[1], 1, 1, device=x_start.device)
                noise = noise + self.offset_noise_strength * offset
            
        # Noisy sample
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Model prediction
        model_output = self.model(x_noisy, t, context)
        
        # Get target based on prediction type
        if self.prediction_type == 'epsilon':
            # Predict noise (standard)
            target = noise
        elif self.prediction_type == 'x0':
            # Predict clean image
            target = x_start
        elif self.prediction_type == 'v':
            # Predict velocity (v-prediction)
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
            sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
            target = sqrt_alpha * noise - sqrt_one_minus_alpha * x_start
        else:
            raise ValueError(f"Unknown prediction type: {self.prediction_type}")
        
        # Compute loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(model_output, target, reduction='none')
        elif self.loss_type == 'l2':
            loss = F.mse_loss(model_output, target, reduction='none')
        elif self.loss_type == 'huber':
            loss = F.smooth_l1_loss(model_output, target, reduction='none')
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        # Apply Min-SNR weighting
        loss_weights = self.get_loss_weights(t)
        weighted_loss = loss * loss_weights
        
        # Reduce to scalar
        return weighted_loss.mean()

    @torch.no_grad()
    def p_sample(self, x, t, context=None):
        """
        Sample x_{t-1} from x_t with proper handling of prediction types
        """
        t_batch = torch.full((x.shape[0],), t, device=x.device, dtype=torch.long)
        
        # Model prediction
        model_output = self.model(x, t_batch, context)
        
        # Convert to noise prediction if needed
        if self.prediction_type == 'epsilon':
            predicted_noise = model_output
        elif self.prediction_type == 'x0':
            # x0 -> noise
            predicted_noise = self.predict_noise_from_start(x, t_batch, model_output)
        elif self.prediction_type == 'v':
            # v -> noise
            sqrt_alpha = self._extract(self.sqrt_alphas_cumprod, t_batch, x.shape)
            sqrt_one_minus_alpha = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
            predicted_noise = sqrt_alpha * model_output + sqrt_one_minus_alpha * x
        
        # Compute x_{t-1}
        sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas[t])
        sqrt_one_minus_alphas_cumprod = self._extract(self.sqrt_one_minus_alphas_cumprod, t_batch, x.shape)
        
        model_mean = sqrt_recip_alphas * (
            x - self.betas[t] * predicted_noise / sqrt_one_minus_alphas_cumprod
        )
        
        if t > 0:
            posterior_variance = self._extract(self.posterior_variance, t_batch, x.shape)
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, shape, context=None, return_intermediates=False):
        """
        Generate samples from pure noise
        
        Args:
            shape: (B, C, H, W)
            context: Optional conditioning
            return_intermediates: If True, return all intermediate steps
        Returns:
            samples: Generated samples
        """
        device = self.betas.device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        intermediates = [] if return_intermediates else None
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='Sampling', total=self.timesteps):
            img = self.p_sample(img, i, context)
            
            if return_intermediates and i % 50 == 0:
                intermediates.append(img.cpu())
        
        if return_intermediates:
            return img, intermediates
        return img

    def _extract(self, a, t, x_shape):
        """Extract coefficients at specified timesteps and reshape for broadcasting"""
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


# ============================================
# Testing
# ============================================

if __name__ == "__main__":
    print("Testing Improved DDPM...\n")
    
    # Dummy model
    class DummyUNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(4, 4, 3, padding=1)
        
        def forward(self, x, t, context=None):
            return self.conv(x)
    
    model = DummyUNet()
    
    # Test 1: Standard epsilon prediction
    print("1. Testing epsilon prediction:")
    ddpm_eps = ImprovedDDPM(
        model,
        timesteps=100,
        prediction_type='epsilon',
        min_snr_gamma=5.0
    )
    
    x = torch.randn(2, 4, 32, 32)
    t = torch.randint(0, 100, (2,))
    
    loss = ddpm_eps.p_losses(x, t)
    print(f"   Loss: {loss.item():.4f}")
    print(f"   Loss has gradient: {loss.requires_grad}")
    
    # Test 2: V-prediction
    print("\n2. Testing v-prediction:")
    ddpm_v = ImprovedDDPM(
        model,
        timesteps=100,
        prediction_type='v',
        min_snr_gamma=5.0
    )
    
    loss_v = ddpm_v.p_losses(x, t)
    print(f"   Loss: {loss_v.item():.4f}")
    
    # Test 3: Offset noise
    print("\n3. Testing offset noise:")
    ddpm_offset = ImprovedDDPM(
        model,
        timesteps=100,
        use_offset_noise=True,
        offset_noise_strength=0.1
    )
    
    loss_offset = ddpm_offset.p_losses(x, t)
    print(f"   Loss with offset noise: {loss_offset.item():.4f}")
    
    # Test 4: SNR weighting
    print("\n4. Testing SNR weights across timesteps:")
    weights_early = ddpm_eps.get_loss_weights(torch.tensor([10]))
    weights_mid = ddpm_eps.get_loss_weights(torch.tensor([50]))
    weights_late = ddpm_eps.get_loss_weights(torch.tensor([90]))
    
    print(f"   Weight at t=10 (early): {weights_early.item():.4f}")
    print(f"   Weight at t=50 (mid): {weights_mid.item():.4f}")
    print(f"   Weight at t=90 (late): {weights_late.item():.4f}")
    
    print("\n✅ All Improved DDPM tests passed!")