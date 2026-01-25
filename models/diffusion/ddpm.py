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
    """
    cosine schedule as proposed in https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)

class DDPM(nn.Module):
    def __init__(self, model, timesteps=1000, beta_schedule='linear', loss_type='l2'):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        self.loss_type = loss_type

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')
            
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        # Helper function to register buffer (saved in state_dict)
        register_buffer = partial(self.register_buffer)

        register_buffer('betas', betas.to(torch.float32))
        register_buffer('alphas_cumprod', alphas_cumprod.to(torch.float32))
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev.to(torch.float32))

        # Calculations for diffusion q(x_t | x_{t-1}) and others
        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod).to(torch.float32))
        register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1. - alphas_cumprod).to(torch.float32))
        
        # Calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        register_buffer('posterior_variance', posterior_variance.to(torch.float32))

    def q_sample(self, x_start, t, noise=None):
        """
        Diffuse the data (t steps)
        x_t = sqrt(alpha_bar_t) * x_0 + sqrt(1-alpha_bar_t) * noise
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        sqrt_alphas_cumprod_t = self._extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape)
        
        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def p_losses(self, x_start, condition, t, noise=None):
        """
        Calculate loss for a batch
        x_start: Target mask (ground truth)
        condition: Input image (CT/MRI)
        """
        if noise is None:
            noise = torch.randn_like(x_start)
            
        # Diffuse mask
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        
        # Predict noise
        # Note: Model expects cat(condition, x_noisy) or handles both. 
        # Update: DMK Model `forward` uses input_proj. We should concatenate here if model expects single input tensor
        # OR model.forward(x_noisy, t, condition) if we update the model signature.
        # Given typical segmentation setups, we usually Concat channel-wise.
        
        model_input = torch.cat([condition, x_noisy], dim=1) # (B, C_img+C_mask, H, W)
        predicted_noise = self.model(model_input, t)
        
        # Loss
        if self.loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif self.loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()
            
        return loss

    @torch.no_grad()
    def p_sample(self, x, t, condition_idx, condition):
        """
        Sample from p(x_{t-1} | x_t)
        """
        betas_t = self._extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self._extract(self.sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = torch.sqrt(1.0 / (1.0 - betas_t))
        
        # Model forward
        model_input = torch.cat([condition, x], dim=1)
        # Note: We pass full batch of t
        t_batch = torch.full((x.shape[0],), t[0], device=x.device, dtype=torch.long)
        
        predicted_noise = self.model(model_input, t_batch)
        
        # Equation: x_{t-1} = 1/sqrt(alpha) * (x_t - beta/sqrt(1-alpha_bar) * noise) + sigma * z
        model_mean = sqrt_recip_alphas_t * (
            x - betas_t * predicted_noise / sqrt_one_minus_alphas_cumprod_t
        )
        
        if t[0] > 0:
            posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Clip variance for stability
            return model_mean + torch.sqrt(posterior_variance_t) * noise
        else:
            return model_mean

    @torch.no_grad()
    def sample(self, condition, shape):
        """
        Generate segmentation mask from pure noise
        condition: Input image
        shape: (B, C, H, W) of target mask
        """
        device = condition.device
        b = shape[0]
        
        # Start from pure noise
        img = torch.randn(shape, device=device)
        
        for i in tqdm(reversed(range(0, self.timesteps)), desc='sampling loop time step', total=self.timesteps):
            t = torch.tensor([i] * b, device=device, dtype=torch.long)
            img = self.p_sample(img, t, i, condition)
            
        return img

    def _extract(self, a, t, x_shape):
        """
        Extract some coefficients at specified timesteps,
        then reshape to [batch_size, 1, 1, 1, 1, ...] for broadcasting purposes.
        """
        batch_size = t.shape[0]
        out = a.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
