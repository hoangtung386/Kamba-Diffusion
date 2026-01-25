import torch
from trainers.base_trainer import BaseTrainer
from models.diffusion.ddpm import DDPM
from evaluators.metrics import SegmentationMetrics
from tqdm import tqdm

class DiffusionTrainer(BaseTrainer):
    def __init__(self, config, model, train_loader, val_loader=None, device=None):
        super().__init__(config, model, train_loader, val_loader, device)
        
        # Wrap model in DDPM if not already
        if not isinstance(self.model, DDPM):
            self.model = DDPM(
                model=self.model,
                timesteps=config.diffusion_steps,
                loss_type='l2' # Configurable later
            )
            self.model = self.model.to(self.device)
            
        self.metrics = SegmentationMetrics(num_classes=config.num_classes, device=self.device)
        
    def train_one_epoch(self, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(self.train_loader, desc=f"Train Epoch {epoch}")
        
        for batch in pbar:
            images = batch['image'].to(self.device) # (B, 1, H, W)
            masks = batch['mask'].to(self.device)   # (B, H, W) or (B, 1, H, W)
            
            # Ensure mask is float and has channels
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float() # DDPM works in continuous space
            
            # Normalize masks to [-1, 1] for diffusion usually? 
            # Standard DDPM works on [-1, 1] data. 
            # If mask is 0/1, maybe ok, but centered is better.
            masks = masks * 2 - 1 
            
            self.optimizer.zero_grad()
            
            # Sample random timesteps
            batch_size = images.shape[0]
            t = torch.randint(0, self.config.diffusion_steps, (batch_size,), device=self.device).long()
            
            # Loss calculation (handled by DDPM class)
            loss = self.model.p_losses(x_start=masks, condition=images, t=t)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
            
        return {'loss': total_loss / len(self.train_loader)}
        
    @torch.no_grad()
    def validate(self, epoch):
        self.model.eval()
        self.metrics.reset()
        
        # Validate on a subset to save time if heavy
        # For diffusion, sampling is slow.
        val_limit = 50 # Limit number of validation samples?
        
        pbar = tqdm(self.val_loader, desc=f"Val Epoch {epoch}")
        count = 0
        
        for batch in pbar:
            if count >= val_limit: break
            
            images = batch['image'].to(self.device)
            masks_target = batch['mask'].to(self.device) # Index or Onehot
            
            # Sample
            # shape match mask
            if masks_target.ndim == 3:
                shape = (images.shape[0], 1, images.shape[2], images.shape[3])
                masks_target_idx = masks_target
            else:
                shape = (images.shape[0], 1, images.shape[2], images.shape[3])
                masks_target_idx = masks_target.squeeze(1) # Assuming single channel encoded
            
            # Run Sampling Loop (Reverse Diffusion)
            sampled_mask = self.model.sample(condition=images, shape=shape)
            
            # Convert [-1, 1] back to [0, 1] logic for metrics
            # Simple thresholding for now
            pred_mask = (sampled_mask > 0).long().squeeze(1)
            
            self.metrics.update(pred_mask, masks_target_idx)
            count += images.shape[0]
            
        dice_class, dice_mean = self.metrics.compute_dice()
        return {'dice': dice_mean.item()}
