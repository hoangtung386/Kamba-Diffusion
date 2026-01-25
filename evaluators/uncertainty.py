import torch
import torch.nn.functional as F
import numpy as np

class UncertaintyEstimator:
    """
    Quantifies uncertainty from multiple diffusion samples (Monte Carlo dropout or Diffusion Sampling).
    """
    def __init__(self, num_samples=5):
        self.num_samples = num_samples
        
    def compute_uncertainty(self, samples):
        """
        Args:
            samples: (N_samples, B, C, H, W) - Raw logits or probabilities
        Returns:
            uncertainty_map: (B, H, W) - Voxel-wise uncertainty
        """
        # Ensure probabilities
        probs = torch.sigmoid(samples) if samples.min() < 0 else samples
        
        # 1. Predictive Entropy
        # Average prob across samples
        mean_prob = probs.mean(dim=0) # (B, C, H, W)
        
        # Entropy = - sum(p * log(p))
        # For binary case: -(p*log(p) + (1-p)*log(1-p))
        epsilon = 1e-6
        entropy = -(mean_prob * torch.log(mean_prob + epsilon) + 
                   (1 - mean_prob) * torch.log(1 - mean_prob + epsilon))
        
        # 2. Predictive Variance
        variance = probs.var(dim=0) # (B, C, H, W)
        
        return {
            'entropy': entropy.mean(dim=1), # Average over channels if multi-class
            'variance': variance.mean(dim=1),
            'mean_pred': mean_prob
        }
        
    def calibrate(self):
        # Placeholder for calibration logic
        pass
