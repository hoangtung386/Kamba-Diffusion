import torch.nn as nn
import torch.nn.functional as F

class DiffusionLoss(nn.Module):
    def __init__(self, loss_type='l2'):
        super().__init__()
        self.loss_type = loss_type
        
    def forward(self, noise_pred, noise_target):
        if self.loss_type == 'l2':
            return F.mse_loss(noise_pred, noise_target)
        elif self.loss_type == 'l1':
            return F.l1_loss(noise_pred, noise_target)
        elif self.loss_type == 'huber':
            return F.smooth_l1_loss(noise_pred, noise_target)
        return F.mse_loss(noise_pred, noise_target)
