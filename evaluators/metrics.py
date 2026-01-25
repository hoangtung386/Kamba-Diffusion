import torch
import torch.nn.functional as F

class SegmentationMetrics:
    def __init__(self, num_classes=2, device='cpu'):
        self.num_classes = num_classes
        self.reset()
        
    def reset(self):
        self.confusion_matrix = 0
        
    def update(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) logits or (B, H, W) class indices
            target: (B, H, W) class indices
        """
        if pred.ndim == 4:
            pred = pred.argmax(dim=1)
            
        # Flatten
        pred = pred.view(-1)
        target = target.view(-1)
        
        # Filter ignore index if necessary (not handled here for simplicity)
        
        # Update confusion matrix
        # Uses bincount trick for speed
        mask = (target >= 0) & (target < self.num_classes)
        hist = torch.bincount(
            self.num_classes * target[mask] + pred[mask],
            minlength=self.num_classes ** 2
        )
        self.confusion_matrix += hist.reshape(self.num_classes, self.num_classes)
        
    def compute_iou(self):
        """Returns per-class IoU and Mean IoU"""
        intersection = torch.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(dim=1) + 
            self.confusion_matrix.sum(dim=0) - 
            intersection
        )
        iou = intersection / (union + 1e-6)
        return iou, iou.mean()
        
    def compute_dice(self):
        """Returns per-class Dice and Mean Dice"""
        intersection = torch.diag(self.confusion_matrix)
        union = (
            self.confusion_matrix.sum(dim=1) + 
            self.confusion_matrix.sum(dim=0)
        )
        dice = 2 * intersection / (union + 1e-6)
        return dice, dice.mean()

def dice_coeff(pred, target, smooth=1.):
    """Functional interface for simple binary/multiclass dice"""
    # pred: (B, C, H, W) - Softmax/Sigmoid output
    # target: (B, C, H, W) - One-hot encoded or same shape
    
    intersection = (pred * target).sum(dim=(2,3))
    union = pred.sum(dim=(2,3)) + target.sum(dim=(2,3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.mean()
