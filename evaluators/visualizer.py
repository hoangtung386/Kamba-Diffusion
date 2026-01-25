import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

class Visualizer:
    def __init__(self, save_dir):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def save_batch(self, images, preds, targets, epoch, batch_idx):
        """
        Save a batch of visualizations.
        images: (B, 1, H, W)
        preds: (B, H, W)
        targets: (B, H, W)
        """
        # Take first sample in batch
        img = images[0].detach().cpu().numpy().squeeze()
        pred = preds[0].detach().cpu().numpy()
        target = targets[0].detach().cpu().numpy()
        
        # Normalize image to 0-255
        img = (img - img.min()) / (img.max() - img.min() + 1e-6) * 255
        img = img.astype(np.uint8)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        
        # Create colored overlays
        # Target: Green
        # Pred: Red
        overlay_target = np.zeros_like(img_rgb)
        overlay_target[:, :, 1] = target * 255 # Green
        
        overlay_pred = np.zeros_like(img_rgb)
        overlay_pred[:, :, 2] = pred * 255 # Red
        
        # Blend
        alpha = 0.3
        vis = img_rgb.copy()
        
        # Apply masks where active
        mask_t = target > 0
        mask_p = pred > 0
        
        vis[mask_t] = vis[mask_t] * (1 - alpha) + overlay_target[mask_t] * alpha
        vis[mask_p] = vis[mask_p] * (1 - alpha) + overlay_pred[mask_p] * alpha
        
        # Save
        filename = os.path.join(self.save_dir, f"epoch_{epoch}_batch_{batch_idx}.png")
        cv2.imwrite(filename, vis)
        
    def plot_uncertainty(self, image, mean_pred, uncertainty, name="unc"):
        """
        Plot Image | Prediction | Uncertainty Heatmap
        """
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Image
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title("Input")
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(mean_pred, cmap='jet', vmin=0, vmax=1)
        axes[1].set_title("Mean Prediction")
        axes[1].axis('off')
        
        # Uncertainty
        im = axes[2].imshow(uncertainty, cmap='inferno')
        axes[2].set_title("Uncertainty (Entropy)")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2])
        
        save_path = os.path.join(self.save_dir, f"{name}_uncertainty.png")
        plt.savefig(save_path)
        plt.close()
