import os
import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from tqdm import tqdm
import time
import datetime
try:
    import wandb
except ImportError:
    wandb = None
from utils.registry import MODEL_REGISTRY

class BaseTrainer(ABC):
    def __init__(self, config, model, train_loader, val_loader=None, device=None):
        self.config = config
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = self.model.to(self.device)
        
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()
        
        # Directories
        self.output_dir = os.path.join(config.output_dir, f"{config.dataset_name}_{config.model_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.log_dir = os.path.join(self.output_dir, 'logs')
        self.ckpt_dir = os.path.join(self.output_dir, 'checkpoints')
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        # State
        self.start_epoch = 0
        self.best_metric = 0.0

        # Wandb Init
        if self.config.use_wandb and wandb is not None:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config=self.config.__dict__,
                name=os.path.basename(self.output_dir)
            )
        
    def _build_optimizer(self):
        if self.config.optimizer == 'adamw':
            return torch.optim.AdamW(
                self.model.parameters(), 
                lr=self.config.learning_rate, 
                weight_decay=self.config.weight_decay
            )
        else:
            return torch.optim.Adam(self.model.parameters(), lr=self.config.learning_rate)
            
    def _build_scheduler(self):
        # ROI: Return dummy or implemented scheduler
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, 
            T_max=self.config.num_epochs
        )
    
    def train(self):
        print(f"Start training on {self.device}...")
        for epoch in range(self.start_epoch, self.config.num_epochs):
            # Train One Epoch
            train_metrics = self.train_one_epoch(epoch)
            
            # Log
            self.log_metrics(epoch, train_metrics, mode='train')
            
            # Scheduler Step
            if self.scheduler:
                self.scheduler.step()
                
            # Validation
            if self.val_loader and (epoch + 1) % self.config.eval_interval == 0:
                val_metrics = self.validate(epoch)
                self.log_metrics(epoch, val_metrics, mode='val')
                
                # Save Best
                curr_metric = val_metrics.get('dice', 0.0)
                if curr_metric > self.best_metric:
                    self.best_metric = curr_metric
                    self.save_checkpoint(epoch, is_best=True)
            
            # Periodic Save
            if (epoch + 1) % self.config.save_interval == 0:
                self.save_checkpoint(epoch)
                
    @abstractmethod
    def train_one_epoch(self, epoch):
        pass
        
    @abstractmethod
    def validate(self, epoch):
        pass
        
    def save_checkpoint(self, epoch, is_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config
        }
        filename = f"checkpoint_epoch_{epoch}.pth"
        path = os.path.join(self.ckpt_dir, filename)
        torch.save(state, path)
        
        if is_best:
            best_path = os.path.join(self.ckpt_dir, "best_model.pth")
            torch.save(state, best_path)
            print(f"Saved Best Model at epoch {epoch} with Dice: {self.best_metric:.4f}")

    def log_metrics(self, epoch, metrics, mode='train'):
        # Simple Console Log
        msg = f"Epoch [{epoch}/{self.config.num_epochs}] {mode.upper()}: "
        msg += " ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        print(msg)
        
        # Wandb Log
        if self.config.use_wandb and wandb is not None:
             wandb.log({f"{mode}/{k}": v for k, v in metrics.items()}, step=epoch)
        
    def load_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.start_epoch = checkpoint['epoch'] + 1
        self.best_metric = checkpoint['best_metric']
        print(f"Loaded checkpoint from {path} at epoch {self.start_epoch}")
