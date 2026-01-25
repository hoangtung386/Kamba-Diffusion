import os
import sys
import argparse
import torch

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.registry import DATASET_REGISTRY, BACKBONE_REGISTRY, DECODER_REGISTRY
from models.dmk_model import UniversalDMK
from trainers.diffusion_trainer import DiffusionTrainer
from datasets.base_dataset import BaseSegmentationDataset

# Configs
from configs.datasets.stroke_config import StrokeConfig

def get_config(args):
    """Factory to get the right config based on dataset name"""
    if args.dataset == 'stroke':
        config = StrokeConfig()
    else:
        raise ValueError(f"Dataset {args.dataset} not configured yet. Implemented: stroke")
        
    # Override config with args
    if args.batch_size: config.batch_size = args.batch_size
    if args.epochs: config.num_epochs = args.epochs
    
    return config

def main():
    parser = argparse.ArgumentParser(description="DMK-Stroke Training")
    parser.add_argument('--dataset', type=str, default='stroke', help='Dataset name (stroke, isic, brats)')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--device', type=str, default='cuda', help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # 1. Setup Config
    config = get_config(args)
    config.device = args.device
    print(f"Loaded config for {config.dataset_name}")
    
    # 2. Setup Dataset
    # Placeholder for actual dataset loading logic since we deferred dataset implementation
    # We will use a Mock Dataset or try to load if implemented.
    try:
        if config.dataset_name in DATASET_REGISTRY.list():
            DatasetClass = DATASET_REGISTRY.get(config.dataset_name)
            train_dataset = DatasetClass(config, split='train')
            val_dataset = DatasetClass(config, split='val')
        else:
            print(f"Warning: Dataset {config.dataset_name} not found in registry. Using MockDataset for testing.")
            # Provide a minimal mock to allow script to run
            from torch.utils.data import TensorDataset
            # (B, 1, 256, 256) image, (B, 256, 256) mask
            dummy_img = torch.randn(10, 1, 256, 256)
            dummy_mask = torch.randint(0, 2, (10, 256, 256))
            train_dataset = {'image': dummy_img, 'mask': dummy_mask} # Not compatible with Trainer dict access yet? 
            # Our Trainer expects dict access from DataLoader.
            # Let's make a Fake Dataset class
            class MockDataset(BaseSegmentationDataset):
                def _build_dataset(self): return [{'image': 'fake.png', 'mask': 'fake.png'}] * 16
                def _load_image(self, p): return torch.randn(1, 256, 256)
                def _load_mask(self, p): return torch.randint(0, 2, (256, 256))
            train_dataset = MockDataset(config, split='train')
            val_dataset = MockDataset(config, split='val')

        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=config.num_workers
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False
        )
    except Exception as e:
        print(f"Error setting up dataset: {e}")
        return

    # 3. Setup Model
    print("Building DMK Model...")
    model = UniversalDMK(config)
    
    # 4. Setup Trainer
    print("Initializing Trainer...")
    trainer = DiffusionTrainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=args.device
    )
    
    # 5. Start Training
    trainer.train()

if __name__ == '__main__':
    main()