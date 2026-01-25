import os
import sys
import argparse
import torch
from tqdm import tqdm

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.registry import DATASET_REGISTRY
from models.dmk_model import UniversalDMK
from utils.checkpoint import load_checkpoint
from evaluators.metrics import SegmentationMetrics
from configs.datasets.stroke_config import StrokeConfig

def evaluate(args):
    device = torch.device(args.device)
    
    # 1. Config
    # Load config from args or default
    # Ideally load config from checkpoint but here we reconstruct
    config = StrokeConfig() 
    config.batch_size = args.batch_size
    
    # 2. Model
    model = UniversalDMK(config)
    model = model.to(device)
    model.eval()
    
    # Load Checkpoint
    load_checkpoint(model, args.checkpoint, device=device)
    print(f"Loaded model from {args.checkpoint}")
    
    # 3. Dataset
    if config.dataset_name in DATASET_REGISTRY.list():
        DatasetClass = DATASET_REGISTRY.get(config.dataset_name)
        val_dataset = DatasetClass(config, split='val') # or test
    else:
        print("Dataset not found, using Mock for demo")
        # Reuse mock logic from train or fail
        return 

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers
    )
    
    # 4. Metrics
    metrics_calc = SegmentationMetrics(num_classes=config.num_classes)
    
    # 5. Loop
    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(val_loader):
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)
            
            # For diffusion model, evaluation usually involves sampling
            # But DMK model might have a 'predict' or we use detailed sampling
            # If using DDPM, we need to sample. 
            # Check if model has direct forward or needs wrapper.
            # Base DMK forward returns logits if not diffusion, or noise if diffusion wrapper?
            # UniversalDMK forward depends on components. 
            # If unwrapped UniversalDMK: it expects (x, t) if diffusion is enabled.
            # To evaluate a diffusion model, we need the Sampler (DDPM/DDIM).
            
            # Ideally we should wrap it again or use Trainer's validate function logic
            # For simplicity here, assuming we use the DDPM wrapper logic or implement simple sampling
            
            pass # Placeholder for complex diffusion eval logic
            # For non-diffusion baseline:
            # preds = model(images)
            # metrics_calc.update(preds, masks)
            
    # print(metrics_calc.compute_dice())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to .pth file')
    parser.add_argument('--dataset', type=str, default='stroke')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    
    evaluate(args)
