import os
import sys
import argparse
import torch
import cv2
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.dmk_model import UniversalDMK
from utils.checkpoint import load_checkpoint
from configs.datasets.stroke_config import StrokeConfig

def preprocess_image(image_path, size=(256, 256)):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, size)
    img = img / 255.0
    img = torch.from_numpy(img).float().unsqueeze(0).unsqueeze(0) # (1, 1, H, W)
    return img

def main(args):
    device = torch.device(args.device)
    config = StrokeConfig()
    
    # Model
    model = UniversalDMK(config)
    model = model.to(device)
    load_checkpoint(model, args.checkpoint, device=device)
    model.eval()
    
    # Input
    img_tensor = preprocess_image(args.image_path).to(device)
    
    # Inference
    # Note: Requires Sampling for diffusion
    print("Running inference...")
    # Placeholder: Assuming deterministic for now or need DDPM wrapper
    # ...
    
    print("Done. (Output saving not implemented for diffusion sampling yet)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()
    main(args)
