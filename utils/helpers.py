import torch
import numpy as np
import random
import os

def set_seed(seed=42):
    """
    Set random seed for reproducibility
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def count_parameters(model):
    """
    Count trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def get_file_list(directory, ext=None):
    """
    Get list of files recursively
    """
    file_list = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if ext and not file.endswith(ext):
                continue
            file_list.append(os.path.join(root, file))
    return file_list

def tensor_to_numpy(t):
    return t.detach().cpu().numpy()
