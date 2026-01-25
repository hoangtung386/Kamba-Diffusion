import os
import torch
import glob

def save_checkpoint(state, save_dir, filename="checkpoint.pth", is_best=False):
    """
    Save checkpoint to disk.
    """
    os.makedirs(save_dir, exist_ok=True)
    filepath = os.path.join(save_dir, filename)
    torch.save(state, filepath)
    
    if is_best:
        best_path = os.path.join(save_dir, "best_model.pth")
        torch.save(state, best_path)

def load_checkpoint(model, filename, optimizer=None, scheduler=None, device='cpu'):
    """
    Load checkpoint from disk.
    """
    if not os.path.isfile(filename):
        raise FileNotFoundError(f"Checkpoint file not found: {filename}")
        
    checkpoint = torch.load(filename, map_location=device)
    
    # Load model
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint) # fallback if just weight saved
        
    # Load optimizer
    if optimizer and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        
    # Load scheduler
    if scheduler and 'scheduler' in checkpoint: # Not every checkpoint has it
        pass # scheduler.load_state_dict(checkpoint['scheduler'])
        
    return checkpoint

def get_last_checkpoint(save_dir):
    """
    Find the latest checkpoint in directory based on modification time or naming convention.
    """
    checkpoints = glob.glob(os.path.join(save_dir, "checkpoint_epoch_*.pth"))
    if not checkpoints:
        return None
    # Sort by modification time
    latest_ckpt = max(checkpoints, key=os.path.getmtime)
    return latest_ckpt
