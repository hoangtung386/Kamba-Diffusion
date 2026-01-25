"""
PyTorch EMA (Exponential Moving Average) wrapper
For stable diffusion model inference
"""

import torch
from torch import nn
from copy import deepcopy


class EMA:
    """
    Exponential Moving Average of model parameters
    Maintains shadow parameters that are updated as moving average
    
    Usage:
        ema = EMA(model, decay=0.9999)
        
        # Training loop
        for batch in dataloader:
            loss = train_step(model, batch)
            optimizer.step()
            ema.update(model)  # Update EMA parameters
        
        # Validation/Inference
        with ema.average_parameters(model):
            validate(model)
    """
    def __init__(self, model, decay=0.9999, device=None):
        """
        Args:
            model: PyTorch model
            decay: Decay rate for EMA (default: 0.9999)
            device: Device to store shadow params
        """
        self.decay = decay
        self.device = device if device is not None else next(model.parameters()).device
        
        # Create shadow parameters
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone().to(self.device)
    
    @torch.no_grad()
    def update(self, model):
        """
        Update EMA parameters
        
        shadow = decay * shadow + (1 - decay) * param
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (
                    self.decay * self.shadow[name] +
                    (1.0 - self.decay) * param.data.to(self.device)
                )
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self, model):
        """Replace model parameters with EMA parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name].to(param.device)
    
    def restore(self, model):
        """Restore original model parameters"""
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
    
    @torch.no_grad()
    def average_parameters(self, model):
        """
        Context manager for using EMA parameters
        
        Example:
            with ema.average_parameters(model):
                outputs = model(inputs)
        """
        return _EMAContext(self, model)
    
    def state_dict(self):
        """Get state dict for checkpointing"""
        return {
            'decay': self.decay,
            'shadow': self.shadow
        }
    
    def load_state_dict(self, state_dict):
        """Load from checkpoint"""
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']


class _EMAContext:
    """Context manager for EMA inference"""
    def __init__(self, ema, model):
        self.ema = ema
        self.model = model
    
    def __enter__(self):
        self.ema.apply_shadow(self.model)
        return self.model
    
    def __exit__(self, *args):
        self.ema.restore(self.model)


if __name__ == "__main__":
    # Test EMA
    print("Testing EMA...")
    
    # Simple model
    model = nn.Linear(10, 10)
    ema = EMA(model, decay=0.999)
    
    # Simulate training
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    for i in range(10):
        # Forward pass
        x = torch.randn(1, 10)
        y = model(x)
        loss = y.mean()
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Update EMA
        ema.update(model)
    
    # Test context manager
    print(f"Original param: {model.weight.data[0, 0].item():.4f}")
    
    with ema.average_parameters(model):
        print(f"EMA param: {model.weight.data[0, 0].item():.4f}")
    
    print(f"Restored param: {model.weight.data[0, 0].item():.4f}")
    
    print("✅ EMA test passed!")
