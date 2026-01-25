import torch.nn as nn
import torch.nn.functional as F
from utils.registry import DECODER_REGISTRY

@DECODER_REGISTRY.register('fpn_decoder')
class FPNDecoder(nn.Module):
    def __init__(self, num_classes=2, **kwargs):
        super().__init__()
        # Simplified FPN
        self.lat_layers = nn.ModuleList([nn.Conv2d(256, 256, 1)])
        
    def forward(self, x, skips):
        # Simplified FPN logic
        return x
