import torch.nn as nn
from utils.registry import BACKBONE_REGISTRY
import torchvision.models as models

@BACKBONE_REGISTRY.register('efficientnet')
class EfficientNetBackbone(nn.Module):
    def __init__(self, in_channels=3, features=None, pretrained=True):
        super().__init__()
        # Load EfficientNet B0
        # available in newer torchvision
        try:
            self.model = models.efficientnet_b0(pretrained=pretrained)
            
             # Modify input
            if in_channels != 3:
                first_conv = self.model.features[0][0]
                self.model.features[0][0] = nn.Conv2d(
                    in_channels, 
                    first_conv.out_channels, 
                    kernel_size=first_conv.kernel_size, 
                    stride=first_conv.stride, 
                    padding=first_conv.padding, 
                    bias=False
                )
        except AttributeError:
             print("Warning: torchvision too old for efficientnet")
             self.model = nn.Identity()

    def forward(self, x):
        # Extract features manually from stages
        # EfficientNet features are sequential
        return [self.model.features(x)] # Simplified return for now
