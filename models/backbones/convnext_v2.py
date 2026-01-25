import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.convnext import ConvNeXt
from utils.registry import BACKBONE_REGISTRY

@BACKBONE_REGISTRY.register('convnext_v2')
class ConvNeXtBackbone(nn.Module):
    """
    ConvNeXt V2 backbone wrapper.
    For this implementation, we rely on timm if available, or a simplified local version if strict no-dep is required.
    We will assume `timm` is allowed as per standard DL projects.
    """
    def __init__(self, in_channels=3, features=[64, 128, 256, 512, 1024], pretrained=True):
        super().__init__()
        # Use timm to create model
        # Note: ConvNeXt in timm (e.g., 'convnext_tiny')
        try:
            import timm
            self.model = timm.create_model(
                'convnextv2_tiny', 
                pretrained=pretrained, 
                features_only=True,
                in_chans=in_channels
            )
            self.out_channels = self.model.feature_info.channels()
        except ImportError:
            print("Warning: timm not installed. Using simple ConvNet as fallback for 'convnext_v2'")
            self.model = nn.Sequential(
                nn.Conv2d(in_channels, 64, 7, 2, 3),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.MaxPool2d(3, 2, 1)
                # ... very dummy fallback
            )
            self.out_channels = [64, 64, 64, 64] # Dummy

    def forward(self, x):
        features = self.model(x)
        # timm features_only=True returns a list of features
        return features
