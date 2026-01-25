import torch.nn as nn
from utils.registry import BACKBONE_REGISTRY
import torchvision.models as models

@BACKBONE_REGISTRY.register('resnet')
@BACKBONE_REGISTRY.register('resnet50')
class ResNetBackbone(nn.Module):
    def __init__(self, in_channels=3, features=None, pretrained=True):
        super().__init__()
        # Load standard ResNet
        base_model = models.resnet50(pretrained=pretrained)
        
        # Modify input layer if in_channels != 3
        if in_channels != 3:
            base_model.conv1 = nn.Conv2d(
                in_channels, 
                64, 
                kernel_size=7, 
                stride=2, 
                padding=3, 
                bias=False
            )
        
        self.encoder = nn.ModuleList([
            nn.Sequential(
                base_model.conv1,
                base_model.bn1,
                base_model.relu,
                base_model.maxpool
            ),
            base_model.layer1,
            base_model.layer2,
            base_model.layer3,
            base_model.layer4
        ])
    
    def forward(self, x):
        features = []
        for layer in self.encoder:
            x = layer(x)
            features.append(x)
        return features
