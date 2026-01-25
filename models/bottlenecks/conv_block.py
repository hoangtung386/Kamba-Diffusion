import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, channels=1024):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.BatchNorm2d(channels),
            nn.ReLU()
        )
        
    def forward(self, x):
        return self.conv(x)
