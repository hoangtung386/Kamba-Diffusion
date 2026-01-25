import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.registry import DECODER_REGISTRY

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

@DECODER_REGISTRY.register('unet_decoder')
class UNetDecoder(nn.Module):
    def __init__(self, encoder_channels, decoder_channels, num_classes=2, **kwargs):
        super().__init__()
        self.encoder_channels = encoder_channels
        self.decoder_channels = decoder_channels
        
        self.up_layers = nn.ModuleList()
        self.conv_layers = nn.ModuleList()
        
        # Latent to first decoder level
        # features list: [f1, f2, f3, f4, f5] where f5 is bottleneck out
        
        # We loop backwards. 
        # For simplicity, assume fixed stages for now or generic matching
        prev_channels = encoder_channels[-1] # Bottleneck channels
        
        for ch in decoder_channels:
            self.up_layers.append(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True))
            # Concat with skip connection
            # We assume skip channels are in reversed encoder_channels list
            # But skips usually come from encoder stages.
            # Simplified for prototype:
            self.conv_layers.append(DoubleConv(prev_channels, ch)) 
            prev_channels = ch
            
    def forward(self, x, skips):
        # x: bottleneck output
        # skips: [f1, f2, f3, f4]
        
        for i, (up, conv) in enumerate(zip(self.up_layers, self.conv_layers)):
            x = up(x)
            
            # Use skip if available
            skip_idx = len(skips) - 1 - i
            if skip_idx >= 0:
                skip = skips[skip_idx]
                if x.shape != skip.shape:
                    x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
                # Usually concat here, but simplified to additive or require channel match logic
                # Just placeholder valid forward
                pass 
                
            x = conv(x)
        return x
