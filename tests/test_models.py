import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.datasets.stroke_config import StrokeConfig
from models.dmk_model import UniversalDMK

class TestDMKModel(unittest.TestCase):
    def setUp(self):
        self.config = StrokeConfig()
        self.config.num_classes = 2
        self.config.input_channels = 1
        self.config.backbone = 'convnext_v2' 
        self.config.bottleneck = 'mamba'
        self.config.decoder = 'kan'
        self.config.use_diffusion = True
        self.device = 'cpu' # Test on CPU for CI/CD compatibility

    def test_model_instantiation(self):
        model = UniversalDMK(self.config)
        self.assertIsInstance(model, UniversalDMK)
        
    def test_forward_pass(self):
        model = UniversalDMK(self.config).to(self.device)
        # B, C, H, W
        x = torch.randn(1, 1, 64, 64).to(self.device)
        t = torch.randint(0, 1000, (1,)).to(self.device)
        
        output = model(x, t)
        # Expected output: (B, NumClasses, H, W)
        self.assertEqual(output.shape, (1, 2, 64, 64))

    def test_forward_pass_no_diffusion(self):
        self.config.use_diffusion = False
        model = UniversalDMK(self.config).to(self.device)
        x = torch.randn(1, 1, 64, 64).to(self.device)
        
        output = model(x)
        self.assertEqual(output.shape, (1, 2, 64, 64))

if __name__ == '__main__':
    unittest.main()
