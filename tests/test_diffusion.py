import unittest
import torch
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.diffusion.ddpm import DDPM

class MockModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x, t):
        # x is condition + noisy_mask
        # return random noise of same shape as target mask
        # assuming x: (B, C_img+C_mask, H, W)
        # return (B, C_mask, H, W)
        # Here we hardcode output for shape check
        return torch.randn(x.shape[0], 1, x.shape[2], x.shape[3])

class TestDiffusion(unittest.TestCase):
    def setUp(self):
        self.model = MockModel()
        self.ddpm = DDPM(self.model, timesteps=10)
        
    def test_q_sample(self):
        x_start = torch.randn(2, 1, 32, 32)
        t = torch.tensor([5, 5])
        
        noisy = self.ddpm.q_sample(x_start, t)
        self.assertEqual(noisy.shape, x_start.shape)
        self.assertFalse(torch.allclose(noisy, x_start)) # Should have noise
        
    def test_loss(self):
        x_start = torch.randn(2, 1, 32, 32) # Mask
        condition = torch.randn(2, 1, 32, 32) # Image
        t = torch.tensor([5, 5])
        
        loss = self.ddpm.p_losses(x_start, condition, t)
        self.assertTrue(isinstance(loss.item(), float))

if __name__ == '__main__':
    unittest.main()
