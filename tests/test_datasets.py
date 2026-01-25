import unittest
import torch
import sys
import os
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datasets.base_dataset import BaseSegmentationDataset
from configs.base_config import BaseConfig

class MockDataset(BaseSegmentationDataset):
    def _build_dataset(self):
        return [{'image': 'dummy_img.png', 'mask': 'dummy_mask.png'}] * 5
        
    def _load_image(self, path):
        return np.zeros((256, 256), dtype=np.uint8)
        
    def _load_mask(self, path):
        return np.zeros((256, 256), dtype=np.uint8)

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.config = BaseConfig() # Abstract but dataclass can instantiate if no abstract methods? 
        # BaseConfig is abstract... wait, it is @dataclass @ABC.
        # Can we instantiate it? Yes if we don't call abstract methods or if we mock it.
        # Actually BaseConfig has abstract methods? No, only Dataset has abstract methods? 
        # BaseConfig has abstract methods get_dataset_config. So we need a Concrete Config.
        
        from configs.datasets.stroke_config import StrokeConfig
        self.config = StrokeConfig()

    def test_dataset_len(self):
        ds = MockDataset(self.config)
        self.assertEqual(len(ds), 5)
        
    def test_getitem(self):
        ds = MockDataset(self.config)
        sample = ds[0]
        self.assertIn('image', sample)
        self.assertIn('mask', sample)
        self.assertIsInstance(sample['image'], torch.Tensor)
        # Check defaults
        self.assertEqual(sample['image'].shape, (1, 256, 256))

if __name__ == '__main__':
    unittest.main()
