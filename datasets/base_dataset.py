import os
import torch
from torch.utils.data import Dataset
from abc import ABC, abstractmethod
import numpy as np
from PIL import Image

class BaseSegmentationDataset(Dataset, ABC):
    """
    Abstract base class for segmentation datasets
    """
    def __init__(self, config, split='train', transform=None):
        self.config = config
        self.split = split
        self.transform = transform
        self.samples = self._build_dataset()
        
    @abstractmethod
    def _build_dataset(self):
        """
        Build list of samples.
        Returns: List of dicts, e.g., [{'image': path, 'mask': path}, ...]
        """
        pass
    
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, index):
        sample = self.samples[index]
        
        # Load data
        image = self._load_image(sample['image'])
        mask = self._load_mask(sample['mask'])
        
        # Function to be implemented by child or generic logic below
        # Here assuming PIL image for generic, but medical usually uses likely nii.gz
        
        if self.transform:
            # Apply transforms
            # Assuming albumentations or similar that takes keywargs
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
            
        # To Tensor
        if not isinstance(image, torch.Tensor):
            image = torch.from_numpy(np.array(image)).float()
            mask = torch.from_numpy(np.array(mask)).long()
            
        if image.ndim == 2:
            image = image.unsqueeze(0)
            
        return {
            'image': image,
            'mask': mask,
            'name': sample.get('id', str(index))
        }

    def _load_image(self, path):
        # Generic loader - override for Medical formats (NIfTI/DICOM)
        return np.array(Image.open(path).convert('L')) # Default grayscale
        
    def _load_mask(self, path):
         return np.array(Image.open(path))