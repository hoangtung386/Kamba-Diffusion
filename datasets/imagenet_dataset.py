"""
ImageNet dataset loader for VAE pretraining
Standard image classification dataset
"""

import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import glob


class ImageNetDataset(Dataset):
    """
    ImageNet dataset for VAE pretraining
    
    Directory structure:
        imagenet/
            train/
                n01440764/
                    n01440764_10026.JPEG
                    ...
                n01443537/
                    ...
            val/
                ...
    """
    def __init__(
        self,
        data_root,
        split='train',
        image_size=256,
        center_crop=True
    ):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # Find all images
        split_dir = os.path.join(data_root, split)
        self.image_paths = []
        
        if os.path.exists(split_dir):
            # Search for images in subdirectories
            for class_dir in os.listdir(split_dir):
                class_path = os.path.join(split_dir, class_dir)
                if os.path.isdir(class_path):
                    images = glob.glob(os.path.join(class_path, '*.JPEG'))
                    images += glob.glob(os.path.join(class_path, '*.jpg'))
                    images += glob.glob(os.path.join(class_path, '*.png'))
                    self.image_paths.extend(images)
        
        print(f"ImageNet {split}: Found {len(self.image_paths)} images")
        
        # Transforms
        if center_crop:
            self.transform = transforms.Compose([
                transforms.Resize(image_size),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) - Normalized to [-1, 1]
        """
        img_path = self.image_paths[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
            return {'image': image}
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a random image instead
            return self.__getitem__((idx + 1) % len(self))


if __name__ == "__main__":
    # Test dataset
    print("\n🧪 Testing ImageNet Dataset...\n")
    
    # Create dummy dataset structure for testing
    import tempfile
    import numpy as np
    from PIL import Image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy structure
        train_dir = os.path.join(tmpdir, 'train', 'n01440764')
        os.makedirs(train_dir, exist_ok=True)
        
        # Create dummy images
        for i in range(5):
            dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            dummy_img.save(os.path.join(train_dir, f'img_{i}.JPEG'))
        
        # Test dataset
        dataset = ImageNetDataset(
            data_root=tmpdir,
            split='train',
            image_size=256
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
        
        print("\n✅ ImageNet dataset tests passed!")
