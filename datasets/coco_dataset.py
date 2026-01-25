"""
COCO Captions dataset loader for text-to-image training
"""

import os
import json
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image


class COCODataset(Dataset):
    """
    COCO Captions dataset for text-to-image generation
    
    Directory structure:
        coco/
            train2017/
                000000000009.jpg
                ...
            val2017/
                ...
            annotations/
                captions_train2017.json
                captions_val2017.json
    """
    def __init__(
        self,
        data_root,
        split='train',
        year='2017',
        image_size=256,
        center_crop=True
    ):
        super().__init__()
        
        self.data_root = data_root
        self.split = split
        self.image_size = image_size
        
        # Paths
        self.image_dir = os.path.join(data_root, f'{split}{year}')
        self.ann_file = os.path.join(data_root, 'annotations', f'captions_{split}{year}.json')
        
        # Load annotations
        print(f"Loading COCO {split} annotations...")
        with open(self.ann_file, 'r') as f:
            coco_data = json.load(f)
        
        # Build image id to annotations mapping
        self.image_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in self.image_id_to_anns:
                self.image_id_to_anns[img_id] = []
            self.image_id_to_anns[img_id].append(ann['caption'])
        
        # Build image id to filename mapping
        self.images = []
        for img_info in coco_data['images']:
            img_id = img_info['id']
            if img_id in self.image_id_to_anns:
                self.images.append({
                    'image_id': img_id,
                    'file_name': img_info['file_name'],
                    'captions': self.image_id_to_anns[img_id]
                })
        
        print(f"COCO {split}: Found {len(self.images)} images with captions")
        
        # Transforms with proper augmentation
        if center_crop:
            # For validation: deterministic center crop
            self.transform = transforms.Compose([
                transforms.Resize(image_size, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
            ])
        else:
            # For training: random augmentation
            self.transform = transforms.Compose([
                transforms.Resize(int(image_size * 1.1), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.RandomCrop(image_size),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1]
            ])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: (3, H, W) - Normalized to [-1, 1]
            caption: str - Random caption from 5 available
        """
        img_info = self.images[idx]
        
        # Load image
        img_path = os.path.join(self.image_dir, img_info['file_name'])
        
        try:
            image = Image.open(img_path).convert('RGB')
            image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return next image
            return self.__getitem__((idx + 1) % len(self))
        
        # Random caption (COCO has 5 captions per image)
        captions = img_info['captions']
        caption = captions[torch.randint(len(captions), (1,)).item()]
        
        return {
            'image': image,
            'caption': caption
        }


if __name__ == "__main__":
    # Test dataset
    print("\n🧪 Testing COCO Dataset...\n")
    
    # Create dummy dataset for testing
    import tempfile
    import numpy as np
    from PIL import Image
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create dummy structure
        img_dir = os.path.join(tmpdir, 'train2017')
        ann_dir = os.path.join(tmpdir, 'annotations')
        os.makedirs(img_dir, exist_ok=True)
        os.makedirs(ann_dir, exist_ok=True)
        
        # Create dummy images
        dummy_images = []
        for i in range(3):
            filename = f'00000000000{i}.jpg'
            dummy_img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
            dummy_img.save(os.path.join(img_dir, filename))
            dummy_images.append({
                'id': i,
                'file_name': filename
            })
        
        # Create dummy annotations
        dummy_anns = []
        for i in range(3):
            for j in range(5):  # 5 captions per image
                dummy_anns.append({
                    'image_id': i,
                    'id': i * 5 + j,
                    'caption': f'This is caption {j} for image {i}'
                })
        
        # Save annotations
        coco_data = {
            'images': dummy_images,
            'annotations': dummy_anns
        }
        
        ann_file = os.path.join(ann_dir, 'captions_train2017.json')
        with open(ann_file, 'w') as f:
            json.dump(coco_data, f)
        
        # Test dataset
        dataset = COCODataset(
            data_root=tmpdir,
            split='train',
            image_size=256
        )
        
        print(f"Dataset size: {len(dataset)}")
        
        # Test loading
        sample = dataset[0]
        print(f"Image shape: {sample['image'].shape}")
        print(f"Caption: {sample['caption']}")
        print(f"Image range: [{sample['image'].min():.2f}, {sample['image'].max():.2f}]")
        
        print("\n✅ COCO dataset tests passed!")
