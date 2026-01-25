@dataclass
class ISICConfig(BaseConfig):
    """ISIC Skin Lesion Dataset Configuration"""
    
    dataset_name: str = "isic"
    num_classes: int = 2
    input_channels: int = 3  # RGB images
    image_size: Tuple[int, int] = (256, 256)
    
    # No multi-slice
    num_slices: int = 1
    
    # No symmetry for skin
    use_symmetry: bool = False
    
    def get_dataset_config(self) -> dict:
        return {
            'data_root': f"{self.data_root}/isic",
            'csv_file': 'train.csv',
            'image_col': 'image',
            'mask_col': 'mask',
            'split_ratio': 0.8,
        }
    
    def get_transform_config(self) -> dict:
        return {
            'resize': self.image_size,
            'normalize': {
                'mean': [0.485, 0.456, 0.406],
                'std': [0.229, 0.224, 0.225]
            },
            'flip_prob': 0.5,
            'rotate_range': 180,  # Skin can be any orientation
            'color_jitter': True,
        }
        