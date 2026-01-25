@dataclass
class BraTSConfig(BaseConfig):
    """BraTS Brain Tumor Dataset Configuration"""
    
    dataset_name: str = "brats"
    num_classes: int = 4  # Background, NCR, ED, ET
    input_channels: int = 4  # T1, T1ce, T2, FLAIR
    image_size: Tuple[int, int] = (240, 240)
    
    # 3D volume
    num_slices: int = 5   # 2T+1, T=2
    use_3d: bool = True
    
    # Symmetry for brain
    use_symmetry: bool = True
    symmetry_axis: int = -1
    
    def get_dataset_config(self) -> dict:
        return {
            'data_root': f"{self.data_root}/brats",
            'modalities': ['t1', 't1ce', 't2', 'flair'],
            'split_file': 'splits.json',
            'num_slices': self.num_slices,
        }
    
    def get_transform_config(self) -> dict:
        return {
            'resize': self.image_size,
            'normalize_per_modality': True,
            'flip_prob': 0.5,
            'rotate_range': 15,
            'elastic_deform': True,
        }
        