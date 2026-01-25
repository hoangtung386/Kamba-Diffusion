@dataclass
class RSNAConfig(BaseConfig):
    """RSNA Pneumonia Detection Configuration"""
    
    dataset_name: str = "rsna"
    num_classes: int = 2
    input_channels: int = 1  # Grayscale X-ray
    image_size: Tuple[int, int] = (512, 512)
    
    num_slices: int = 1  # Single image
    use_symmetry: bool = True  # Left-right lung
    symmetry_axis: int = -1
    
    def get_dataset_config(self) -> dict:
        return {
            'data_root': f"{self.data_root}/rsna",
            'csv_file': 'train.csv',
            'image_dir': 'stage_2_train_images',
            'dicom_format': True,
        }
    
    def get_transform_config(self) -> dict:
        return {
            'resize': self.image_size,
            'normalize': {'mean': [0.5], 'std': [0.5]},
            'flip_prob': 0.5,
            'rotate_range': 10,
            'clahe': True,  # Enhance contrast
        }
        