class Registry:
    """
    Registry for models, datasets, losses, etc.
    Allows easy plugin of new components
    """
    
    def __init__(self, name: str):
        self.name = name
        self._registry = {}
    
    def register(self, name: str):
        """Decorator to register a class"""
        def wrapper(cls):
            self._registry[name] = cls
            return cls
        return wrapper
    
    def get(self, name: str):
        """Get registered class by name"""
        if name not in self._registry:
            raise KeyError(f"{name} not found in {self.name} registry")
        return self._registry[name]
    
    def list(self):
        """List all registered names"""
        return list(self._registry.keys())


# Create global registries
DATASET_REGISTRY = Registry('dataset')
MODEL_REGISTRY = Registry('model')
BACKBONE_REGISTRY = Registry('backbone')
DECODER_REGISTRY = Registry('decoder')
LOSS_REGISTRY = Registry('loss')


# Example usage:
# @DATASET_REGISTRY.register('stroke')
# class StrokeDataset(BaseSegmentationDataset):
