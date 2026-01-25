# 🗂️ Complete DMK-Stroke Framework Structure

## 📁 Folder Tree

```
DMK-Stroke/
│
├── 📄 README.md
├── 📄 requirements.txt
├── 📄 setup.pys
├── 📄 .gitignore
│
├── 📂 configs/                       # ⚙️ Configuration files
│   ├── __init__.py
│   ├── base_config.py               # Abstract base config
│   │
│   ├── 📂 datasets/                  # Dataset configs
│   │   ├── __init__.py
│   │   ├── stroke_config.py         # Brain stroke
│   │   ├── isic_config.py           # Skin lesion
│   │   ├── brats_config.py          # Brain tumor
│   │   ├── rsna_config.py           # Pneumonia
│   │   └── synapse_config.py        # Multi-organ
│   │
│   └── 📂 experiments/               # Experiment configs
│       ├── dmk_stroke.yaml
│       ├── dmk_isic.yaml
│       └── ablation_studies.yaml
│
├── 📂 data/                          # 💾 Data directory
│   ├── stroke/
│   │   ├── images/
│   │   │   ├── patient_001/
│   │   │   │   ├── slice_001.png
│   │   │   │   └── ...
│   │   │   └── patient_002/
│   │   └── masks/
│   │       └── (same structure)
│   │
│   ├── isic/
│   │   ├── images/
│   │   │   ├── ISIC_0000000.jpg
│   │   │   └── ...
│   │   └── masks/
│   │
│   ├── brats/
│   │   ├── BraTS20_Training_001/
│   │   │   ├── t1.nii.gz
│   │   │   ├── t1ce.nii.gz
│   │   │   ├── t2.nii.gz
│   │   │   ├── flair.nii.gz
│   │   │   └── seg.nii.gz
│   │   └── ...
│   │
│   └── rsna/
│       ├── stage_2_train_images/
│       ├── train.csv
│       └── ...
│
├── 📂 datasets/                      # 🔄 Dataset loaders
│   ├── __init__.py
│   ├── base_dataset.py              # Abstract base class
│   ├── stroke_dataset.py            # ✅ Implement this
│   ├── isic_dataset.py              # ✅ Implement this
│   ├── brats_dataset.py             # ✅ Implement this
│   ├── rsna_dataset.py              # ✅ Implement this
│   │
│   └── 📂 transforms/                # Data augmentation
│       ├── __init__.py
│       ├── spatial.py               # Flip, rotate, elastic
│       ├── intensity.py             # Brightness, contrast
│       └── composition.py           # Compose transforms
│
├── 📂 models/                        # 🧠 Model architectures
│   ├── __init__.py
│   │
│   ├── 📂 backbones/                 # Encoder backbones
│   │   ├── __init__.py
│   │   ├── convnext_v2.py          # ✅ ConvNeXt V2
│   │   ├── resnet.py               # ResNet variants
│   │   └── efficientnet.py         # EfficientNet
│   │
│   ├── 📂 bottlenecks/               # Bottleneck modules
│   │   ├── __init__.py
│   │   ├── mamba_block.py          # ✅ Mamba SSM (NOVEL)
│   │   ├── transformer_block.py    # Transformer (ablation)
│   │   └── conv_block.py           # Conv (ablation)
│   │
│   ├── 📂 decoders/                  # Decoder modules
│   │   ├── __init__.py
│   │   ├── kan_decoder.py          # ✅ KAN decoder (NOVEL)
│   │   ├── unet_decoder.py         # U-Net decoder
│   │   └── fpn_decoder.py          # FPN decoder
│   │
│   ├── 📂 modules/                   # Utility modules
│   │   ├── __init__.py
│   │   ├── symmetry.py             # ✅ Symmetry fusion
│   │   ├── attention.py            # Attention blocks
│   │   └── embedding.py            # ✅ Time/position embeddings
│   │
│   ├── 📂 diffusion/                 # 🌊 Diffusion components
│   │   ├── __init__.py
│   │   ├── ddpm.py                 # ✅ DDPM implementation
│   │   ├── ddim.py                 # ✅ DDIM sampling
│   │   ├── scheduler.py            # ✅ Noise schedulers
│   │   └── losses.py               # Diffusion losses
│   │
│   └── dmk_model.py                # ✅ Main DMK model
│
├── 📂 trainers/                      # 🏃 Training logic
│   ├── __init__.py
│   ├── base_trainer.py             # Abstract trainer
│   ├── diffusion_trainer.py        # ✅ Diffusion training
│   └── direct_trainer.py           # Direct segmentation
│
├── 📂 evaluators/                    # 📊 Evaluation
│   ├── __init__.py
│   ├── metrics.py                  # Dice, IoU, HD95, etc.
│   ├── visualizer.py               # Visualization tools
│   └── uncertainty.py              # Uncertainty quantification
│
├── 📂 utils/                         # 🛠️ Utilities
│   ├── __init__.py
│   ├── logger.py                   # Logging utilities
│   ├── checkpoint.py               # Save/load checkpoints
│   ├── distributed.py              # DDP support
│   ├── registry.py                 # ✅ Registry pattern
│   └── helpers.py                  # Helper functions
│
├── 📂 scripts/                       # 🚀 Executable scripts
│   ├── train.py                    # ✅ Main training script
│   ├── evaluate.py                 # Evaluation script
│   ├── inference.py                # Inference script
│   ├── download_datasets.py        # Auto-download datasets
│   └── convert_dataset.py          # Convert dataset formats
│
├── 📂 experiments/                   # 💼 Experiment outputs
│   ├── stroke_dmk/
│   │   ├── checkpoints/
│   │   │   ├── best_model.pth
│   │   │   └── epoch_100.pth
│   │   ├── logs/
│   │   │   └── train.log
│   │   ├── wandb/
│   │   └── visualizations/
│   │
│   ├── isic_dmk/
│   └── brats_dmk/
│
├── 📂 notebooks/                     # 📓 Jupyter notebooks
│   ├── 01_demo_stroke.ipynb
│   ├── 02_demo_isic.ipynb
│   ├── 03_visualization.ipynb
│   └── 04_uncertainty_analysis.ipynb
│
└── 📂 tests/                         # 🧪 Unit tests
    ├── __init__.py
    ├── test_models.py
    ├── test_datasets.py
    ├── test_diffusion.py
    └── test_registry.py
```

---

## 📝 Implementation Priority

### ✅ **PHASE 1: Core Components (Week 1-2)**

Must implement first:

1. **configs/base_config.py** - Base configuration
2. **configs/datasets/stroke_config.py** - Stroke dataset config
3. **utils/registry.py** - Registry pattern
4. **datasets/base_dataset.py** - Abstract dataset class
5. **datasets/stroke_dataset.py** - Stroke dataset loader

### ✅ **PHASE 2: Model Architecture (Week 3-4)**

6. **models/backbones/convnext_v2.py** - Encoder
7. **models/bottlenecks/mamba_block.py** - Mamba SSM (NOVEL!)
8. **models/decoders/kan_decoder.py** - KAN decoder (NOVEL!)
9. **models/modules/embedding.py** - Time embeddings
10. **models/modules/symmetry.py** - Symmetry fusion
11. **models/dmk_model.py** - Main model

### ✅ **PHASE 3: Diffusion (Week 5-6)**

12. **models/diffusion/ddpm.py** - DDPM framework
13. **models/diffusion/ddim.py** - DDIM sampling
14. **models/diffusion/scheduler.py** - Noise schedulers
15. **models/diffusion/losses.py** - Diffusion losses

### ✅ **PHASE 4: Training Pipeline (Week 7-8)**

16. **trainers/base_trainer.py** - Base trainer
17. **trainers/diffusion_trainer.py** - Diffusion training
18. **evaluators/metrics.py** - Evaluation metrics
19. **scripts/train.py** - Main training script
20. **scripts/evaluate.py** - Evaluation script

---

## 🎯 Key Files to Implement

### **Priority 1: Must-Have (Core)**

| File | Purpose | Complexity | Novel |
|------|---------|------------|-------|
| `configs/base_config.py` | Configuration system | ⭐⭐ | ❌ |
| `utils/registry.py` | Plugin architecture | ⭐⭐ | ❌ |
| `datasets/base_dataset.py` | Dataset interface | ⭐⭐⭐ | ❌ |
| `models/dmk_model.py` | Main architecture | ⭐⭐⭐⭐ | ✅ |
| `models/bottlenecks/mamba_block.py` | Mamba SSM | ⭐⭐⭐⭐⭐ | ✅✅✅ |
| `models/decoders/kan_decoder.py` | KAN decoder | ⭐⭐⭐⭐⭐ | ✅✅✅ |
| `models/diffusion/ddpm.py` | DDPM framework | ⭐⭐⭐⭐ | ✅ |
| `trainers/diffusion_trainer.py` | Training loop | ⭐⭐⭐⭐ | ✅ |

### **Priority 2: Dataset Loaders**

| File | Dataset | Input Type | Classes |
|------|---------|------------|---------|
| `datasets/stroke_dataset.py` | Brain Stroke | CT (1ch) | 2 |
| `datasets/isic_dataset.py` | Skin Lesion | RGB (3ch) | 2 |
| `datasets/brats_dataset.py` | Brain Tumor | Multi-modal (4ch) | 4 |
| `datasets/rsna_dataset.py` | Pneumonia | X-ray (1ch) | 2 |

### **Priority 3: Ablation Studies**

| File | Purpose | Compare Against |
|------|---------|-----------------|
| `models/bottlenecks/transformer_block.py` | Baseline | Mamba |
| `models/bottlenecks/conv_block.py` | Baseline | Mamba |
| `models/decoders/unet_decoder.py` | Baseline | KAN |
| `trainers/direct_trainer.py` | Baseline | Diffusion |

---

## 🔧 Example Implementation

### **File: configs/datasets/stroke_config.py**

```python
from dataclasses import dataclass
from configs.base_config import BaseConfig

@dataclass
class StrokeConfig(BaseConfig):
    # Dataset info
    dataset_name: str = "stroke"
    num_classes: int = 2
    input_channels: int = 1
    
    # Multi-slice CT
    num_slices: int = 3  # 2T+1
    use_symmetry: bool = True
    
    def get_dataset_config(self):
        return {
            'data_root': './data/stroke',
            'image_dir': 'images',
            'mask_dir': 'masks',
        }
```

### **File: datasets/stroke_dataset.py**

```python
from datasets.base_dataset import BaseSegmentationDataset
from utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register('stroke')
class StrokeDataset(BaseSegmentationDataset):
    def _build_dataset(self):
        # Scan patient folders
        # Build sample list
        pass
    
    def _load_image(self, index):
        # Load CT slice(s)
        pass
    
    def _load_mask(self, index):
        # Load segmentation mask
        pass
```

---

## 🚀 How to Use

### **1. Training on Stroke Dataset**

```bash
python scripts/train.py \
    --dataset stroke \
    --backbone convnext_v2 \
    --bottleneck mamba \
    --decoder kan \
    --use_diffusion \
    --batch_size 4 \
    --epochs 300
```

### **2. Training on ISIC Dataset**

```bash
python scripts/train.py \
    --dataset isic \
    --backbone efficientnet \
    --decoder kan \
    --batch_size 8
```

### **3. Ablation Study**

```bash
# Baseline: Transformer + UNet
python scripts/train.py --dataset stroke --bottleneck transformer --decoder unet

# Ours: Mamba + KAN
python scripts/train.py --dataset stroke --bottleneck mamba --decoder kan
```

---

## 📦 Next Steps

**Bạn muốn tôi làm gì tiếp theo?**

**Option A:** Viết code chi tiết cho **core files** (Mamba, KAN, DDPM)

**Option B:** Viết code cho **dataset loaders** (Stroke, ISIC, BraTS)

**Option C:** Viết code cho **training pipeline** (trainer + scripts)

**Option D:** Tất cả! Tôi sẽ tạo từng file một theo priority

Bạn chọn option nào? 🤔
