# 🎨 DMK-ImageGen: Latent Diffusion with Mamba + KAN

**Text-to-Image Generation using Mamba SSM and KAN Architecture**

> A novel approach to latent diffusion models combining Mamba (linear complexity SSM) with KAN (Kolmogorov-Arnold Networks) for efficient and interpretable image generation.

---

## 🌟 Key Features

- **Latent Diffusion Model (LDM)** - Efficient generation in compressed latent space (8x downsampling)
- **Mamba U-Net** - Linear complexity O(N) denoiser vs Transformer's O(N²)
- **KAN Decoder** - Interpretable VAE decoder using Kolmogorov-Arnold Networks
- **CLIP Text Conditioning** - Powerful text-to-image alignment
- **Classifier-Free Guidance** - High-quality controlled generation

---

## 🏗️ Architecture

```
Text Prompt → CLIP Encoder → Context (768-dim)
                                ↓
Image → VAE Encoder → Latent (4-ch, 32x32)
                                ↓
        Latent + Context + Timestep
                                ↓
              Mamba U-Net Denoiser
                                ↓
           Denoised Latent (4-ch, 32x32)
                                ↓
           KAN Decoder → Image (3-ch, 256x256)
```

### Novel Components

1. **Mamba-based Denoiser** - First LDM using State Space Models for attention
2. **KAN VAE Decoder** - Interpretable upsampling with learned activation functions
3. **Cross-Attention Fusion** - Text conditioning via multi-head cross-attention

---

## 📦 Installation

```bash
# Clone repository
git clone https://github.com/yourusername/DMK-ImageGen
cd DMK-ImageGen

# Install dependencies
pip install -r requirements.txt

# Install Mamba (requires CUDA)
pip install mamba-ssm

# Install CLIP
pip install transformers
```

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- CUDA >= 11.8 (for Mamba)
- GPU with >= 24GB VRAM (recommended: A6000, RTX 4090)

---

## 🚀 Quick Start

### 1. Train VAE (Stage 1)

```bash
python scripts/train_vae.py \
    --data_root /path/to/imagenet \
    --image_size 256 \
    --batch_size 256 \
    --epochs 100 \
    --use_kan \
    --exp_name vae_imagenet_kan
```

**Expected time:** 1-2 weeks on A6000

### 2. Train LDM (Stage 2)

```bash
python scripts/train_ldm.py \
    --data_root /path/to/coco \
    --vae_checkpoint experiments/vae_imagenet_kan/checkpoints/vae_best.pth \
    --image_size 256 \
    --batch_size 128 \
    --epochs 500 \
    --exp_name ldm_coco_mamba
```

**Expected time:** 4-6 weeks on A6000

### 3. Generate Images

```bash
python scripts/generate.py \
    --vae_checkpoint experiments/vae_imagenet_kan/checkpoints/vae_best.pth \
    --checkpoint experiments/ldm_coco_mamba/checkpoints/ldm_best.pth \
    --prompt "A beautiful sunset over mountains" \
    --num_steps 50 \
    --guidance_scale 7.5 \
    --num_samples 4 \
    --output_dir outputs/
```

---

## 📂 Project Structure

```
DMK-ImageGen/
│
├── models/
│   ├── autoencoders/
│   │   ├── vae.py                 # VAE with KAN decoder
│   │   └── losses.py              # Perceptual + KL loss
│   │
│   ├── text_encoders/
│   │   └── clip_encoder.py        # CLIP text encoder
│   │
│   ├── denoisers/
│   │   └── mamba_unet.py          # Mamba-based U-Net
│   │
│   ├── diffusion/
│   │   ├── ddpm.py                # DDPM framework
│   │   └── guidance.py            # Classifier-free guidance
│   │
│   ├── bottlenecks/
│   │   └── mamba_block.py         # Mamba SSM blocks
│   │
│   ├── modules/
│   │   ├── cross_attention.py     # Cross-attention layers
│   │   └── embedding.py           # Time embeddings
│   │
│   └── ldm_model.py               # Main LDM pipeline
│
├── datasets/
│   ├── imagenet_dataset.py        # ImageNet for VAE
│   ├── coco_dataset.py            # COCO Captions
│   └── laion_dataset.py           # LAION (TODO)
│
├── scripts/
│   ├── train_vae.py               # Stage 1: VAE training
│   ├── train_ldm.py               # Stage 2: LDM training
│   └── generate.py                # Inference
│
└── experiments/                   # Training outputs
```

---

## 🎯 Training Datasets

| Dataset | Purpose | Size | Download |
|---------|---------|------|----------|
| **ImageNet** | VAE pretraining | 1.2M images | [Link](http://www.image-net.org/) |
| **COCO Captions** | LDM training (starter) | 120K images | [Link](https://cocodataset.org/) |
| **LAION-Aesthetics** | LDM training (full) | 600K-12M | [Link](https://laion.ai/) |

---

## 📊 Model Configurations

### VAE

- **Encoder:** ResNet-style, 4 downsampling levels
- **Latent:** 4 channels, 32x32 (8x compression)
- **Decoder:** KAN-based upsampling (novel!)
- **Loss:** Reconstruction + Perceptual + KL (weight: 1e-6)

### Mamba U-Net

- **Base channels:** 320
- **Channel multipliers:** [1, 2, 4, 4]
- **Attention resolutions:** [1, 2, 3]
- **Mamba d_state:** 16
- **Cross-attention heads:** 8

### Diffusion

- **Timesteps:** 1000 (training), 50 (inference)
- **Schedule:** Linear beta schedule
- **Sampling:** DDIM with classifier-free guidance
- **Guidance scale:** 7.5 (default)

---

## 🎨 Example Results

(Add generated images here after training)

---

## 📈 Comparison with Stable Diffusion

| Metric | Stable Diffusion 1.5 | DMK-ImageGen (Ours) |
|--------|---------------------|---------------------|
| **Architecture** | Transformer U-Net | Mamba U-Net |
| **Complexity** | O(N²) | O(N) ✅ |
| **Decoder** | Conv Upsampling | KAN (Interpretable) ✅ |
| **Speed (512x512)** | ~2.5s/img | ~1.8s/img (est.) |
| **VRAM (inference)** | ~8GB | ~6GB (est.) |
| **FID Score** | 10-12 | TBD (target: <15) |

---

## 🔬 Research Contributions

1. **Linear Complexity LDM** - First latent diffusion model using Mamba SSM
2. **Interpretable VAE** - KAN-based decoder for understanding generation
3. **Efficient Attention** - Mamba replaces Transformer in U-Net
4. **Novel Training Strategy** - Two-stage approach with frozen components

### Potential Publications

- "Mamba-Diffusion: Linear-Complexity Latent Diffusion Models"
- "KAN-VAE: Interpretable Image Autoencoders with Kolmogorov-Arnold Networks"
- "Scaling Laws of SSM-based Generative Models"

---

## 🛠️ Advanced Usage

### Multi-GPU Training

```bash
# Use PyTorch DDP
torchrun --nproc_per_node=4 scripts/train_ldm.py \
    --data_root /path/to/coco \
    --vae_checkpoint vae_best.pth \
    --batch_size 32
```

### Custom Prompts File

```bash
# Create prompts.txt with one prompt per line
python scripts/generate.py \
    --vae_checkpoint vae_best.pth \
   --checkpoint ldm_best.pth \
    --prompts_file prompts.txt \
    --output_dir custom_outputs/
```

### Different Guidance Scales

```bash
# Lower guidance (more diverse)
python scripts/generate.py ... --guidance_scale 3.0

# Higher guidance (more aligned to prompt)
python scripts/generate.py ... --guidance_scale 10.0
```

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@misc{dmk-imagegen2026,
  title={DMK-ImageGen: Latent Diffusion with Mamba and KAN},
  author={Your Name},
  year={2026},
  howpublished={\url{https://github.com/yourusername/DMK-ImageGen}}
}
```

---

## 🙏 Acknowledgments

- **Mamba** - State Space Models architecture
- **Stable Diffusion** - Latent diffusion framework inspiration
- **CLIP** - Text encoder from OpenAI
- **KAN** - Kolmogorov-Arnold Networks

---

## 📜 License

MIT License - see LICENSE file for details

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

---

## 📧 Contact

For questions or collaborations:
- GitHub Issues: [Create an issue](https://github.com/hoangtung386/Kamba-Diffusion/issues)
- Email: levuhoangtung1542003@gmail.com

---

**🎯 Project Status:** Active Development

**💪 GPU Tested:** NVIDIA A6000 (48GB VRAM)

**⚡ Next Steps:**
- [ ] Full LAION training
- [ ] 512x512 resolution support
- [ ] LoRA/DreamBooth fine-tuning
- [ ] DPM-Solver integration
- [ ] Web demo

---

Made with ❤️ for the open-source AI community
