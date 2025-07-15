# Custom 4D-STEM Autoencoder

A ResNet-based convolutional autoencoder for dimensionality reduction and analysis of 4D Scanning Transmission Electron Microscopy (4D-STEM) diffraction patterns.

## 🎯 Key Features

- **Image-size agnostic processing** - Works with any input size (32×32 to 512×512+)
- **ResNet-based architecture** with skip connections and adaptive pooling
- **Regularized training** with multiple loss components (MSE + L1 + contrastive + divergence)
- **Sparse embedding layer** with non-negative activations (32D latent space)
- **Comprehensive evaluation** with PSNR, SSIM, and MSE metrics
- **Professional STEM visualization** with virtual bright/dark field imaging

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Convert Data
```bash
python scripts/convert_dm4.py \
    --input data/Diffraction_SI.dm4 \
    --output data/train_tensor.pt
```

### 3. Train Model
```bash
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 50 \
    --device cpu
```

### 4. Generate Embeddings
```bash
python scripts/generate_embeddings.py \
    --input data/train_tensor.pt \
    --checkpoint outputs/ae.ckpt \
    --output outputs/embeddings.pt
```

### 5. Visualize Results
```bash
python scripts/visualise_scan_latents.py \
    --raw data/train_tensor.pt \
    --latents outputs/embeddings.pt \
    --scan 42 114 \
    --outfig outputs/results.png
```

## 📊 Model Performance

- **Parameter count**: 3,914,404 parameters (consistent across all input sizes)
- **Latent dimension**: 32D sparse embeddings
- **Input flexibility**: 32×32 to 512×512+ diffraction patterns
- **Reconstruction quality**: PSNR, SSIM, and MSE tracking

## 📁 Project Structure

```
Custom_4DSTEM_AE/
├── models/           # Neural network architectures
├── scripts/          # Training and utility scripts  
├── data/             # Input data files
├── outputs/          # Generated results and checkpoints
├── tests/            # Architecture and training tests
├── docs/             # Detailed documentation
└── experiments.ipynb # Interactive analysis notebook
```

## 📚 Documentation

- **[Model Architecture](docs/models/README.md)** - Detailed network architecture and components
- **[Scripts Usage](docs/scripts/README.md)** - Complete guide to all scripts and utilities
- **[Training Guide](docs/training/README.md)** - Training procedures and hyperparameters
- **[Visualization](docs/visualization/README.md)** - STEM analysis and visualization features
- **[API Reference](docs/API.md)** - Complete API documentation

## 🔬 Architecture Overview

```
Input (any size) → Encoder → 32D Latent → Decoder → Reconstruction (original size)
                      ↓
                 3 ResNet Blocks + Adaptive Pooling
                      ↓
                 Sparse Embeddings (ReLU)
                      ↓
                 3 ResNet Upsampling Blocks
```

**Key Architecture Features:**
- Adaptive pooling handles variable input sizes
- Conditional batch normalization for small feature maps
- Fixed parameter count across all input resolutions
- Skip connections and residual learning

## 🧪 Loss Function

```
L = MSE(y, ŷ) + λ_act·L₁(a) + λ_sim·L_sim + λ_div·L_div
```

- **MSE**: Reconstruction fidelity
- **L₁**: Sparsity regularization  
- **L_sim**: Contrastive similarity (embedding diversity)
- **L_div**: Activation divergence (prevents mode collapse)

## 📈 Output Files

### Training Outputs:
- `ae.ckpt` - Trained model checkpoint
- `loss_curve.png` - Training progression
- `*_reconstruction_comparison.png` - Before/after comparisons
- `*_stem_visualization.png` - Virtual field analysis
- `tb_logs/` - TensorBoard logs

### Analysis Outputs:
- `embeddings.pt` - Latent space embeddings
- `latent_mosaic.png` - Spatial latent maps

## 🔧 Requirements

- PyTorch 2.2+
- PyTorch Lightning 2.2+
- scikit-image (SSIM/PSNR)
- matplotlib, numpy, h5py, tqdm
- hyperspy (for .dm4 files)

## 📝 Citation

```bibtex
[TBD]
```

---

**For detailed documentation on specific components, see the [docs/](docs/) directory.**