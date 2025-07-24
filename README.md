# Custom 4D-STEM Autoencoder

A ResNet-based convolutional autoencoder for dimensionality reduction and analysis of 4D Scanning Transmission Electron Microscopy (4D-STEM) diffraction patterns.

## 🎯 Key Features

- **Image-size agnostic processing** - Works with any input size (256×256 to 1024×1024+)
- **ResNet-based architecture** with skip connections and adaptive pooling
- **HDF5 data format** - Efficient, compressed storage with integrated metadata
- **Regularized training** with multiple loss components (MSE + L1 + contrastive + divergence)
- **Sparse embedding layer** with non-negative activations (variable dimension latent space)
- **Mixed precision training** - Support for bf16 (A100) and float16 precision
- **Checkpoint resuming** - Resume training from any saved epoch
- **Comprehensive evaluation** with PSNR, SSIM, and MSE metrics

## 🚀 Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Convert Data
```bash
python scripts/preprocessing/convert_dm4.py \
    --input data/Diffraction_SI.dm4 \
    --output data/train_data.zarr \
    --downsample 2
```
*Note: Output will be automatically saved as `train_data.h5` with HDF5 format*

### 3. Train Model
```bash
python scripts/training/train.py \
    --data data/train_data.h5 \
    --output_dir outputs \
    --epochs 50 \
    --batch 16 \
    --latent 128 \
    --precision bf16
```

### 4. Resume Training (Optional)
```bash
python scripts/training/train.py \
    --data data/train_data.h5 \
    --output_dir outputs \
    --epochs 100 \
    --resume_from_checkpoint outputs/checkpoints/ae_e050_*.ckpt
```

### 5. Generate Embeddings
```bash
python scripts/generate_embeddings.py \
    --input data/train_data.h5 \
    --checkpoint outputs/ae_*_final.ckpt \
    --output outputs/embeddings.pt
```

## 📊 Model Performance

- **Latent dimension**: Configurable (32, 64, 128, 256...)
- **Input flexibility**: 256×256 to 1024×1024+ diffraction patterns  
- **Data format**: HDF5 with gzip compression (~10x size reduction)
- **Training speed**: Mixed precision (bf16/float16) for 2x speedup
- **Reconstruction quality**: PSNR, SSIM, and MSE tracking

## 📁 Project Structure

```
4DSTEM_AE_MSc/
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
- **[STEM Visualization](scripts/stem_visualization.py)** - STEM analysis and visualization features

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
- `stem_visualization.png` - Virtual field analysis
- `tb_logs/` - TensorBoard logs

### Analysis Outputs:
- `embeddings.pt` - Latent space embeddings
- `latent_mosaic.png` - Spatial latent maps

## 🔧 Requirements

- PyTorch 2.2+ with PyTorch Lightning 2.2+
- scikit-image (SSIM/PSNR)
- matplotlib, numpy, h5py, tqdm
- hyperspy (for .dm4 files)

## 📝 Citation

```bibtex
[TBD]
```

---

**For detailed documentation on specific components, see the [docs/](docs/) directory.**