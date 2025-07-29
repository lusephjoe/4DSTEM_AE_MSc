# 4D-STEM Autoencoder

A ResNet-based convolutional autoencoder for dimensionality reduction and analysis of 4D Scanning Transmission Electron Microscopy (4D-STEM) diffraction patterns.

## Key Features

- Image-size agnostic processing (256×256 to 1024×1024+)
- ResNet-based architecture with skip connections and adaptive pooling
- HDF5 data format with efficient compression and metadata storage
- Regularized training with multiple loss components (MSE + L1 + contrastive + divergence) 
- Sparse embedding layer with configurable latent dimensions
- Mixed precision training support (bf16/float16)
- Comprehensive test suite with 59+ tests for robustness
- Advanced coordinate handling for irregular scan patterns

## Quick Start

### 1. Installation
```bash
pip install -r requirements.txt
```

### 2. Convert Data
```bash
python scripts/preprocessing/convert_dm4.py \
    --input data/Diffraction_SI.dm4 \
    --output data/train_data.h5 \
    --downsample 2 \
    --mode bin
```

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

### 4. Generate Embeddings
```bash
python scripts/visualization/generate_embeddings.py \
    --input data/train_data.h5 \
    --checkpoint outputs/ae_*_final.ckpt \
    --output outputs/embeddings.npz
```

### 5. Visualize Results
```bash
python scripts/visualization/visualise_scan_latents.py \
    --embeddings outputs/embeddings.npz \
    --scan 194 209
```

## Model Performance

- Latent dimension: Configurable (32, 64, 128, 256...)
- Input flexibility: 256×256 to 1024×1024+ diffraction patterns  
- Data format: HDF5 with gzip compression (~10x size reduction)
- Training speed: Mixed precision (bf16/float16) for 2x speedup
- Reconstruction quality: PSNR, SSIM, and MSE tracking

## Project Structure

```
4DSTEM_AE_MSc/
├── models/                    # Neural network architectures
├── scripts/
│   ├── preprocessing/         # Data conversion and preprocessing
│   │   └── tests/            # Comprehensive test suite (59+ tests)
│   ├── training/             # Model training scripts
│   └── visualization/        # Analysis and visualization tools
├── data/                     # Input data files
└── outputs/                  # Generated results and checkpoints
```

## Data Processing

### Supported Formats
- Input: Gatan .dm4 files
- Output: HDF5 with metadata and compression
- Coordinate handling: Raster and serpentine scan patterns

### Downsampling Methods
- `stride`: Pixel skipping (fastest)
- `bin`: Mean pooling (recommended)
- `gauss`: Gaussian filtering + stride (anti-aliasing)
- `fft`: Fourier cropping (highest quality)

## Architecture Overview

```
Input (any size) → Encoder → Configurable Latent → Decoder → Reconstruction
                      ↓
              Initial Conv Layers (64→128 channels)
                      ↓  
              3 ResNet Blocks (4x pooling each: 64x→16x→4x)
                      ↓
              Final Conv Layers (128→64→1 channels)
                      ↓
              Adaptive Pool (4x4) + Embedding (ReLU)
                      ↓
              Adaptive Decoder (variable output size)
```

Key features: Adaptive pooling for variable input sizes, conditional batch normalization, skip connections, fixed parameter count across resolutions.

## Loss Function

```
L = Reconstruction + λ_lp·Lp_reg + λ_contrast·L_contrast + λ_div·L_div
```

- Reconstruction: MSE, MAE, or Huber loss options
- Lp_reg: Lp norm regularization on latent representations (L1/L2)
- L_contrast: Contrastive similarity regularization (embedding diversity)
- L_div: Activation divergence regularization (prevents mode collapse)

## Testing

Run the comprehensive test suite:

```bash
# Run all tests
python scripts/preprocessing/tests/run_tests.py

# Run only fast tests  
python scripts/preprocessing/tests/run_tests.py --fast

# Run specific test groups
python scripts/preprocessing/tests/run_tests.py --group downsample
```

## Requirements

- PyTorch 2.2+ with PyTorch Lightning 2.2+
- scikit-image (SSIM/PSNR)
- matplotlib, numpy, h5py, tqdm
- hyperspy (for .dm4 file loading)

## Citations & Acknowledgements
This work and model was heavily inspired by / based on the following paper and the Python package m3_learning (https://github.com/m3-learning)

```bibtex
@article{Ludacka_He_Qin_Zahn_Christiansen_Hunnestad_Zhang_Yan_Bourret_Kézsmárki_et al._2024, 
    title={Imaging and structure analysis of ferroelectric domains, domain walls, and vortices by scanning electron diffraction}, 
    volume={10}, 
    DOI={10.1038/s41524-024-01265-y}, 
    number={1}, 
    journal={npj Computational Materials}, 
    author={Ludacka, Ursula and He, Jiali and Qin, Shuyu and Zahn, Manuel and Christiansen, Emil Frang and Hunnestad, Kasper A. and Zhang, Xinqiao and Yan, Zewu and Bourret, Edith and Kézsmárki, István and et al.}, 
    year={2024}, 
    month={May}
}
``` 

## AI Disclosure Statement

Claude Code Assistant was used in the development of this project.