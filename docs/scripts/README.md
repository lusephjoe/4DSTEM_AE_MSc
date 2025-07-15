# Scripts Documentation

This document provides comprehensive documentation for all utility scripts in the Custom 4D-STEM Autoencoder project.

## 📁 Scripts Overview

The `scripts/` directory contains five main utilities for data processing, training, and analysis:

1. **`train.py`** - Main training script with PyTorch Lightning
2. **`convert_dm4.py`** - Digital Micrograph file converter  
3. **`generate_embeddings.py`** - Latent space embedding generator
4. **`visualise_scan_latents.py`** - Latent space visualization tool
5. **`preprocess.py`** - Additional data preprocessing utilities

## 🚀 train.py

**Purpose**: Train the autoencoder model with comprehensive logging and evaluation

### Usage
```bash
python scripts/train.py [OPTIONS]
```

### Key Arguments
```bash
--data PATH              # Path to training data (.pt file)
--output_dir PATH        # Directory for outputs and checkpoints
--epochs INT             # Number of training epochs (default: 15)
--batch INT              # Batch size (default: 2048)
--latent INT             # Latent dimension (default: 32)
--lr FLOAT               # Learning rate (default: 1e-3)
--device STRING          # Device: 'cpu', 'gpu', or 'auto' (default: auto)
--input_size INT         # Input image size (default: 256)

# Regularization parameters
--lambda_act FLOAT       # L1 regularization coefficient (default: 1e-4)
--lambda_sim FLOAT       # Contrastive regularization (default: 5e-5)
--lambda_div FLOAT       # Divergence regularization (default: 2e-4)

# Evaluation options
--realtime_metrics       # Enable real-time PSNR/SSIM tracking
--no_realtime_metrics    # Disable real-time metrics (faster training)
```

### Example Usage
```bash
# Basic training
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 50 \
    --device cpu

# Advanced training with custom hyperparameters
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 100 \
    --batch 1024 \
    --latent 64 \
    --lr 5e-4 \
    --lambda_act 2e-4 \
    --realtime_metrics
```

### Output Files
- `ae.ckpt` - Trained model checkpoint
- `loss_curve.png` - Training loss progression
- `reconstruction_comparison.png` - Pre-training pattern comparison
- `stem_visualization.png` - Pre-training STEM analysis
- `final_reconstruction_comparison.png` - Post-training comparison
- `final_stem_visualization.png` - Post-training STEM analysis
- `tb_logs/` - TensorBoard logging files

### Training Features
- **80/20 train/validation split** for proper evaluation
- **Multi-component regularized loss** with adjustable coefficients
- **Optional real-time metrics** (PSNR/SSIM) - disable for faster training
- **Comprehensive logging** with TensorBoard integration
- **Automatic final evaluation** with detailed metrics and visualizations

## 🔄 convert_dm4.py

**Purpose**: Convert Digital Micrograph (.dm4) files to PyTorch tensors

### Usage
```bash
python scripts/convert_dm4.py [OPTIONS]
```

### Arguments
```bash
--input PATH             # Input .dm4 file path
--output PATH            # Output .pt file path
--downsample INT         # Downsampling factor (default: 1)
--mode STRING            # Downsampling mode: 'bin' or 'avg' (default: bin)
--normalize              # Normalize to [0,1] range (default: True)
```

### Example Usage
```bash
# Basic conversion
python scripts/convert_dm4.py \
    --input data/Diffraction_SI.dm4 \
    --output data/train_tensor.pt

# With downsampling for memory efficiency
python scripts/convert_dm4.py \
    --input data/Diffraction_SI.dm4 \
    --output data/train_tensor.pt \
    --downsample 8 \
    --mode bin
```

### Features
- **Automatic format detection** and conversion
- **Memory-efficient processing** with progress tracking
- **Downsampling options** for large datasets
- **Flexible binning and averaging** modes
- **Automatic normalization** to [0,1] range

## 🧠 generate_embeddings.py

**Purpose**: Generate latent space embeddings from trained models

### Usage
```bash
python scripts/generate_embeddings.py [OPTIONS]
```

### Arguments
```bash
--input PATH             # Input data file (.pt)
--checkpoint PATH        # Trained model checkpoint (.ckpt)
--output PATH            # Output embeddings file (.pt)
--batch_size INT         # Processing batch size (default: 2048)
--device STRING          # Device: 'cpu', 'gpu', or 'auto' (default: auto)
```

### Example Usage
```bash
# Generate embeddings
python scripts/generate_embeddings.py \
    --input data/train_tensor.pt \
    --checkpoint outputs/ae.ckpt \
    --output outputs/embeddings.pt \
    --batch_size 1024

# For large datasets
python scripts/generate_embeddings.py \
    --input data/large_dataset.pt \
    --checkpoint outputs/ae.ckpt \
    --output outputs/embeddings.pt \
    --batch_size 512 \
    --device gpu
```

### Features
- **Batch processing** for memory efficiency
- **Automatic checkpoint loading** with device handling
- **Progress tracking** for large datasets
- **Flexible output formats** (.pt, .npy compatible)

## 📊 visualise_scan_latents.py

**Purpose**: Create spatial maps of latent dimensions across scan positions

### Usage
```bash
python scripts/visualise_scan_latents.py [OPTIONS]
```

### Arguments
```bash
--raw PATH               # Raw data file (.pt)
--latents PATH           # Latent embeddings file (.pt)
--scan INT INT           # Scan grid dimensions (height width)
--virtual STRING         # Virtual field mode: 'bf' (bright field)
--outfig PATH            # Output figure path (.png)
--dpi INT                # Figure DPI (default: 300)
```

### Example Usage
```bash
# Basic latent visualization
python scripts/visualise_scan_latents.py \
    --raw data/train_tensor.pt \
    --latents outputs/embeddings.pt \
    --scan 42 114 \
    --virtual bf \
    --outfig outputs/latent_mosaic.png

# High-resolution output
python scripts/visualise_scan_latents.py \
    --raw data/train_tensor.pt \
    --latents outputs/embeddings.pt \
    --scan 64 64 \
    --virtual bf \
    --outfig outputs/high_res_latents.png \
    --dpi 600
```

### Features
- **Configurable scan grid** dimensions
- **Multiple visualization modes** (bright-field, etc.)
- **Mosaic generation** for all latent dimensions
- **High-quality output** with adjustable DPI
- **Automatic spatial mapping** of latent features

### Output
- **Latent mosaic**: Grid showing spatial distribution of each latent dimension
- **Virtual field images**: Reconstructed bright/dark field visualizations
- **Scan position maps**: Spatial arrangement of embeddings

## 🔧 preprocess.py

**Purpose**: Additional data preprocessing and preparation utilities

### Features
- Data normalization and standardization
- Batch processing utilities
- Quality control and validation
- Format conversion helpers

## 📋 Workflow Examples

### Complete Training Pipeline
```bash
# 1. Convert data
python scripts/convert_dm4.py \
    --input data/raw_data.dm4 \
    --output data/train_tensor.pt \
    --downsample 4

# 2. Train model
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 50 \
    --realtime_metrics

# 3. Generate embeddings
python scripts/generate_embeddings.py \
    --input data/train_tensor.pt \
    --checkpoint outputs/ae.ckpt \
    --output outputs/embeddings.pt

# 4. Visualize results
python scripts/visualise_scan_latents.py \
    --raw data/train_tensor.pt \
    --latents outputs/embeddings.pt \
    --scan 42 114 \
    --outfig outputs/analysis.png
```

### Hyperparameter Tuning
```bash
# Test different latent dimensions
for latent in 16 32 64 128; do
    python scripts/train.py \
        --data data/train_tensor.pt \
        --output_dir outputs/latent_${latent} \
        --latent $latent \
        --epochs 30
done

# Test different regularization
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs/strong_reg \
    --lambda_act 5e-4 \
    --lambda_sim 1e-4 \
    --lambda_div 5e-4
```

## ⚡ Performance Tips

### Memory Optimization
- **Reduce batch size** for large inputs or limited memory
- **Use CPU training** for very large datasets
- **Enable gradient checkpointing** for deeper models

### Speed Optimization  
- **Disable real-time metrics** during training for 2-3x speedup
- **Use GPU** when available for faster training
- **Increase batch size** on high-memory systems

### Quality Optimization
- **Enable real-time metrics** to monitor reconstruction quality
- **Use larger latent dimensions** for complex patterns
- **Adjust regularization** based on dataset characteristics

## 🐛 Troubleshooting

### Common Issues
1. **Out of memory**: Reduce batch size or use CPU
2. **Slow training**: Disable real-time metrics
3. **Poor reconstruction**: Increase latent dimension or adjust regularization
4. **File not found**: Check data paths and file extensions

### Error Messages
- `CUDA out of memory`: Use `--device cpu` or smaller batch size
- `File not readable`: Ensure .dm4 files are accessible and valid
- `Invalid scan dimensions`: Check scan grid matches data dimensions

---

*For implementation details, see the source code in the `scripts/` directory.*