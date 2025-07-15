# Training Guide

This document provides comprehensive guidance for training the Custom 4D-STEM Autoencoder, including best practices, hyperparameter tuning, and troubleshooting.

## 🎯 Training Overview

The training process uses PyTorch Lightning with a multi-component regularized loss function designed for high-quality reconstruction of 4D-STEM diffraction patterns.

## 🏃‍♂️ Quick Start Training

### Basic Training Command
```bash
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 50 \
    --device cpu
```

### Recommended Settings
```bash
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 100 \
    --batch 1024 \
    --latent 32 \
    --lr 1e-3 \
    --lambda_act 1e-4 \
    --lambda_sim 5e-5 \
    --lambda_div 2e-4 \
    --realtime_metrics
```

## 🧪 Loss Function Details

### Multi-Component Loss
```
L = MSE(y, ŷ) + λ_act·L₁(a) + λ_sim·L_sim + λ_div·L_div
```

#### Component Breakdown
1. **MSE Loss**: Pixel-wise reconstruction fidelity
2. **L1 Regularization** (`λ_act`): Promotes sparsity in latent activations
3. **Contrastive Loss** (`λ_sim`): Encourages embedding diversity
4. **Divergence Loss** (`λ_div`): Prevents mode collapse

#### Default Hyperparameters
| Parameter | Default Value | Purpose |
|-----------|---------------|---------|
| `λ_act` | `1e-4` | L1 sparsity regularization |
| `λ_sim` | `5e-5` | Contrastive similarity penalty |
| `λ_div` | `2e-4` | Activation divergence regularization |

## ⚙️ Hyperparameter Guide

### Learning Rate (`--lr`)
**Default**: `1e-3`

**Recommendations**:
- **Large datasets**: `1e-3` to `5e-3`
- **Small datasets**: `1e-4` to `5e-4`
- **Fine-tuning**: `1e-5` to `1e-4`

```bash
# Conservative learning rate
--lr 5e-4

# Aggressive learning rate  
--lr 2e-3
```

### Batch Size (`--batch`)
**Default**: `2048`

**Recommendations**:
- **GPU training**: `1024` to `4096`
- **CPU training**: `512` to `2048`
- **Limited memory**: `256` to `512`

```bash
# High-memory system
--batch 4096

# Limited memory
--batch 512
```

### Latent Dimension (`--latent`)
**Default**: `32`

**Guidelines**:
- **Simple patterns**: `16` to `32`
- **Complex patterns**: `32` to `64`
- **Very complex patterns**: `64` to `128`

```bash
# For complex diffraction patterns
--latent 64

# For memory efficiency
--latent 16
```

### Regularization Parameters

#### L1 Regularization (`--lambda_act`)
**Default**: `1e-4`

**Effect**: Controls sparsity of latent activations
- **Higher values** (1e-3): Strong sparsity, may hurt reconstruction
- **Lower values** (1e-5): Weak sparsity, may overfit

```bash
# Strong sparsity
--lambda_act 5e-4

# Weak sparsity
--lambda_act 1e-5
```

#### Contrastive Regularization (`--lambda_sim`)
**Default**: `5e-5`

**Effect**: Promotes diversity in embeddings
- **Higher values** (1e-4): Strong diversity, may hurt reconstruction
- **Lower values** (1e-6): Weak diversity, may cause mode collapse

```bash
# Strong diversity
--lambda_sim 1e-4

# Minimal diversity
--lambda_sim 1e-6
```

#### Divergence Regularization (`--lambda_div`)
**Default**: `2e-4`

**Effect**: Prevents mode collapse
- **Higher values** (5e-4): Strong prevention, may increase noise
- **Lower values** (1e-5): Weak prevention, risk of collapse

```bash
# Strong mode collapse prevention
--lambda_div 5e-4

# Minimal prevention
--lambda_div 1e-5
```

## 📊 Training Monitoring

### Real-time Metrics (`--realtime_metrics`)
**Purpose**: Track PSNR and SSIM during training

**Trade-offs**:
- ✅ **Enables**: Real-time quality monitoring
- ❌ **Cost**: 2-3x slower training

```bash
# Enable for quality monitoring
--realtime_metrics

# Disable for faster training
--no_realtime_metrics
```

### TensorBoard Logging
Automatically enabled - view with:
```bash
tensorboard --logdir outputs/tb_logs
```

**Logged Metrics**:
- Training and validation loss components
- PSNR and SSIM (if enabled)
- Learning rate and regularization values

## 🎨 Training Strategies

### Progressive Training
Start with basic settings, then refine:

```bash
# Phase 1: Initial training
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs/phase1 \
    --epochs 30 \
    --batch 2048 \
    --no_realtime_metrics

# Phase 2: Quality refinement
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs/phase2 \
    --epochs 50 \
    --batch 1024 \
    --lr 5e-4 \
    --realtime_metrics
```

### Hyperparameter Sweeps
Systematic exploration of parameter space:

```bash
# Latent dimension sweep
for latent in 16 32 64; do
    python scripts/train.py \
        --data data/train_tensor.pt \
        --output_dir outputs/latent_${latent} \
        --latent $latent \
        --epochs 50
done

# Regularization sweep
for lambda_act in 1e-5 1e-4 1e-3; do
    python scripts/train.py \
        --data data/train_tensor.pt \
        --output_dir outputs/reg_${lambda_act} \
        --lambda_act $lambda_act \
        --epochs 50
done
```

### Transfer Learning
Fine-tune from pre-trained models:

```bash
# Fine-tune from checkpoint
python scripts/train.py \
    --data data/new_dataset.pt \
    --output_dir outputs/finetune \
    --epochs 20 \
    --lr 1e-4 \
    --checkpoint outputs/pretrained/ae.ckpt
```

## 📈 Performance Optimization

### Memory Optimization
```bash
# For limited memory systems
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --batch 256 \
    --device cpu \
    --no_realtime_metrics
```

### Speed Optimization
```bash
# For maximum training speed
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --batch 4096 \
    --device gpu \
    --no_realtime_metrics \
    --workers 8
```

### Quality Optimization
```bash
# For maximum reconstruction quality
python scripts/train.py \
    --data data/train_tensor.pt \
    --output_dir outputs \
    --epochs 200 \
    --batch 512 \
    --latent 64 \
    --lr 5e-4 \
    --lambda_act 2e-4 \
    --realtime_metrics
```

## 📋 Best Practices

### Dataset Preparation
1. **Normalize data** to [0,1] range
2. **Check data quality** - remove corrupted patterns
3. **Consider downsampling** for memory efficiency
4. **Validate scan dimensions** match data shape

### Training Process
1. **Start with default parameters** for baseline
2. **Monitor early epochs** for convergence issues
3. **Use real-time metrics** for quality assessment
4. **Save multiple checkpoints** for comparison
5. **Validate on unseen data** after training

### Regularization Tuning
1. **Start with default values** (1e-4, 5e-5, 2e-4)
2. **Adjust L1 first** - most impactful on reconstruction
3. **Tune contrastive loss** for embedding diversity
4. **Modify divergence** if mode collapse occurs

## 🐛 Troubleshooting

### Common Issues

#### Poor Reconstruction Quality
**Symptoms**: High MSE, low PSNR/SSIM
**Solutions**:
- Increase latent dimension (`--latent 64`)
- Reduce L1 regularization (`--lambda_act 1e-5`)
- Lower learning rate (`--lr 5e-4`)
- Train longer (`--epochs 200`)

#### Mode Collapse
**Symptoms**: All embeddings become similar
**Solutions**:
- Increase divergence regularization (`--lambda_div 5e-4`)
- Increase contrastive loss (`--lambda_sim 1e-4`)
- Use different initialization

#### Training Instability
**Symptoms**: Loss oscillations, NaN values
**Solutions**:
- Reduce learning rate (`--lr 1e-4`)
- Reduce batch size (`--batch 512`)
- Check data normalization
- Reduce regularization strength

#### Memory Issues
**Symptoms**: CUDA out of memory, system crashes
**Solutions**:
- Reduce batch size (`--batch 256`)
- Use CPU training (`--device cpu`)
- Disable real-time metrics (`--no_realtime_metrics`)
- Reduce input size during conversion

### Diagnostic Commands

#### Check Data Shape
```python
import torch
data = torch.load('data/train_tensor.pt')
print(f"Data shape: {data.shape}")
print(f"Data range: [{data.min():.3f}, {data.max():.3f}]")
```

#### Monitor GPU Memory
```bash
# During training
nvidia-smi -l 1
```

#### Validate Checkpoint
```python
from models.autoencoder import Autoencoder
model = Autoencoder.load_from_checkpoint('outputs/ae.ckpt')
print(f"Model loaded successfully")
```

## 📊 Evaluation Metrics

### Training Metrics
- **Total Loss**: Combined multi-component loss
- **MSE Loss**: Reconstruction fidelity
- **L1 Regularization**: Sparsity term
- **Contrastive Loss**: Embedding diversity
- **Divergence Loss**: Mode collapse prevention

### Quality Metrics (if enabled)
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index (0-1)
- **MSE**: Mean Squared Error

### Success Criteria
- **Training loss**: Decreasing trend
- **Validation loss**: Following training loss
- **PSNR**: > 20 dB for good quality
- **SSIM**: > 0.7 for good structural similarity
- **Visual inspection**: Clear reconstruction quality

---

*For more details on model architecture and implementation, see [Model Documentation](../models/README.md).*