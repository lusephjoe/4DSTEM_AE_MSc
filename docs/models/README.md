# Model Architecture Documentation

This document provides detailed technical documentation for the neural network architectures used in the Custom 4D-STEM Autoencoder project.

## 🏗️ Architecture Overview

The autoencoder uses a ResNet-based architecture with adaptive components designed for **image-size agnostic processing**.

## 📁 Model Files

### `models/autoencoder.py`
Main autoencoder implementation with three primary classes:

#### `Encoder` Class
**Purpose**: Encodes input diffraction patterns into 32D latent representations

**Architecture Flow**:
```
Input (1, H, W) 
→ conv_input + BN + ReLU (64 channels)
→ conv_pre + BN + ReLU (128 channels)
→ ResNet Block 1 (adaptive pooling)
→ ResNet Block 2 (adaptive pooling)
→ ResNet Block 3 (adaptive pooling)
→ conv_post + [conditional BN] + ReLU (64 channels)
→ conv_final + [conditional BN] + ReLU (1 channel)
→ Adaptive Pooling (4×4)
→ Embedding Layer (32D output)
```

**Key Features**:
- **Adaptive pooling**: Handles any input size from 32×32 to 512×512+
- **Conditional batch normalization**: Only applied when spatial dimensions > 1×1

#### `Decoder` Class
**Purpose**: Reconstructs diffraction patterns from 32D latent vectors

**Architecture Flow**:
```
32D Latent Vector
→ Linear (32 → 2048) 
→ Reshape (128, 4, 4)
→ conv_initial (128 channels)
→ ResNet Up Block 1 (4×4 → 16×16)
→ ResNet Up Block 2 (16×16 → 64×64)
→ ResNet Up Block 3 (64×64 → 256×256)
→ conv_final + Sigmoid (1 channel)
→ Interpolate to target size
```

**Key Features**:
- **Fixed base size**: Always starts from 4×4 feature maps
- **Consistent parameters**: Same parameter count regardless of target size
- **Adaptive output**: Final interpolation to match desired output size

#### `Autoencoder` Class
**Purpose**: Complete model combining encoder and decoder with regularized loss

**Methods**:
- `forward(x)`: Full encode-decode cycle
- `embed(x)`: Encode only (returns latent vectors)
- `compute_loss(x, x_hat, z)`: Multi-component loss calculation

### `models/blocks.py`
Modular building blocks for the neural network

#### `ConvBlock` Class
**Purpose**: Residual convolutional block with skip connections

**Architecture**:
```
Input → Conv1 → [Conditional BN] → ReLU
      → Conv2 → [Conditional BN] → ReLU  
      → Conv3 → [Conditional BN] → Add Skip → ReLU
```

**Features**:
- 3 sequential convolutional layers
- Skip connection with dimension matching
- Conditional batch normalization for small feature maps

#### `IdentityBlock` Class
**Purpose**: Single convolution with normalization

**Architecture**:
```
Input → Conv → [Conditional BN] → ReLU
```

#### `ResNetBlock` Class
**Purpose**: Complete ResNet block with adaptive pooling

**Architecture**:
```
Input → ConvBlock → IdentityBlock → Adaptive Pooling
```

**Key Innovation**: 
- Replaces fixed MaxPool2d with adaptive pooling
- Calculates target size: `max(1, current_size // pool_size)`
- Prevents "output size too small" errors

#### `ResNetUpBlock` Class  
**Purpose**: Upsampling ResNet block for decoder

**Architecture**:
```
Input → Bilinear Upsample → ConvBlock → IdentityBlock
```

#### `EmbeddingLayer` Class
**Purpose**: Projects flattened features to latent space

**Architecture**:
```
Input → Linear → ReLU (non-negative activations)
```

#### `AdaptiveDecoder` Class
**Purpose**: Size-agnostic decoder with consistent parameters

**Key Design**:
- **Fixed base size**: Always uses 4×4 starting point
- **Consistent parameters**: 2048 features regardless of target size
- **Adaptive output**: Final interpolation to target dimensions

#### Loss Components

##### `ContrastiveLoss` Class
**Purpose**: Promotes diversity in latent embeddings

**Implementation**:
```python
# Compute cosine similarity matrix
embeddings_norm = F.normalize(embeddings, p=2, dim=1)
similarity_matrix = torch.matmul(embeddings_norm, embeddings_norm.t())

# Penalize high similarity between different samples
loss = F.relu(similarity - margin)
```

##### `DivergenceLoss` Class  
**Purpose**: Prevents mode collapse by encouraging activation variance

**Implementation**:
```python
variance = torch.var(embeddings, dim=0)
loss = torch.mean(1.0 / (variance + 1e-8))
```

### `models/summary.py`
Model analysis and performance evaluation tools

#### Key Functions:
- `show()`: Enhanced model summary with performance metrics
- `calculate_metrics()`: PSNR, SSIM, and MSE computation
- `save_comparison_images()`: Visual reconstruction comparisons
- `save_stem_visualization()`: Professional STEM analysis visualizations

## 🧪 Loss Function Details

### Multi-Component Loss
```python
L = MSE(y, ŷ) + λ_act·L₁(a) + λ_sim·L_sim + λ_div·L_div
```

**Components**:
1. **MSE Loss**: Pixel-wise reconstruction fidelity
2. **L1 Regularization**: Promotes sparsity in latent activations
3. **Contrastive Loss**: Encourages embedding diversity
4. **Divergence Loss**: Prevents mode collapse

**Default Hyperparameters**:
- `λ_act = 1e-4` (L1 regularization)
- `λ_sim = 5e-5` (contrastive similarity)
- `λ_div = 2e-4` (divergence regularization)

## 🔧 Technical Implementation Details

### Adaptive Pooling Strategy
The model uses adaptive pooling to handle variable input sizes:

```python
# Calculate target size based on current input and pool factor
current_size = x.size(-1)
target_size = max(1, current_size // pool_size)

# Use adaptive pooling instead of fixed MaxPool2d
x = F.adaptive_avg_pool2d(x, (target_size, target_size))
```

### Conditional Batch Normalization
Batch normalization is only applied when spatial dimensions are greater than 1×1:

```python
out = self.conv(x)
if out.size(-1) > 1 or out.size(-2) > 1:
    out = self.bn(out)
out = self.relu(out)
```

This prevents the "Expected more than 1 value per channel" error that occurs with 1×1 feature maps.

### Parameter Consistency
The model maintains exactly **3,914,404 parameters** across all input sizes by:
- Using fixed 4×4 base size in decoder
- Consistent channel dimensions throughout
- Adaptive final interpolation rather than size-dependent layers

## 📊 Model Performance

### Input Size Flexibility
| Input Size | Parameters | Latent Shape | Output Shape |
|------------|------------|--------------|--------------|
| 32×32      | 3,914,404  | (B, 32)      | (B, 1, 32, 32) |
| 64×64      | 3,914,404  | (B, 32)      | (B, 1, 64, 64) |
| 128×128    | 3,914,404  | (B, 32)      | (B, 1, 128, 128) |
| 256×256    | 3,914,404  | (B, 32)      | (B, 1, 256, 256) |
| 512×512    | 3,914,404  | (B, 32)      | (B, 1, 512, 512) |

### Architecture Validation
All model components pass comprehensive tests:
- ✅ Encoder output consistency across input sizes
- ✅ Decoder output size matching
- ✅ Autoencoder reconstruction consistency
- ✅ Embedding layer non-negativity
- ✅ Gradient flow validation
- ✅ Parameter count consistency

## 🚀 Usage Examples

### Basic Model Creation
```python
from models.autoencoder import Autoencoder

# Create model for 256×256 inputs
model = Autoencoder(latent_dim=32, out_shape=(256, 256))

# Model works with any input size
x = torch.randn(2, 1, 256, 256)
reconstruction = model(x)
latent = model.embed(x)
```

### Loss Computation
```python
# Forward pass
z = model.embed(x)
x_hat = model(x)

# Compute regularized loss
loss_dict = model.compute_loss(x, x_hat, z, 
                              lambda_act=1e-4,
                              lambda_sim=5e-5, 
                              lambda_div=2e-4)
```

### Model Summary
```python
from models.summary import show

# Display detailed model summary
show(model, input_size=(1, 256, 256))
```

## 📈 Performance Metrics

The model tracks comprehensive reconstruction quality:
- **PSNR**: Peak Signal-to-Noise Ratio (dB)
- **SSIM**: Structural Similarity Index (0-1)
- **MSE**: Mean Squared Error
- **Loss components**: Individual tracking of all loss terms

---

*For implementation details, see the source code in the `models/` directory.*