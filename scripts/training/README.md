# 4D-STEM Autoencoder Training Pipeline

This directory contains the refactored training pipeline for 4D-STEM autoencoder models with clean, object-oriented architecture and comprehensive regularization support.

## Architecture Overview

### Core Components

**`train.py`** - Main training orchestrator (681 lines)
- Clean object-oriented design with single responsibility classes
- Comprehensive configuration management via `TrainingConfig` dataclass
- Modular pipeline with `DatasetManager`, `ModelManager`, `TrainerManager`, and `TrainingPipeline`

**`dataset.py`** - HDF5Dataset implementation
- Optimized lazy loading for HDF5-compressed 4D-STEM data
- Automatic normalization statistics computation and caching
- Memory-efficient chunked processing with Welford's algorithm

**`lightning_model.py`** - PyTorch Lightning model wrapper
- Scale-aligned loss computation in log space
- Comprehensive metrics tracking (MSE, PSNR, SSIM, diffraction metrics)
- Flexible regularization system integration

## Usage

### Basic Training
```bash
python train.py --data data/train_data.h5 --output_dir outputs --epochs 50
```

### Advanced Configuration
```bash
# High-performance training with regularization
python train.py \
    --data data/train_data.h5 \
    --output_dir outputs \
    --epochs 100 \
    --batch 64 \
    --latent 256 \
    --lr 0.001 \
    --precision 16 \
    --lambda_act 1e-4 \
    --lambda_l2 1e-6 \
    --compile \
    --num_workers 8
```

### Regularization Options
```bash
# Light regularization (recommended for stable training)
python train.py --data data.h5 --output_dir outputs \
    --lambda_act 1e-5 --lambda_l2 1e-6

# Heavy regularization (prevent overfitting)
python train.py --data data.h5 --output_dir outputs \
    --lambda_act 1e-3 --lambda_l2 1e-4 --lambda_div 1e-4

# Contrastive learning setup
python train.py --data data.h5 --output_dir outputs \
    --lambda_sim 1e-4 --lambda_div 5e-5

# Multiple regularizations
python train.py --data data.h5 --output_dir outputs \
    --lambda_act 1e-4 --lambda_sim 5e-5 --lambda_div 2e-4 \
    --lambda_l2 1e-6 --lambda_kl 1e-3
```

## Configuration Options

### Data Parameters
- `--data`: Path to training data (.h5 or .pt file)
- `--output_dir`: Output directory for results and checkpoints
- `--no_normalization`: Skip z-score normalization, train on raw log data
- `--no_validation`: Use entire dataset for training (no validation split)

### Model Parameters
- `--latent`: Latent dimension size (default: 128)
- `--input_size`: Input image size, assumes square (default: 256)

### Training Parameters
- `--epochs`: Number of training epochs (default: 50)
- `--batch`: Batch size (default: 128)
- `--lr`: Learning rate (default: 1e-3)
- `--no_scheduler`: Disable learning rate scheduler
- `--precision`: Training precision - 32, 16, bf16 (default: 16)
- `--accumulate_grad_batches`: Gradient accumulation batches (default: 1)

### Regularization Parameters
- `--lambda_act`: Lp regularization coefficient (default: 1e-5)
- `--lambda_sim`: Contrastive similarity coefficient (default: 0)
- `--lambda_div`: Activation divergence coefficient (default: 0)
- `--lambda_l2`: L2 regularization coefficient (default: 0)
- `--lambda_kl`: KL divergence coefficient (default: 0)

### Hardware Parameters
- `--device`: Device to use - auto, cpu, cuda, mps (default: auto)
- `--gpus`: Number of GPUs to use (default: 1)
- `--compile`: Use torch.compile for optimization (Linux only)

### Data Loading Parameters
- `--num_workers`: Number of data loading workers (default: 4)
- `--pin_memory`: Pin memory for faster GPU transfer
- `--persistent_workers`: Keep workers alive between epochs

## Regularization System

The training pipeline supports five types of regularization:

### 1. Lp Regularization (`--lambda_act`)
- **Purpose**: Activation magnitude regularization
- **Default**: `1e-5`
- **Usage**: General regularization to prevent activation explosion
- **Typical values**: `1e-6` to `1e-3`

### 2. Contrastive Regularization (`--lambda_sim`)
- **Purpose**: Contrastive similarity learning
- **Default**: `0` (disabled)
- **Usage**: Encourage similar patterns to have similar representations
- **Typical values**: `1e-5` to `1e-4`

### 3. Divergence Regularization (`--lambda_div`)
- **Purpose**: Activation diversity encouragement
- **Default**: `0` (disabled)
- **Usage**: Prevent representation collapse, encourage diverse activations
- **Typical values**: `1e-5` to `2e-4`

### 4. L2 Regularization (`--lambda_l2`)
- **Purpose**: Weight decay / L2 penalty
- **Default**: `0` (disabled)
- **Usage**: Prevent overfitting by penalizing large weights
- **Typical values**: `1e-7` to `1e-4`

### 5. KL Divergence (`--lambda_kl`)
- **Purpose**: Probabilistic regularization
- **Default**: `0` (disabled)
- **Usage**: Variational autoencoder-style regularization
- **Typical values**: `1e-4` to `1e-3`

## Output Structure

Training generates the following outputs in the specified `--output_dir`:

```
outputs/
├── checkpoints/                    # Model checkpoints
│   ├── ae_e050_1201_1430_epoch050_valloss0.0123.ckpt
│   └── ...
├── tb_logs/                        # TensorBoard logs
│   └── version_0/
├── ae_e050_1201_1430_final.ckpt   # Final model checkpoint
├── ae_e050_1201_1430_loss.png     # Training loss curve
├── ae_e050_1201_1430_reconstruction.png  # Sample reconstructions
└── training_log_20241201_143022.txt      # Detailed training log
```

## Performance Features

### Memory Optimization
- **Lazy HDF5 loading**: Patterns loaded on-demand
- **Chunked statistics computation**: Memory-efficient normalization
- **Optimized data loaders**: Configurable multiprocessing with fallback
- **Mixed precision training**: Reduced memory usage with float16

### Speed Optimization
- **torch.compile**: Model compilation for faster execution (Linux)
- **Persistent workers**: Reduce data loading overhead
- **Gradient accumulation**: Effective larger batch sizes
- **Automatic device selection**: Best available hardware

### Monitoring
- **Real-time metrics**: MSE, PSNR, SSIM during training
- **Diffraction-specific metrics**: Peak preservation, log correlation
- **TensorBoard integration**: Comprehensive training visualization
- **Structured logging**: Detailed training logs with timestamps

## Data Format Support

### HDF5 Format (Recommended)
- **File extension**: `.h5`
- **Dataset name**: `patterns`
- **Shape**: `(n_patterns, height, width)`
- **Metadata**: Automatic detection of data type and scaling
- **Normalization**: Automatic computation and caching of statistics

### PyTorch Tensor Format
- **File extension**: `.pt`
- **Shape**: `(n_patterns, channels, height, width)`
- **Loading**: Direct tensor loading via `torch.load()`

## Testing

The training pipeline includes a comprehensive test suite with 95 tests:

```bash
# Run all tests
pytest scripts/training/tests/

# Run only unit tests
pytest scripts/training/tests/ -m unit

# Run regularization tests
pytest scripts/training/tests/test_regularization* -v

# Run with coverage
pytest scripts/training/tests/ --cov=scripts.training.train
```

### Test Coverage
- Configuration validation and device detection
- Dataset loading for HDF5 and tensor formats
- Data loader creation with multiprocessing fallback
- Model creation and compilation
- Trainer setup with different accelerators
- Complete training pipeline orchestration
- **Regularization system** (all 5 types with 30 dedicated tests)
- Command-line interface parsing
- Error handling and edge cases

## Migration from Legacy Code

The refactored system maintains full backward compatibility:

```bash
# Legacy usage (still works)
python train.py --data data.h5 --output_dir outputs --epochs 50

# New features available
python train.py --data data.h5 --output_dir outputs --epochs 50 \
    --lambda_act 1e-4 --precision 16 --compile
```

## Dependencies

### Core Requirements
- **PyTorch Lightning**: Training framework and utilities
- **PyTorch**: Deep learning framework
- **h5py**: HDF5 file format support
- **NumPy**: Numerical computations
- **matplotlib**: Loss curve visualization

### Optional Dependencies
- **tensorboard**: Training visualization
- **psutil**: System monitoring
- **tqdm**: Progress bars

## Architecture Benefits

### Maintainability
- **Single Responsibility**: Each class has a clear, focused purpose
- **Dependency Injection**: Configuration passed through constructors
- **Separation of Concerns**: Data, model, and training logic cleanly separated
- **Type Hints**: Full type annotations for better IDE support

### Extensibility
- **Pluggable Components**: Easy to swap implementations
- **Configuration-Driven**: All parameters centralized in TrainingConfig
- **Clean Interfaces**: Well-defined methods for extending functionality
- **Testable Design**: Each component can be unit tested independently

### Reliability
- **Comprehensive Testing**: 95 tests covering all components
- **Error Handling**: Graceful degradation and proper error propagation
- **Logging**: Structured logging with proper context
- **Validation**: Input validation and configuration checking

This refactored training pipeline provides a solid foundation for 4D-STEM autoencoder research with professional-grade architecture, comprehensive regularization support, and extensive testing coverage.