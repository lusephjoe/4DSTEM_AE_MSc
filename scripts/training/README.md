# Training Scripts

This directory contains scripts for training 4D-STEM autoencoders.

## Scripts

### `train.py`
**Traditional tensor-based training script**
- Loads preprocessed `.pt` tensor files
- Uses PyTorch Lightning for training
- Supports CUDA optimizations and mixed precision
- Includes memory monitoring and performance optimizations

**Usage:**
```bash
python train.py --data processed_data.pt --output_dir outputs --epochs 50
```

**Key Features:**
- Automatic device selection (CUDA/MPS/CPU)
- Mixed precision training (float16/float32)
- Model compilation with torch.compile
- Advanced optimizers (AdamW + scheduler)
- Memory-efficient data loading
- Real-time metrics and progress tracking

### `train_hyperspy.py`
**HyperSpy-based training script (RECOMMENDED)**
- Loads `.hspy` files directly with lazy loading
- Memory-efficient training for large datasets
- On-the-fly preprocessing and normalization
- Chunk-based loading for scalability

**Usage:**
```bash
python train_hyperspy.py --data data.hspy --output_dir outputs --use_chunked
```

**Key Features:**
- Lazy loading - minimal memory usage
- Chunked dataset support
- On-the-fly downsampling and normalization
- Configurable preprocessing parameters
- Memory monitoring and optimization

### `hyperspy_dataset.py`
**PyTorch dataset implementations for HyperSpy**
- `HyperSpyDataset`: Individual pattern loading
- `ChunkedHyperSpyDataset`: Chunk-based loading with caching

**Classes:**
- `HyperSpyDataset`: Loads patterns on-demand
- `ChunkedHyperSpyDataset`: Loads patterns in chunks with LRU caching

## Training Options

### Memory Efficiency
- **Small datasets (< 1GB)**: Use `train.py` with regular tensors
- **Large datasets (> 1GB)**: Use `train_hyperspy.py` with chunked loading

### Performance Options
```bash
# High performance CUDA training
python train_hyperspy.py --data data.hspy --output_dir outputs \
    --precision 16 --compile --persistent_workers --num_workers 4

# Memory-efficient training
python train_hyperspy.py --data data.hspy --output_dir outputs \
    --use_chunked --chunk_size 32 --batch 16
```

### Preprocessing During Training
```bash
# Downsample and subsample during training
python train_hyperspy.py --data data.hspy --output_dir outputs \
    --downsample 2 --scan_step 2 --downsample_mode bin
```

## Output

Both scripts generate:
- **Model checkpoints**: `ae.ckpt` and epoch-specific checkpoints
- **Training logs**: TensorBoard logs in `tb_logs/`
- **Loss curves**: `loss_curve.png`
- **Model summaries**: Architecture and parameter information
- **Performance metrics**: PSNR, SSIM, MSE tracking

## Dependencies

- PyTorch Lightning
- HyperSpy (for train_hyperspy.py)
- tqdm (for progress bars)
- psutil (for memory monitoring)

## Migration

**From old workflow:**
```bash
# Old
python preprocess.py --input data.dm4 --output processed.pt
python train.py --data processed.pt --output_dir outputs

# New (recommended)
python ../preprocessing/convert_dm4_to_hspy.py --input data.dm4 --output data.hspy
python train_hyperspy.py --data data.hspy --output_dir outputs
```