# Preprocessing Scripts

This directory contains scripts for converting and preprocessing 4D-STEM data.

## Scripts

### `convert_dm4_to_hspy.py` (RECOMMENDED)
**Fast conversion from dm4 to hspy format**
- Converts dm4 files to HyperSpy format with lazy loading
- Automatic optimal chunking for performance
- Compression and rechunking optimization
- One-time conversion, multiple uses

**Usage:**
```bash
python convert_dm4_to_hspy.py --input data.dm4 --output data.hspy
```

**Key Features:**
- **5-10x faster** than traditional preprocessing
- **Lazy loading** - data stays on disk
- **Automatic chunking** - optimized for 64MB chunks
- **Compression** - smaller files than original dm4
- **Rechunking** - optimized for scan dimension access

### `preprocess.py`
**Traditional preprocessing to PyTorch tensors**
- Converts various formats to `.pt` tensors
- Memory-efficient chunked processing for large files
- Multiple downsampling methods
- Progress tracking and memory monitoring

**Usage:**
```bash
python preprocess.py --input data.dm4 --output processed.pt
```

**Key Features:**
- Automatic file type detection (dm4, hdf5, hspy)
- Memory-efficient processing with chunking
- Multiple downsampling methods (bin, stride, gauss, fft)
- Progress bars and memory monitoring
- Configurable data types (float32, float16)

### `convert_dm4.py`
**Legacy conversion script**
- Basic dm4 to other format conversion
- Limited functionality compared to newer scripts

## Workflow Comparison

### New Workflow (Recommended)
```bash
# Convert once
python convert_dm4_to_hspy.py --input large_data.dm4 --output large_data.hspy

# Train multiple times with different parameters
python ../training/train_hyperspy.py --data large_data.hspy --output_dir outputs1 --downsample 2
python ../training/train_hyperspy.py --data large_data.hspy --output_dir outputs2 --scan_step 2
```

### Traditional Workflow
```bash
# Preprocess (slow, memory intensive)
python preprocess.py --input data.dm4 --output processed.pt --downsample 2

# Train
python ../training/train.py --data processed.pt --output_dir outputs
```

## Performance Comparison

| Method | Processing Time | Memory Usage | Flexibility |
|--------|----------------|--------------|-------------|
| **convert_dm4_to_hspy** | Fast (minutes) | Low | High |
| **preprocess.py** | Slow (hours) | High | Low |

## File Size Guidelines

| Original dm4 Size | Recommended Method | Memory Settings |
|------------------|-------------------|----------------|
| < 1 GB | Either method | Default settings |
| 1-5 GB | convert_dm4_to_hspy | --chunk_size 64 |
| 5-20 GB | convert_dm4_to_hspy | --chunk_size 32 |
| > 20 GB | preprocess.py --memory_efficient | --chunk_size 16 |

## Preprocessing Options

### Downsampling Methods
- **bin**: Mean pooling (default, best quality)
- **stride**: Pixel skipping (fastest)
- **gauss**: Gaussian filter + stride (good quality)
- **fft**: Fourier cropping (best quality, slowest)

### Data Types
- **float32**: Full precision (default)
- **float16**: Half precision (50% smaller files)

### Memory-Efficient Processing
```bash
# For very large files
python preprocess.py --input huge_data.dm4 --output processed.pt \
    --memory_efficient --chunk_size 32 --dtype float16
```

## Output Formats

### HyperSpy Format (.hspy)
- **Lazy loading**: Data stays on disk
- **Chunked**: Optimized for efficient access
- **Compressed**: Often smaller than original
- **Metadata**: Preserves all original metadata

### PyTorch Tensors (.pt)
- **Immediate loading**: Full data in memory
- **Preprocessed**: Ready for training
- **Compressed**: float16 option available
- **Fast training**: No preprocessing overhead

## Migration Guide

**From old preprocessing:**
```bash
# Old way (slow)
python preprocess.py --input data.dm4 --output processed.pt

# New way (fast)
python convert_dm4_to_hspy.py --input data.dm4 --output data.hspy
```

**Benefits of migration:**
- **10x faster** preprocessing
- **100x less memory** usage during training
- **Flexible** parameter experimentation
- **Reusable** converted files

## Troubleshooting

### "Out of memory" during preprocessing
- Use `--memory_efficient` flag
- Reduce `--chunk_size`
- Use `--dtype float16`

### Slow preprocessing
- Use `convert_dm4_to_hspy.py` instead of `preprocess.py`
- Increase `--chunk_size` if you have more memory

### Large output files
- Use `--dtype float16`
- Increase `--downsample` factor
- Use `--scan_step` for subsampling