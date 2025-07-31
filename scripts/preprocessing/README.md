# Preprocessing Scripts

Data conversion and preprocessing tools for 4D-STEM datasets.

## Scripts

### `preprocess.py`
**Legacy tensor preprocessing**

Converts various formats to PyTorch tensors with preprocessing options.

```bash
python preprocess.py --input data.dm4 --output processed.pt --downsample 2
```

**Features:**
- Multiple input formats (dm4, hdf5, hspy)
- Downsampling methods (bin, stride, gaussian, fft)
- Memory-efficient chunked processing
- Configurable data types (float32, float16)

### `convert_dm4.py`
**Main conversion utility**

Basic DM4 conversion with limited functionality.

## Usage Examples

### Basic Conversion
```bash
# DM4 to HDF5 for training pipeline
python convert_dm4_to_hspy.py --input scan_data.dm4 --output scan_data.h5

# DM4 to tensor with downsampling
python preprocess.py --input scan_data.dm4 --output processed.pt --downsample 2
```

### Memory-Efficient Processing
```bash
# For large files (>5GB)
python convert_dm4_to_hspy.py --input large_scan.dm4 --output large_scan.h5 --chunk_size 32

# Tensor preprocessing with memory efficiency
python preprocess.py --input large_scan.dm4 --output processed.pt \
    --memory_efficient --chunk_size 16 --dtype float16
```

## Configuration Options

### Downsampling Methods
- **bin**: Mean pooling (default, best quality)
- **stride**: Pixel skipping (fastest)
- **gaussian**: Gaussian filter + downsampling (balanced)
- **fft**: Fourier domain cropping (highest quality)

### Data Types
- **float32**: Full precision (default)
- **float16**: Half precision (reduces file size by 50%)

### Chunking Options
- **chunk_size**: Memory usage control (16, 32, 64 MB chunks)
- **memory_efficient**: Enable for large datasets
- **scan_step**: Subsample scan positions for smaller datasets

## Output Formats

### HDF5 (.h5)
- Compatible with training pipeline
- Efficient lazy loading
- Preserves metadata and coordinates
- Optimized chunking for fast access

### PyTorch Tensors (.pt)
- Direct tensor format
- Immediate loading
- Preprocessed and ready for training
- Supports float16 compression

## Common Issues

### Memory Errors
Use memory-efficient processing:
```bash
python preprocess.py --input large_file.dm4 --output output.pt \
    --memory_efficient --chunk_size 16 --dtype float16
```

### Large Output Files
Reduce file size with downsampling and compression:
```bash
python preprocess.py --input data.dm4 --output compressed.pt \
    --downsample 2 --dtype float16
```

For detailed options: `python script_name.py --help`