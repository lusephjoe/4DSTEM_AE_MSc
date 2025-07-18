"""Preprocess 4D-STEM datasets with automatic file type detection including .zarr - data processing only (no format conversion)."""
import argparse, h5py, numpy as np
from pathlib import Path
try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False
try:
    import zarr
    HAS_ZARR = True
except ImportError:
    HAS_ZARR = False
from scipy.ndimage import gaussian_filter
from tqdm import tqdm
import gc
import psutil
import os
import json

def get_memory_usage():
    """Get current memory usage in GB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

def load_data_lazy(file_path: Path):
    """Load data lazily for memory-efficient processing."""
    file_type = detect_file_type(file_path)
    print(f"Loading {file_type} file lazily: {file_path}")
    
    if file_type == "hdf5":
        f = h5py.File(file_path, "r")
        dataset = f["/data"]
        data_shape = dataset.shape
        print(f"Dataset shape: {data_shape}")
        return f, dataset, data_shape
        
    elif file_type in ["dm4", "hspy"]:
        if not HAS_HYPERSPY:
            raise ImportError("HyperSpy required for dm4/hspy files. Install with: pip install hyperspy")
        
        print(f"Loading {file_type} file lazily...")
        sig = hs.load(file_path.as_posix(), lazy=True)  # Load lazily!
        data_shape = sig.data.shape
        print(f"Dataset shape: {data_shape}")
        return sig, sig.data, data_shape
        
    elif file_type == "zarr":
        if not HAS_ZARR:
            raise ImportError("Zarr required for .zarr files. Install with: pip install zarr")
        
        print(f"Loading zarr file lazily...")
        z = zarr.open(str(file_path), mode="r")
        data_shape = z.shape
        print(f"Dataset shape: {data_shape}")
        
        # Load metadata if available
        metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
        metadata = None
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            print(f"Loaded metadata: {metadata}")
        
        return z, z, data_shape, metadata
    else:
        raise ValueError(f"Unsupported file type: {file_type}")

def compute_normalization_stats(data_source, data_shape, chunk_size: int = 64, metadata=None):
    """Compute normalization statistics in a streaming fashion."""
    if len(data_shape) == 4:
        ny, nx, qy, qx = data_shape
        total_patterns = ny * nx
        is_4d = True
    else:
        # Already reshaped zarr data (patterns, height, width)
        total_patterns, qy, qx = data_shape
        is_4d = False
    
    # First pass: compute mean and std for log-scaled data
    count = 0
    mean_sum = 0.0
    var_sum = 0.0
    
    if is_4d:
        with tqdm(total=ny, desc="Computing statistics", unit="row") as pbar:
            for i in range(0, ny, chunk_size):
                end_i = min(i + chunk_size, ny)
                
                # Load chunk
                if hasattr(data_source, 'compute'):  # Dask array
                    chunk_data = data_source[i:end_i].compute()
                else:  # Regular array or h5py dataset
                    chunk_data = data_source[i:end_i]
                
                # Convert to float32 and apply log scaling
                chunk_data = chunk_data.astype("float32")
                chunk_data = np.log(chunk_data + 1)
                
                # Update running statistics
                chunk_flat = chunk_data.reshape(-1)
                chunk_count = len(chunk_flat)
                chunk_mean = np.mean(chunk_flat)
                chunk_var = np.var(chunk_flat)
                
                # Welford's online algorithm for numerical stability
                delta = chunk_mean - mean_sum / max(count, 1)
                mean_sum += chunk_count * chunk_mean
                count += chunk_count
                mean = mean_sum / count
                var_sum += chunk_count * chunk_var + chunk_count * delta * delta * (count - chunk_count) / count
                
                pbar.update(end_i - i)
                
                # Clean up
                del chunk_data, chunk_flat
                gc.collect()
    else:
        # Process zarr data pattern by pattern
        with tqdm(total=total_patterns, desc="Computing statistics", unit="pattern") as pbar:
            for i in range(0, total_patterns, chunk_size):
                end_i = min(i + chunk_size, total_patterns)
                
                # Load chunk
                chunk_data = np.array(data_source[i:end_i])
                
                # Handle zarr dequantization if needed
                if metadata and metadata.get("dtype") == "uint16":
                    data_min = metadata["data_min"]
                    data_range = metadata["data_range"]
                    chunk_data = chunk_data.astype("float32") / 65535.0 * data_range + data_min
                
                # Convert to float32 and apply log scaling
                chunk_data = chunk_data.astype("float32")
                chunk_data = np.log(chunk_data + 1)
                
                # Update running statistics
                chunk_flat = chunk_data.reshape(-1)
                chunk_count = len(chunk_flat)
                chunk_mean = np.mean(chunk_flat)
                chunk_var = np.var(chunk_flat)
                
                # Welford's online algorithm for numerical stability
                delta = chunk_mean - mean_sum / max(count, 1)
                mean_sum += chunk_count * chunk_mean
                count += chunk_count
                mean = mean_sum / count
                var_sum += chunk_count * chunk_var + chunk_count * delta * delta * (count - chunk_count) / count
                
                pbar.update(end_i - i)
                
                # Clean up
                del chunk_data, chunk_flat
                gc.collect()
    
    std = np.sqrt(var_sum / count)
    return mean, std

def process_data_chunked(data_source, data_shape, mean: float, std: float, 
                        downsample: int, mode: str, sigma: float, scan_step: int, 
                        chunk_size: int = 64, metadata=None):
    """Process data in chunks: normalize, subsample, and downsample."""
    if len(data_shape) == 4:
        ny, nx, qy, qx = data_shape
        is_4d = True
        
        # Apply scan step subsampling to dimensions
        if scan_step > 1:
            ny, nx = ny // scan_step, nx // scan_step
    else:
        # Already reshaped zarr data
        total_patterns, qy, qx = data_shape
        is_4d = False
        ny, nx = 1, total_patterns  # Treat as single row for processing
    
    # Determine final pattern size after downsampling
    if downsample > 1:
        final_qy, final_qx = qy // downsample, qx // downsample
    else:
        final_qy, final_qx = qy, qx
    
    all_chunks = []
    
    if is_4d:
        with tqdm(total=ny, desc="Processing data", unit="row") as pbar:
            for i in range(0, ny, chunk_size):
                end_i = min(i + chunk_size, ny)
                
                # Load chunk with scan step subsampling
                if scan_step > 1:
                    actual_i = i * scan_step
                    actual_end_i = min(end_i * scan_step, data_shape[0])
                    if hasattr(data_source, 'compute'):  # Dask array
                        chunk_data = data_source[actual_i:actual_end_i:scan_step, ::scan_step].compute()
                    else:  # Regular array or h5py dataset
                        chunk_data = data_source[actual_i:actual_end_i:scan_step, ::scan_step]
                else:
                    if hasattr(data_source, 'compute'):  # Dask array
                        chunk_data = data_source[i:end_i].compute()
                    else:  # Regular array or h5py dataset
                        chunk_data = data_source[i:end_i]
                
                # Apply normalization
                chunk_data = chunk_data.astype("float32")
                chunk_data = np.log(chunk_data + 1)
                chunk_data = (chunk_data - mean) / (std + 1e-8)
                
                # Apply downsampling if needed
                if downsample > 1:
                    if mode == "stride":
                        chunk_data = chunk_data[..., ::downsample, ::downsample]
                    elif mode == "bin":
                        chunk_data = block_bin_mean(chunk_data, downsample)
                    elif mode == "gauss":
                        chunk_data = gaussian_downsample(chunk_data, downsample, sigma)
                    elif mode == "fft":
                        chunk_data = fft_crop(chunk_data, downsample)
                
                # Reshape chunk to samples format
                chunk_ny, chunk_nx = chunk_data.shape[:2]
                chunk_data = chunk_data.reshape(chunk_ny * chunk_nx, 1, final_qy, final_qx)
                
                all_chunks.append(chunk_data)
                pbar.update(end_i - i)
                
                # Clean up
                del chunk_data
                gc.collect()
    else:
        # Process zarr data
        with tqdm(total=total_patterns, desc="Processing data", unit="pattern") as pbar:
            for i in range(0, total_patterns, chunk_size):
                end_i = min(i + chunk_size, total_patterns)
                
                # Load chunk
                chunk_data = np.array(data_source[i:end_i])
                
                # Handle zarr dequantization if needed
                if metadata and metadata.get("dtype") == "uint16":
                    data_min = metadata["data_min"]
                    data_range = metadata["data_range"]
                    chunk_data = chunk_data.astype("float32") / 65535.0 * data_range + data_min
                
                # Apply normalization
                chunk_data = chunk_data.astype("float32")
                chunk_data = np.log(chunk_data + 1)
                chunk_data = (chunk_data - mean) / (std + 1e-8)
                
                # Apply downsampling if needed
                if downsample > 1:
                    if mode == "stride":
                        chunk_data = chunk_data[..., ::downsample, ::downsample]
                    elif mode == "bin":
                        chunk_data = block_bin_mean(chunk_data, downsample)
                    elif mode == "gauss":
                        chunk_data = gaussian_downsample(chunk_data, downsample, sigma)
                    elif mode == "fft":
                        chunk_data = fft_crop(chunk_data, downsample)
                
                # Add channel dimension
                if chunk_data.ndim == 3:
                    chunk_data = chunk_data[:, np.newaxis, :, :]
                
                all_chunks.append(chunk_data)
                pbar.update(end_i - i)
                
                # Clean up
                del chunk_data
                gc.collect()
    
    return np.concatenate(all_chunks, axis=0)

def normalise(x: np.ndarray) -> np.ndarray:
    """Z-score normalization with log scaling like m3_learning"""
    print("Normalizing data...")
    x = x.astype("float32")
    x = np.log(x + 1)  # Log scaling with +1 to avoid log(0) like m3_learning
    # Apply z-score normalization (mean=0, std=1)
    x_flat = x.reshape(-1)
    mean = np.mean(x_flat)
    std = np.std(x_flat)
    x = (x - mean) / (std + 1e-8)  # Small epsilon to avoid division by zero
    return x

def block_bin_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Reduce spatial resolution by k*k mean pooling."""
    *lead, qy, qx = arr.shape
    qy2, qx2 = (qy // k) * k, (qx // k) * k  # drop edges so divisible by k
    trimmed = arr[..., :qy2, :qx2]
    newshape = (*lead, qy2 // k, k, qx2 // k, k)
    return trimmed.reshape(newshape).mean(axis=(-1, -3))

def gaussian_downsample(arr: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Gaussian low-pass filter then stride-based down-sample."""
    blurred = gaussian_filter(arr, sigma=(0, 0, sigma, sigma), mode="reflect")
    return blurred[..., ::k, ::k]

def fft_crop(arr: np.ndarray, k: int) -> np.ndarray:
    """Down-sample via Fourier cropping (slow, high fidelity)."""
    *lead, qy, qx = arr.shape
    fq = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    cy, cx = qy // 2, qx // 2
    ny, nx = qy // k, qx // k
    fq_crop = fq[..., cy - ny // 2: cy + (ny + 1) // 2,
                  cx - nx // 2: cx + (nx + 1) // 2]
    img = np.fft.ifft2(np.fft.ifftshift(fq_crop, axes=(-2, -1)),
                       s=(ny, nx), axes=(-2, -1)).real
    return img.astype(arr.dtype)

def downsample_patterns(data: np.ndarray, k: int, mode: str, sigma: float) -> np.ndarray:
    """Apply downsampling to diffraction patterns."""
    if k <= 1:
        return data
    print(f"Downsampling patterns using {mode} method...")
    if mode == "stride":
        return data[..., ::k, ::k]
    if mode == "bin":
        return block_bin_mean(data, k)
    if mode == "gauss":
        return gaussian_downsample(data, k, sigma)
    if mode == "fft":
        return fft_crop(data, k)
    raise ValueError(f"Unknown down-sampling mode: {mode}")

def detect_file_type(file_path: Path) -> str:
    """Detect file type based on extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".dm4", ".dm3"]:
        return "dm4"
    elif suffix in [".hdf5", ".h5"]:
        return "hdf5"
    elif suffix == ".hspy":
        return "hspy"
    elif suffix == ".zarr" or file_path.is_dir():
        return "zarr"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def load_data(file_path: Path) -> np.ndarray:
    """Load data based on file type."""
    file_type = detect_file_type(file_path)
    print(f"Loading {file_type} file: {file_path}")
    
    if file_type == "hdf5":
        with h5py.File(file_path, "r") as f:
            data = f["/data"][:]  # shape = (Ny, Nx, Qy, Qx)
    elif file_type in ["dm4", "hspy"]:
        if not HAS_HYPERSPY:
            raise ImportError("HyperSpy required for dm4/hspy files. Install with: pip install hyperspy")
        sig = hs.load(file_path.as_posix(), lazy=False)
        data = sig.data  # shape = (Ny, Nx, Qy, Qx)
    elif file_type == "zarr":
        if not HAS_ZARR:
            raise ImportError("Zarr required for .zarr files. Install with: pip install zarr")
        z = zarr.open(str(file_path), mode="r")
        data = np.array(z)  # Load all data
        
        # Handle zarr dequantization if needed
        metadata_path = file_path.parent / f"{file_path.stem}_metadata.json"
        if metadata_path.exists():
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            if metadata.get("dtype") == "uint16":
                data_min = metadata["data_min"]
                data_range = metadata["data_range"]
                data = data.astype("float32") / 65535.0 * data_range + data_min
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return data

def main():
    p = argparse.ArgumentParser(description="Preprocess 4D-STEM files including .zarr (data processing only)")
    p.add_argument("--input", type=Path, required=True, help="Input file (.dm4, .hdf5, .hspy, .zarr)")
    p.add_argument("--downsample", type=int, default=1, metavar="k",
                   help="Factor k; diffraction pattern size becomes Q/k * Q/k")
    p.add_argument("--mode", choices=["bin", "stride", "gauss", "fft"], default="bin",
                   help="Down-sampling strategy: bin (mean pooling), stride (pixel skip), gauss (Gaussian+stride), fft (Fourier crop)")
    p.add_argument("--sigma", type=float, default=0.8, metavar="sigma",
                   help="Gaussian sigma (pixels) if --mode gauss (default 0.8)")
    p.add_argument("--scan_step", type=int, default=1, metavar="n",
                   help="Take every n-th probe position along both scan axes")
    p.add_argument("--chunk_size", type=int, default=64, help="Chunk size for memory-efficient processing")
    p.add_argument("--memory_efficient", action="store_true", help="Enable memory-efficient processing for large files")
    args = p.parse_args()

    print("\n" + "="*60)
    print("4D-STEM PREPROCESSING PIPELINE (DATA PROCESSING ONLY)")
    print("="*60)
    
    # Check file size and enable memory-efficient processing if needed
    if args.input.is_dir():
        # For zarr directories, sum all file sizes
        file_size_gb = sum(f.stat().st_size for f in args.input.rglob('*') if f.is_file()) / (1024 ** 3)
    else:
        file_size_gb = args.input.stat().st_size / (1024 ** 3)
    
    print(f"Input file size: {file_size_gb:.2f} GB")
    
    # Enable memory-efficient processing for large files or if explicitly requested
    if file_size_gb > 2.0 or args.memory_efficient:
        print("🔧 Using memory-efficient processing for large file")
        print(f"📊 Initial memory usage: {get_memory_usage():.2f} GB")
        
        # Load data lazily
        load_result = load_data_lazy(args.input)
        if len(load_result) == 4:
            file_handle, data_source, data_shape, metadata = load_result
        else:
            file_handle, data_source, data_shape = load_result
            metadata = None
            
        print(f"✓ Loaded data shape: {data_shape} from {detect_file_type(args.input)} file (lazy)")
        
        # Compute normalization statistics first
        print("Computing normalization statistics...")
        mean, std = compute_normalization_stats(data_source, data_shape, args.chunk_size, metadata)
        print(f"✓ Computed statistics (mean={mean:.4f}, std={std:.4f})")
        
        # Process data in chunks
        print("Processing data in chunks...")
        processed_data = process_data_chunked(
            data_source, data_shape, mean, std, 
            args.downsample, args.mode, args.sigma, args.scan_step, 
            args.chunk_size, metadata
        )
        
        print(f"✓ Processed data shape: {processed_data.shape}")
        print(f"📊 Peak memory usage: {get_memory_usage():.2f} GB")
        
        # Close file handle
        if hasattr(file_handle, 'close'):
            file_handle.close()
        
    else:
        print("🔧 Using standard processing for small file")
        print(f"📊 Initial memory usage: {get_memory_usage():.2f} GB")
        
        # Load data with automatic file type detection
        data = load_data(args.input)
        print(f"✓ Loaded data shape: {data.shape} from {detect_file_type(args.input)} file")

        # Handle zarr data that's already in (patterns, height, width) format
        if len(data.shape) == 3:
            # Already in pattern format - add fake spatial dimensions for processing
            total_patterns, qy, qx = data.shape
            # Reshape to (1, total_patterns, qy, qx) for processing
            data = data.reshape(1, total_patterns, qy, qx)
            print(f"✓ Reshaped zarr data to 4D format: {data.shape}")

        # Optional scan grid thinning
        if args.scan_step > 1:
            data = data[::args.scan_step, ::args.scan_step, ...]
            print(f"✓ Scan grid subsampled by {args.scan_step} → new grid {data.shape[:2]}")

        # Pattern down-sampling
        if args.downsample > 1:
            data = downsample_patterns(data, args.downsample, args.mode, args.sigma)
            print(f"✓ Downsampled patterns by {args.downsample}x using {args.mode} method")

        # Intensity normalisation with log scaling
        data = normalise(data)
        print(f"✓ Normalized data (mean=0, std=1)")

        # Reshape to samples × 1 × Qy × Qx
        ny, nx, qy, qx = data.shape
        processed_data = data.reshape(ny*nx, 1, qy, qx)
        print(f"✓ Reshaped to PyTorch format: {processed_data.shape}")
        
        print(f"📊 Peak memory usage: {get_memory_usage():.2f} GB")

    # Print final statistics
    print(f"\n📊 FINAL PROCESSED DATA STATISTICS:")
    print(f"   Shape: {processed_data.shape}")
    print(f"   Data type: {processed_data.dtype}")
    print(f"   Mean: {processed_data.mean():.6f}")
    print(f"   Std: {processed_data.std():.6f}")
    print(f"   Min: {processed_data.min():.6f}")
    print(f"   Max: {processed_data.max():.6f}")
    print(f"   Memory usage: {processed_data.nbytes / (1024**3):.2f} GB")
    
    # Clean up
    del processed_data
    gc.collect()
    
    print(f"📊 Final memory usage: {get_memory_usage():.2f} GB")
    
    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()