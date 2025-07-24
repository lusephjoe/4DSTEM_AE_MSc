"""Convert Gatan .dm4 4D-STEM stacks to the PyTorch tensor format used by `train.py`.

Down-sampling strategies implemented (choose with `--mode`):

* **stride**  - plain pixel skipping (fastest, strong aliasing)
* **bin**     - k*k mean-pooling (good compromise, default)
* **gauss**   - Gaussian low-pass filter followed by stride (anti-alias)
* **fft**     - Fourier cropping (gold-standard frequency fidelity, slow)

Extra features
--------------
* Optional **scan grid sub-sampling** (take every *n*-th probe position) to thin very dense scans.
* All heavy lifting done with NumPy / SciPy - no GPU required, but script plays nicely inside
  a CUDA environment because the tensors are saved in CPU-compatible `.pt` format.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import os

import numpy as np
import torch
import hyperspy.api as hs  # pip install hyperspy
from scipy.ndimage import gaussian_filter
import gc
import dask.array as da
import json
from tqdm import tqdm
import h5py

# ─────────────────────────── helpers ────────────────────────────

def normalise(x: np.ndarray) -> np.ndarray:
    """Scale array to [0,1] float32 with log scaling"""
    x = x.astype("float32", copy=False)
    x = np.log(x + 1e-6)  # Log scaling with small epsilon to avoid log(0)
    x -= x.min()
    x /= x.max() + 1e-6
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
    if k <= 1:
        return data
    if mode == "stride":
        return data[..., ::k, ::k]
    if mode == "bin":
        return block_bin_mean(data, k)
    if mode == "gauss":
        return gaussian_downsample(data, k, sigma)
    if mode == "fft":
        return fft_crop(data, k)
    raise ValueError(f"Unknown down-sampling mode: {mode}")


def process_chunk_single_threaded(sig, row_start, row_end, scan_step, downsample, mode, sigma):
    """Process a chunk of rows in single-threaded mode."""
    try:
        # Apply scan step to row indices
        scan_row_start = row_start * scan_step
        scan_row_end = row_end * scan_step
        
        # Load this chunk of rows
        chunk_data = sig.data[scan_row_start:scan_row_end:scan_step, ::scan_step].compute()
        
        # Apply downsampling
        if downsample > 1:
            chunk_data = downsample_patterns(chunk_data, downsample, mode, sigma)
        
        # Apply normalization
        chunk_data = normalise(chunk_data)
        
        # Reshape to (patterns, qy, qx)
        chunk_rows, chunk_cols, qy_processed, qx_processed = chunk_data.shape
        chunk_data = chunk_data.reshape(chunk_rows * chunk_cols, qy_processed, qx_processed)
        
        # Force garbage collection to free memory
        gc.collect()
        
        return chunk_data
        
    except Exception as e:
        print(f"Error processing chunk rows {row_start}-{row_end}: {e}")
        return None


# ───────────────────────────── CLI ─────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Convert .dm4 4D-STEM file → Zarr compressed tensor")
    p.add_argument("--input", type=Path, required=True, help="Input .dm4 file")
    p.add_argument("--output", type=Path, required=True, help="Output .zarr file")
    p.add_argument("--downsample", type=int, default=1, metavar="k",
                   help="Factor k; diffraction pattern size becomes Q/k * Q/k")
    p.add_argument("--mode", choices=["bin", "stride", "gauss", "fft"], default="bin",
                   help="Down-sampling strategy - see script docstring for details")
    p.add_argument("--sigma", type=float, default=0.8, metavar="sigma",
                   help="Gaussian sigma (pixels) if --mode gauss (default 0.8)")
    p.add_argument("--scan_step", type=int, default=1, metavar="n",
                   help="Take every n-th probe position along both scan axes")
    p.add_argument("--chunk_size", type=int, default=128, metavar="n",
                   help="Chunk size for zarr storage (patterns per chunk)")
    p.add_argument("--dtype", choices=["uint16", "float16", "float32"], default="float16",
                   help="Output data type for compression")
    p.add_argument("--compression_level", type=int, default=4, metavar="n",
                   help="Compression level (1-9, higher = more compression)")
    args = p.parse_args()
    
    print("Using HDF5-based streaming compression for memory efficiency")

    # Load .dm4 with lazy loading
    print(f"Loading {args.input} (lazy mode)...")
    sig = hs.load(args.input.as_posix(), lazy=True)
    original_shape = sig.data.shape
    print(f"Original shape: {original_shape}")
    
    # Apply scan step to get final shape
    if args.scan_step > 1:
        ny, nx = original_shape[0] // args.scan_step, original_shape[1] // args.scan_step
        print(f"Scan grid subsampled by {args.scan_step} → new grid ({ny}, {nx})")
    else:
        ny, nx = original_shape[:2]
    
    qy, qx = original_shape[2], original_shape[3]
    
    # Calculate final pattern size after downsampling
    if args.downsample > 1:
        qy_final = qy // args.downsample
        qx_final = qx // args.downsample
        print(f"Patterns will be downsampled from {qy}x{qx} to {qy_final}x{qx_final}")
    else:
        qy_final, qx_final = qy, qx
    
    total_patterns = ny * nx
    print(f"Total patterns: {total_patterns}")
    print(f"Final pattern size: {qy_final}x{qx_final}")
    
    # Create dask array from lazy hyperspy signal
    print("Creating dask array...")
    raw_data = sig.data
    
    # Apply scan step and reshaping
    if args.scan_step > 1:
        raw_data = raw_data[::args.scan_step, ::args.scan_step]
    
    # Rechunk the existing dask array to smaller chunks to avoid 2GB limit
    # Calculate chunk size to stay under 2GB for float16 data
    max_chunk_size_gb = 1.5  # Use 1.5GB as safety margin
    bytes_per_pattern = qy * qx * 2  # float16 = 2 bytes
    max_patterns_per_chunk = int(max_chunk_size_gb * 1024**3 / bytes_per_pattern)
    safe_chunk_size = min(args.chunk_size, max_patterns_per_chunk)
    
    print(f"Using chunk size: {safe_chunk_size} patterns (max {max_patterns_per_chunk} for 2GB limit)")
    
    # Rechunk to smaller chunks
    dask_data = da.rechunk(raw_data, chunks=(safe_chunk_size, safe_chunk_size, qy, qx))
    
    # Reshape to (patterns, height, width) format
    dask_data = dask_data.reshape(total_patterns, qy, qx)
    
    # Rechunk again after reshape to ensure proper chunking
    dask_data = da.rechunk(dask_data, chunks=(safe_chunk_size, qy, qx))
    
    # Apply downsampling if needed
    if args.downsample > 1:
        print(f"Applying downsampling with mode: {args.mode}")
        # Process downsampling in chunks to manage memory
        def downsample_chunk(chunk):
            return downsample_patterns(chunk, args.downsample, args.mode, args.sigma)
        
        # Calculate new chunk size after downsampling
        new_safe_chunk_size = safe_chunk_size  # Keep same number of patterns
        
        dask_data = dask_data.map_blocks(downsample_chunk, 
                                        dtype=dask_data.dtype,
                                        drop_axis=None,
                                        new_axis=None,
                                        chunks=(new_safe_chunk_size, qy_final, qx_final))
    
    # Skip min/max computation - handle precision later
    print(f"Converting to {args.dtype} (skipping min/max computation)...")
    
    # Use default range for metadata
    data_min, data_max = 0.0, 1.0
    data_range = data_max - data_min
    
    with tqdm(total=1, desc=f"Converting to {args.dtype}", unit="operation") as pbar:
        if args.dtype == "uint16":
            # Convert to uint16 without scaling (assumes data already in 0-1 range)
            processed_data = (dask_data * 65535).astype("uint16")
        elif args.dtype == "float16":
            # Convert to float16 without normalization
            processed_data = dask_data.astype("float16")
        else:  # float32
            # Keep as float32 without normalization
            processed_data = dask_data.astype("float32")
        pbar.update(1)
    
    # Save to HDF5
    print(f"Saving to HDF5 format: {args.output}")
    print(f"Compression: gzip level {args.compression_level}")
    
    # Quick validation before conversion
    print(f"Pre-flight check:")
    print(f"  Data shape: {processed_data.shape}")  
    print(f"  Data chunks: {processed_data.chunks}")
    print(f"  Final chunk size: {processed_data.chunks[0][-1]} patterns")
    
    # Test a small slice to catch errors early
    try:
        test_slice = processed_data[:2].compute()
        print(f"  Test slice successful: {test_slice.shape}, dtype: {test_slice.dtype}")
    except Exception as e:
        print(f"ERROR: Test slice failed: {e}")
        return
    
    # Save to HDF5 with compression
    print("Saving data to HDF5...")
    print("Note: This operation may take several minutes without progress updates")
    
    # Change output extension to .h5
    h5_output = str(args.output).replace('.zarr', '.h5')
    
    # Remove existing file if it exists
    if Path(h5_output).exists():
        print(f"Removing existing file at {h5_output}")
        Path(h5_output).unlink()
    
    print("Converting dask array to numpy (this may take time)...")
    numpy_data = processed_data.compute()
    print(f"Converted to numpy: {numpy_data.shape}, dtype: {numpy_data.dtype}")
    
    # Save to HDF5 with compression
    print("Writing to HDF5 with compression...")
    with h5py.File(h5_output, 'w') as f:
        # Create compressed dataset
        dset = f.create_dataset('patterns', 
                               data=numpy_data,
                               compression='gzip', 
                               compression_opts=args.compression_level,
                               chunks=True,
                               shuffle=True)
        
        # Store metadata as attributes
        dset.attrs['original_shape'] = original_shape
        dset.attrs['final_shape'] = numpy_data.shape
        dset.attrs['data_min'] = float(data_min)
        dset.attrs['data_max'] = float(data_max)
        dset.attrs['data_range'] = float(data_range)
        dset.attrs['dtype'] = args.dtype
        dset.attrs['downsample'] = args.downsample
        dset.attrs['scan_step'] = args.scan_step
        dset.attrs['mode'] = args.mode
        dset.attrs['sigma'] = args.sigma
    
    print(f"HDF5 file saved successfully: {h5_output}")
    
    # Update args.output for metadata saving
    args.output = Path(h5_output)
    
    # Save metadata for reconstruction
    metadata = {
        "original_shape": original_shape,
        "final_shape": (total_patterns, qy_final, qx_final),
        "data_min": float(data_min),
        "data_max": float(data_max),
        "data_range": float(data_range),
        "dtype": args.dtype,
        "downsample": args.downsample,
        "scan_step": args.scan_step,
        "mode": args.mode,
        "sigma": args.sigma
    }
    
    metadata_path = args.output.parent / f"{args.output.stem}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"Metadata saved to: {metadata_path}")
    
    # Check file size
    if args.output.exists():
        # For HDF5 files, get the actual file size directly
        file_size_bytes = args.output.stat().st_size
        file_size_gb = file_size_bytes / 1024**3
        print(f"Output file size: {file_size_gb:.2f} GB")
        
        # Calculate compression ratio (avoid division by zero)
        if file_size_gb > 0:
            original_size_gb = total_patterns * qy_final * qx_final * 4 / 1024**3  # float32 size
            compression_ratio = original_size_gb / file_size_gb
            print(f"Compression ratio: {compression_ratio:.1f}x (from {original_size_gb:.2f} GB to {file_size_gb:.2f} GB)")
        else:
            print("Warning: Could not determine file size for compression ratio calculation")
    
    print(f"Saved {total_patterns} patterns of size {qy_final}x{qx_final} → {args.output}")

if __name__ == "__main__":
    main()