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
    p = argparse.ArgumentParser(description="Convert .dm4 4D-STEM file → PyTorch tensor")
    p.add_argument("--input", type=Path, required=True, help="Input .dm4 file")
    p.add_argument("--output", type=Path, required=True, help="Output .pt tensor file")
    p.add_argument("--downsample", type=int, default=1, metavar="k",
                   help="Factor k; diffraction pattern size becomes Q/k * Q/k")
    p.add_argument("--mode", choices=["bin", "stride", "gauss", "fft"], default="bin",
                   help="Down-sampling strategy - see script docstring for details")
    p.add_argument("--sigma", type=float, default=0.8, metavar="sigma",
                   help="Gaussian sigma (pixels) if --mode gauss (default 0.8)")
    p.add_argument("--scan_step", type=int, default=1, metavar="n",
                   help="Take every n-th probe position along both scan axes")
    p.add_argument("--max_memory_gb", type=float, default=8.0,
                   help="Maximum memory to use during conversion (GB)")
    p.add_argument("--compress", action="store_true",
                   help="Enable compression when saving (reduces file size)")
    args = p.parse_args()
    
    print("Using single-threaded processing for memory efficiency")

    # Load .dm4 with lazy loading to check size first
    print(f"Loading {args.input} (lazy mode to check size)...")
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
    
    # Estimate memory requirements
    total_patterns = ny * nx
    pattern_size_mb = (qy_final * qx_final * 4) / (1024**2)  # float32 size
    estimated_memory_gb = (total_patterns * pattern_size_mb) / 1024
    
    print(f"Total patterns: {total_patterns}")
    print(f"Final pattern size: {qy_final}x{qx_final}")
    print(f"Estimated memory needed: {estimated_memory_gb:.2f} GB")
    
    # Use single-threaded chunked processing for memory efficiency
    print(f"Large file detected ({estimated_memory_gb:.2f} GB estimated memory)")
    print("Processing with single-threaded chunked approach to stay within memory limits...")
    
    # Calculate chunk size based on available memory
    # Use conservative estimate: process chunks that fit in available memory
    memory_safety_factor = 0.3  # Use 30% of available memory as safety margin
    available_memory_gb = args.max_memory_gb * memory_safety_factor
    
    patterns_per_row = nx
    # Account for processing overhead (normalization, downsampling need extra memory)
    processing_overhead = 4.0  # 4x memory overhead for processing
    effective_pattern_size_mb = pattern_size_mb * processing_overhead
    
    rows_per_chunk = max(1, int(available_memory_gb * 1024 / (patterns_per_row * effective_pattern_size_mb)))
    rows_per_chunk = min(rows_per_chunk, ny)
    
    # Ensure we don't create chunks that are too small (minimum 1 row)
    if rows_per_chunk < 1:
        rows_per_chunk = 1
        print(f"WARNING: Chunk size is very small. Consider increasing --max_memory_gb or using --downsample")
    
    print(f"Processing {rows_per_chunk} rows per chunk (using {available_memory_gb:.1f} GB safely)...")
    
    # Create work chunks
    work_chunks = []
    for row_start in range(0, ny, rows_per_chunk):
        row_end = min(row_start + rows_per_chunk, ny)
        work_chunks.append((row_start, row_end))
    
    print(f"Created {len(work_chunks)} work chunks")
    
    # Process chunks sequentially
    all_tensors = []
    
    for i, (row_start, row_end) in enumerate(work_chunks):
        print(f"Processing chunk {i+1}/{len(work_chunks)} (rows {row_start}-{row_end-1})...")
        
        chunk_data = process_chunk_single_threaded(
            sig, row_start, row_end, args.scan_step, args.downsample, args.mode, args.sigma
        )
        
        if chunk_data is not None:
            chunk_tensor = torch.from_numpy(chunk_data).unsqueeze(1)
            all_tensors.append(chunk_tensor)
            del chunk_data, chunk_tensor
            gc.collect()  # Force garbage collection after each chunk
    
    # Concatenate all chunks
    print("Concatenating all chunks...")
    tensor = torch.cat(all_tensors, dim=0)
    del all_tensors
    
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Memory usage: {tensor.element_size() * tensor.nelement() / 1024**3:.2f} GB")
    
    # Save tensor
    print(f"Saving to {args.output}...")
    if args.compress:
        # Use new ZIP-based serialization for compression
        torch.save(tensor, args.output, _use_new_zipfile_serialization=True)
        print("Saved with compression enabled")
    else:
        torch.save(tensor, args.output)
        print("Saved without compression")
    
    # Check file size
    file_size_gb = args.output.stat().st_size / 1024**3
    print(f"Output file size: {file_size_gb:.2f} GB")
    print(f"Saved {tensor.shape[0]} patterns of size {tensor.shape[2]}x{tensor.shape[3]} → {args.output}")

if __name__ == "__main__":
    main()