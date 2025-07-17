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

import numpy as np
import torch
import hyperspy.api as hs  # pip install hyperspy
from scipy.ndimage import gaussian_filter

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
    
    if estimated_memory_gb > args.max_memory_gb:
        print(f"Large file detected ({estimated_memory_gb:.2f} GB > {args.max_memory_gb:.2f} GB)")
        print("Processing in chunks to reduce RAM usage...")
        
        # Calculate chunk size based on memory limit
        chunk_size = max(1, int(args.max_memory_gb * 1024 / pattern_size_mb))
        chunk_size = min(chunk_size, total_patterns)  # Don't exceed total patterns
        
        print(f"Processing {chunk_size} patterns at a time...")
        
        # Process in chunks by loading rectangular regions
        all_tensors = []
        
        # Calculate how many scan positions we can fit in a chunk
        patterns_per_row = nx
        rows_per_chunk = max(1, chunk_size // patterns_per_row)
        
        print(f"Processing {rows_per_chunk} rows at a time...")
        
        for row_start in range(0, ny, rows_per_chunk):
            row_end = min(row_start + rows_per_chunk, ny)
            patterns_in_chunk = (row_end - row_start) * nx
            
            print(f"Processing rows {row_start} to {row_end-1} ({patterns_in_chunk} patterns)...")
            
            # Load rectangular chunk of data
            scan_row_start = row_start * args.scan_step
            scan_row_end = row_end * args.scan_step
            
            # Load chunk as rectangular region
            chunk_data = sig.data[scan_row_start:scan_row_end:args.scan_step, ::args.scan_step].compute()
            
            # Reshape to (patterns, qy, qx)
            chunk_rows, chunk_cols, qy, qx = chunk_data.shape
            chunk_data = chunk_data.reshape(chunk_rows * chunk_cols, qy, qx)
            
            # Apply downsampling to chunk
            if args.downsample > 1:
                # Reshape to match downsample_patterns expected input (ny, nx, qy, qx)
                chunk_data = chunk_data.reshape(chunk_rows, chunk_cols, qy, qx)
                chunk_data = downsample_patterns(chunk_data, args.downsample, args.mode, args.sigma)
                # Reshape back to (patterns, qy, qx)
                chunk_data = chunk_data.reshape(chunk_rows * chunk_cols, chunk_data.shape[2], chunk_data.shape[3])
            
            # Apply normalization to chunk
            chunk_data = normalise(chunk_data)
            
            # Convert to tensor with channel dimension
            chunk_tensor = torch.from_numpy(chunk_data).unsqueeze(1)  # Shape: (chunk_size, 1, qy, qx)
            all_tensors.append(chunk_tensor)
            
            # Clear memory
            del chunk_data, chunk_tensor
        
        # Concatenate all chunks
        print("Concatenating all chunks...")
        tensor = torch.cat(all_tensors, dim=0)
        del all_tensors
        
    else:
        # Small enough to process all at once (original behavior)
        print("Processing all data at once...")
        data = sig.data.compute()
        
        # Apply scan step
        if args.scan_step > 1:
            data = data[::args.scan_step, ::args.scan_step, ...]
            print(f"Scan grid subsampled by {args.scan_step} → new grid {data.shape[:2]}")
        
        # Apply downsampling
        data = downsample_patterns(data, args.downsample, args.mode, args.sigma)
        
        # Apply normalization
        data = normalise(data)
        
        # Convert to tensor
        ny, nx, qy, qx = data.shape
        tensor = torch.from_numpy(data.reshape(ny * nx, 1, qy, qx))
    
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