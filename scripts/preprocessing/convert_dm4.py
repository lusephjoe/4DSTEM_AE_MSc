#!/usr/bin/env python3
"""Convert Gatan .dm4 4D-STEM stacks to HDF5 format for training.

Down-sampling strategies implemented (choose with `--mode`):

* **stride**  - plain pixel skipping (fastest, strong aliasing)
* **bin**     - k*k mean-pooling (good compromise, default)
* **gauss**   - Gaussian low-pass filter followed by stride (anti-alias)
* **fft**     - Fourier cropping (gold-standard frequency fidelity, slow)

Features:
- Optional scan grid sub-sampling (take every n-th probe position)
- Memory-efficient streaming processing with HDF5 compression
- Preserves raw intensity values (normalization handled by training)
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Tuple, Optional
import gc

import numpy as np
import torch
import hyperspy.api as hs
import h5py
import dask.array as da
from scipy.ndimage import gaussian_filter
from tqdm import tqdm


# Configuration constants
DEFAULT_CHUNK_SIZE = 128
MAX_MEMORY_GB = 1.5  # Safety margin for chunk size calculation
BYTES_PER_FLOAT16 = 2


class DownsampleStrategy:
    """Encapsulates different downsampling strategies for diffraction patterns."""
    
    @staticmethod
    def stride(data: np.ndarray, factor: int) -> np.ndarray:
        """Plain pixel skipping (fastest, strong aliasing)."""
        return data[..., ::factor, ::factor]
    
    @staticmethod
    def bin_mean(data: np.ndarray, factor: int) -> np.ndarray:
        """k*k mean pooling (good compromise)."""
        *lead, qy, qx = data.shape
        qy2, qx2 = (qy // factor) * factor, (qx // factor) * factor
        trimmed = data[..., :qy2, :qx2]
        newshape = (*lead, qy2 // factor, factor, qx2 // factor, factor)
        result = trimmed.reshape(newshape).mean(axis=(-1, -3))
        # Preserve input dtype
        return result.astype(data.dtype)
    
    @staticmethod
    def gaussian(data: np.ndarray, factor: int, sigma: float) -> np.ndarray:
        """Gaussian low-pass filter then stride-based down-sample."""
        # Create sigma tuple matching data dimensions
        if data.ndim == 2:
            sigma_tuple = (sigma, sigma)
        elif data.ndim == 4:
            sigma_tuple = (0, 0, sigma, sigma)  # Don't blur spatial dimensions
        else:
            # For other dimensions, only blur last 2 dimensions
            sigma_tuple = (0,) * (data.ndim - 2) + (sigma, sigma)
        
        blurred = gaussian_filter(data, sigma=sigma_tuple, mode="reflect")
        result = blurred[..., ::factor, ::factor]
        # Preserve input dtype
        return result.astype(data.dtype)
    
    @staticmethod
    def fft_crop(data: np.ndarray, factor: int) -> np.ndarray:
        """Down-sample via Fourier cropping (slow, high fidelity)."""
        *lead, qy, qx = data.shape
        fq = np.fft.fftshift(np.fft.fft2(data, axes=(-2, -1)), axes=(-2, -1))
        cy, cx = qy // 2, qx // 2
        ny, nx = qy // factor, qx // factor
        fq_crop = fq[..., cy - ny // 2: cy + (ny + 1) // 2,
                      cx - nx // 2: cx + (nx + 1) // 2]
        img = np.fft.ifft2(np.fft.ifftshift(fq_crop, axes=(-2, -1)),
                           s=(ny, nx), axes=(-2, -1)).real
        return img.astype(data.dtype)
    
    @classmethod
    def apply(cls, data: np.ndarray, factor: int, mode: str, sigma: float = 0.8) -> np.ndarray:
        """Apply downsampling strategy based on mode."""
        if factor <= 1:
            return data
        
        strategies = {
            "stride": cls.stride,
            "bin": cls.bin_mean,
            "gauss": lambda d, f: cls.gaussian(d, f, sigma),
            "fft": cls.fft_crop
        }
        
        if mode not in strategies:
            raise ValueError(f"Unknown downsampling mode: {mode}. Available: {list(strategies.keys())}")
        
        return strategies[mode](data, factor)


class DM4Converter:
    """Handles conversion of .dm4 files to HDF5 format with configurable processing."""
    
    def __init__(self, input_path: Path, output_path: Path, 
                 downsample: int = 1, mode: str = "bin", sigma: float = 0.8,
                 scan_step: int = 1, chunk_size: int = DEFAULT_CHUNK_SIZE,
                 dtype: str = "float16", compression_level: int = 4):
        """Initialize converter with processing parameters."""
        self.input_path = input_path
        self.output_path = self._ensure_h5_extension(output_path)
        self.downsample = downsample
        self.mode = mode
        self.sigma = sigma
        self.scan_step = scan_step
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.compression_level = compression_level
        
        # Will be set during processing
        self.original_shape: Optional[Tuple[int, ...]] = None
        self.final_shape: Optional[Tuple[int, ...]] = None
        self.data_stats: Optional[dict] = None
    
    @staticmethod
    def _ensure_h5_extension(path: Path) -> Path:
        """Ensure output path has .h5 extension."""
        return path.with_suffix('.h5') if path.suffix != '.h5' else path
    
    def _load_dm4_lazy(self) -> hs.signals.Signal2D:
        """Load .dm4 file with lazy loading for memory efficiency."""
        print(f"Loading {self.input_path} (lazy mode)...")
        signal = hs.load(self.input_path.as_posix(), lazy=True)
        self.original_shape = signal.data.shape
        print(f"Original shape: {self.original_shape}")
        return signal
    
    def _calculate_processing_dimensions(self) -> Tuple[int, int, int, int, int]:
        """Calculate dimensions after scan step and downsampling."""
        ny_orig, nx_orig, qy_orig, qx_orig = self.original_shape
        
        # Apply scan step
        ny = ny_orig // self.scan_step if self.scan_step > 1 else ny_orig
        nx = nx_orig // self.scan_step if self.scan_step > 1 else nx_orig
        
        if self.scan_step > 1:
            print(f"Scan grid subsampled by {self.scan_step} → new grid ({ny}, {nx})")
        
        # Apply downsampling
        qy_final = qy_orig // self.downsample if self.downsample > 1 else qy_orig
        qx_final = qx_orig // self.downsample if self.downsample > 1 else qx_orig
        
        if self.downsample > 1:
            print(f"Patterns downsampled from {qy_orig}x{qx_orig} to {qy_final}x{qx_final}")
        
        total_patterns = ny * nx
        print(f"Total patterns: {total_patterns}")
        print(f"Final pattern size: {qy_final}x{qx_final}")
        
        return ny, nx, qy_final, qx_final, total_patterns
    
    def _calculate_safe_chunk_size(self, qy: int, qx: int) -> int:
        """Calculate chunk size to stay under memory limits."""
        bytes_per_pattern = qy * qx * BYTES_PER_FLOAT16
        max_patterns_per_chunk = int(MAX_MEMORY_GB * 1024**3 / bytes_per_pattern)
        safe_chunk_size = min(self.chunk_size, max_patterns_per_chunk)
        
        print(f"Using chunk size: {safe_chunk_size} patterns "
              f"(max {max_patterns_per_chunk} for {MAX_MEMORY_GB}GB limit)")
        
        return safe_chunk_size
    
    def _prepare_dask_array(self, signal: hs.signals.Signal2D, 
                           ny: int, nx: int, qy: int, qx: int,
                           total_patterns: int) -> da.Array:
        """Prepare and reshape dask array with proper chunking."""
        print("Creating dask array...")
        raw_data = signal.data
        
        # Apply scan step
        if self.scan_step > 1:
            raw_data = raw_data[::self.scan_step, ::self.scan_step]
        
        # Calculate safe chunking
        safe_chunk_size = self._calculate_safe_chunk_size(qy, qx)
        
        # Rechunk and reshape
        dask_data = da.rechunk(raw_data, chunks=(safe_chunk_size, safe_chunk_size, qy, qx))
        dask_data = dask_data.reshape(total_patterns, qy, qx)
        dask_data = da.rechunk(dask_data, chunks=(safe_chunk_size, qy, qx))
        
        return dask_data
    
    def _apply_downsampling(self, dask_data: da.Array, qy_final: int, qx_final: int) -> da.Array:
        """Apply downsampling to dask array if needed."""
        if self.downsample <= 1:
            return dask_data
        
        print(f"Applying downsampling with mode: {self.mode}")
        
        def downsample_chunk(chunk):
            return DownsampleStrategy.apply(chunk, self.downsample, self.mode, self.sigma)
        
        return dask_data.map_blocks(
            downsample_chunk,
            dtype=dask_data.dtype,
            drop_axis=None,
            new_axis=None,
            chunks=(dask_data.chunks[0], qy_final, qx_final)
        )
    
    def _convert_dtype(self, dask_data: da.Array) -> da.Array:
        """Convert data to specified dtype."""
        print(f"Converting to {self.dtype} (preserving raw intensities)...")
        
        with tqdm(total=1, desc=f"Converting to {self.dtype}", unit="operation") as pbar:
            if self.dtype == "uint16":
                processed_data = (dask_data * 65535).astype("uint16")
            elif self.dtype == "float16":
                processed_data = dask_data.astype("float16")
            else:  # float32
                processed_data = dask_data.astype("float32")
            pbar.update(1)
        
        return processed_data
    
    def _validate_data(self, processed_data: da.Array) -> bool:
        """Validate processed data before saving."""
        print("Pre-flight check:")
        print(f"  Data shape: {processed_data.shape}")
        print(f"  Data chunks: {processed_data.chunks}")
        print(f"  Final chunk size: {processed_data.chunks[0][-1]} patterns")
        
        try:
            test_slice = processed_data[:2].compute()
            print(f"  Test slice successful: {test_slice.shape}, dtype: {test_slice.dtype}")
            return True
        except Exception as e:
            print(f"ERROR: Test slice failed: {e}")
            return False
    
    def _save_hdf5(self, processed_data: da.Array) -> dict:
        """Save processed data to HDF5 with compression."""
        print("Converting dask array to numpy (this may take time)...")
        numpy_data = processed_data.compute()
        print(f"Converted to numpy: {numpy_data.shape}, dtype: {numpy_data.dtype}")
        
        # Prepare output
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        if self.output_path.exists():
            print(f"Removing existing file at {self.output_path}")
            self.output_path.unlink()
        
        # Save to HDF5
        print(f"Writing to HDF5 with gzip compression level {self.compression_level}...")
        with h5py.File(self.output_path, 'w') as f:
            dset = f.create_dataset(
                'patterns',
                data=numpy_data,
                compression='gzip',
                compression_opts=self.compression_level,
                chunks=True,
                shuffle=True
            )
            
            # Store metadata as attributes
            data_min = float(numpy_data.min())
            data_max = float(numpy_data.max())
            data_range = data_max - data_min
            
            dset.attrs.update({
                'original_shape': self.original_shape,
                'final_shape': numpy_data.shape,
                'data_min': data_min,
                'data_max': data_max,
                'data_range': data_range,
                'dtype': self.dtype,
                'downsample': self.downsample,
                'scan_step': self.scan_step,
                'mode': self.mode,
                'sigma': self.sigma
            })
        
        self.final_shape = numpy_data.shape
        self.data_stats = {
            'data_min': data_min,
            'data_max': data_max,
            'data_range': data_range
        }
        
        print(f"HDF5 file saved successfully: {self.output_path}")
        return self.data_stats
    
    def _save_metadata(self) -> None:
        """Save processing metadata to JSON file."""
        metadata = {
            "original_shape": self.original_shape,
            "final_shape": self.final_shape,
            "dtype": self.dtype,
            "downsample": self.downsample,
            "scan_step": self.scan_step,
            "mode": self.mode,
            "sigma": self.sigma,
            **self.data_stats
        }
        
        metadata_path = self.output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"Metadata saved to: {metadata_path}")
    
    def _report_compression_stats(self) -> None:
        """Report file size and compression statistics."""
        if not self.output_path.exists():
            print("Warning: Output file not found for compression stats")
            return
        
        file_size_bytes = self.output_path.stat().st_size
        file_size_gb = file_size_bytes / 1024**3
        print(f"Output file size: {file_size_gb:.2f} GB")
        
        if file_size_gb > 0 and self.final_shape:
            # Calculate uncompressed size (as float32)
            total_patterns, qy_final, qx_final = self.final_shape
            original_size_gb = total_patterns * qy_final * qx_final * 4 / 1024**3
            compression_ratio = original_size_gb / file_size_gb
            print(f"Compression ratio: {compression_ratio:.1f}x "
                  f"(from {original_size_gb:.2f} GB to {file_size_gb:.2f} GB)")
    
    def convert(self) -> None:
        """Main conversion pipeline."""
        print("=== DM4 to HDF5 Conversion ===")
        print("Using HDF5-based streaming compression for memory efficiency")
        
        # Load and analyze input
        signal = self._load_dm4_lazy()
        ny, nx, qy_final, qx_final, total_patterns = self._calculate_processing_dimensions()
        
        # Prepare data pipeline
        dask_data = self._prepare_dask_array(signal, ny, nx, 
                                           self.original_shape[2], self.original_shape[3],
                                           total_patterns)
        
        # Apply processing steps
        dask_data = self._apply_downsampling(dask_data, qy_final, qx_final)
        processed_data = self._convert_dtype(dask_data)
        
        # Validate and save
        if not self._validate_data(processed_data):
            print("ERROR: Data validation failed. Aborting conversion.")
            return
        
        self._save_hdf5(processed_data)
        self._save_metadata()
        self._report_compression_stats()
        
        print(f"\n✓ Conversion complete: {total_patterns} patterns of size "
              f"{qy_final}x{qx_final} → {self.output_path}")


def create_argument_parser() -> argparse.ArgumentParser:
    """Create and configure argument parser."""
    parser = argparse.ArgumentParser(
        description="Convert .dm4 4D-STEM file to HDF5 format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Input/Output
    parser.add_argument("--input", type=Path, required=True, 
                       help="Input .dm4 file")
    parser.add_argument("--output", type=Path, required=True, 
                       help="Output .h5 file")
    
    # Processing options
    parser.add_argument("--downsample", type=int, default=1, metavar="k",
                       help="Downsampling factor (default: 1, no downsampling)")
    parser.add_argument("--mode", choices=["bin", "stride", "gauss", "fft"], default="bin",
                       help="Downsampling strategy (default: bin)")
    parser.add_argument("--sigma", type=float, default=0.8, metavar="σ",
                       help="Gaussian sigma for 'gauss' mode (default: 0.8)")
    parser.add_argument("--scan_step", type=int, default=1, metavar="n",
                       help="Take every n-th probe position (default: 1)")
    
    # Storage options  
    parser.add_argument("--chunk_size", type=int, default=DEFAULT_CHUNK_SIZE, metavar="n",
                       help=f"Chunk size for processing (default: {DEFAULT_CHUNK_SIZE})")
    parser.add_argument("--dtype", choices=["uint16", "float16", "float32"], default="float16",
                       help="Output data type (default: float16)")
    parser.add_argument("--compression_level", type=int, default=4, metavar="1-9",
                       help="HDF5 gzip compression level (default: 4)")
    
    return parser


def main():
    """Main entry point."""
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Validate input
    if not args.input.exists():
        print(f"ERROR: Input file not found: {args.input}")
        return
    
    if not args.input.suffix.lower() == '.dm4':
        print(f"WARNING: Input file doesn't have .dm4 extension: {args.input}")
    
    # Create converter and run
    converter = DM4Converter(
        input_path=args.input,
        output_path=args.output,
        downsample=args.downsample,
        mode=args.mode,
        sigma=args.sigma,
        scan_step=args.scan_step,
        chunk_size=args.chunk_size,
        dtype=args.dtype,
        compression_level=args.compression_level
    )
    
    try:
        converter.convert()
    except KeyboardInterrupt:
        print("\nConversion interrupted by user")
    except Exception as e:
        print(f"ERROR: Conversion failed: {e}")
        raise


if __name__ == "__main__":
    main()