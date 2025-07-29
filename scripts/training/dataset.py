"""
HDF5Dataset implementation for 4D-STEM data loading.

Extracted from the original train.py for better modularity and testability.
"""
from __future__ import annotations

import json
import datetime
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import h5py
from torch.utils.data import Dataset


class HDF5Dataset(Dataset):
    """Optimized dataset for lazy loading of HDF5-compressed 4D-STEM data with optional normalization."""
    
    def __init__(self, data_path: Path, metadata_path: Optional[Path] = None, use_normalization: bool = True):
        self.data_path = Path(data_path)
        self.use_normalization = use_normalization
        
        # Only support HDF5 files
        if self.data_path.suffix != '.h5':
            raise ValueError(f"Only HDF5 files (.h5) are supported, got: {self.data_path.suffix}")
        
        # Don't open HDF5 file in __init__ to avoid pickling issues
        self.h5_file = None
        self.arr = None
        
        # Get metadata and shape from a temporary file handle
        with h5py.File(str(data_path), 'r') as temp_file:
            temp_arr = temp_file['patterns']
            self.shape = temp_arr.shape
            
            # Load metadata from HDF5 attributes or fallback to defaults
            if hasattr(temp_arr, 'attrs') and len(temp_arr.attrs) > 0:
                self.metadata = dict(temp_arr.attrs)
                print(f"Loaded HDF5 dataset with metadata: {list(self.metadata.keys())}")
            else:
                print("Warning: No metadata found in HDF5 file, using defaults")
                self.metadata = {"data_min": 0.0, "data_max": 1.0, "data_range": 1.0, "dtype": "float16"}
        
        # Load or compute global normalization statistics if enabled
        if self.use_normalization:
            print("Loading/computing normalization statistics...")
            self._load_or_compute_global_stats()
        else:
            print("Normalization disabled - training directly on log data")
            self.global_log_mean = 0.0
            self.global_log_std = 1.0
        
        print(f"Loaded HDF5 dataset: {self.shape} patterns, dtype: {self.metadata['dtype']}")
        print(f"Data range: {self.metadata['data_min']:.3f} to {self.metadata['data_min'] + self.metadata['data_range']:.3f}")
        print(f"Global normalization: mean={self.global_log_mean:.4f}, std={self.global_log_std:.4f}")
    
    def _ensure_file_open(self):
        """Ensure HDF5 file is open. Called lazily to avoid pickling issues."""
        if self.h5_file is None or not self.h5_file.id.valid:
            # Optimized HDF5 settings for better I/O performance
            self.h5_file = h5py.File(
                str(self.data_path), 'r',
                rdcc_nbytes=1024*1024*64,  # 64MB chunk cache
                rdcc_nslots=10007          # More cache slots
            )
            self.arr = self.h5_file['patterns']
    
    def __getstate__(self):
        """Custom pickling to exclude HDF5 file handles."""
        state = self.__dict__.copy()
        # Remove unpicklable HDF5 objects
        state['h5_file'] = None
        state['arr'] = None
        return state
    
    def __setstate__(self, state):
        """Custom unpickling to restore state without HDF5 file handles."""
        self.__dict__.update(state)
        # HDF5 file will be reopened lazily in _ensure_file_open()
    
    def __del__(self):
        """Ensure HDF5 file is properly closed."""
        try:
            if hasattr(self, 'h5_file') and self.h5_file is not None:
                self.h5_file.close()
        except:
            pass  # Ignore errors during cleanup
    
    def _load_or_compute_global_stats(self):
        """Load pre-computed global statistics or compute them once."""
        stats_path = self.data_path.parent / f"{self.data_path.stem}_normalization_stats.json"
        
        if stats_path.exists():
            print("Loading pre-computed normalization statistics...")
            with open(stats_path, 'r') as f:
                stats = json.load(f)
                self.global_log_mean = stats['log_mean']
                self.global_log_std = stats['log_std']
        else:
            print("Computing global normalization statistics (one-time cost)...")
            self._compute_and_save_global_stats(stats_path)
    
    def _compute_and_save_global_stats(self, stats_path: Path):
        """Compute global log-space statistics efficiently using streaming approach."""
        total_patterns = self.shape[0]
        
        print(f"Computing normalization statistics from ALL {total_patterns} patterns...")
        print("Using optimized streaming approach with chunked processing...")
        
        # Optimized chunked processing parameters
        chunk_size = min(1000, max(100, total_patterns // 1000))
        pixel_stride = 50  # Sample every 50th pixel for speed while maintaining accuracy
        
        # Streaming computation using Welford's algorithm
        running_mean = 0.0
        running_m2 = 0.0  # For variance calculation
        total_count = 0
        
        print(f"Processing in chunks of {chunk_size} patterns, sampling every {pixel_stride}th pixel...")
        
        # Open file temporarily for statistics computation
        with h5py.File(str(self.data_path), 'r') as temp_file:
            temp_arr = temp_file['patterns']
            
            # Process all patterns in chunks
            for chunk_start in range(0, total_patterns, chunk_size):
                chunk_end = min(chunk_start + chunk_size, total_patterns)
                
                # Progress reporting
                progress = (chunk_start / total_patterns) * 100
                print(f"Processing chunk {chunk_start//chunk_size + 1}/{(total_patterns + chunk_size - 1)//chunk_size} "
                      f"(patterns {chunk_start}-{chunk_end-1}, {progress:.1f}% complete)")
                
                try:
                    # Load chunk of patterns
                    chunk_data = np.array(temp_arr[chunk_start:chunk_end])
                    
                    # Apply same dequantization as in training
                    if self.metadata["dtype"] == "uint16":
                        chunk_data = chunk_data.astype("float32") / 65535.0 * self.metadata["data_range"] + self.metadata["data_min"]
                    elif self.metadata["dtype"] == "float16":
                        chunk_data = chunk_data.astype("float32")
                    
                    # Apply log scaling (must match training transform)
                    log_chunk = np.log(chunk_data + 1)
                    
                    # Sample pixels efficiently
                    flat_chunk = log_chunk.flatten()
                    sampled_pixels = flat_chunk[::pixel_stride]
                    
                    # Update running statistics using Welford's algorithm
                    for value in sampled_pixels:
                        total_count += 1
                        delta = value - running_mean
                        running_mean += delta / total_count
                        delta2 = value - running_mean
                        running_m2 += delta * delta2
                    
                    # Force garbage collection to free memory
                    del chunk_data, log_chunk, flat_chunk, sampled_pixels
                    
                except Exception as e:
                    print(f"Warning: Failed to process chunk {chunk_start}-{chunk_end}: {e}")
                    continue
        
        # Finalize statistics
        self.global_log_mean = float(running_mean)
        self.global_log_std = float(np.sqrt(running_m2 / (total_count - 1)) if total_count > 1 else 1.0)
        
        # Save for future use
        stats = {
            'log_mean': self.global_log_mean,
            'log_std': self.global_log_std,
            'total_patterns_used': total_patterns,
            'pixels_sampled': total_count,
            'pixel_stride': pixel_stride,
            'chunk_size': chunk_size,
            'computed_on': datetime.datetime.now().isoformat(),
            'method': 'full_dataset_streaming_welford'
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"Global stats computed and saved: mean={self.global_log_mean:.4f}, std={self.global_log_std:.4f}")
        print(f"Used ALL {total_patterns} patterns with {total_count} pixel samples (every {pixel_stride}th pixel)")
        print(f"Statistics saved to: {stats_path}")
    
    def __len__(self):
        return self.shape[0]
    
    def __getitem__(self, idx):
        """Get a single data sample with optimized preprocessing."""
        if torch.is_tensor(idx):
            idx = idx.item()
        
        # Ensure file is open (lazy loading)
        self._ensure_file_open()
        
        # Direct conversion with minimal copies
        pattern = np.array(self.arr[idx])
        x = torch.from_numpy(pattern).float()
        
        # Apply dequantization
        x = self._dequantize_fast(x)
        
        # Apply log transform
        x = torch.log(x + 1)
        
        # Apply normalization only if enabled
        if self.use_normalization:
            x = (x - self.global_log_mean) / (self.global_log_std + 1e-8)
        
        # Ensure channel dimension
        if x.dim() == 2:
            x = x.unsqueeze(0)
        
        return (x,)
    
    def _dequantize_fast(self, x: torch.Tensor) -> torch.Tensor:
        """Optimized dequantization with minimal branching."""
        dtype = self.metadata["dtype"]
        if dtype == "uint16":
            return x * (self.metadata["data_range"] / 65535.0) + self.metadata["data_min"]
        elif dtype == "float16":
            return x  # Already float32 from torch.from_numpy().float()
        return x  # float32 ready