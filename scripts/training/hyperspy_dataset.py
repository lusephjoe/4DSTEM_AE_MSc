"""PyTorch dataset for efficient lazy loading of HyperSpy 4D-STEM data."""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Union
import warnings
warnings.filterwarnings("ignore")

try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False

class HyperSpyDataset(Dataset):
    """PyTorch dataset for lazy loading of HyperSpy 4D-STEM data."""
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 scan_step: int = 1,
                 downsample: int = 1,
                 downsample_mode: str = "bin",
                 sigma: float = 0.8,
                 normalize: bool = True,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize HyperSpy dataset for lazy loading.
        
        Args:
            file_path: Path to .hspy file
            scan_step: Subsample scan positions by this factor
            downsample: Downsample diffraction patterns by this factor
            downsample_mode: Downsampling method ('bin', 'stride', 'gauss', 'fft')
            sigma: Gaussian sigma for 'gauss' mode
            normalize: Whether to apply log normalization
            dtype: Output tensor dtype
        """
        if not HAS_HYPERSPY:
            raise ImportError("HyperSpy required. Install with: pip install hyperspy")
        
        self.file_path = Path(file_path)
        self.scan_step = scan_step
        self.downsample = downsample
        self.downsample_mode = downsample_mode
        self.sigma = sigma
        self.normalize = normalize
        self.dtype = dtype
        
        # Load signal lazily
        print(f"Loading {self.file_path} lazily...")
        self.signal = hs.load(str(self.file_path), lazy=True)
        
        # Get original shape
        self.original_shape = self.signal.data.shape
        print(f"Original shape: {self.original_shape}")
        
        # Calculate effective shape after subsampling
        ny, nx, qy, qx = self.original_shape
        self.ny = ny // scan_step
        self.nx = nx // scan_step
        self.qy = qy // downsample if downsample > 1 else qy
        self.qx = qx // downsample if downsample > 1 else qx
        
        # Total number of samples
        self.total_samples = self.ny * self.nx
        
        print(f"Dataset shape after processing: ({self.ny}, {self.nx}, {self.qy}, {self.qx})")
        print(f"Total samples: {self.total_samples}")
        
        # Precompute normalization statistics if needed
        self.mean, self.std = None, None
        if normalize:
            self._compute_normalization_stats()
    
    def _compute_normalization_stats(self):
        """Compute normalization statistics from a sample of the data."""
        print("Computing normalization statistics...")
        
        # Sample a subset of the data for statistics
        sample_size = min(1000, self.total_samples)
        indices = np.random.choice(self.total_samples, sample_size, replace=False)
        
        all_samples = []
        for idx in indices:
            row = idx // self.nx
            col = idx % self.nx
            
            # Get scan position with subsampling
            scan_row = row * self.scan_step
            scan_col = col * self.scan_step
            
            # Load single pattern
            pattern = self.signal.data[scan_row, scan_col].compute()
            
            # Apply log scaling
            pattern = np.log(pattern.astype(np.float32) + 1)
            all_samples.append(pattern.flatten())
        
        # Compute global statistics
        all_data = np.concatenate(all_samples)
        self.mean = float(np.mean(all_data))
        self.std = float(np.std(all_data))
        
        print(f"✓ Computed normalization stats: mean={self.mean:.4f}, std={self.std:.4f}")
    
    def _downsample_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Apply downsampling to a single pattern."""
        if self.downsample <= 1:
            return pattern
            
        if self.downsample_mode == "stride":
            return pattern[::self.downsample, ::self.downsample]
        elif self.downsample_mode == "bin":
            # Simple binning
            k = self.downsample
            qy, qx = pattern.shape
            qy2, qx2 = (qy // k) * k, (qx // k) * k
            trimmed = pattern[:qy2, :qx2]
            return trimmed.reshape(qy2 // k, k, qx2 // k, k).mean(axis=(1, 3))
        elif self.downsample_mode == "gauss":
            from scipy.ndimage import gaussian_filter
            blurred = gaussian_filter(pattern, sigma=self.sigma)
            return blurred[::self.downsample, ::self.downsample]
        else:
            # Default to stride
            return pattern[::self.downsample, ::self.downsample]
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single sample by index."""
        # Convert flat index to 2D coordinates
        row = idx // self.nx
        col = idx % self.nx
        
        # Get scan position with subsampling
        scan_row = row * self.scan_step
        scan_col = col * self.scan_step
        
        # Load single pattern from lazy array
        pattern = self.signal.data[scan_row, scan_col].compute()
        
        # Convert to float32
        pattern = pattern.astype(np.float32)
        
        # Apply normalization
        if self.normalize:
            pattern = np.log(pattern + 1)
            pattern = (pattern - self.mean) / (self.std + 1e-8)
        
        # Apply downsampling
        if self.downsample > 1:
            pattern = self._downsample_pattern(pattern)
        
        # Convert to tensor with channel dimension
        pattern = torch.from_numpy(pattern).unsqueeze(0)  # Add channel dimension
        
        return pattern.to(self.dtype)
    
    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of individual samples: (channels, height, width)."""
        return (1, self.qy, self.qx)
    
    def close(self):
        """Close the HyperSpy signal."""
        if hasattr(self.signal, 'close'):
            self.signal.close()


class ChunkedHyperSpyDataset(Dataset):
    """Memory-efficient dataset that loads data in chunks."""
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 chunk_size: int = 64,
                 scan_step: int = 1,
                 downsample: int = 1,
                 downsample_mode: str = "bin",
                 sigma: float = 0.8,
                 normalize: bool = True,
                 dtype: torch.dtype = torch.float32):
        """
        Initialize chunked HyperSpy dataset.
        
        Args:
            file_path: Path to .hspy file
            chunk_size: Number of patterns to load at once
            scan_step: Subsample scan positions by this factor
            downsample: Downsample diffraction patterns by this factor
            downsample_mode: Downsampling method
            sigma: Gaussian sigma for 'gauss' mode
            normalize: Whether to apply log normalization
            dtype: Output tensor dtype
        """
        self.base_dataset = HyperSpyDataset(
            file_path, scan_step, downsample, downsample_mode, 
            sigma, normalize, dtype
        )
        self.chunk_size = chunk_size
        self.cache = {}
        self.cache_keys = []
        self.max_cache_size = 4  # Keep max 4 chunks in memory
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def _get_chunk_key(self, idx: int) -> int:
        """Get chunk key for given index."""
        return idx // self.chunk_size
    
    def _load_chunk(self, chunk_key: int) -> dict:
        """Load a chunk of data."""
        start_idx = chunk_key * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.base_dataset))
        
        chunk_data = {}
        for i in range(start_idx, end_idx):
            chunk_data[i] = self.base_dataset[i]
        
        return chunk_data
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item with caching."""
        chunk_key = self._get_chunk_key(idx)
        
        # Check if chunk is in cache
        if chunk_key not in self.cache:
            # Load chunk
            chunk_data = self._load_chunk(chunk_key)
            
            # Manage cache size
            if len(self.cache) >= self.max_cache_size:
                # Remove oldest chunk
                oldest_key = self.cache_keys.pop(0)
                del self.cache[oldest_key]
            
            # Add to cache
            self.cache[chunk_key] = chunk_data
            self.cache_keys.append(chunk_key)
        
        return self.cache[chunk_key][idx]
    
    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of individual samples."""
        return self.base_dataset.get_shape()
    
    def close(self):
        """Close the base dataset."""
        self.base_dataset.close()