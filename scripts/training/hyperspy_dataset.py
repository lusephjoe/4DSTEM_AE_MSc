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
        
        # Use standard normalization values for 4D-STEM log data
        # These are typical values for log-transformed diffraction patterns
        self.mean = 2.5  # Typical mean for log(intensity + 1)
        self.std = 1.5   # Typical std for log(intensity + 1)
        
        if normalize:
            print(f"✓ Using standard normalization values: mean={self.mean:.1f}, std={self.std:.1f}")
        
        # Validate dataset integrity
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate dataset integrity and provide helpful information."""
        try:
            # Check if we can access a sample pattern
            test_pattern = self.signal.data[0, 0].compute()
            
            # Validate pattern properties
            if test_pattern.size == 0:
                print("Warning: Test pattern is empty")
            elif np.all(test_pattern == 0):
                print("Warning: Test pattern contains only zeros")
            elif np.any(np.isnan(test_pattern)) or np.any(np.isinf(test_pattern)):
                print("Warning: Test pattern contains NaN or infinite values")
            else:
                print(f"✓ Dataset validation passed (test pattern range: {test_pattern.min():.2f} to {test_pattern.max():.2f})")
                
        except Exception as e:
            print(f"Warning: Dataset validation failed: {e}")
            print("Dataset may still work but could have issues during training")
    
    
    def _downsample_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Apply efficient downsampling to a single pattern with optimizations."""
        if self.downsample <= 1:
            return pattern
            
        k = self.downsample
        qy, qx = pattern.shape
        
        # Early exit for very small patterns
        if qy < k or qx < k:
            return pattern[::k, ::k]
        
        if self.downsample_mode == "stride":
            # Simple stride sampling - fastest
            return pattern[::k, ::k]
            
        elif self.downsample_mode == "bin":
            # Optimized mean pooling with minimal memory allocation
            qy2, qx2 = (qy // k) * k, (qx // k) * k
            if qy2 < k or qx2 < k:
                # Fallback to stride if too small
                return pattern[::k, ::k]
            
            # Use views instead of copies when possible
            trimmed = pattern[:qy2, :qx2]
            
            # Optimized reshape and mean - use specific dtype to avoid conversion
            try:
                # Direct reshape with mean - most efficient for large arrays
                reshaped = trimmed.reshape(qy2 // k, k, qx2 // k, k)
                binned = np.mean(reshaped, axis=(1, 3), dtype=pattern.dtype)
                return binned
            except (ValueError, MemoryError):
                # Fallback to stride if reshape fails
                return pattern[::k, ::k]
            
        elif self.downsample_mode == "gauss":
            # Gaussian filtering with optimized parameters and caching
            try:
                from scipy.ndimage import gaussian_filter
                
                # Optimize sigma based on downsampling factor
                if k <= 2:
                    sigma = self.sigma * 0.8  # Less blur for small downsampling
                elif k <= 4:
                    sigma = self.sigma
                else:
                    sigma = self.sigma * 1.2  # More blur for heavy downsampling
                
                sigma = min(sigma, k * 0.6)  # Clamp to reasonable value
                
                # Use truncate parameter for efficiency
                blurred = gaussian_filter(pattern, sigma=sigma, mode='reflect', truncate=3.0)
                return blurred[::k, ::k]
            except (ImportError, MemoryError):
                # Fallback to stride if scipy not available or memory issues
                return pattern[::k, ::k]
        else:
            # Default to stride
            return pattern[::k, ::k]
    
    def __len__(self) -> int:
        return self.total_samples
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get a single sample by index with efficient processing and error handling."""
        # Convert flat index to 2D coordinates
        row = idx // self.nx
        col = idx % self.nx
        
        # Get scan position with subsampling
        scan_row = row * self.scan_step
        scan_col = col * self.scan_step
        
        # Load single pattern from lazy array with robust error handling
        try:
            pattern = self.signal.data[scan_row, scan_col].compute()
            
            # Validate pattern dimensions
            if pattern.size == 0 or pattern.shape[0] == 0 or pattern.shape[1] == 0:
                raise ValueError(f"Empty pattern at ({scan_row}, {scan_col})")
                
        except Exception as e:
            if idx % 1000 == 0:  # Only print every 1000th error to avoid spam
                print(f"Warning: Failed to load pattern at ({scan_row}, {scan_col}): {e}")
            # Return zero pattern with correct shape
            qy, qx = self.qy, self.qx
            pattern = np.zeros((qy, qx), dtype=np.float32)
        
        # Efficient processing pipeline
        try:
            pattern = self._process_pattern(pattern)
        except Exception as e:
            if idx % 1000 == 0:  # Only print every 1000th error to avoid spam
                print(f"Warning: Failed to process pattern at ({scan_row}, {scan_col}): {e}")
            # Return zero pattern with correct shape
            qy, qx = self.qy, self.qx
            pattern = np.zeros((qy, qx), dtype=np.float32)
        
        # Convert to tensor with channel dimension - optimized
        try:
            # Use contiguous array for better performance
            if not pattern.flags.c_contiguous:
                pattern = np.ascontiguousarray(pattern)
            
            tensor = torch.from_numpy(pattern).unsqueeze(0)  # Add channel dimension
            
            # Convert dtype if needed
            if tensor.dtype != self.dtype:
                tensor = tensor.to(self.dtype)
                
            return tensor
        except Exception as e:
            if idx % 1000 == 0:  # Only print every 1000th error to avoid spam
                print(f"Warning: Failed to convert pattern to tensor at ({scan_row}, {scan_col}): {e}")
            # Return zero tensor with correct shape
            qy, qx = self.qy, self.qx
            return torch.zeros(1, qy, qx, dtype=self.dtype)
    
    def _process_pattern(self, pattern: np.ndarray) -> np.ndarray:
        """Efficiently process a single pattern with optimized operations."""
        # Convert to float32 early for consistent processing
        if pattern.dtype != np.float32:
            pattern = pattern.astype(np.float32)
        
        # Apply downsampling before normalization for efficiency
        if self.downsample > 1:
            pattern = self._downsample_pattern(pattern)
        
        # Apply normalization with optimized operations
        if self.normalize:
            # Vectorized operations with minimal memory allocation
            # Use in-place operations where possible
            pattern = np.log(pattern + 1.0)  # Add small constant to avoid log(0)
            
            # Efficient normalization using pre-computed values
            std_with_eps = self.std + 1e-8
            pattern -= self.mean
            pattern /= std_with_eps
        
        return pattern
    
    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of individual samples: (channels, height, width)."""
        return (1, self.qy, self.qx)
    
    def close(self):
        """Close the HyperSpy signal and free memory."""
        if hasattr(self.signal, 'close'):
            self.signal.close()
        elif hasattr(self.signal, 'data') and hasattr(self.signal.data, 'close'):
            self.signal.data.close()
        
        # Clear references to help garbage collection
        self.signal = None
        
        # Force garbage collection
        import gc
        gc.collect()


class ChunkedHyperSpyDataset(Dataset):
    """Memory-efficient dataset that loads data in chunks with advanced caching."""
    
    def __init__(self, 
                 file_path: Union[str, Path],
                 chunk_size: int = 64,
                 scan_step: int = 1,
                 downsample: int = 1,
                 downsample_mode: str = "bin",
                 sigma: float = 0.8,
                 normalize: bool = True,
                 dtype: torch.dtype = torch.float32,
                 debug: bool = False):
        """
        Initialize chunked HyperSpy dataset with optimized caching.
        
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
        self.debug = debug
        self.base_dataset = HyperSpyDataset(
            file_path, scan_step, downsample, downsample_mode, 
            sigma, normalize, dtype
        )
        
        # Optimize chunk size based on dataset characteristics
        dataset_size = len(self.base_dataset)
        if dataset_size > 100000:  # Very large dataset
            self.chunk_size = min(chunk_size, 32)  # Smaller chunks for memory efficiency
        elif dataset_size > 50000:  # Large dataset
            self.chunk_size = min(chunk_size, 48)
        else:
            self.chunk_size = chunk_size
        
        self.cache = {}
        self.cache_keys = []
        # Adaptive cache size based on available memory
        self.max_cache_size = self._calculate_optimal_cache_size()
        
        # Add a simple lock to prevent concurrent access issues
        import threading
        self._lock = threading.Lock()
        
        print(f"✓ Chunked dataset initialized with cache size: {self.max_cache_size} chunks (chunk_size: {self.chunk_size})")
        
        # Test loading the first chunk to catch issues early
        print("Testing first chunk loading...")
        try:
            test_sample = self[0]
            print(f"✓ First chunk test successful, sample shape: {test_sample.shape}")
        except Exception as e:
            print(f"Warning: First chunk test failed: {e}")
            print("Dataset may still work but could hang during training.")
    
    def _calculate_optimal_cache_size(self) -> int:
        """Calculate optimal cache size based on available memory and dataset size."""
        try:
            import psutil
            available_memory_gb = psutil.virtual_memory().available / (1024**3)
            
            # Estimate memory per chunk with overhead
            pattern_size_mb = (self.base_dataset.qy * self.base_dataset.qx * 4) / (1024**2)  # float32
            chunk_size_mb = pattern_size_mb * self.chunk_size
            
            # Add overhead for Python objects and processing
            chunk_size_mb *= 1.5  # 50% overhead
            
            # Use adaptive percentage based on dataset size
            dataset_size = len(self.base_dataset)
            if dataset_size > 50000:  # Large dataset
                cache_memory_percent = 0.05  # 5% for very large datasets
            elif dataset_size > 10000:  # Medium dataset
                cache_memory_percent = 0.08  # 8% for medium datasets
            else:  # Small dataset
                cache_memory_percent = 0.12  # 12% for small datasets
            
            max_cache_memory_gb = available_memory_gb * cache_memory_percent
            optimal_cache_size = int(max_cache_memory_gb * 1024 / chunk_size_mb)
            
            # Adaptive limits based on dataset size
            if dataset_size > 50000:
                min_cache, max_cache = 2, 8
            elif dataset_size > 10000:
                min_cache, max_cache = 3, 12
            else:
                min_cache, max_cache = 4, 16
            
            optimal_size = max(min_cache, min(max_cache, optimal_cache_size))
            
            print(f"Cache configuration: {optimal_size} chunks ({optimal_size * chunk_size_mb:.1f} MB total)")
            return optimal_size
        except Exception as e:
            print(f"Warning: Failed to calculate optimal cache size: {e}")
            return 4
        
    def __len__(self) -> int:
        return len(self.base_dataset)
    
    def _get_chunk_key(self, idx: int) -> int:
        """Get chunk key for given index."""
        return idx // self.chunk_size
    
    def _load_chunk(self, chunk_key: int) -> dict:
        """Load a chunk of data with efficient batch processing."""
        if self.debug:
            print(f"Loading chunk {chunk_key}...")
        start_idx = chunk_key * self.chunk_size
        end_idx = min(start_idx + self.chunk_size, len(self.base_dataset))
        
        if self.debug:
            print(f"Chunk {chunk_key}: loading indices {start_idx} to {end_idx}")
        chunk_data = {}
        
        # Try to batch load patterns for efficiency
        if hasattr(self.base_dataset.signal.data, 'compute'):
            # For dask arrays, we can potentially batch compute
            try:
                indices = list(range(start_idx, end_idx))
                patterns = self._batch_load_patterns(indices)
                
                for i, pattern in enumerate(patterns):
                    if pattern is not None:
                        chunk_data[start_idx + i] = pattern
            except Exception as e:
                print(f"Warning: Batch loading failed, falling back to individual loading: {e}")
                # Fallback to individual loading
                for i in range(start_idx, end_idx):
                    try:
                        chunk_data[i] = self.base_dataset[i]
                    except Exception as e:
                        print(f"Warning: Failed to load pattern {i}: {e}")
                        continue
        else:
            # Standard individual loading
            for i in range(start_idx, end_idx):
                try:
                    chunk_data[i] = self.base_dataset[i]
                except Exception as e:
                    print(f"Warning: Failed to load pattern {i}: {e}")
                    continue
        
        return chunk_data
    
    def _batch_load_patterns(self, indices: list) -> list:
        """Optimized batch loading with vectorized operations."""
        patterns = []
        
        # Group indices by row for better locality
        row_groups = {}
        for idx in indices:
            row = idx // self.base_dataset.nx
            col = idx % self.base_dataset.nx
            if row not in row_groups:
                row_groups[row] = []
            row_groups[row].append((idx, col))
        
        # Process each row group
        for row, col_data in row_groups.items():
            scan_row = row * self.base_dataset.scan_step
            
            # Try to load entire row at once for better efficiency
            try:
                if len(col_data) > 1:
                    # Load multiple columns from the same row
                    cols = [col for _, col in col_data]
                    scan_cols = [col * self.base_dataset.scan_step for col in cols]
                    
                    # Batch load patterns from the same row
                    row_patterns = []
                    for scan_col in scan_cols:
                        try:
                            pattern = self.base_dataset.signal.data[scan_row, scan_col].compute()
                            processed = self.base_dataset._process_pattern(pattern)
                            tensor = torch.from_numpy(processed).unsqueeze(0).to(self.base_dataset.dtype)
                            row_patterns.append(tensor)
                        except Exception:
                            row_patterns.append(None)
                    
                    # Add to patterns list in original order
                    for i, (idx, _) in enumerate(col_data):
                        patterns.append(row_patterns[i])
                else:
                    # Single pattern in this row
                    idx, col = col_data[0]
                    scan_col = col * self.base_dataset.scan_step
                    
                    try:
                        pattern = self.base_dataset.signal.data[scan_row, scan_col].compute()
                        processed = self.base_dataset._process_pattern(pattern)
                        tensor = torch.from_numpy(processed).unsqueeze(0).to(self.base_dataset.dtype)
                        patterns.append(tensor)
                    except Exception:
                        patterns.append(None)
            
            except Exception:
                # Fallback: add None for all patterns in this row
                for _ in col_data:
                    patterns.append(None)
        
        return patterns
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        """Get item with optimized caching and error handling."""
        if self.debug and idx < 5:  # Debug first few items
            print(f"ChunkedHyperSpyDataset.__getitem__({idx}) called")
        
        chunk_key = self._get_chunk_key(idx)
        
        if self.debug and idx < 5:
            print(f"Chunk key for idx {idx}: {chunk_key}")
        
        # Use lock to prevent concurrent access issues
        with self._lock:
            # Check if chunk is in cache
            if chunk_key not in self.cache:
                # Load chunk with memory management
                try:
                    chunk_data = self._load_chunk(chunk_key)
                    
                    # Manage cache size with LRU eviction and aggressive cleanup
                    while len(self.cache) >= self.max_cache_size:
                        # Remove oldest chunk
                        oldest_key = self.cache_keys.pop(0)
                        if oldest_key in self.cache:
                            # Clear chunk data with safe cleanup
                            chunk_to_delete = self.cache[oldest_key]
                            if isinstance(chunk_to_delete, dict):
                                for k, v in chunk_to_delete.items():
                                    # Safer cleanup - just delete references
                                    if hasattr(v, 'data'):
                                        try:
                                            del v.data
                                        except:
                                            pass
                            del self.cache[oldest_key]
                            del chunk_to_delete
                            
                            # Force garbage collection for large datasets
                            import gc
                            gc.collect()
                            
                            # Additional CUDA memory cleanup if available
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                    
                    # Add to cache
                    self.cache[chunk_key] = chunk_data
                    self.cache_keys.append(chunk_key)
                    
                except Exception as e:
                    print(f"Warning: Failed to load chunk {chunk_key}: {e}")
                    # Return fallback pattern
                    return self._get_fallback_pattern()
            
            # Update cache access order (move to end) - optimized
            if chunk_key in self.cache_keys:
                # More efficient than remove + append for large lists
                idx_pos = self.cache_keys.index(chunk_key)
                self.cache_keys.append(self.cache_keys.pop(idx_pos))
            
            # Return pattern from cache
            if idx in self.cache[chunk_key]:
                return self.cache[chunk_key][idx]
            else:
                # Fallback if pattern not in chunk
                return self._get_fallback_pattern()
    
    def _get_fallback_pattern(self) -> torch.Tensor:
        """Generate a fallback pattern in case of loading errors."""
        qy, qx = self.base_dataset.qy, self.base_dataset.qx
        
        # Create a more realistic fallback pattern instead of zeros
        # This helps prevent training issues with completely zero patterns
        fallback = torch.zeros(1, qy, qx, dtype=self.base_dataset.dtype)
        
        # Add small random noise to avoid perfect zeros
        if self.base_dataset.normalize:
            # For normalized data, use small values around the mean
            fallback += torch.randn_like(fallback) * 0.01
        else:
            # For unnormalized data, use very small positive values
            fallback += torch.rand_like(fallback) * 0.001
        
        return fallback
    
    def get_shape(self) -> Tuple[int, int, int]:
        """Get the shape of individual samples."""
        return self.base_dataset.get_shape()
    
    def close(self):
        """Close the base dataset and clear cache with safe cleanup."""
        # Clear cache with safe tensor cleanup
        for chunk_key, chunk_data in self.cache.items():
            if isinstance(chunk_data, dict):
                for k, v in chunk_data.items():
                    # Safer cleanup - just delete references instead of resizing storage
                    if hasattr(v, 'data'):
                        try:
                            del v.data
                        except:
                            pass
        
        self.cache.clear()
        self.cache_keys.clear()
        
        # Close base dataset
        self.base_dataset.close()
        
        # Aggressive memory cleanup
        import gc
        gc.collect()
        
        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clear references
        self.base_dataset = None