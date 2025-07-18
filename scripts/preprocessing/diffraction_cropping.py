#!/usr/bin/env python3
"""
Automatic Diffraction Pattern Cropping for 4D-STEM Data

This module implements automatic cropping of diffraction patterns from 4D-STEM datasets,
reducing data size while retaining 95-99% of original intensities.

Key Features:
- Analysis pass: Compute optimal cropping radius
- Cropping pass: Uniform crop based on global radius
- Visualization: Before/after comparison
- Performance: Chunked processing for memory efficiency
- GPU acceleration support with CuPy

Author: Claude Code Assistant
Date: 2025-01-18
"""

from __future__ import annotations
import numpy as np
import torch
from typing import Tuple, Optional, Dict, Any
from pathlib import Path
import warnings
from dataclasses import dataclass
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import time

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

try:
    import dask.array as da
    HAS_DASK = True
except ImportError:
    HAS_DASK = False
    da = None


@dataclass
class CroppingConfig:
    """Configuration for diffraction pattern cropping."""
    target_retention: float = 0.98  # Target intensity retention (98%)
    min_retention: float = 0.95     # Minimum acceptable retention (95%)
    margin_pixels: int = 3          # Additional pixels around computed radius
    chunk_size: int = 1000         # Patterns per chunk for memory efficiency
    use_gpu: bool = False          # Enable GPU acceleration
    center_method: str = "centroid" # "centroid" or "center" or "manual"
    manual_center: Optional[Tuple[int, int]] = None  # Manual center (y, x)
    visualization: bool = True      # Enable visualization
    verbose: bool = True           # Enable verbose output
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not (0 < self.target_retention <= 1):
            raise ValueError("target_retention must be between 0 and 1")
        
        if not (0 < self.min_retention <= 1):
            raise ValueError("min_retention must be between 0 and 1")
        
        if self.min_retention > self.target_retention:
            raise ValueError("min_retention cannot be greater than target_retention")
        
        if self.margin_pixels < 0:
            raise ValueError("margin_pixels must be non-negative")
        
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        
        if self.center_method not in ["centroid", "center", "manual"]:
            raise ValueError("center_method must be 'centroid', 'center', or 'manual'")
        
        if self.center_method == "manual" and self.manual_center is None:
            raise ValueError("manual_center must be provided when center_method is 'manual'")


class DiffractionCropper:
    """
    Automatic diffraction pattern cropper for 4D-STEM data.
    
    This class implements a two-pass algorithm:
    1. Analysis pass: Determine optimal cropping radius
    2. Cropping pass: Apply uniform cropping to all patterns
    """
    
    def __init__(self, config: CroppingConfig = None):
        """
        Initialize the diffraction cropper.
        
        Args:
            config: Configuration object with cropping parameters
        """
        self.config = config or CroppingConfig()
        self.xp = cp if (self.config.use_gpu and HAS_CUPY) else np
        
        # Results storage
        self.analysis_results: Optional[Dict[str, Any]] = None
        self.cropped_data: Optional[torch.Tensor] = None
        
        if self.config.use_gpu and not HAS_CUPY:
            warnings.warn("CuPy not available, falling back to CPU processing")
            self.config.use_gpu = False
            self.xp = np
    
    def _to_array(self, data: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy array."""
        if isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        return data
    
    def _from_array(self, data: np.ndarray) -> torch.Tensor:
        """Convert numpy array to tensor."""
        if self.config.use_gpu and HAS_CUPY and isinstance(data, cp.ndarray):
            data = cp.asnumpy(data)
        return torch.from_numpy(data)
    
    def _compute_centroid(self, pattern: np.ndarray) -> Tuple[float, float]:
        """
        Compute the centroid of a diffraction pattern.
        
        Args:
            pattern: 2D diffraction pattern
            
        Returns:
            Tuple of (center_y, center_x) coordinates
        """
        if self.config.use_gpu and HAS_CUPY:
            pattern = cp.asarray(pattern)
        
        # Create coordinate grids
        h, w = pattern.shape
        y_coords, x_coords = self.xp.mgrid[0:h, 0:w]
        
        # Compute weighted centroid
        total_intensity = self.xp.sum(pattern)
        if total_intensity == 0:
            return h / 2, w / 2
        
        center_y = self.xp.sum(y_coords * pattern) / total_intensity
        center_x = self.xp.sum(x_coords * pattern) / total_intensity
        
        if self.config.use_gpu and HAS_CUPY:
            center_y = float(cp.asnumpy(center_y))
            center_x = float(cp.asnumpy(center_x))
        
        return center_y, center_x
    
    def _compute_radial_profile(self, pattern: np.ndarray, 
                              center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute radial intensity profile from diffraction pattern.
        
        Args:
            pattern: 2D diffraction pattern
            center: Center coordinates (y, x)
            
        Returns:
            Tuple of (radii, intensities) arrays
        """
        if self.config.use_gpu and HAS_CUPY:
            pattern = cp.asarray(pattern)
        
        h, w = pattern.shape
        center_y, center_x = center
        
        # Create coordinate grids
        y_coords, x_coords = self.xp.mgrid[0:h, 0:w]
        
        # Compute radial distances
        r = self.xp.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Compute maximum radius
        max_radius = int(self.xp.max(r)) + 1
        
        # Bin intensities by radius
        radii = self.xp.arange(max_radius)
        radial_profile = self.xp.zeros(max_radius)
        
        for i in range(max_radius):
            mask = (r >= i) & (r < i + 1)
            if self.xp.any(mask):
                radial_profile[i] = self.xp.mean(pattern[mask])
        
        if self.config.use_gpu and HAS_CUPY:
            radii = cp.asnumpy(radii)
            radial_profile = cp.asnumpy(radial_profile)
        
        return radii, radial_profile
    
    def _compute_cumulative_intensity(self, pattern: np.ndarray, 
                                    center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute cumulative intensity as a function of radius.
        
        Args:
            pattern: 2D diffraction pattern
            center: Center coordinates (y, x)
            
        Returns:
            Tuple of (radii, cumulative_intensities) arrays
        """
        if self.config.use_gpu and HAS_CUPY:
            pattern = cp.asarray(pattern)
        
        h, w = pattern.shape
        center_y, center_x = center
        
        # Create coordinate grids
        y_coords, x_coords = self.xp.mgrid[0:h, 0:w]
        
        # Compute radial distances
        r = self.xp.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
        
        # Compute maximum radius (ensure we capture all pixels)
        max_radius = int(self.xp.ceil(self.xp.max(r))) + 1
        
        # Compute cumulative intensity
        radii = self.xp.arange(max_radius)
        cumulative_intensity = self.xp.zeros(max_radius)
        total_intensity = self.xp.sum(pattern)
        
        if total_intensity == 0:
            # Handle edge case of zero intensity
            cumulative_intensity[:] = 1.0
        else:
            for i in range(max_radius):
                mask = r <= i
                cumulative_intensity[i] = self.xp.sum(pattern[mask]) / total_intensity
        
        if self.config.use_gpu and HAS_CUPY:
            radii = cp.asnumpy(radii)
            cumulative_intensity = cp.asnumpy(cumulative_intensity)
        
        return radii, cumulative_intensity
    
    def _find_retention_radius(self, cumulative_intensity: np.ndarray, 
                             target_retention: float) -> int:
        """
        Find radius that retains target percentage of intensity.
        
        Args:
            cumulative_intensity: Cumulative intensity array
            target_retention: Target retention percentage (0-1)
            
        Returns:
            Radius that achieves target retention
        """
        # Find first radius where cumulative intensity >= target
        indices = np.where(cumulative_intensity >= target_retention)[0]
        if len(indices) == 0:
            return len(cumulative_intensity) - 1
        return indices[0]
    
    def analyze_patterns(self, data: torch.Tensor) -> Dict[str, Any]:
        """
        Analysis pass: Compute optimal cropping radius.
        
        Args:
            data: Input tensor of shape (N, C, H, W) or (N, H, W)
            
        Returns:
            Dictionary with analysis results
        """
        if self.config.verbose:
            print("Starting analysis pass...")
        
        # Convert to numpy and handle dimensions
        np_data = self._to_array(data)
        if len(np_data.shape) == 4:
            # Remove channel dimension if present
            np_data = np_data.squeeze(1)
        
        n_patterns, h, w = np_data.shape
        
        if self.config.verbose:
            print(f"Analyzing {n_patterns} patterns of size {h}x{w}")
        
        # Determine center method
        if self.config.center_method == "manual" and self.config.manual_center:
            center = self.config.manual_center
            centers = [center] * n_patterns
        elif self.config.center_method == "center":
            center = (h // 2, w // 2)
            centers = [center] * n_patterns
        else:
            # Compute centroids
            if self.config.verbose:
                print("Computing centroids...")
            centers = []
            for i in range(n_patterns):
                if (i + 1) % 1000 == 0 and self.config.verbose:
                    print(f"  Processed {i + 1}/{n_patterns} patterns")
                centers.append(self._compute_centroid(np_data[i]))
        
        # Compute retention radii for each pattern
        if self.config.verbose:
            print("Computing retention radii...")
        
        retention_radii = []
        sample_profiles = []
        
        # Process in chunks to manage memory
        chunk_size = self.config.chunk_size
        
        for chunk_start in range(0, n_patterns, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_patterns)
            
            if self.config.verbose:
                print(f"  Processing chunk {chunk_start//chunk_size + 1}/{(n_patterns-1)//chunk_size + 1}")
            
            for i in range(chunk_start, chunk_end):
                pattern = np_data[i]
                center = centers[i]
                
                # Compute cumulative intensity
                radii, cum_intensity = self._compute_cumulative_intensity(pattern, center)
                
                # Find radius for target retention
                radius = self._find_retention_radius(cum_intensity, self.config.target_retention)
                retention_radii.append(radius)
                
                # Store sample profiles for visualization
                if len(sample_profiles) < 10:
                    sample_profiles.append({
                        'radii': radii,
                        'cumulative': cum_intensity,
                        'center': center,
                        'retention_radius': radius
                    })
        
        # Compute global cropping radius
        max_radius = max(retention_radii)
        global_radius = max_radius + self.config.margin_pixels
        
        # Ensure radius doesn't exceed image bounds (leave some margin)
        max_possible_radius = min(h, w) // 2 - 1
        global_radius = min(global_radius, max_possible_radius)
        
        # Compute statistics
        retention_stats = {
            'mean': np.mean(retention_radii),
            'std': np.std(retention_radii),
            'min': np.min(retention_radii),
            'max': np.max(retention_radii),
            'median': np.median(retention_radii)
        }
        
        results = {
            'global_radius': global_radius,
            'retention_radii': retention_radii,
            'retention_stats': retention_stats,
            'centers': centers,
            'sample_profiles': sample_profiles,
            'original_shape': (h, w),
            'cropped_shape': (2 * global_radius, 2 * global_radius),
            'n_patterns': n_patterns
        }
        
        self.analysis_results = results
        
        if self.config.verbose:
            print(f"Analysis complete!")
            print(f"  Global cropping radius: {global_radius}")
            print(f"  Original size: {h}x{w}")
            print(f"  Cropped size: {2*global_radius}x{2*global_radius}")
            print(f"  Retention radius stats: mean={retention_stats['mean']:.1f}, "
                  f"std={retention_stats['std']:.1f}, range=[{retention_stats['min']}-{retention_stats['max']}]")
        
        return results
    
    def crop_patterns(self, data: torch.Tensor, 
                     analysis_results: Optional[Dict[str, Any]] = None) -> torch.Tensor:
        """
        Cropping pass: Apply uniform cropping to all patterns.
        
        Args:
            data: Input tensor of shape (N, C, H, W) or (N, H, W)
            analysis_results: Results from analysis pass (optional)
            
        Returns:
            Cropped tensor
        """
        if analysis_results is None:
            if self.analysis_results is None:
                raise ValueError("Must run analyze_patterns first or provide analysis_results")
            analysis_results = self.analysis_results
        
        if self.config.verbose:
            print("Starting cropping pass...")
        
        # Convert to numpy and handle dimensions
        np_data = self._to_array(data)
        original_shape = np_data.shape
        
        if len(np_data.shape) == 4:
            n_patterns, c, h, w = np_data.shape
            np_data = np_data.squeeze(1)
        else:
            n_patterns, h, w = np_data.shape
            c = 1
        
        global_radius = analysis_results['global_radius']
        centers = analysis_results['centers']
        crop_size = 2 * global_radius
        
        # Initialize output array
        cropped_data = np.zeros((n_patterns, crop_size, crop_size), dtype=np_data.dtype)
        
        if self.config.verbose:
            print(f"Cropping {n_patterns} patterns to {crop_size}x{crop_size}")
        
        # Process in chunks
        chunk_size = self.config.chunk_size
        
        for chunk_start in range(0, n_patterns, chunk_size):
            chunk_end = min(chunk_start + chunk_size, n_patterns)
            
            if self.config.verbose:
                print(f"  Processing chunk {chunk_start//chunk_size + 1}/{(n_patterns-1)//chunk_size + 1}")
            
            for i in range(chunk_start, chunk_end):
                pattern = np_data[i]
                center_y, center_x = centers[i]
                
                # Compute crop bounds
                y_min = int(center_y - global_radius)
                y_max = int(center_y + global_radius)
                x_min = int(center_x - global_radius)
                x_max = int(center_x + global_radius)
                
                # Handle boundary conditions
                pad_y_min = max(0, -y_min)
                pad_y_max = max(0, y_max - h)
                pad_x_min = max(0, -x_min)
                pad_x_max = max(0, x_max - w)
                
                y_min = max(0, y_min)
                y_max = min(h, y_max)
                x_min = max(0, x_min)
                x_max = min(w, x_max)
                
                # Extract pattern
                cropped_pattern = pattern[y_min:y_max, x_min:x_max]
                
                # Create output array of exact size
                output_pattern = np.zeros((crop_size, crop_size), dtype=np_data.dtype)
                
                # Calculate where to place the extracted pattern
                dest_y_start = pad_y_min
                dest_y_end = dest_y_start + cropped_pattern.shape[0]
                dest_x_start = pad_x_min
                dest_x_end = dest_x_start + cropped_pattern.shape[1]
                
                # Ensure we don't exceed bounds
                dest_y_end = min(dest_y_end, crop_size)
                dest_x_end = min(dest_x_end, crop_size)
                
                # Place the pattern
                output_pattern[dest_y_start:dest_y_end, dest_x_start:dest_x_end] = \
                    cropped_pattern[:dest_y_end-dest_y_start, :dest_x_end-dest_x_start]
                
                cropped_data[i] = output_pattern
        
        # Restore original tensor format
        if len(original_shape) == 4:
            cropped_data = cropped_data[:, np.newaxis, :, :]
        
        cropped_tensor = self._from_array(cropped_data)
        self.cropped_data = cropped_tensor
        
        if self.config.verbose:
            print(f"Cropping complete! Output shape: {cropped_tensor.shape}")
        
        return cropped_tensor
    
    def validate_retention(self, original_data: torch.Tensor, 
                         cropped_data: torch.Tensor,
                         analysis_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Validate intensity retention after cropping.
        
        Args:
            original_data: Original tensor data
            cropped_data: Cropped tensor data
            analysis_results: Results from analysis pass
            
        Returns:
            Dictionary with validation results
        """
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if self.config.verbose:
            print("Validating intensity retention...")
        
        # Convert to numpy
        orig_np = self._to_array(original_data)
        crop_np = self._to_array(cropped_data)
        
        if len(orig_np.shape) == 4:
            orig_np = orig_np.squeeze(1)
        if len(crop_np.shape) == 4:
            crop_np = crop_np.squeeze(1)
        
        n_patterns = orig_np.shape[0]
        centers = analysis_results['centers']
        global_radius = analysis_results['global_radius']
        
        retentions = []
        
        for i in range(n_patterns):
            orig_pattern = orig_np[i]
            crop_pattern = crop_np[i]
            center_y, center_x = centers[i]
            
            # Create circular mask for original pattern
            h, w = orig_pattern.shape
            y_coords, x_coords = np.mgrid[0:h, 0:w]
            r = np.sqrt((y_coords - center_y)**2 + (x_coords - center_x)**2)
            mask = r <= global_radius
            
            # Compute intensities
            orig_total = np.sum(orig_pattern)
            orig_within_radius = np.sum(orig_pattern[mask])
            crop_total = np.sum(crop_pattern)
            
            if orig_total > 0:
                retention = crop_total / orig_total
            else:
                retention = 1.0
            
            retentions.append(retention)
        
        retentions = np.array(retentions)
        
        validation_results = {
            'mean_retention': np.mean(retentions),
            'std_retention': np.std(retentions),
            'min_retention': np.min(retentions),
            'max_retention': np.max(retentions),
            'median_retention': np.median(retentions),
            'retentions': retentions,
            'passes_min_threshold': np.sum(retentions >= self.config.min_retention),
            'passes_target_threshold': np.sum(retentions >= self.config.target_retention),
            'n_patterns': n_patterns
        }
        
        if self.config.verbose:
            print(f"Validation complete!")
            print(f"  Mean retention: {validation_results['mean_retention']:.3f}")
            print(f"  Min retention: {validation_results['min_retention']:.3f}")
            print(f"  Patterns passing min threshold ({self.config.min_retention:.2f}): "
                  f"{validation_results['passes_min_threshold']}/{n_patterns}")
            print(f"  Patterns passing target threshold ({self.config.target_retention:.2f}): "
                  f"{validation_results['passes_target_threshold']}/{n_patterns}")
        
        return validation_results
    
    def visualize_results(self, original_data: torch.Tensor,
                         cropped_data: torch.Tensor,
                         analysis_results: Optional[Dict[str, Any]] = None,
                         save_path: Optional[Path] = None) -> None:
        """
        Visualize cropping results with before/after comparison.
        
        Args:
            original_data: Original tensor data
            cropped_data: Cropped tensor data
            analysis_results: Results from analysis pass
            save_path: Path to save visualization
        """
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if self.config.verbose:
            print("Creating visualization...")
        
        # Convert to numpy
        orig_np = self._to_array(original_data)
        crop_np = self._to_array(cropped_data)
        
        if len(orig_np.shape) == 4:
            orig_np = orig_np.squeeze(1)
        if len(crop_np.shape) == 4:
            crop_np = crop_np.squeeze(1)
        
        # Compute average patterns
        avg_original = np.mean(orig_np, axis=0)
        avg_cropped = np.mean(crop_np, axis=0)
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original pattern
        im1 = axes[0, 0].imshow(avg_original, cmap='hot', origin='lower')
        axes[0, 0].set_title('Average Original Pattern')
        axes[0, 0].set_xlabel('X (pixels)')
        axes[0, 0].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Add cropping circle
        global_radius = analysis_results['global_radius']
        center_y, center_x = analysis_results['original_shape'][0] // 2, analysis_results['original_shape'][1] // 2
        circle = Circle((center_x, center_y), global_radius, fill=False, color='cyan', linewidth=2)
        axes[0, 0].add_patch(circle)
        
        # Cropped pattern
        im2 = axes[0, 1].imshow(avg_cropped, cmap='hot', origin='lower')
        axes[0, 1].set_title('Average Cropped Pattern')
        axes[0, 1].set_xlabel('X (pixels)')
        axes[0, 1].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Difference
        # Resize cropped to match original for comparison
        from scipy.ndimage import zoom
        zoom_factor = avg_original.shape[0] / avg_cropped.shape[0]
        cropped_resized = zoom(avg_cropped, zoom_factor, order=1)
        
        # Ensure same size
        if cropped_resized.shape != avg_original.shape:
            min_size = min(avg_original.shape[0], cropped_resized.shape[0])
            avg_original_trimmed = avg_original[:min_size, :min_size]
            cropped_resized_trimmed = cropped_resized[:min_size, :min_size]
        else:
            avg_original_trimmed = avg_original
            cropped_resized_trimmed = cropped_resized
        
        difference = avg_original_trimmed - cropped_resized_trimmed
        im3 = axes[0, 2].imshow(difference, cmap='RdBu', origin='lower')
        axes[0, 2].set_title('Difference (Original - Cropped)')
        axes[0, 2].set_xlabel('X (pixels)')
        axes[0, 2].set_ylabel('Y (pixels)')
        plt.colorbar(im3, ax=axes[0, 2])
        
        # Retention statistics
        if 'sample_profiles' in analysis_results:
            for i, profile in enumerate(analysis_results['sample_profiles'][:5]):
                axes[1, 0].plot(profile['radii'], profile['cumulative'], 
                               alpha=0.7, label=f'Pattern {i+1}')
            axes[1, 0].axhline(y=self.config.target_retention, color='red', linestyle='--', 
                              label=f'Target ({self.config.target_retention:.2f})')
            axes[1, 0].set_xlabel('Radius (pixels)')
            axes[1, 0].set_ylabel('Cumulative Intensity Fraction')
            axes[1, 0].set_title('Sample Cumulative Intensity Profiles')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        
        # Retention radius histogram
        retention_radii = analysis_results['retention_radii']
        axes[1, 1].hist(retention_radii, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(x=global_radius, color='red', linestyle='--', 
                          label=f'Global Radius ({global_radius})')
        axes[1, 1].set_xlabel('Retention Radius (pixels)')
        axes[1, 1].set_ylabel('Frequency')
        axes[1, 1].set_title('Distribution of Retention Radii')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        # Summary statistics
        stats_text = f"""
        Analysis Summary:
        • Patterns processed: {analysis_results['n_patterns']:,}
        • Original size: {analysis_results['original_shape'][0]}×{analysis_results['original_shape'][1]}
        • Cropped size: {analysis_results['cropped_shape'][0]}×{analysis_results['cropped_shape'][1]}
        • Global radius: {global_radius} pixels
        • Size reduction: {(1 - np.prod(analysis_results['cropped_shape'])/np.prod(analysis_results['original_shape'])):.1%}
        
        Retention Radius Statistics:
        • Mean: {analysis_results['retention_stats']['mean']:.1f} pixels
        • Std: {analysis_results['retention_stats']['std']:.1f} pixels
        • Range: [{analysis_results['retention_stats']['min']}-{analysis_results['retention_stats']['max']}] pixels
        • Median: {analysis_results['retention_stats']['median']:.1f} pixels
        """
        
        axes[1, 2].text(0.05, 0.95, stats_text, transform=axes[1, 2].transAxes, 
                        verticalalignment='top', fontfamily='monospace', fontsize=10)
        axes[1, 2].set_xlim(0, 1)
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Summary Statistics')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            if self.config.verbose:
                print(f"Visualization saved to {save_path}")
        
        plt.show()
    
    def process_dataset(self, data: torch.Tensor, 
                       save_path: Optional[Path] = None) -> Tuple[torch.Tensor, Dict[str, Any]]:
        """
        Complete processing pipeline: analyze, crop, and validate.
        
        Args:
            data: Input tensor data
            save_path: Optional path to save results
            
        Returns:
            Tuple of (cropped_data, results_dict)
        """
        start_time = time.time()
        
        # Analysis pass
        analysis_results = self.analyze_patterns(data)
        
        # Cropping pass
        cropped_data = self.crop_patterns(data, analysis_results)
        
        # Validation
        validation_results = self.validate_retention(data, cropped_data, analysis_results)
        
        # Visualization
        if self.config.visualization:
            viz_path = save_path.parent / f"{save_path.stem}_visualization.png" if save_path else None
            self.visualize_results(data, cropped_data, analysis_results, viz_path)
        
        # Combine results
        results = {
            'analysis': analysis_results,
            'validation': validation_results,
            'processing_time': time.time() - start_time,
            'config': self.config
        }
        
        # Save cropped data if requested
        if save_path:
            torch.save(cropped_data, save_path)
            
            # Save results metadata
            results_path = save_path.parent / f"{save_path.stem}_results.json"
            import json
            
            # Convert numpy arrays to lists for JSON serialization
            def convert_numpy_types(obj):
                """Recursively convert numpy types to JSON-serializable types."""
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, (np.integer, np.int64, np.int32)):
                    return int(obj)
                elif isinstance(obj, (np.floating, np.float64, np.float32)):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {k: convert_numpy_types(v) for k, v in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                elif isinstance(obj, tuple):
                    return tuple(convert_numpy_types(item) for item in obj)
                elif hasattr(obj, '__dict__'):
                    # Handle objects with __dict__ (like CroppingConfig)
                    return convert_numpy_types(obj.__dict__)
                else:
                    return obj
            
            json_results = convert_numpy_types(results)
            
            with open(results_path, 'w') as f:
                json.dump(json_results, f, indent=2)
            
            if self.config.verbose:
                print(f"Results saved to {save_path}")
                print(f"Metadata saved to {results_path}")
        
        return cropped_data, results


def create_test_data(n_patterns: int = 100, 
                    image_size: int = 256,
                    noise_level: float = 0.1) -> torch.Tensor:
    """
    Create synthetic test data with Airy disk patterns.
    
    Args:
        n_patterns: Number of patterns to generate
        image_size: Size of each pattern
        noise_level: Amount of noise to add
        
    Returns:
        Tensor of synthetic diffraction patterns
    """
    patterns = []
    
    for i in range(n_patterns):
        # Create coordinate grids
        x = np.linspace(-1, 1, image_size)
        y = np.linspace(-1, 1, image_size)
        X, Y = np.meshgrid(x, y)
        
        # Add small random center shift
        center_shift = np.random.normal(0, 0.1, 2)
        X += center_shift[0]
        Y += center_shift[1]
        
        # Create Airy disk pattern
        r = np.sqrt(X**2 + Y**2)
        r[r == 0] = 1e-10  # Avoid division by zero
        
        # Airy disk formula: (2*J1(r)/r)^2 where J1 is first-order Bessel function
        from scipy.special import j1
        airy_radius = 0.3 + np.random.normal(0, 0.05)  # Slight variation
        airy_pattern = (2 * j1(np.pi * r / airy_radius) / (np.pi * r / airy_radius))**2
        
        # Add noise
        noise = np.random.normal(0, noise_level, airy_pattern.shape)
        pattern = airy_pattern + noise
        pattern[pattern < 0] = 0  # Ensure non-negative
        
        patterns.append(pattern)
    
    return torch.tensor(np.array(patterns), dtype=torch.float32)[:, None, :, :]


if __name__ == "__main__":
    # Demo with synthetic data
    print("Creating synthetic test data...")
    test_data = create_test_data(n_patterns=1000, image_size=256)
    
    print("Testing diffraction pattern cropping...")
    config = CroppingConfig(
        target_retention=0.98,
        margin_pixels=3,
        chunk_size=200,
        use_gpu=False,
        verbose=True
    )
    
    cropper = DiffractionCropper(config)
    cropped_data, results = cropper.process_dataset(test_data)
    
    print(f"Processing complete!")
    print(f"Original shape: {test_data.shape}")
    print(f"Cropped shape: {cropped_data.shape}")
    print(f"Size reduction: {(1 - cropped_data.numel()/test_data.numel()):.1%}")