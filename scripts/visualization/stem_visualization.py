#!/usr/bin/env python3
"""
STEM Visualization Module

Enhanced visualization tools for 4D-STEM data including:
- Virtual bright/dark field imaging
- Diffraction pattern analysis
- Support for .dm4 files
- Interactive visualization capabilities
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Tuple, Optional, Union
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hyperspy.api as hs
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks

# py4DSTEM imports
try:
    import py4DSTEM
    from py4DSTEM.process.calibration.origin import get_origin_single_dp
    from py4DSTEM.process.calibration.probe import get_probe_size
    from py4DSTEM.datacube import DataCube
    PY4DSTEM_AVAILABLE = True
except ImportError:
    PY4DSTEM_AVAILABLE = False
    print("Warning: py4DSTEM not available. Using fallback implementations.")


class STEMVisualizer:
    """Enhanced STEM data visualization with interactive capabilities."""
    
    def __init__(self, data: Union[np.ndarray, torch.Tensor], scan_shape: Optional[Tuple[int, int]] = None, 
                 scalebar_info: Optional[dict] = None):
        """
        Initialize STEM visualizer.
        
        Args:
            data: Diffraction patterns (N, H, W) or (N, C, H, W)
            scan_shape: Shape of the scan grid (scan_y, scan_x)
            scalebar_info: Dict with keys 'width', 'scale_length', 'units'
        """
        self.data = self._prepare_data(data)
        self.scan_shape = scan_shape or self._infer_scan_shape()
        self.pattern_shape = self.data.shape[-2:]
        self.center = (self.pattern_shape[0] // 2, self.pattern_shape[1] // 2)
        
        # Find direct beam position
        self.direct_beam_position = self._find_direct_beam_position()
        
        # Detect Bragg spot characteristics
        self.bragg_radius = self._detect_bragg_spot_radius()
        
        # Default field regions based on Bragg spot analysis
        self.bright_field_region = self._default_bright_field_region()
        self.dark_field_region = self._default_dark_field_region()
        self.haadf_region = self._default_haadf_region()
        
        # Scalebar information
        self.scalebar_info = scalebar_info or {
            'width': max(self.scan_shape) * 10,  # Default: 10nm per pixel
            'scale_length': 100,  # Default: 100nm scalebar
            'units': 'nm'
        }
        
        # Create py4DSTEM DataCube if available
        self._py4dstem_datacube = None
        if PY4DSTEM_AVAILABLE:
            self._create_py4dstem_datacube()
        
    def _prepare_data(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input data to numpy array with proper shape and preprocess."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Handle batch and channel dimensions
        if data.ndim == 4:
            data = data.squeeze(1)  # Remove channel dimension
        
        # Convert to float32 if needed for scipy compatibility
        original_dtype = data.dtype
        if data.dtype in [np.float16, np.int8, np.int16, np.uint8, np.uint16]:
            print(f"Converting data from {original_dtype} to float32 for processing compatibility")
            data = data.astype(np.float32)
            self.original_dtype = original_dtype
        else:
            self.original_dtype = data.dtype
        
        # Enhanced preprocessing for better virtual imaging
        # 1. Handle negative values from reconstructions
        data_positive = np.maximum(data, 0)
        
        # 2. Remove extreme outliers (hot pixels) using percentile clipping
        p99 = np.percentile(data_positive, 99.9)
        data_clipped = np.minimum(data_positive, p99)
        
        # 3. Apply median filter to reduce hot pixel artifacts
        from scipy.ndimage import median_filter
        data_filtered = np.zeros_like(data_clipped)
        for i in range(data_clipped.shape[0]):
            data_filtered[i] = median_filter(data_clipped[i], size=3)
        
        # Store both raw and processed data
        self.data_raw = data.copy()  # Keep original for fallback
        self.log_data = np.log(data_filtered + 1)  # Log-transformed for visualization
        
        return data_filtered
    
    def _create_py4dstem_datacube(self):
        """Create py4DSTEM DataCube from the data."""
        try:
            # Reshape data to match py4DSTEM expectations (R_Nx, R_Ny, Q_Nx, Q_Ny)
            R_Nx, R_Ny = self.scan_shape
            Q_Nx, Q_Ny = self.pattern_shape
            
            # Reshape from (N, H, W) to (R_Nx, R_Ny, Q_Nx, Q_Ny)
            data_4d = self.data.reshape(R_Nx, R_Ny, Q_Nx, Q_Ny)
            
            # Create DataCube
            self._py4dstem_datacube = DataCube(data=data_4d)
            
            # Initialize calibration
            self._py4dstem_datacube.calibration = py4DSTEM.Calibration()
            
        except Exception as e:
            print(f"Warning: Failed to create py4DSTEM DataCube: {e}")
            self._py4dstem_datacube = None
    
    def _infer_scan_shape(self) -> Tuple[int, int]:
        """Infer scan shape from data dimensions."""
        n_patterns = self.data.shape[0]
        # Try to find square or rectangular scan
        factors = []
        for i in range(1, int(np.sqrt(n_patterns)) + 1):
            if n_patterns % i == 0:
                factors.append((i, n_patterns // i))
        # Choose the most square-like factor pair
        return min(factors, key=lambda x: abs(x[0] - x[1]))
    
    def _find_direct_beam_position(self) -> Tuple[int, int]:
        """Find the direct beam position optimized for bullseye/Bragg patterns."""
        # Calculate mean diffraction pattern for beam position detection
        # Use positive values only to handle reconstructed data with negatives
        mean_dp = np.mean(np.maximum(self.data, 0), axis=0)
        
        if PY4DSTEM_AVAILABLE:
            try:
                # Use py4DSTEM's optimized origin finding
                # First get probe size
                r, _, _ = get_probe_size(mean_dp)
                
                # Find origin using py4DSTEM
                qx0, qy0 = get_origin_single_dp(mean_dp, r)
                
                return int(round(qx0)), int(round(qy0))
                
            except Exception as e:
                print(f"Warning: py4DSTEM origin detection failed ({e}), using fallback")
        
        # Enhanced fallback implementation for bullseye patterns
        # Use radial symmetry analysis to find the center
        min_size = min(self.pattern_shape)
        
        # First pass: rough center estimation using maximum intensity
        rough_center_y, rough_center_x = np.unravel_index(np.argmax(mean_dp), mean_dp.shape)
        
        # Second pass: refine using radial symmetry
        # Test a small grid around the rough center
        search_radius = min_size // 20
        best_center = (rough_center_y, rough_center_x)
        best_symmetry_score = 0
        
        y_indices, x_indices = np.indices(mean_dp.shape)
        
        for dy in range(-search_radius, search_radius + 1, 2):
            for dx in range(-search_radius, search_radius + 1, 2):
                test_y = rough_center_y + dy
                test_x = rough_center_x + dx
                
                # Check bounds
                if test_y < 0 or test_y >= mean_dp.shape[0] or test_x < 0 or test_x >= mean_dp.shape[1]:
                    continue
                
                # Calculate radial distances from this test center
                distances = np.hypot(x_indices - test_x, y_indices - test_y)
                
                # Test symmetry by comparing opposite quadrants
                symmetry_score = self._calculate_radial_symmetry(mean_dp, test_y, test_x, distances)
                
                if symmetry_score > best_symmetry_score:
                    best_symmetry_score = symmetry_score
                    best_center = (test_y, test_x)
        
        center_y, center_x = best_center
        
        # Final refinement using center of mass on the central peak
        refined_r = min_size // 30  # Small radius for center refinement
        mask = np.hypot(x_indices - center_x, y_indices - center_y) <= refined_r
        
        # Center of mass calculation on the central region
        total_intensity = np.sum(mean_dp * mask)
        if total_intensity > 0:
            x_com = np.sum(x_indices * mean_dp * mask) / total_intensity
            y_com = np.sum(y_indices * mean_dp * mask) / total_intensity
            
            # Ensure the refined center is reasonable
            if abs(x_com - center_x) < refined_r and abs(y_com - center_y) < refined_r:
                return int(round(y_com)), int(round(x_com))
        
        return center_y, center_x
    
    def _calculate_radial_symmetry(self, pattern: np.ndarray, center_y: int, center_x: int, distances: np.ndarray) -> float:
        """Calculate radial symmetry score for a potential beam center."""
        # Define annular regions for symmetry testing
        min_radius = 5
        max_radius = min(pattern.shape) // 4
        n_rings = 8
        
        symmetry_scores = []
        
        for r in np.linspace(min_radius, max_radius, n_rings):
            # Create annular mask
            inner_r = r - 2
            outer_r = r + 2
            ring_mask = (distances >= inner_r) & (distances <= outer_r)
            
            if np.sum(ring_mask) < 10:  # Skip if too few pixels
                continue
                
            # Get intensities in this ring
            ring_intensities = pattern[ring_mask]
            
            # Calculate angular positions
            y_coords, x_coords = np.where(ring_mask)
            angles = np.arctan2(y_coords - center_y, x_coords - center_x)
            
            # Sort by angle
            sorted_indices = np.argsort(angles)
            sorted_intensities = ring_intensities[sorted_indices]
            
            # Calculate correlation with shifted version (tests for rotational symmetry)
            if len(sorted_intensities) > 8:
                shift = len(sorted_intensities) // 4  # 90 degree rotation
                correlation = np.corrcoef(sorted_intensities[:-shift], sorted_intensities[shift:])[0, 1]
                if not np.isnan(correlation):
                    symmetry_scores.append(abs(correlation))
        
        return np.mean(symmetry_scores) if symmetry_scores else 0.0
    
    def _detect_bragg_spot_radius(self) -> float:
        """Detect the typical distance to the first Bragg spots from the beam center."""
        mean_dp = np.mean(np.maximum(self.data, 0), axis=0)
        center_y, center_x = self.direct_beam_position
        
        # Create radial profile
        y_indices, x_indices = np.indices(mean_dp.shape)
        distances = np.hypot(x_indices - center_x, y_indices - center_y)
        
        # Calculate radial average
        max_radius = min(mean_dp.shape) // 2
        radii = np.arange(1, max_radius)
        radial_profile = []
        
        for r in radii:
            ring_mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if np.sum(ring_mask) > 0:
                radial_profile.append(np.mean(mean_dp[ring_mask]))
            else:
                radial_profile.append(0)
        
        radial_profile = np.array(radial_profile)
        
        # Find peaks in the radial profile (excluding the central peak)
        peaks, properties = find_peaks(radial_profile, height=np.max(radial_profile) * 0.1, distance=5)
        
        if len(peaks) > 0:
            # Return the radius of the first significant peak
            first_bragg_radius = radii[peaks[0]]
            print(f"Detected first Bragg spot at radius: {first_bragg_radius:.1f} pixels")
            return float(first_bragg_radius)
        else:
            # Fallback estimate
            fallback_radius = min(mean_dp.shape) // 8
            print(f"No clear Bragg peaks detected, using fallback radius: {fallback_radius}")
            return float(fallback_radius)
    
    def _default_bright_field_region(self) -> Tuple[int, int, int, int]:
        """Default bright field region centered on direct beam (kept for compatibility)."""
        center_y, center_x = self.direct_beam_position
        region_size = min(self.pattern_shape) // 16  # Conservative size
        return (center_y - region_size, center_y + region_size,
                center_x - region_size, center_x + region_size)
    
    def create_bright_field_image(self, radius: Optional[float] = None) -> np.ndarray:
        """
        Create bright field image using py4DSTEM if available, fallback to custom implementation.
        
        Args:
            radius: Radius of circular bright field detector. If None, uses Bragg-informed default.
        
        Returns:
            Bright field image (scan_y, scan_x)
        """
        if PY4DSTEM_AVAILABLE and self._py4dstem_datacube is not None:
            try:
                center_y, center_x = self.direct_beam_position
                
                if radius is None:
                    radius = self.bragg_radius * 0.8  # Stop before Bragg spots
                
                # Use py4DSTEM's virtual imaging
                virtual_image = self._py4dstem_datacube.get_virtual_image(
                    mode='circle',
                    geometry=((center_x, center_y), radius),
                    centered=False,
                    calibrated=False,
                    verbose=False
                )
                
                return virtual_image.data
                
            except Exception as e:
                print(f"Warning: py4DSTEM bright field imaging failed ({e}), using fallback")
        
        # Fallback implementation (original custom approach)
        center_y, center_x = self.direct_beam_position
        
        if radius is None:
            radius = self.bragg_radius * 0.8  # Stop before Bragg spots
        
        # Create circular mask
        y_indices, x_indices = np.indices(self.pattern_shape)
        mask = np.hypot(x_indices - center_x, y_indices - center_y) <= radius
        
        # Apply mask to all diffraction patterns and sum
        virtual_image = np.zeros(self.scan_shape)
        
        # Ensure proper scan coordinate mapping
        for i in range(self.data.shape[0]):
            # Convert linear index to 2D scan coordinates
            scan_y = i // self.scan_shape[1]  
            scan_x = i % self.scan_shape[1]
            virtual_image[scan_y, scan_x] = np.sum(self.data[i] * mask)
        
        return virtual_image
    
    def _default_dark_field_region(self) -> Tuple[int, int, float, float]:
        """Default conventional dark field region based on Bragg spot analysis."""
        center_y, center_x = self.direct_beam_position
        # Conventional DF: sample around the Bragg spots
        inner_radius = self.bragg_radius * 0.9  # Just inside Bragg spots
        outer_radius = self.bragg_radius * 1.1  # Just outside Bragg spots
        return (center_y, center_x, inner_radius, outer_radius)
    
    def _default_haadf_region(self) -> Tuple[int, int, float, float]:
        """Default HAADF region for high-angle incoherent scattering."""
        center_y, center_x = self.direct_beam_position
        # HAADF: high-angle region beyond coherent Bragg scattering
        inner_radius = self.bragg_radius * 1.5  # Well beyond Bragg spots
        outer_radius = min(self.pattern_shape) // 3  # Extend to detector edge
        return (center_y, center_x, inner_radius, outer_radius)
    
    def set_bright_field_region(self, region: Tuple[int, int, int, int]):
        """Set bright field region (y_min, y_max, x_min, x_max)."""
        self.bright_field_region = region
    
    def set_dark_field_region(self, region: Union[Tuple[int, int, int, int], Tuple[int, int, float, float]]):
        """
        Set dark field region. Supports both formats for backward compatibility:
        - Old format: (y_min, y_max, x_min, x_max) - rectangular region
        - New format: (center_y, center_x, inner_radius, outer_radius) - annular region
        """
        self.dark_field_region = region
        
    def set_dark_field_annular(self, center_y: int, center_x: int, inner_radius: float, outer_radius: float):
        """Set dark field annular detector parameters."""
        self.dark_field_region = (center_y, center_x, inner_radius, outer_radius)
    
    def create_annular_mask(self, center_y: int, center_x: int, inner_radius: float, outer_radius: float) -> np.ndarray:
        """
        Create annular (ring-shaped) mask for dark field imaging.
        
        Args:
            center_y, center_x: Center position of the annulus
            inner_radius: Inner radius (excludes direct beam)
            outer_radius: Outer radius
            
        Returns:
            Boolean mask with True for pixels in the annular region
        """
        y_indices, x_indices = np.indices(self.pattern_shape)
        distances = np.hypot(x_indices - center_x, y_indices - center_y)
        
        # Annular mask: pixels between inner and outer radius
        mask = (distances >= inner_radius) & (distances <= outer_radius)
        return mask
    
    def create_dark_field_image(self, inner_radius: Optional[float] = None, outer_radius: Optional[float] = None) -> np.ndarray:
        """
        Create dark field image using annular mask to exclude direct beam.
        
        Args:
            inner_radius: Inner radius of annular detector (excludes direct beam)
            outer_radius: Outer radius of annular detector
        
        Returns:
            Dark field image (scan_y, scan_x)
        """
        center_y, center_x = self.direct_beam_position
        
        # Default radii based on pattern size
        if inner_radius is None:
            inner_radius = min(self.pattern_shape) // 25  # Exclude direct beam
        if outer_radius is None:
            outer_radius = min(self.pattern_shape) // 6   # Collect scattered electrons
            
        if PY4DSTEM_AVAILABLE and self._py4dstem_datacube is not None:
            try:
                # Use py4DSTEM's annular virtual imaging
                virtual_image = self._py4dstem_datacube.get_virtual_image(
                    mode='annular',
                    geometry=((center_x, center_y), (inner_radius, outer_radius)),
                    centered=False,
                    calibrated=False,
                    verbose=False
                )
                
                return virtual_image.data
                
            except Exception as e:
                print(f"Warning: py4DSTEM annular imaging failed ({e}), using fallback")
        
        # Fallback implementation using custom annular mask
        mask = self.create_annular_mask(center_y, center_x, inner_radius, outer_radius)
        
        # Apply mask to all diffraction patterns and sum
        virtual_image = np.zeros(self.scan_shape)
        
        # Ensure proper scan coordinate mapping
        for i in range(self.data.shape[0]):
            # Convert linear index to 2D scan coordinates
            scan_y = i // self.scan_shape[1]  
            scan_x = i % self.scan_shape[1]
            virtual_image[scan_y, scan_x] = np.sum(self.data[i] * mask)
        
        return virtual_image
    
    def create_haadf_image(self, inner_radius: Optional[float] = None, outer_radius: Optional[float] = None) -> np.ndarray:
        """
        Create High-Angle Annular Dark Field (HAADF) image for Z-contrast imaging.
        
        Args:
            inner_radius: Inner radius of HAADF detector (well beyond Bragg spots)
            outer_radius: Outer radius of HAADF detector (extends to detector edge)
        
        Returns:
            HAADF image (scan_y, scan_x)
        """
        center_y, center_x = self.direct_beam_position
        
        # HAADF defaults: high-angle incoherent scattering region
        if inner_radius is None:
            inner_radius = self.bragg_radius * 1.5  # Well beyond Bragg spots
        if outer_radius is None:
            outer_radius = min(self.pattern_shape) // 3  # Extend toward detector edge
            
        if PY4DSTEM_AVAILABLE and self._py4dstem_datacube is not None:
            try:
                # Use py4DSTEM's annular virtual imaging
                virtual_image = self._py4dstem_datacube.get_virtual_image(
                    mode='annular',
                    geometry=((center_x, center_y), (inner_radius, outer_radius)),
                    centered=False,
                    calibrated=False,
                    verbose=False
                )
                
                return virtual_image.data
                
            except Exception as e:
                print(f"Warning: py4DSTEM HAADF imaging failed ({e}), using fallback")
        
        # Fallback implementation using custom annular mask
        mask = self.create_annular_mask(center_y, center_x, inner_radius, outer_radius)
        
        # Apply mask to all diffraction patterns and sum
        virtual_image = np.zeros(self.scan_shape)
        
        # Ensure proper scan coordinate mapping
        for i in range(self.data.shape[0]):
            # Convert linear index to 2D scan coordinates
            scan_y = i // self.scan_shape[1]  
            scan_x = i % self.scan_shape[1]
            virtual_image[scan_y, scan_x] = np.sum(self.data[i] * mask)
        
        return virtual_image
    
    def create_virtual_field_image(self, field_region: Tuple[int, int, int, int]) -> np.ndarray:
        """
        Create virtual field image by integrating over specified rectangular region.
        
        Args:
            field_region: Region for integration (y_min, y_max, x_min, x_max)
        
        Returns:
            Virtual field image (scan_y, scan_x)
        """
        if PY4DSTEM_AVAILABLE and self._py4dstem_datacube is not None:
            try:
                y_min, y_max, x_min, x_max = field_region
                
                # Use py4DSTEM's rectangular virtual imaging
                virtual_image = self._py4dstem_datacube.get_virtual_image(
                    mode='rectangle',
                    geometry=(x_min, x_max, y_min, y_max),
                    centered=False,
                    calibrated=False,
                    verbose=False
                )
                
                return virtual_image.data
                
            except Exception as e:
                print(f"Warning: py4DSTEM virtual field imaging failed ({e}), using fallback")
        
        # Fallback implementation (original custom approach)
        y_min, y_max, x_min, x_max = field_region
        
        # Extract the field region from raw data and sum intensities
        field_data = self.data[:, y_min:y_max, x_min:x_max]
        virtual_image = np.sum(field_data.reshape(self.data.shape[0], -1), axis=1)
        
        # Reshape to scan dimensions
        virtual_image = virtual_image.reshape(self.scan_shape)
        
        return virtual_image
    
    def _add_scalebar(self, ax, pattern_scale=False):
        """Add scalebar to the plot following m3_learning style."""
        if pattern_scale:
            # For diffraction patterns - typically in reciprocal space
            scale_size = 20  # Example: 20 pixels
            scale_length = 1.0  # Example: 1.0 nm⁻¹
            units = "nm⁻¹"
            image_size = max(self.pattern_shape)
        else:
            # For real space images
            scale_size = self.scalebar_info['scale_length']
            scale_length = scale_size
            units = self.scalebar_info['units']
            image_size = self.scalebar_info['width']
        
        # Get axis limits
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        x_size = int(np.abs(x_lim[1] - x_lim[0]))
        y_size = int(np.abs(y_lim[1] - y_lim[0]))
        
        # Calculate scalebar position (bottom right)
        fract = scale_size / image_size
        x_start = x_lim[0] + 0.9 * x_size
        x_end = x_lim[0] + (0.9 - fract) * x_size
        y_start = y_lim[0] + 0.1 * y_size
        y_end = y_lim[0] + 0.125 * y_size
        y_label = y_lim[0] + 0.175 * y_size
        
        # Draw scalebar
        ax.plot([x_end, x_start], [y_start, y_start], 'w-', linewidth=3)
        ax.plot([x_end, x_start], [y_start, y_start], 'k-', linewidth=1)
        ax.plot([x_end, x_end], [y_start, y_end], 'w-', linewidth=3)
        ax.plot([x_end, x_end], [y_start, y_end], 'k-', linewidth=1)
        ax.plot([x_start, x_start], [y_start, y_end], 'w-', linewidth=3)
        ax.plot([x_start, x_start], [y_start, y_end], 'k-', linewidth=1)
        
        # Add text label
        ax.text((x_start + x_end) / 2, y_label, f"{scale_length} {units}",
                ha='center', va='bottom', color='white', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    def plot_mean_diffraction_pattern(self, ax=None, show_regions=True, use_log=True, add_scalebar=False):
        """Plot mean diffraction pattern with field regions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use log data for visualization
        if use_log and hasattr(self, 'log_data'):
            mean_pattern = np.mean(self.log_data, axis=0)
        else:
            mean_pattern = np.mean(self.data, axis=0)
            
        im = ax.imshow(mean_pattern, cmap='viridis')
        ax.set_title('Mean Diffraction Pattern')
        ax.axis('off')
        
        if show_regions:
            # Mark direct beam position and Bragg radius
            y_beam, x_beam = self.direct_beam_position
            ax.plot(x_beam, y_beam, 'w+', markersize=15, markeredgewidth=3)
            ax.plot(x_beam, y_beam, 'k+', markersize=12, markeredgewidth=2)
            
            # Show Bragg spot radius for reference
            from matplotlib.patches import Circle
            bragg_circle = Circle((x_beam, y_beam), self.bragg_radius,
                                fill=False, edgecolor='yellow', linewidth=1, linestyle=':')
            ax.add_patch(bragg_circle)
            ax.text(x_beam + self.bragg_radius + 5, y_beam - self.bragg_radius/2, 
                   f'Bragg\nr={self.bragg_radius:.1f}', color='yellow', fontsize=8, weight='bold')
            
            # Bright field region (circular)
            bf_radius = self.bragg_radius * 0.8  # Default BF radius
            bf_circle = Circle((x_beam, y_beam), bf_radius,
                             fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(bf_circle)
            ax.text(x_beam - bf_radius - 15, y_beam, 'BF', color='red', fontsize=12, weight='bold')
            
            # Conventional Dark field region (annular around Bragg spots)
            center_y, center_x, inner_radius, outer_radius = self.dark_field_region
            inner_circle = Circle((center_x, center_y), inner_radius, 
                                fill=False, edgecolor='blue', linewidth=2, linestyle='--')
            outer_circle = Circle((center_x, center_y), outer_radius,
                                fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)
            ax.text(center_x + outer_radius + 5, center_y + outer_radius/2, 'DF', 
                   color='blue', fontsize=12, weight='bold')
            
            # HAADF region (high-angle annular)
            haadf_center_y, haadf_center_x, haadf_inner, haadf_outer = self.haadf_region
            haadf_inner_circle = Circle((haadf_center_x, haadf_center_y), haadf_inner,
                                      fill=False, edgecolor='orange', linewidth=2, linestyle='--')
            haadf_outer_circle = Circle((haadf_center_x, haadf_center_y), haadf_outer,
                                      fill=False, edgecolor='orange', linewidth=2)
            ax.add_patch(haadf_inner_circle)
            ax.add_patch(haadf_outer_circle)
            ax.text(haadf_center_x + haadf_outer + 5, haadf_center_y - haadf_outer/2, 'HAADF',
                   color='orange', fontsize=12, weight='bold')
        
        if add_scalebar:
            self._add_scalebar(ax, pattern_scale=True)
        
        return im
    
    def plot_virtual_images(self, figsize=(12, 4)):
        """Plot virtual bright and dark field images."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Mean diffraction pattern (log scale)
        self.plot_mean_diffraction_pattern(axes[0], use_log=True)
        axes[0].set_title('Mean Diffraction Pattern (Log)')
        
        # Bright field image (using circular mask)
        bf_image = self.create_bright_field_image()
        im1 = axes[1].imshow(bf_image, cmap='gray')
        axes[1].set_title('Virtual Bright Field')
        axes[1].axis('off')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # Dark field image - use proper annular detector
        if isinstance(self.dark_field_region[2], float):
            # New annular format
            center_y, center_x, inner_radius, outer_radius = self.dark_field_region
            df_image = self.create_dark_field_image(inner_radius, outer_radius)
        else:
            # Old rectangular format for backward compatibility
            df_image = self.create_virtual_field_image(self.dark_field_region)
            
        im2 = axes[2].imshow(df_image, cmap='hot')
        axes[2].set_title('Virtual Dark Field (Annular)')
        axes[2].axis('off')
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        plt.tight_layout()
        return fig
    
    def interactive_region_selector(self):
        """Interactive region selector for virtual field imaging."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mean_pattern = np.mean(self.data, axis=0)
        im = ax.imshow(mean_pattern, cmap='viridis')
        ax.set_title('Select Virtual Field Regions (Click and drag)')
        
        # Store selected regions
        self.selected_regions = []
        
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Ensure proper order
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            region = (y_min, y_max, x_min, x_max)
            self.selected_regions.append(region)
            
            print(f"Selected region: {region}")
            print(f"To use: visualizer.set_bright_field_region({region})")
        
        selector = RectangleSelector(ax, onselect, useblit=True,
                                   button=[1], minspanx=5, minspany=5,
                                   spancoords='pixels', interactive=True)
        
        plt.colorbar(im)
        plt.show()
        
        return selector
    
    def save_complete_visualization(self, output_path: str, comparison_data: Optional[np.ndarray] = None):
        """Save complete STEM visualization with all components."""
        # Determine subplot configuration
        n_plots = 3  # Mean pattern + BF + DF
        if comparison_data is not None:
            n_plots += 3  # + reconstructed versions
        
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        # Ensure axes is 2D
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        ax_idx = 0
        
        # Original data visualizations
        self.plot_mean_diffraction_pattern(axes[ax_idx // cols, ax_idx % cols], use_log=True)
        axes[ax_idx // cols, ax_idx % cols].set_title('Mean Diffraction Pattern (Log)')
        ax_idx += 1
        
        # Bright field (using circular mask)
        bf_image = self.create_bright_field_image()
        im = axes[ax_idx // cols, ax_idx % cols].imshow(bf_image, cmap='gray')
        axes[ax_idx // cols, ax_idx % cols].set_title('Virtual Bright Field')
        axes[ax_idx // cols, ax_idx % cols].axis('off')
        divider = make_axes_locatable(axes[ax_idx // cols, ax_idx % cols])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        # Dark field
        df_image = self.create_virtual_field_image(self.dark_field_region)
        im = axes[ax_idx // cols, ax_idx % cols].imshow(df_image, cmap='hot')
        axes[ax_idx // cols, ax_idx % cols].set_title('Virtual Dark Field')
        axes[ax_idx // cols, ax_idx % cols].axis('off')
        divider = make_axes_locatable(axes[ax_idx // cols, ax_idx % cols])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        # Comparison data if provided
        if comparison_data is not None:
            comp_viz = STEMVisualizer(comparison_data, self.scan_shape)
            comp_viz.bright_field_region = self.bright_field_region
            comp_viz.dark_field_region = self.dark_field_region
            
            # Reconstructed mean pattern
            comp_viz.plot_mean_diffraction_pattern(axes[ax_idx // cols, ax_idx % cols], show_regions=False)
            axes[ax_idx // cols, ax_idx % cols].set_title('Reconstructed Mean Pattern')
            ax_idx += 1
            
            # Reconstructed bright field (using circular mask)
            comp_bf = comp_viz.create_bright_field_image()
            axes[ax_idx // cols, ax_idx % cols].imshow(comp_bf, cmap='gray')
            axes[ax_idx // cols, ax_idx % cols].set_title('Reconstructed Bright Field')
            axes[ax_idx // cols, ax_idx % cols].axis('off')
            ax_idx += 1
            
            # Reconstructed dark field
            comp_df = comp_viz.create_virtual_field_image(self.dark_field_region)
            axes[ax_idx // cols, ax_idx % cols].imshow(comp_df, cmap='hot')
            axes[ax_idx // cols, ax_idx % cols].set_title('Reconstructed Dark Field')
            axes[ax_idx // cols, ax_idx % cols].axis('off')
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, rows * cols):
            axes[i // cols, i % cols].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"STEM visualization saved to: {output_path}")


def load_dm4_data(filepath: str) -> Tuple[np.ndarray, dict, Tuple[int, int]]:
    """
    Load 4D-STEM data from .dm4 file following m3_learning approach.
    
    Args:
        filepath: Path to .dm4 file
    
    Returns:
        Tuple of (data_array, metadata, scan_shape)
    """
    # Load using hyperspy
    signal = hs.load(filepath)
    
    # Extract data and metadata
    data = signal.data
    metadata = signal.metadata.as_dictionary()
    
    # Reshape if needed - 4D STEM data should be (scan_y, scan_x, det_y, det_x)
    if data.ndim == 4:
        scan_y, scan_x, det_y, det_x = data.shape
        # Reshape to (N, det_y, det_x) format for processing
        data = data.reshape(scan_y * scan_x, det_y, det_x)
        scan_shape = (scan_y, scan_x)
    else:
        scan_shape = None
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Scan shape: {scan_shape}")
    print(f"Pattern shape: {data.shape[-2:]}")
    
    return data, metadata, scan_shape


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description='STEM Data Visualization Tool')
    parser.add_argument('input_file', type=str, help='Input file (.dm4, .pt, .npy)')
    parser.add_argument('--output', '-o', type=str, default='stem_visualization.png',
                       help='Output visualization file')
    parser.add_argument('--comparison', '-c', type=str, default=None,
                       help='Comparison data file (reconstructed data)')
    parser.add_argument('--scan_shape', nargs=2, type=int, default=None,
                       help='Scan shape (scan_y scan_x)')
    parser.add_argument('--bf_region', nargs=4, type=int, default=None,
                       help='Bright field region (y_min y_max x_min x_max)')
    parser.add_argument('--df_region', nargs=4, type=int, default=None,
                       help='Dark field region (y_min y_max x_min x_max)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive region selection')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_file)
    if input_path.suffix == '.dm4':
        data, metadata, scan_shape = load_dm4_data(args.input_file)
        print(f"Loaded .dm4 file with shape: {data.shape}")
        print(f"Inferred scan shape: {scan_shape}")
    elif input_path.suffix == '.pt':
        data = torch.load(args.input_file)
        scan_shape = tuple(args.scan_shape) if args.scan_shape else None
        print(f"Loaded .pt file with shape: {data.shape}")
    elif input_path.suffix == '.npy':
        data = np.load(args.input_file)
        scan_shape = tuple(args.scan_shape) if args.scan_shape else None
        print(f"Loaded .npy file with shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Create visualizer
    visualizer = STEMVisualizer(data, scan_shape)
    
    # Set custom regions if provided
    if args.bf_region:
        visualizer.set_bright_field_region(tuple(args.bf_region))
    if args.df_region:
        visualizer.set_dark_field_region(tuple(args.df_region))
    
    # Interactive mode
    if args.interactive:
        print("Starting interactive region selector...")
        print("Click and drag to select regions on the diffraction pattern")
        selector = visualizer.interactive_region_selector()
        return
    
    # Load comparison data if provided
    comparison_data = None
    if args.comparison:
        comp_path = Path(args.comparison)
        if comp_path.suffix == '.pt':
            comparison_data = torch.load(args.comparison)
        elif comp_path.suffix == '.npy':
            comparison_data = np.load(args.comparison)
        else:
            raise ValueError(f"Unsupported comparison file format: {comp_path.suffix}")
        print(f"Loaded comparison data with shape: {comparison_data.shape}")
    
    # Create main 3-panel visualization with scalebars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Mean diffraction pattern (log scale) with scalebar
    visualizer.plot_mean_diffraction_pattern(axes[0], use_log=True, add_scalebar=True)
    axes[0].set_title('Mean Diffraction Pattern (Log)')
    
    # Bright field image with scalebar (using circular mask)
    bf_image = visualizer.create_bright_field_image()
    axes[1].imshow(bf_image, cmap='gray')
    axes[1].set_title('Virtual Bright Field')
    axes[1].axis('off')
    visualizer._add_scalebar(axes[1], pattern_scale=False)
    
    # Dark field image with scalebar - use proper annular detector
    if isinstance(visualizer.dark_field_region[2], float):
        # New annular format
        center_y, center_x, inner_radius, outer_radius = visualizer.dark_field_region
        df_image = visualizer.create_dark_field_image(inner_radius, outer_radius)
    else:
        # Old rectangular format for backward compatibility
        df_image = visualizer.create_virtual_field_image(visualizer.dark_field_region)
        
    axes[2].imshow(df_image, cmap='hot')
    axes[2].set_title('Virtual Dark Field (Annular)')
    axes[2].axis('off')
    visualizer._add_scalebar(axes[2], pattern_scale=False)
    
    plt.tight_layout()
    fig.savefig(args.output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"STEM visualization saved to: {args.output}")
    
    # If comparison data provided, create comparison visualization
    if comparison_data is not None:
        visualizer.save_complete_visualization(args.output.replace('.png', '_comparison.png'), comparison_data)
        print(f"Comparison visualization saved to: {args.output.replace('.png', '_comparison.png')}")


if __name__ == "__main__":
    main()