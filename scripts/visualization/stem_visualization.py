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
        
        # Pre-calculate common arrays for efficiency
        self.y_indices, self.x_indices = np.indices(self.pattern_shape)
        
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
        """Convert input data to numpy array with minimal preprocessing."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Handle batch and channel dimensions
        if data.ndim == 4:
            data = data.squeeze(1)  # Remove channel dimension
        
        # Store original dtype - keep it unchanged for efficiency
        self.original_dtype = data.dtype
        print(f"Using original data type: {self.original_dtype}")
        
        # Minimal preprocessing - work with original dtype
        # 1. Handle negative values (works with any dtype)
        data_positive = np.maximum(data, 0)
        
        # 2. Simple outlier handling without type conversion
        # Use numpy functions that preserve dtype
        p99 = np.percentile(data_positive.astype(np.float64), 99.9)  # Only convert for percentile calc
        data_clipped = np.minimum(data_positive, p99)
        
        # 3. Skip median filter for now - it forces float64 conversion
        # Instead use a simpler approach that preserves dtype
        
        # Store processed data in original dtype
        self.data_raw = data.copy()  # Keep original for fallback
        
        return data_clipped
    
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
        # Work with original dtype, only convert for specific operations that need it
        mean_dp = np.mean(np.maximum(self.data, 0), axis=0).astype(np.float32)
        
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
        mean_dp = np.mean(np.maximum(self.data, 0), axis=0).astype(np.float32)
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
        
        # Create circular mask using cached indices
        mask = np.hypot(self.x_indices - center_x, self.y_indices - center_y) <= radius
        
        # Apply mask to all diffraction patterns and sum (vectorized - much faster!)
        # Multiply all patterns by mask at once, then sum over detector dimensions
        masked_data = self.data.astype(np.float64) * mask[np.newaxis, :, :]
        flat_intensities = np.sum(masked_data, axis=(1, 2))
        
        # Reshape to scan grid with proper coordinate mapping
        virtual_image = flat_intensities.reshape(self.scan_shape).astype(np.float32)
        
        return virtual_image
    
    def _default_dark_field_region(self) -> Tuple[int, int, float, float]:
        """Default conventional dark field region based on Bragg spot analysis."""
        center_y, center_x = self.direct_beam_position
        # Conventional DF: sample around the Bragg spots with wider range
        inner_radius = self.bragg_radius * 1.2  # Start beyond the Bragg spots
        outer_radius = self.bragg_radius * 2.0  # Wider annular region
        return (center_y, center_x, inner_radius, outer_radius)
    
    def _default_haadf_region(self) -> Tuple[int, int, float, float]:
        """Default HAADF region for high-angle incoherent scattering."""
        center_y, center_x = self.direct_beam_position
        # HAADF: high-angle region beyond coherent Bragg scattering
        inner_radius = self.bragg_radius * 2.5  # Well beyond Bragg spots
        outer_radius = min(self.pattern_shape) // 4.5  # Reduce outer radius by ~1/3
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
        # Use cached indices for efficiency
        distances = np.hypot(self.x_indices - center_x, self.y_indices - center_y)
        
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
        
        # Apply mask to all diffraction patterns and sum (vectorized - much faster!)
        # Multiply all patterns by mask at once, then sum over detector dimensions
        masked_data = self.data.astype(np.float64) * mask[np.newaxis, :, :]
        flat_intensities = np.sum(masked_data, axis=(1, 2))
        
        # Reshape to scan grid with proper coordinate mapping
        virtual_image = flat_intensities.reshape(self.scan_shape).astype(np.float32)
        
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
        
        # Apply mask to all diffraction patterns and sum (vectorized - much faster!)
        # Multiply all patterns by mask at once, then sum over detector dimensions
        masked_data = self.data.astype(np.float64) * mask[np.newaxis, :, :]
        flat_intensities = np.sum(masked_data, axis=(1, 2))
        
        # Reshape to scan grid with proper coordinate mapping
        virtual_image = flat_intensities.reshape(self.scan_shape).astype(np.float32)
        
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
    
    def create_radial_profile(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create radial intensity profile from the mean diffraction pattern.
        
        Returns:
            Tuple of (radii, radial_profile) arrays
        """
        mean_pattern = np.mean(np.maximum(self.data, 0), axis=0).astype(np.float32)
        center_y, center_x = self.direct_beam_position
        y_indices, x_indices = np.indices(mean_pattern.shape)
        distances = np.hypot(x_indices - center_x, y_indices - center_y)
        
        # Calculate radial profile
        max_radius = min(mean_pattern.shape) // 2
        radii = np.arange(1, max_radius)
        radial_profile = []
        
        for r in radii:
            ring_mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if np.sum(ring_mask) > 0:
                radial_profile.append(np.mean(mean_pattern[ring_mask]))
            else:
                radial_profile.append(0)
        
        return radii, np.array(radial_profile)
    
    def plot_radial_profile(self, ax=None) -> plt.Axes:
        """
        Plot radial intensity profile with detector region markers.
        
        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        
        radii, radial_profile = self.create_radial_profile()
        
        ax.plot(radii, radial_profile, 'b-', linewidth=2)
        ax.axvline(self.bragg_radius, color='yellow', linestyle='--', label=f'Bragg: {self.bragg_radius:.1f}')
        ax.axvline(self.bragg_radius * 0.8, color='red', linestyle=':', label='BF limit')
        ax.axvline(self.bragg_radius * 1.5, color='orange', linestyle=':', label='HAADF start')
        ax.set_xlabel('Radius (pixels)')
        ax.set_ylabel('Intensity')
        ax.set_title('Radial Profile Analysis')
        ax.legend(fontsize=8)
        ax.set_yscale('log')
        
        # Add more tick marks with numerical labels on y-axis (intensity)
        from matplotlib.ticker import LogLocator, LogFormatterMathText
        ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=10))
        ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=[2, 3, 4, 5, 6, 7, 8, 9], numticks=10))
        ax.yaxis.set_major_formatter(LogFormatterMathText(base=10.0))
        ax.grid(True, which='major', alpha=0.3)
        ax.grid(True, which='minor', alpha=0.1)
        
        return ax
    
    def create_detector_schematic(self, ax=None, size=(100, 100)) -> plt.Axes:
        """
        Create a schematic diagram showing detector geometries.
        
        Args:
            ax: Matplotlib axes to plot on. If None, creates new figure.
            size: Tuple of (width, height) for the schematic
            
        Returns:
            Matplotlib axes object
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
            
        ax.set_xlim(0, size[0])
        ax.set_ylim(0, size[1])
        ax.set_aspect('equal')
        
        # Draw detector schematic - scale relative to pattern
        center_x, center_y = size[0] // 2, size[1] // 2
        scale_factor = min(size) / min(self.pattern_shape) * 0.8  # 80% of available space
        
        # Scale detector radii to schematic size
        bf_r = self.bragg_radius * 0.8 * scale_factor
        df_inner = self.dark_field_region[2] * scale_factor
        df_outer = self.dark_field_region[3] * scale_factor
        haadf_inner = self.haadf_region[2] * scale_factor
        haadf_outer = self.haadf_region[3] * scale_factor
        
        # Create detector elements
        bf_circle = plt.Circle((center_x, center_y), bf_r, fill=True, alpha=0.3, color='red', label='BF')
        df_ring = plt.Circle((center_x, center_y), df_outer, fill=False, color='blue', linewidth=3, label='DF')
        df_inner_circle = plt.Circle((center_x, center_y), df_inner, fill=False, color='blue', linewidth=1, linestyle='--')
        haadf_ring = plt.Circle((center_x, center_y), haadf_outer, fill=False, color='orange', linewidth=3, label='HAADF')
        haadf_inner_circle = plt.Circle((center_x, center_y), haadf_inner, fill=False, color='orange', linewidth=1, linestyle='--')
        
        # Add patches to axes
        ax.add_patch(bf_circle)
        ax.add_patch(df_ring)
        ax.add_patch(df_inner_circle)
        ax.add_patch(haadf_ring)
        ax.add_patch(haadf_inner_circle)
        ax.plot(center_x, center_y, 'k+', markersize=10, markeredgewidth=2)
        
        ax.set_title('Detector Geometry')
        ax.legend(loc='upper right', fontsize=8)
        ax.axis('off')
        
        return ax
    
    def create_composite_image(self, bf_image: np.ndarray, haadf_image: np.ndarray, 
                              mode: str = 'rgb') -> np.ndarray:
        """
        Create composite visualization of multiple virtual detector images.
        
        Args:
            bf_image: Bright field image
            haadf_image: HAADF image
            mode: Composite mode ('rgb', 'overlay', 'difference')
            
        Returns:
            Composite image array
        """
        # Normalize both images for comparison
        bf_norm = (bf_image - bf_image.min()) / (bf_image.max() - bf_image.min())
        haadf_norm = (haadf_image - haadf_image.min()) / (haadf_image.max() - haadf_image.min())
        
        if mode == 'rgb':
            # Create RGB composite: BF=red, HAADF=green
            composite = np.zeros((*bf_image.shape, 3))
            composite[..., 0] = bf_norm  # Red channel = BF
            composite[..., 1] = haadf_norm  # Green channel = HAADF
            composite[..., 2] = 0.5 * (bf_norm + haadf_norm)  # Blue = average
        elif mode == 'overlay':
            # Simple alpha blending
            composite = 0.6 * bf_norm + 0.4 * haadf_norm
        elif mode == 'difference':
            # Show differences between techniques
            composite = np.abs(bf_norm - haadf_norm)
        else:
            raise ValueError(f"Unknown composite mode: {mode}")
        
        return composite
    
    def create_mean_pattern_comparison(self, figsize=(15, 5)) -> plt.Figure:
        """
        Create comparison of mean diffraction patterns (linear vs log scale).
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Mean diffraction pattern with detector regions (linear scale)
        self.plot_mean_diffraction_pattern(axes[0], show_regions=True, use_log=False, add_scalebar=True)
        axes[0].set_title('Mean Diffraction Pattern\n(With Detector Regions)')
        
        # 2. Mean pattern - log scale
        mean_pattern = np.mean(np.maximum(self.data, 0), axis=0).astype(np.float32)
        im = axes[1].imshow(np.log(mean_pattern + 1e-6), cmap='viridis')
        axes[1].set_title('Mean Pattern (Log Scale)')
        axes[1].axis('off')
        cbar = plt.colorbar(im, ax=axes[1], shrink=0.8)
        cbar.set_label('Intensity (log)', rotation=270, labelpad=15)
        
        # 3. Radial profile analysis
        self.plot_radial_profile(axes[2])
        
        plt.tight_layout()
        return fig
    
    def create_virtual_detector_suite(self, figsize=(20, 5)) -> plt.Figure:
        """
        Create visualization of all virtual detector images.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 4, figsize=figsize)
        
        # 1. Bright field image
        bf_image = self.create_bright_field_image()
        im1 = axes[0].imshow(bf_image, cmap='gray')
        axes[0].set_title('Virtual Bright Field')
        axes[0].axis('off')
        cbar1 = plt.colorbar(im1, ax=axes[0], shrink=0.8)
        cbar1.set_label('Intensity (counts)', rotation=270, labelpad=15)
        
        # 2. Dark field image
        center_y, center_x, inner_radius, outer_radius = self.dark_field_region
        df_image = self.create_dark_field_image(inner_radius, outer_radius)
        im2 = axes[1].imshow(df_image, cmap='hot')
        axes[1].set_title('Virtual Dark Field')
        axes[1].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[1], shrink=0.8)
        cbar2.set_label('Intensity (counts)', rotation=270, labelpad=15)
        
        # 3. HAADF image
        haadf_image = self.create_haadf_image()
        im3 = axes[2].imshow(haadf_image, cmap='plasma')
        axes[2].set_title('Virtual HAADF (Z-contrast)')
        axes[2].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[2], shrink=0.8)
        cbar3.set_label('Intensity (counts)', rotation=270, labelpad=15)
        
        # 4. Composite: BF vs HAADF
        composite = self.create_composite_image(bf_image, haadf_image, mode='rgb')
        axes[3].imshow(composite)
        axes[3].set_title('BF/HAADF Composite\n(Red=BF, Green=HAADF)')
        axes[3].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_analysis_dashboard(self, figsize=(15, 5)) -> plt.Figure:
        """
        Create analysis dashboard with radial profile and detector geometry.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # 1. Radial profile analysis
        self.plot_radial_profile(axes[0])
        
        # 2. Detector geometry schematic
        self.create_detector_schematic(axes[1])
        
        # 3. Mean pattern with regions overlay
        self.plot_mean_diffraction_pattern(axes[2], show_regions=True, use_log=True, add_scalebar=False)
        axes[2].set_title('Detector Regions Overlay')
        
        plt.tight_layout()
        return fig
    
    def create_detector_comparison(self, figsize=(12, 8)) -> plt.Figure:
        """
        Create side-by-side comparison of different detector configurations.
        
        Args:
            figsize: Figure size tuple
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # Create images
        bf_image = self.create_bright_field_image()
        center_y, center_x, inner_radius, outer_radius = self.dark_field_region
        df_image = self.create_dark_field_image(inner_radius, outer_radius)
        haadf_image = self.create_haadf_image()
        
        # Plot images with consistent colormaps and scaling
        im1 = axes[0, 0].imshow(bf_image, cmap='gray')
        axes[0, 0].set_title(f'Bright Field (r≤{self.bragg_radius*0.8:.1f}px)')
        axes[0, 0].axis('off')
        plt.colorbar(im1, ax=axes[0, 0], shrink=0.6)
        
        im2 = axes[0, 1].imshow(df_image, cmap='hot')
        axes[0, 1].set_title(f'Dark Field (r={inner_radius:.1f}-{outer_radius:.1f}px)')
        axes[0, 1].axis('off')
        plt.colorbar(im2, ax=axes[0, 1], shrink=0.6)
        
        im3 = axes[1, 0].imshow(haadf_image, cmap='plasma')
        haadf_inner, haadf_outer = self.haadf_region[2:4]
        axes[1, 0].set_title(f'HAADF (r={haadf_inner:.1f}-{haadf_outer:.1f}px)')
        axes[1, 0].axis('off')
        plt.colorbar(im3, ax=axes[1, 0], shrink=0.6)
        
        # Composite image
        composite = self.create_composite_image(bf_image, haadf_image, mode='rgb')
        axes[1, 1].imshow(composite)
        axes[1, 1].set_title('RGB Composite\n(Red=BF, Green=HAADF)')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        return fig
    
    def create_comprehensive_visualization(self, figsize=(20, 10), include_metadata=True, 
                                         metadata=None) -> plt.Figure:
        """
        Create comprehensive 2x4 visualization layout with all detector types and analysis.
        
        This method combines individual visualization components for a complete overview.
        For more focused visualizations, use the individual methods:
        - create_mean_pattern_comparison()
        - create_virtual_detector_suite()  
        - create_analysis_dashboard()
        - create_detector_comparison()
        
        Args:
            figsize: Figure size tuple
            include_metadata: Whether to include metadata text
            metadata: Optional metadata dictionary
            
        Returns:
            Matplotlib figure object
        """
        fig, axes = plt.subplots(2, 4, figsize=figsize)
        
        # Row 1: Mean patterns and analysis
        # 1. Mean diffraction pattern with detector regions (linear scale)
        self.plot_mean_diffraction_pattern(axes[0, 0], show_regions=True, use_log=False, add_scalebar=True)
        axes[0, 0].set_title('Mean Diffraction Pattern\n(With Detector Regions)')
        
        # 2. Radial profile analysis
        self.plot_radial_profile(axes[0, 1])
        
        # 3. Mean pattern - log scale
        mean_pattern = np.mean(np.maximum(self.data, 0), axis=0).astype(np.float32)
        im2 = axes[0, 2].imshow(np.log(mean_pattern + 1e-6), cmap='viridis')
        axes[0, 2].set_title('Mean Pattern (Log Scale)')
        axes[0, 2].axis('off')
        cbar2 = plt.colorbar(im2, ax=axes[0, 2], shrink=0.6)
        cbar2.set_label('Intensity (log)', rotation=270, labelpad=15)
        
        # 4. Detector geometry schematic
        self.create_detector_schematic(axes[0, 3])
        
        # Row 2: Virtual images from all detector types
        # 1. Bright field image
        bf_image = self.create_bright_field_image()
        im3 = axes[1, 0].imshow(bf_image, cmap='gray')
        axes[1, 0].set_title('Virtual Bright Field')
        axes[1, 0].axis('off')
        cbar3 = plt.colorbar(im3, ax=axes[1, 0], shrink=0.6)
        cbar3.set_label('Intensity (counts)', rotation=270, labelpad=15)
        
        # 2. Conventional Dark field image
        df_image = self.create_dark_field_image(*self.dark_field_region[2:4])
        im4 = axes[1, 1].imshow(df_image, cmap='hot')
        axes[1, 1].set_title('Virtual Dark Field')
        axes[1, 1].axis('off')
        cbar4 = plt.colorbar(im4, ax=axes[1, 1], shrink=0.6)
        cbar4.set_label('Intensity (counts)', rotation=270, labelpad=15)
        
        # 3. HAADF image
        haadf_image = self.create_haadf_image()
        im5 = axes[1, 2].imshow(haadf_image, cmap='plasma')
        axes[1, 2].set_title('Virtual HAADF (Z-contrast)')
        axes[1, 2].axis('off')
        cbar5 = plt.colorbar(im5, ax=axes[1, 2], shrink=0.6)
        cbar5.set_label('Intensity (counts)', rotation=270, labelpad=15)
        
        # 4. Composite: BF vs HAADF
        composite = self.create_composite_image(bf_image, haadf_image, mode='rgb')
        axes[1, 3].imshow(composite)
        axes[1, 3].set_title('BF/HAADF Composite\n(Red=BF, Green=HAADF)')
        axes[1, 3].axis('off')
        
        # Add metadata and statistics if requested
        if include_metadata:
            if metadata:
                info_text = []
                if 'original_shape' in metadata:
                    info_text.append(f"Original: {metadata['original_shape']}")
                if 'compression_ratio' in metadata:
                    info_text.append(f"Compression: {metadata['compression_ratio']:.1f}x")
                if 'normalization_method' in metadata:
                    info_text.append(f"Norm: {metadata['normalization_method']}")
                    
                if info_text:
                    fig.suptitle(' | '.join(info_text), y=0.02, fontsize=10)
            
            # Add statistics
            stats_text = f"Data: {self.data.shape} | Range: [{self.data.min():.1f}, {self.data.max():.1f}] | Direct beam: {self.direct_beam_position}"
            fig.suptitle(stats_text, y=0.98, fontsize=10)
        
        # Improve spacing between subplots
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.08, hspace=0.3, wspace=0.25)
        
        return fig
    
    def interactive_beam_center_detection(self) -> Tuple[Tuple[int, int], float]:
        """
        Interactive beam center and radius detection.
        Click 1: Center of the direct beam
        Click 2: Edge of the direct beam disc (for radius measurement)
        
        Returns:
            Tuple of ((center_y, center_x), radius)
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        
        # Calculate mean pattern
        mean_pattern = np.mean(np.maximum(self.data, 0), axis=0)
        
        # Interactive click detection
        clicks = []
        center_pos = None
        radius = None
        
        def onclick(event):
            nonlocal center_pos, radius
            if event.inaxes is None or event.button != 1:  # Only left mouse button
                return
                
            x, y = int(round(event.xdata)), int(round(event.ydata))
            clicks.append((y, x))  # Store as (row, col) = (y, x)
            
            print(f"Click {len(clicks)}: Position ({y}, {x})")
            
            if len(clicks) == 1:
                # First click: beam center
                center_pos = (y, x)
                ax.plot(x, y, 'r+', markersize=20, markeredgewidth=3, label='Beam Center')
                ax.plot(x, y, 'w+', markersize=16, markeredgewidth=2)
                ax.legend()
                fig.canvas.draw()
                print("Click on the edge of the direct beam disc to set radius...")
                
            elif len(clicks) == 2:
                # Second click: edge for radius calculation
                center_y, center_x = center_pos
                
                # Calculate radius
                radius = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                
                # Draw circle showing the detected region
                from matplotlib.patches import Circle
                circle = Circle((center_x, center_y), radius, 
                              fill=False, edgecolor='red', linewidth=2, linestyle='--')
                ax.add_patch(circle)
                ax.plot(x, y, 'ro', markersize=8, label=f'Edge (r={radius:.1f}px)')
                ax.legend()
                fig.canvas.draw()
                
                print(f"Beam center: ({center_y}, {center_x})")
                print(f"Detected radius: {radius:.1f} pixels")
                print("Auto-closing window...")
                
                # Auto-close immediately after detection
                plt.close(fig)
                return  # Exit the onclick function early
        
        def on_done(_):
            plt.close(fig)
        
        # Create interactive plot
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Show mean pattern in log scale for better visibility
        im = ax.imshow(np.log(mean_pattern + 1e-6), cmap='viridis')
        ax.set_title('Interactive Beam Center Detection\n1. Click beam center\n2. Click beam edge')
        plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Add done button
        ax_button = plt.axes([0.81, 0.01, 0.08, 0.05])
        button = Button(ax_button, 'Done')
        button.on_clicked(on_done)
        
        # Connect click event
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        print("Interactive beam center detection:")
        print("1. Click on the CENTER of the direct beam")
        print("2. Click on the EDGE of the direct beam disc")
        print("Window will auto-close after second click")
        
        # Use blocking show - simpler and more reliable
        plt.show()
        
        # Force close all figures to ensure cleanup
        plt.close('all')
        
        # Validate results
        if center_pos is None:
            print("Warning: No center position detected, using pattern center")
            center_pos = (self.pattern_shape[0] // 2, self.pattern_shape[1] // 2)
        
        if radius is None:
            print("Warning: No radius detected, using default estimate")
            radius = min(self.pattern_shape) // 8
        
        print(f"Interactive detection returning: center={center_pos}, radius={radius:.1f}")
        return center_pos, radius
    
    def apply_interactive_detection(self):
        """Apply interactive beam center detection and update internal parameters."""
        center_pos, radius = self.interactive_beam_center_detection()
        
        # Update internal parameters
        self.direct_beam_position = center_pos
        self.bragg_radius = radius
        
        # Force recalculation of cached indices
        self.y_indices, self.x_indices = np.indices(self.pattern_shape)
        
        # Recalculate detector regions with new parameters
        self.bright_field_region = self._default_bright_field_region()
        self.dark_field_region = self._default_dark_field_region()
        self.haadf_region = self._default_haadf_region()
        
        print(f"Updated STEMVisualizer with interactive detection results")
        print(f"Center: {self.direct_beam_position}, Radius: {self.bragg_radius:.1f}")
    
    def _add_scalebar(self, ax, pattern_scale=False):
        """Add simple, clean scalebar to the plot."""
        if pattern_scale:
            # For diffraction patterns - reciprocal space units
            pattern_size = min(self.pattern_shape)
            scale_pixels = max(20, pattern_size // 10)  # Adaptive size
            # Convert pixels to reciprocal space units (nm^-1)
            # Assuming typical STEM detector calibration: ~0.01 nm^-1 per pixel
            reciprocal_scale = scale_pixels * 0.01
            scale_length = f"{reciprocal_scale:.2f} nm⁻¹"
            image_size = pattern_size
        else:
            # For real space images
            scale_pixels = self.scalebar_info['scale_length'] 
            scale_length = f"{scale_pixels} {self.scalebar_info['units']}"
            image_size = self.scalebar_info['width']
        
        # Get axis limits
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        x_size = abs(x_lim[1] - x_lim[0])
        y_size = abs(y_lim[1] - y_lim[0])
        
        # Calculate scalebar position (bottom left to avoid legend conflict)
        margin_x = 0.05  # 5% margin from edges
        margin_y = 0.08  # 8% margin from bottom
        
        # Scalebar length as fraction of image
        bar_length_frac = scale_pixels / image_size if image_size > 0 else 0.15
        bar_length_frac = min(bar_length_frac, 0.25)  # Cap at 25% of image width
        
        # Position scalebar (bottom left)
        bar_width = bar_length_frac * x_size
        x_start = x_lim[0] + margin_x * x_size
        x_end = x_start + bar_width
        y_bar = y_lim[0] + margin_y * y_size
        
        # Simple, clean scalebar design
        bar_thickness = 3
        
        # Main bar (white with black outline)
        ax.plot([x_start, x_end], [y_bar, y_bar], 'white', linewidth=bar_thickness+2, solid_capstyle='round')
        ax.plot([x_start, x_end], [y_bar, y_bar], 'black', linewidth=bar_thickness, solid_capstyle='round')
        
        # Simple end ticks
        tick_height = 4
        ax.plot([x_start, x_start], [y_bar - tick_height/2, y_bar + tick_height/2], 'white', linewidth=bar_thickness+1)
        ax.plot([x_start, x_start], [y_bar - tick_height/2, y_bar + tick_height/2], 'black', linewidth=bar_thickness-1)
        ax.plot([x_end, x_end], [y_bar - tick_height/2, y_bar + tick_height/2], 'white', linewidth=bar_thickness+1)
        ax.plot([x_end, x_end], [y_bar - tick_height/2, y_bar + tick_height/2], 'black', linewidth=bar_thickness-1)
        
        # Simple text label with better spacing to avoid overlap
        label_y = y_bar + tick_height/2 + 0.035 * y_size  # Increased spacing
        ax.text((x_start + x_end) / 2, label_y, scale_length,
                ha='center', va='bottom', color='white', fontsize=8, weight='bold',
                bbox=dict(boxstyle='round,pad=0.15', facecolor='black', alpha=0.7))

    def plot_mean_diffraction_pattern(self, ax=None, show_regions=True, use_log=True, add_scalebar=False):
        """Plot mean diffraction pattern with field regions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use log scale for visualization, calculated on-demand to avoid storing extra data
        mean_pattern = np.mean(np.maximum(self.data, 0), axis=0).astype(np.float32)
        if use_log:
            mean_pattern = np.log(mean_pattern + 1e-6)
            
        im = ax.imshow(mean_pattern, cmap='viridis')
        ax.set_title('Mean Diffraction Pattern')
        ax.axis('off')
        
        if show_regions:
            # Mark direct beam position with refined crosshair
            y_beam, x_beam = self.direct_beam_position
            
            # Smaller, cleaner crosshair
            cross_size = min(self.pattern_shape) // 25  # Adaptive size
            ax.plot([x_beam - cross_size, x_beam + cross_size], [y_beam, y_beam], 
                   'white', linewidth=2.5, solid_capstyle='round', label='Beam Center')
            ax.plot([x_beam, x_beam], [y_beam - cross_size, y_beam + cross_size], 
                   'white', linewidth=2.5, solid_capstyle='round')
            ax.plot([x_beam - cross_size, x_beam + cross_size], [y_beam, y_beam], 
                   'black', linewidth=1.5, solid_capstyle='round')
            ax.plot([x_beam, x_beam], [y_beam - cross_size, y_beam + cross_size], 
                   'black', linewidth=1.5, solid_capstyle='round')
            
            # Show Bragg spot radius for reference
            from matplotlib.patches import Circle
            bragg_circle = Circle((x_beam, y_beam), self.bragg_radius,
                                fill=False, edgecolor='yellow', linewidth=1.5, linestyle=':', 
                                label='Bragg radius')
            ax.add_patch(bragg_circle)
            
            # Bright field region (circular)
            bf_radius = self.bragg_radius * 0.8  # Default BF radius
            bf_circle = Circle((x_beam, y_beam), bf_radius,
                             fill=False, edgecolor='red', linewidth=2,
                             label='BF detector')
            ax.add_patch(bf_circle)
            
            # Conventional Dark field region (annular around Bragg spots)
            center_y, center_x, inner_radius, outer_radius = self.dark_field_region
            inner_circle = Circle((center_x, center_y), inner_radius, 
                                fill=False, edgecolor='blue', linewidth=1.5, linestyle='--')
            outer_circle = Circle((center_x, center_y), outer_radius,
                                fill=False, edgecolor='blue', linewidth=2,
                                label='DF detector')
            ax.add_patch(inner_circle)
            ax.add_patch(outer_circle)
            
            # HAADF region (high-angle annular)
            haadf_center_y, haadf_center_x, haadf_inner, haadf_outer = self.haadf_region
            haadf_inner_circle = Circle((haadf_center_x, haadf_center_y), haadf_inner,
                                      fill=False, edgecolor='orange', linewidth=1.5, linestyle='--')
            haadf_outer_circle = Circle((haadf_center_x, haadf_center_y), haadf_outer,
                                      fill=False, edgecolor='orange', linewidth=2,
                                      label='HAADF detector')
            ax.add_patch(haadf_inner_circle)
            ax.add_patch(haadf_outer_circle)
            
            # Add compact legend with detector information - moved further down
            ax.legend(loc='lower right', fontsize=5, framealpha=0.9, 
                     bbox_to_anchor=(0.98, -0.02), markerscale=0.6)
        
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