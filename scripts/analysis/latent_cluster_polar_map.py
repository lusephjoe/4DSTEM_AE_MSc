#!/usr/bin/env python3
import argparse
import time
import json
import warnings
from pathlib import Path
from typing import Tuple, Optional, Dict, List, Union, Any
from datetime import datetime
import hashlib
import sys

import numpy as np
import h5py
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.colors import hsv_to_rgb
import tifffile
from scipy import ndimage
from scipy.signal import medfilt
from skimage.metrics import structural_similarity as ssim
import yaml

# Optional dependencies
try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False
    cp = None

try:
    import dask.array as da
    DASK_AVAILABLE = True
except ImportError:
    DASK_AVAILABLE = False
    da = None

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=RuntimeWarning)

class Timer:
    """Context manager for timing operations with visible display."""
    def __init__(self, description: str, verbose: bool = True):
        self.description = description
        self.verbose = verbose
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        if self.verbose:
            print(f"Running {self.description}...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"Completed {self.description} in {elapsed:.2f}s")
        return False
    
    @property
    def elapsed(self):
        if self.end_time is None:
            return time.time() - self.start_time if self.start_time else 0
        return self.end_time - self.start_time

class H5Dataset:
    """
    Lazy loader for 4D-STEM HDF5 datasets with chunked access and file handle caching.
    
    This class provides memory-efficient access to large 4D-STEM datasets stored
    in HDF5 format. It supports both (Ny, Nx, Ky, Kx) and (N, Ky, Kx) layouts
    and handles automatic reshaping as needed.
    
    The key advantages are that it doesn't load the entire dataset into memory,
    instead loading chunks on-demand during processing, and it caches the file
    handle to reduce I/O overhead across multiple read operations.
    
    Parameters:
    -----------
    filepath : Path
        Path to the HDF5 file containing 4D-STEM data
    dataset_name : str  
        Name of the dataset within the HDF5 file (e.g., 'patterns', '/entry/data')
    scan_shape : Tuple[int, int]
        Expected scan dimensions (Ny, Nx) for reshaping if needed
        
    Attributes:
    -----------
    Ky, Kx : int
        Detector dimensions in pixels
    total_frames : int
        Total number of diffraction patterns
    dtype : numpy.dtype
        Data type of the stored patterns
    """
    
    def __init__(self, filepath: Path, dataset_name: str, scan_shape: Tuple[int, int]):
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.scan_shape = scan_shape
        self.Ny, self.Nx = scan_shape
        self.total_frames = self.Ny * self.Nx
        self._file_handle = None
        
        # Open file to get metadata
        with h5py.File(filepath, 'r') as f:
            if dataset_name not in f:
                available = list(f.keys())
                raise KeyError(f"Dataset '{dataset_name}' not found. Available: {available}")
            
            self.dataset_shape = f[dataset_name].shape
            self.dtype = f[dataset_name].dtype
            
            # Determine if we need to reshape
            if len(self.dataset_shape) == 4:
                # Already in (Ny, Nx, Ky, Kx) format
                self.Ky, self.Kx = self.dataset_shape[2], self.dataset_shape[3]
            elif len(self.dataset_shape) == 3:
                # In (N, Ky, Kx) format - need to reshape
                self.Ky, self.Kx = self.dataset_shape[1], self.dataset_shape[2]
                if self.dataset_shape[0] != self.total_frames:
                    raise ValueError(f"Frame count mismatch: {self.dataset_shape[0]} vs {self.total_frames}")
            else:
                raise ValueError(f"Unexpected dataset shape: {self.dataset_shape}")
        
        print(f"Loaded H5 dataset: {self.dataset_shape} -> scan {scan_shape}, detector {self.Ky}x{self.Kx}")
    
    def _get_file_handle(self):
        """Get cached file handle, opening if necessary."""
        if self._file_handle is None:
            self._file_handle = h5py.File(self.filepath, 'r')
        return self._file_handle
    
    def close(self):
        """Close the file handle if open."""
        if self._file_handle is not None:
            self._file_handle.close()
            self._file_handle = None
    
    def __del__(self):
        """Ensure file is closed when object is deleted."""
        self.close()
    
    def __getitem__(self, indices):
        """Load frames by linear indices."""
        f = self._get_file_handle()
        dataset = f[self.dataset_name]
        
        if len(self.dataset_shape) == 4:
            # (Ny, Nx, Ky, Kx) format
            if isinstance(indices, slice):
                start, stop, step = indices.indices(self.total_frames)
                indices = np.arange(start, stop, step)
            
            # Convert linear indices to 2D coordinates
            y_coords = indices // self.Nx
            x_coords = indices % self.Nx
            
            # Load frames
            frames = np.empty((len(indices), self.Ky, self.Kx), dtype=self.dtype)
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                frames[i] = dataset[y, x, :, :]
            
            return frames
        else:
            # (N, Ky, Kx) format
            return dataset[indices]
    
    def load_chunk(self, start_idx: int, chunk_size: int):
        """Load a chunk of frames efficiently."""
        end_idx = min(start_idx + chunk_size, self.total_frames)
        actual_size = end_idx - start_idx
        
        f = self._get_file_handle()
        dataset = f[self.dataset_name]
        
        if len(self.dataset_shape) == 4:
            # Need to convert indices to coordinates
            indices = np.arange(start_idx, end_idx)
            y_coords = indices // self.Nx
            x_coords = indices % self.Nx
            
            frames = np.empty((actual_size, self.Ky, self.Kx), dtype=self.dtype)
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                frames[i] = dataset[y, x, :, :]
            
            return frames
        else:
            return dataset[start_idx:end_idx]

class Preprocessor:
    """
    Handles data preprocessing operations for diffraction patterns.
    
    Preprocessing is crucial for stable template generation and contrast analysis.
    The operations are applied in the following order:
    1. Background subtraction (if enabled)
    2. Log transform: log(I + 1) to compress dynamic range
    3. Central beam masking to prevent template domination by direct beam
    
    The preprocessor is applied during both template building and contrast mapping
    to ensure consistency. For DPC analysis, raw (unprocessed) intensities are used
    to preserve accurate center-of-mass calculations.
    
    Parameters:
    -----------
    config : Dict
        Configuration dictionary with preprocessing parameters
    detector_shape : Tuple[int, int]
        Shape of the detector (Ky, Kx) for creating masks
    center : Optional[Tuple[int, int]]
        Beam center coordinates (cy, cx). If None, uses detector center.
    """
    
    def __init__(self, config: Dict, detector_shape: Tuple[int, int], center: Optional[Tuple[int, int]] = None):
        self.config = config
        self.background = config.get('background_subtract', False)
        self.log_transform = config.get('log_transform', False)
        self.normalize = config.get('normalize', True)
        
        # Create central mask if specified
        central_mask_px = config.get('central_mask_px', None)
        if central_mask_px is not None:
            Ky, Kx = detector_shape
            yy, xx = np.mgrid[:Ky, :Kx]
            if center is not None:
                cy, cx = center
            else:
                cy, cx = Ky//2, Kx//2
            rr = np.hypot(yy - cy, xx - cx)
            self.detector_mask = (rr > central_mask_px).astype(np.float32)
        else:
            self.detector_mask = None
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Apply preprocessing to frames."""
        processed = frames.astype(np.float32)
        
        # Background subtraction
        if self.background:
            processed = processed - np.mean(processed, axis=0, keepdims=True)
        
        # Log transform using log1p for better accuracy
        if self.log_transform:
            processed = np.log1p(processed)
        
        # Apply detector mask
        if self.detector_mask is not None:
            processed = processed * self.detector_mask[np.newaxis, :, :]
        
        return processed

def interactive_beam_center_picker(sample_pattern: np.ndarray, title: str = "Click to select beam center") -> Tuple[int, int]:
    """
    Interactive beam center picking using matplotlib.
    
    Opens a window displaying the sample pattern and waits for a single click.
    The window closes automatically after the center is selected.
    
    Parameters:
    -----------
    sample_pattern : np.ndarray
        2D diffraction pattern to display for center picking
    title : str, optional
        Window title (default: "Click to select beam center")
        
    Returns:
    --------
    center : Tuple[int, int]
        Selected beam center coordinates (cy, cx)
    """
    center_coords = [None, None]
    
    def on_click(event):
        if event.inaxes and event.button == 1:  # Left mouse button
            center_coords[0] = int(round(event.ydata))
            center_coords[1] = int(round(event.xdata))
            plt.close(fig)  # Close specific figure
    
    # Create figure and display pattern
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(sample_pattern, cmap='viridis')
    ax.set_title(f"{title}\n(Click center, window will close automatically)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    
    # Add crosshairs at current center estimate
    cy_auto, cx_auto = np.unravel_index(np.argmax(ndimage.gaussian_filter(sample_pattern, 3)), sample_pattern.shape)
    ax.axhline(cy_auto, color='red', linestyle='--', alpha=0.7, label=f'Auto-detected: ({cy_auto}, {cx_auto})')
    ax.axvline(cx_auto, color='red', linestyle='--', alpha=0.7)
    ax.legend()
    
    # Connect click event
    fig.canvas.mpl_connect('button_press_event', on_click)
    
    print(f"Interactive beam center picker opened. Click on the beam center in the displayed pattern.")
    print(f"Auto-detected center is shown as red dashed lines at ({cy_auto}, {cx_auto})")
    
    plt.show(block=True)
    
    # Window should already be closed by event handler, but ensure cleanup
    try:
        plt.close(fig)
    except:
        pass
    
    # Small delay to ensure matplotlib cleanup is complete
    import time
    time.sleep(0.1)
    
    if center_coords[0] is None:
        print("No center selected, using auto-detected center")
        return cy_auto, cx_auto
    
    print(f"Selected beam center: ({center_coords[0]}, {center_coords[1]})")
    return center_coords[0], center_coords[1]


def interactive_kspace_calibration(sample_pattern: np.ndarray, center: Tuple[int, int], 
                                 known_spacing_angstrom: float = None, 
                                 title: str = "Click opposite Bragg peaks for k-space calibration") -> Tuple[float, float]:
    """
    Interactive k-space calibration by clicking opposite Bragg peaks or ring features.
    
    Allows the user to click two points (opposite Bragg peaks or ring boundaries) to
    automatically determine the k-space pixel sizes. If a known d-spacing is provided,
    calculates absolute calibration; otherwise provides relative calibration.
    
    Parameters:
    -----------
    sample_pattern : np.ndarray
        2D diffraction pattern to display for calibration
    center : Tuple[int, int]
        Beam center coordinates (cy, cx)
    known_spacing_angstrom : float, optional
        Known d-spacing in Angstroms for absolute calibration
    title : str, optional
        Window title
        
    Returns:
    --------
    pixel_size_ky, pixel_size_kx : Tuple[float, float]
        Calibrated k-space pixel sizes (1/Å or relative units)
    """
    clicks = []
    cy, cx = center
    
    def on_click(event):
        if event.inaxes and event.button == 1:  # Left mouse button
            y_click = int(round(event.ydata))
            x_click = int(round(event.xdata))
            clicks.append((y_click, x_click))
            
            # Plot the click
            ax.plot(x_click, y_click, 'ro', markersize=8)
            ax.annotate(f'Point {len(clicks)}', (x_click, y_click), 
                       xytext=(5, 5), textcoords='offset points', color='red', fontweight='bold')
            
            if len(clicks) == 1:
                ax.set_title(f"{title}\nClick the opposite peak (2/2)")
            elif len(clicks) == 2:
                # Draw line between points
                ax.plot([clicks[0][1], clicks[1][1]], [clicks[0][0], clicks[1][0]], 'r-', linewidth=2)
                ax.set_title("Calibration complete - window will close automatically")
                
                # Calculate calibration
                dy1 = clicks[0][0] - cy
                dx1 = clicks[0][1] - cx
                dy2 = clicks[1][0] - cy  
                dx2 = clicks[1][1] - cx
                
                # Distance from center for each point
                r1 = np.sqrt(dy1**2 + dx1**2)
                r2 = np.sqrt(dy2**2 + dx2**2)
                
                # Average radius (assuming opposite peaks)
                avg_radius_pixels = (r1 + r2) / 2
                
                if known_spacing_angstrom is not None:
                    # Absolute calibration: k = 1/d, k_pixels = k / pixel_size_k
                    # So pixel_size_k = k / k_pixels = (1/d) / (radius_pixels)
                    k_magnitude = 1.0 / known_spacing_angstrom  # 1/Å
                    pixel_size_k = k_magnitude / avg_radius_pixels
                    print(f"Absolute calibration: {pixel_size_k:.6f} Å⁻¹/pixel")
                    print(f"d-spacing: {known_spacing_angstrom:.3f} Å")
                else:
                    # Relative calibration: normalize so this radius = 1.0
                    pixel_size_k = 1.0 / avg_radius_pixels
                    print(f"Relative calibration: {pixel_size_k:.6f} relative units/pixel")
                
                print(f"Average peak radius: {avg_radius_pixels:.1f} pixels")
                print(f"Individual radii: {r1:.1f}, {r2:.1f} pixels")
                
                # Update display 
                fig.canvas.draw()
                fig.canvas.flush_events()  # Ensure display updates
                
            fig.canvas.draw()
            
            if len(clicks) >= 2:
                # Close window immediately after second click
                plt.close(fig)
                return  # Exit event handler after second click
    
    def on_key(event):
        if len(clicks) >= 2:
            plt.close(fig)  # Close specific figure
    
    # Create figure and display pattern
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Use log scale for better visibility of peaks
    display_pattern = np.log1p(sample_pattern)
    im = ax.imshow(display_pattern, cmap='viridis')
    plt.colorbar(im, ax=ax, label='Log intensity')
    
    # Mark the beam center
    ax.plot(cx, cy, '+', color='white', markersize=15, markeredgewidth=3, label=f'Center ({cy}, {cx})')
    
    # Add circles to guide peak selection
    circle_radii = [20, 40, 60, 80, 100]
    for radius in circle_radii:
        if radius < min(sample_pattern.shape) // 2:
            circle = plt.Circle((cx, cy), radius, fill=False, color='white', alpha=0.3, linestyle='--')
            ax.add_patch(circle)
    
    ax.set_title(f"{title}\nClick the first Bragg peak (1/2)")
    ax.set_xlabel("X (pixels)")
    ax.set_ylabel("Y (pixels)")
    ax.legend()
    
    # Connect events
    fig.canvas.mpl_connect('button_press_event', on_click)
    fig.canvas.mpl_connect('key_press_event', on_key)
    
    print("Interactive k-space calibration:")
    print("1. Click on a Bragg peak or ring feature")
    print("2. Click on the opposite peak (180° around center)")
    if known_spacing_angstrom:
        print(f"3. Using known d-spacing: {known_spacing_angstrom:.3f} Å")
    else:
        print("3. Relative calibration will be computed")
    
    plt.show(block=True)
    
    # Window should already be closed by event handler, but ensure cleanup
    try:
        plt.close(fig)
    except:
        pass
    
    # Small delay to ensure matplotlib cleanup is complete
    import time
    time.sleep(0.1)
    
    if len(clicks) < 2:
        print("Calibration cancelled - insufficient points")
        return 1.0, 1.0
    
    # Calculate final calibration values
    dy1 = clicks[0][0] - cy
    dx1 = clicks[0][1] - cx
    dy2 = clicks[1][0] - cy
    dx2 = clicks[1][1] - cx
    
    r1 = np.sqrt(dy1**2 + dx1**2)
    r2 = np.sqrt(dy2**2 + dx2**2)
    avg_radius_pixels = (r1 + r2) / 2
    
    if known_spacing_angstrom is not None:
        pixel_size_k = (1.0 / known_spacing_angstrom) / avg_radius_pixels
    else:
        pixel_size_k = 1.0 / avg_radius_pixels
    
    # For now, assume isotropic calibration (same for both axes)
    # Future enhancement could support anisotropic by measuring different directions
    return pixel_size_k, pixel_size_k


def interactive_annular_detector_optimization(sample_pattern: np.ndarray, center: Tuple[int, int],
                                            title: str = "Annular Detector Optimization") -> Dict[str, float]:
    """
    Interactive optimization of annular detector parameters using radial analysis.
    
    This function analyzes the radial intensity profile and azimuthal anisotropy to
    automatically suggest and allow manual adjustment of optimal detector parameters
    for conventional DPC analysis. Supports both single and two-ring configurations.
    
    The algorithm:
    1. Computes radial mean intensity I(r) and its derivative dI/dr
    2. Finds central disk edge R0 as first strong negative peak in dI/dr
    3. Computes azimuthal anisotropy ANI(r) = σ_θ(r) / (I(r) + ε)
    4. Suggests parameters based on anisotropy analysis
    5. Allows interactive adjustment between single/two-ring modes
    
    Parameters:
    -----------
    sample_pattern : np.ndarray
        2D diffraction pattern for analysis
    center : Tuple[int, int]
        Beam center coordinates (cy, cx)
    title : str
        Window title
        
    Returns:
    --------
    detector_params : Dict[str, float]
        Optimized detector parameters with keys:
        - 'mode': 'single' or 'two_ring'
        - 'r1_inner', 'r1_outer': First ring radii
        - 'r2_inner', 'r2_outer': Second ring radii (if two_ring mode)  
        - 'ring2_weight': Weight for second ring (if two_ring mode)
    """
    # Enable interactive mode and force backend
    plt.ion()  # Interactive mode on
    
    cy, cx = center
    Ky, Kx = sample_pattern.shape
    
    # Create radial coordinate arrays
    yy, xx = np.mgrid[:Ky, :Kx]
    rr = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    theta = np.arctan2(yy - cy, xx - cx)
    
    # Compute radial profile
    max_r = min(cy, cx, Ky-cy, Kx-cx) - 5  # Stay within detector bounds
    r_bins = np.arange(0, max_r, 0.5)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    # Radial mean intensity
    I_radial = []
    sigma_radial = []
    
    for i, r in enumerate(r_centers):
        r_min, r_max = r_bins[i], r_bins[i+1]
        mask = (rr >= r_min) & (rr < r_max)
        if np.sum(mask) > 0:
            intensities = sample_pattern[mask]
            I_radial.append(np.mean(intensities))
            sigma_radial.append(np.std(intensities))
        else:
            I_radial.append(0.0)
            sigma_radial.append(0.0)
    
    I_radial = np.array(I_radial)
    sigma_radial = np.array(sigma_radial)
    
    # Smooth derivative to find central disk edge
    from scipy import ndimage
    I_smooth = ndimage.gaussian_filter1d(I_radial, sigma=2)
    dI_dr = np.gradient(I_smooth)
    
    # Find central disk edge R0 (first strong negative peak)
    # Look for negative peaks beyond r=2 to avoid center artifacts
    valid_region = r_centers > 2
    if np.any(valid_region):
        dI_valid = dI_dr[valid_region]
        r_valid = r_centers[valid_region]
        
        # Find local minima in derivative
        from scipy.signal import find_peaks
        neg_peaks, _ = find_peaks(-dI_valid, height=0.1*np.abs(dI_valid).max())
        
        if len(neg_peaks) > 0:
            R0 = r_valid[neg_peaks[0]]
        else:
            # Fallback: find steepest negative slope
            min_idx = np.argmin(dI_valid)
            R0 = r_valid[min_idx]
    else:
        R0 = 5.0  # Default fallback
    
    # Compute azimuthal anisotropy ANI(r)
    ANI = []
    for i, r in enumerate(r_centers):
        r_min, r_max = r_bins[i], r_bins[i+1]
        ring_mask = (rr >= r_min) & (rr < r_max)
        if np.sum(ring_mask) > 10:  # Need enough points for meaningful statistics
            ring_intensities = sample_pattern[ring_mask]
            ring_theta = theta[ring_mask]
            
            # Compute azimuthal standard deviation
            theta_bins = np.linspace(-np.pi, np.pi, 36)  # 10-degree bins
            theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
            azimuthal_I = []
            
            for j in range(len(theta_bins)-1):
                theta_mask = (ring_theta >= theta_bins[j]) & (ring_theta < theta_bins[j+1])
                if np.sum(theta_mask) > 0:
                    azimuthal_I.append(np.mean(ring_intensities[theta_mask]))
            
            if len(azimuthal_I) > 1:
                sigma_theta = np.std(azimuthal_I)
                mean_I = I_radial[i]
                ani = sigma_theta / (mean_I + 0.01 * I_radial.max())
            else:
                ani = 0.0
        else:
            ani = 0.0
        
        ANI.append(ani)
    
    ANI = np.array(ANI)
    
    # Suggest optimal parameters for both single and two-ring modes
    delta = 2.0  # pixels beyond R0
    
    # Single ring suggestions
    r1_inner_suggested = R0 + delta
    ani_threshold = np.median(ANI) + 2 * np.std(ANI)  # Robust threshold
    intensity_threshold = 0.03 * I_radial.max()  # 3% of peak intensity
    
    candidates = r_centers[r_centers > r1_inner_suggested]
    if len(candidates) > 0:
        for r in candidates:
            idx = np.argmin(np.abs(r_centers - r))
            if ANI[idx] > ani_threshold or I_radial[idx] < intensity_threshold:
                r1_outer_suggested = r
                break
        else:
            r1_outer_suggested = candidates[len(candidates)//2]
    else:
        r1_outer_suggested = max_r * 0.7
    
    # Two-ring suggestions - detect where anisotropy calms down after Bragg band
    # Find the high anisotropy region (Bragg band)
    high_ani_mask = ANI > ani_threshold
    if np.any(high_ani_mask):
        high_ani_end = r_centers[high_ani_mask][-1] if np.any(high_ani_mask) else r1_outer_suggested
        
        # Look for calm region after Bragg band
        calm_candidates = r_centers[r_centers > high_ani_end + 5]  # 5px buffer
        if len(calm_candidates) > 10:  # Need reasonable range
            # Find where anisotropy drops below threshold again
            calm_region_start = None
            for r in calm_candidates:
                idx = np.argmin(np.abs(r_centers - r))
                if ANI[idx] < ani_threshold * 0.7:  # 70% of threshold
                    calm_region_start = r
                    break
            
            if calm_region_start:
                r2_inner_suggested = calm_region_start
                # Extend to reasonable outer radius or detector limit
                r2_outer_suggested = min(calm_region_start + 15, max_r * 0.9)
            else:
                # Fallback: use region beyond Bragg band
                r2_inner_suggested = high_ani_end + 8
                r2_outer_suggested = min(r2_inner_suggested + 15, max_r * 0.9)
        else:
            # No clear calm region - use default extended range
            r2_inner_suggested = r1_outer_suggested + 10
            r2_outer_suggested = min(r2_inner_suggested + 15, max_r * 0.9)
    else:
        # No clear Bragg band detected - use default spacing
        r2_inner_suggested = r1_outer_suggested + 10
        r2_outer_suggested = min(r2_inner_suggested + 15, max_r * 0.9)
    
    # Default two-ring weight (outer ring gets less weight to reduce noise)
    ring2_weight_suggested = 0.5
    
    # Create interactive plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Display pattern with current detector
    ax1 = axes[0, 0]
    im = ax1.imshow(np.log1p(sample_pattern), cmap='viridis')
    
    # Initialize with two-ring mode for better starting point
    detector_mode = 'two_ring'
    
    # Plot circles for current detector  
    detector_circles = {}
    ring1_inner = plt.Circle((cx, cy), r1_inner_suggested, fill=False, color='red', linewidth=2, label='Ring1 inner')
    ring1_outer = plt.Circle((cx, cy), r1_outer_suggested, fill=False, color='red', linewidth=2, linestyle='--', label='Ring1 outer')
    ring2_inner = plt.Circle((cx, cy), r2_inner_suggested, fill=False, color='blue', linewidth=2, label='Ring2 inner') 
    ring2_outer = plt.Circle((cx, cy), r2_outer_suggested, fill=False, color='blue', linewidth=2, linestyle='--', label='Ring2 outer')
    
    ax1.add_patch(ring1_inner)
    ax1.add_patch(ring1_outer)
    ax1.add_patch(ring2_inner)
    ax1.add_patch(ring2_outer)
    
    detector_circles = {
        'ring1_inner': ring1_inner,
        'ring1_outer': ring1_outer, 
        'ring2_inner': ring2_inner,
        'ring2_outer': ring2_outer
    }
    
    ax1.set_title('Diffraction Pattern with Two-Ring Detector')
    ax1.legend(loc='upper right', fontsize='small')
    ax1.axis('equal')
    
    # Radial profile  
    ax2 = axes[0, 1]
    line_I, = ax2.plot(r_centers, I_radial, 'b-', label='I(r)', linewidth=1.5)
    ax2.axvline(R0, color='green', linestyle='--', alpha=0.8, label=f'R0 = {R0:.1f}')
    
    # Show both rings
    ax2.axvspan(r1_inner_suggested, r1_outer_suggested, alpha=0.2, color='red', label=f'Ring1: [{r1_inner_suggested:.1f}, {r1_outer_suggested:.1f}]')
    ax2.axvspan(r2_inner_suggested, r2_outer_suggested, alpha=0.2, color='blue', label=f'Ring2: [{r2_inner_suggested:.1f}, {r2_outer_suggested:.1f}]')
    
    ax2.set_xlabel('Radius (pixels)')
    ax2.set_ylabel('Mean Intensity') 
    ax2.set_title('Radial Intensity Profile')
    ax2.legend(fontsize='small')
    ax2.grid(True, alpha=0.3)
    
    # Derivative and anisotropy
    ax3 = axes[1, 0]
    ax3.plot(r_centers, dI_dr, 'g-', label='dI/dr')
    ax3.axvline(R0, color='green', linestyle='--', alpha=0.7, label=f'Central disk edge')
    ax3.set_xlabel('Radius (pixels)')
    ax3.set_ylabel('dI/dr')
    ax3.set_title('Radial Derivative (Central Disk Detection)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    ax4 = axes[1, 1]
    ax4.plot(r_centers, ANI, 'm-', label='Anisotropy', linewidth=1.5)
    ax4.axhline(ani_threshold, color='orange', linestyle='--', alpha=0.7, label='High ANI threshold')
    
    # Highlight Bragg band and calm regions
    bragg_mask = ANI > ani_threshold
    if np.any(bragg_mask):
        bragg_region = r_centers[bragg_mask]
        if len(bragg_region) > 0:
            ax4.axvspan(bragg_region[0], bragg_region[-1], alpha=0.3, color='orange', label='Bragg band')
    
    # Show ring regions
    ax4.axvspan(r1_inner_suggested, r1_outer_suggested, alpha=0.2, color='red', label='Ring1')
    ax4.axvspan(r2_inner_suggested, r2_outer_suggested, alpha=0.2, color='blue', label='Ring2')
    
    ax4.set_xlabel('Radius (pixels)')
    ax4.set_ylabel('ANI(r)')
    ax4.set_title('Azimuthal Anisotropy (Bragg Detection)')
    ax4.legend(fontsize='small')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Interactive adjustment - support both single and two-ring modes
    current_params = {
        'mode': detector_mode,
        'r1_inner': r1_inner_suggested, 
        'r1_outer': r1_outer_suggested,
        'r2_inner': r2_inner_suggested,
        'r2_outer': r2_outer_suggested,
        'ring2_weight': ring2_weight_suggested
    }
    
    def update_detector():
        # Update circles based on mode
        if current_params['mode'] == 'single':
            # Show only first ring, hide second
            detector_circles['ring1_inner'].set_radius(current_params['r1_inner'])
            detector_circles['ring1_outer'].set_radius(current_params['r1_outer'])
            detector_circles['ring2_inner'].set_visible(False)
            detector_circles['ring2_outer'].set_visible(False)
            ax1.set_title('Diffraction Pattern with Single-Ring Detector')
        else:  # two_ring mode
            # Show both rings
            detector_circles['ring1_inner'].set_radius(current_params['r1_inner'])
            detector_circles['ring1_outer'].set_radius(current_params['r1_outer'])
            detector_circles['ring2_inner'].set_radius(current_params['r2_inner'])
            detector_circles['ring2_outer'].set_radius(current_params['r2_outer'])
            detector_circles['ring2_inner'].set_visible(True)
            detector_circles['ring2_outer'].set_visible(True)
            ax1.set_title(f'Diffraction Pattern with Two-Ring Detector (Ring2 weight: {current_params["ring2_weight"]:.1f})')
        
        # Update radial profile spans
        ax2.clear()
        ax2.plot(r_centers, I_radial, 'b-', label='I(r)', linewidth=1.5)
        ax2.axvline(R0, color='green', linestyle='--', alpha=0.8, label=f'R0 = {R0:.1f}')
        
        ax2.axvspan(current_params['r1_inner'], current_params['r1_outer'], alpha=0.3, color='red', 
                   label=f'Ring1: [{current_params["r1_inner"]:.1f}, {current_params["r1_outer"]:.1f}]')
        
        if current_params['mode'] == 'two_ring':
            ax2.axvspan(current_params['r2_inner'], current_params['r2_outer'], alpha=0.3, color='blue',
                       label=f'Ring2: [{current_params["r2_inner"]:.1f}, {current_params["r2_outer"]:.1f}] (w={current_params["ring2_weight"]:.1f})')
        
        ax2.set_xlabel('Radius (pixels)')
        ax2.set_ylabel('Mean Intensity')
        ax2.set_title('Radial Intensity Profile')
        ax2.legend(fontsize='small')
        ax2.grid(True, alpha=0.3)
        
        # Update anisotropy plot spans
        ax4.clear()
        ax4.plot(r_centers, ANI, 'm-', label='Anisotropy', linewidth=1.5)
        ax4.axhline(ani_threshold, color='orange', linestyle='--', alpha=0.7, label='High ANI threshold')
        
        if np.any(bragg_mask):
            bragg_region = r_centers[bragg_mask]
            if len(bragg_region) > 0:
                ax4.axvspan(bragg_region[0], bragg_region[-1], alpha=0.3, color='orange', label='Bragg band')
        
        ax4.axvspan(current_params['r1_inner'], current_params['r1_outer'], alpha=0.3, color='red', label='Ring1')
        if current_params['mode'] == 'two_ring':
            ax4.axvspan(current_params['r2_inner'], current_params['r2_outer'], alpha=0.3, color='blue', label='Ring2')
        
        ax4.set_xlabel('Radius (pixels)')
        ax4.set_ylabel('ANI(r)')
        ax4.set_title('Azimuthal Anisotropy (Bragg Detection)')
        ax4.legend(fontsize='small')
        ax4.grid(True, alpha=0.3)
        
        # Force redraw
        fig.canvas.draw()
        fig.canvas.flush_events()
    
    def on_key(event):
        if event is None or not hasattr(event, 'key') or event.key is None:
            return
            
        key = event.key.lower()
        step = 0.1 if 'shift' in event.key.lower() else 1.0
        weight_step = 0.05 if 'shift' in event.key.lower() else 0.1
        
        # Flag to track if we should update the display
        should_update = False
        
        # Removed debug output
        
        # Mode switching
        if key == 's':
            current_params['mode'] = 'single' if current_params['mode'] == 'two_ring' else 'two_ring'
            should_update = True
        
        # Ring 1 adjustments (always available)
        elif key == '1':
            current_params['r1_inner'] = max(current_params['r1_inner'] - step, 1.0)
            should_update = True
        elif key == '2':
            current_params['r1_inner'] = min(current_params['r1_inner'] + step, current_params['r1_outer'] - 1)
            should_update = True
        elif key == '3':
            current_params['r1_outer'] = max(current_params['r1_outer'] - step, current_params['r1_inner'] + 1)
            should_update = True
        elif key == '4':
            current_params['r1_outer'] = min(current_params['r1_outer'] + step, max_r)
            should_update = True
        
        # Ring 2 adjustments (only in two_ring mode)
        elif current_params['mode'] == 'two_ring':
            if key == 'z':
                current_params['r2_inner'] = max(current_params['r2_inner'] - step, current_params['r1_outer'] + 1)
                should_update = True
            elif key == 'x':
                current_params['r2_inner'] = min(current_params['r2_inner'] + step, current_params['r2_outer'] - 1)
                should_update = True
            elif key == 'c':
                current_params['r2_outer'] = max(current_params['r2_outer'] - step, current_params['r2_inner'] + 1)
                should_update = True
            elif key == 'v':
                current_params['r2_outer'] = min(current_params['r2_outer'] + step, max_r)
                should_update = True
            elif key == 'a':
                current_params['ring2_weight'] = max(current_params['ring2_weight'] - weight_step, 0.1)
                should_update = True
            elif key == 'd':
                current_params['ring2_weight'] = min(current_params['ring2_weight'] + weight_step, 2.0)
                should_update = True
        
        # Accept parameters
        elif key == 'enter':
            plt.close(fig)
            return
        
        # Help key
        elif key == 'h':
            print("Interactive controls:")
            print("  s: Switch between single/two-ring modes")
            print("  Ring1: 1/2 (inner -/+), 3/4 (outer -/+)")
            print("  Ring2: z/x (inner -/+), c/v (outer -/+), a/d (weight -/+)")
            print("  Shift + key: fine adjustment")
            print("  Enter: accept parameters")
        
        # Only update if we actually changed something
        if should_update:
            update_detector()
            print_current_status()
    
    def print_current_status():
        """Print current detector parameters once to avoid spam."""
        if current_params['mode'] == 'single':
            status = f"Mode: Single-ring | Ring1: [{current_params['r1_inner']:.1f}, {current_params['r1_outer']:.1f}]"
        else:
            status = (f"Mode: Two-ring | Ring1: [{current_params['r1_inner']:.1f}, {current_params['r1_outer']:.1f}] | " +
                     f"Ring2: [{current_params['r2_inner']:.1f}, {current_params['r2_outer']:.1f}] (weight: {current_params['ring2_weight']:.2f})")
        
        # Only print if status has changed (prevent spam)
        if not hasattr(print_current_status, 'last_status') or print_current_status.last_status != status:
            print(status)
            print_current_status.last_status = status
    
    # Connect event handler and make sure figure has focus
    cid = fig.canvas.mpl_connect('key_press_event', on_key)
    
    # Also try button press events in case key events don't work
    def on_click(event):
        print("Figure clicked - keyboard focus should be active now. Try pressing 'h'", flush=True)
    
    cid_click = fig.canvas.mpl_connect('button_press_event', on_click)
    
    try:
        fig.canvas.set_window_title(title + " - Press any key to interact")
    except AttributeError:
        # Some backends (like Mac) don't support set_window_title
        fig.suptitle(title + " - Press any key to interact", fontsize=14)
    
    # Make figure focusable
    fig.canvas.manager.show()
    plt.draw()
    
    print(f"\nTwo-Ring Annular Detector Optimization:")
    print(f"Detected central disk edge R0 = {R0:.1f} pixels")
    print(f"Suggested Ring1: [{r1_inner_suggested:.1f}, {r1_outer_suggested:.1f}] (avoids central disk)")
    print(f"Suggested Ring2: [{r2_inner_suggested:.1f}, {r2_outer_suggested:.1f}] (skips Bragg band)")
    print(f"Default Ring2 weight: {ring2_weight_suggested:.1f} (reduces noise)")
    print()
    print_current_status()
    print()
    print("Interactive controls:")
    print("  h: Show help")
    print("  s: Switch between single/two-ring modes")
    print("  Ring1: 1/2 (inner -/+), 3/4 (outer -/+)")
    print("  Ring2: z/x (inner -/+), c/v (outer -/+), a/d (weight -/+)")  
    print("  Shift + key: fine adjustment (0.1 px or 0.05 weight)")
    print("  Enter: accept parameters")
    print("\n*** Click on the figure window to activate keyboard controls ***\n")
    
    # Show figure and wait for user interaction
    plt.show(block=True)
    
    # Cleanup
    try:
        fig.canvas.mpl_disconnect(cid)
        fig.canvas.mpl_disconnect(cid_click)
        plt.close(fig)
    except:
        pass
    
    # Turn off interactive mode
    plt.ioff()
    
    return current_params


class BaseDPCAnalyzer:
    """
    Base class for 4D-STEM DPC analysis with common functionality.
    
    This class provides the foundation for different DPC analysis approaches:
    - Data loading and preprocessing
    - Beam center detection and k-space calibration
    - Basic DPC/CoM computation utilities
    - Output saving and visualization
    
    Subclasses implement specific analysis strategies:
    - ClusterDPCAnalyzer: Uses cluster-based virtual detectors
    - ConventionalDPCAnalyzer: Uses conventional DPC approaches
    """
    
    def __init__(self, config: Dict):
        self.config = config
        self._validate_calibration_config()
        self.device = config.get('device', 'cpu')
        self.use_gpu = self.device.startswith('cuda') and CUPY_AVAILABLE
        
        if self.use_gpu:
            print(f"Using GPU acceleration: {self.device}")
            self.xp = cp
        else:
            print("Using CPU computation")
            self.xp = np
            
        # Global statistics for normalization
        self.mu_global = None
        self.sigma_global = None

    def _validate_calibration_config(self):
        """Validate calibration parameters and log usage information."""
        cal = self.config.get('calibration', {})
        
        pixel_size = cal.get('pixel_size')
        pixel_size_units = cal.get('pixel_size_units', 'nm')
        pixel_size_ky = cal.get('pixel_size_ky')
        pixel_size_kx = cal.get('pixel_size_kx')
        convergence_angle = cal.get('convergence_angle')
        
        if pixel_size is not None:
            print(f"Config validation: pixel_size={pixel_size} {pixel_size_units}/pixel will take precedence over calculated values")
            
        if convergence_angle is not None:
            print(f"Config validation: convergence_angle={convergence_angle} mrad is available for calculations")
            
        # Check for potential conflicts (warn but don't fail)
        if pixel_size is not None and pixel_size_ky is not None:
            if abs(pixel_size - pixel_size_ky) > 1e-6:
                print(f"Warning: pixel_size ({pixel_size}) differs from pixel_size_ky ({pixel_size_ky})")
                print("  -> pixel_size will take precedence")
                
        if pixel_size is not None and pixel_size_kx is not None:
            if abs(pixel_size - pixel_size_kx) > 1e-6:
                print(f"Warning: pixel_size ({pixel_size}) differs from pixel_size_kx ({pixel_size_kx})")
                print("  -> pixel_size will take precedence")
    
    def _convert_pixel_size_to_k_space(self, pixel_size: float, units: str) -> float:
        """
        Convert real-space pixel size to k-space pixel size.
        
        Parameters:
        - pixel_size: Real-space pixel size value
        - units: "nm" or "angstrom"
        
        Returns:
        - k-space pixel size in (1/Å)/pixel
        
        Note: This assumes the pixel_size represents the sampling in diffraction space,
        not real space. For 4D-STEM, this is typically the angular sampling converted
        to reciprocal space units.
        """
        if units.lower() in ['nm', 'nanometer', 'nanometers']:
            # Convert nm to Angstrom, then take reciprocal
            pixel_size_angstrom = pixel_size * 10.0  # nm to Angstrom
            return 1.0 / pixel_size_angstrom  # (1/Å)/pixel
        elif units.lower() in ['angstrom', 'ang', 'å', 'a']:
            # Direct reciprocal
            return 1.0 / pixel_size  # (1/Å)/pixel
        else:
            print(f"Warning: Unknown pixel_size_units '{units}', assuming nm")
            pixel_size_angstrom = pixel_size * 10.0
            return 1.0 / pixel_size_angstrom

    def load_data(self, data_path: Path, dataset_name: str, scan_shape: Tuple[int, int]) -> H5Dataset:
        """Load 4D-STEM data."""
        return H5Dataset(data_path, dataset_name, scan_shape)
    
    def detect_beam_center(self, dataset: H5Dataset) -> Tuple[int, int]:
        """
        Detect beam center with interactive and automatic modes.
        
        Returns:
        --------
        center : Tuple[int, int]
            Beam center coordinates (cy, cx)
        """
        cal = self.config.get('calibration', {})
        cy, cx = cal.get('center_y'), cal.get('center_x')
        interactive_mode = self.config.get('interactive_center', False)
        
        # Load sample data if needed for detection
        sample_frames = None
        sample_mean = None
        if (cy is None or cx is None or interactive_mode):
            sample_frames = dataset.load_chunk(0, min(100, dataset.total_frames))
            sample_mean = np.mean(sample_frames, axis=0).astype(np.float32)
        
        # Priority 1: Use config values if both are specified
        if cy is not None and cx is not None:
            print(f"Using beam center from config: ({cy}, {cx})")
        else:
            # Priority 2: Interactive mode if enabled and config incomplete
            if interactive_mode:
                print("Interactive beam center picking enabled...")
                cy_picked, cx_picked = interactive_beam_center_picker(sample_mean)
                cy, cx = cy_picked, cx_picked
                self.config.setdefault('calibration', {})['center_y'] = cy
                self.config.setdefault('calibration', {})['center_x'] = cx
                print(f"Interactively selected beam center: ({cy}, {cx})")
            else:
                # Priority 3: Auto-detect missing coordinates
                cy_det, cx_det = np.unravel_index(np.argmax(ndimage.gaussian_filter(sample_mean, 3)), sample_mean.shape)
                if cy is None:
                    cy = int(cy_det)
                    self.config.setdefault('calibration', {})['center_y'] = cy
                if cx is None:
                    cx = int(cx_det)
                    self.config.setdefault('calibration', {})['center_x'] = cx
                print(f"Auto-detected beam center: ({cy}, {cx})")
        
        print(f"FINAL BEAM CENTER: ({cy}, {cx}) - This will be used for all calculations")
        return cy, cx

    def calibrate_k_space(self, dataset: H5Dataset, center: Tuple[int, int]) -> None:
        """
        Perform k-space calibration with interactive and automatic modes.
        """
        interactive_kspace_mode = self.config.get('interactive_kspace', False)
        known_spacing = self.config.get('known_spacing', None)
        
        if interactive_kspace_mode:
            print("Interactive k-space calibration enabled...")
            # Load sample data for calibration
            sample_frames = dataset.load_chunk(0, min(100, dataset.total_frames))
            sample_mean = np.mean(sample_frames, axis=0).astype(np.float32)
            
            pixel_size_ky, pixel_size_kx = interactive_kspace_calibration(
                sample_mean, center, known_spacing
            )
            # Update config with new calibration
            self.config.setdefault('calibration', {})['pixel_size_ky'] = pixel_size_ky
            self.config.setdefault('calibration', {})['pixel_size_kx'] = pixel_size_kx
            
            # If isotropic (ky == kx), also set the direct pixel_size for future precedence
            if abs(pixel_size_ky - pixel_size_kx) < 1e-10:
                self.config.setdefault('calibration', {})['pixel_size'] = pixel_size_ky
                print(f"K-space calibration updated: isotropic pixel_size={pixel_size_ky:.6f}")
            else:
                print(f"K-space calibration updated: anisotropic ky={pixel_size_ky:.6f}, kx={pixel_size_kx:.6f}")

    def make_k_grids(self, detector_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create k-space coordinate grids."""
        Ky, Kx = detector_shape
        cal = self.config.get('calibration', {})
        cy, cx = cal.get('center_y'), cal.get('center_x')

        if cy is None or cx is None:
            mu = self.mu_global if self.mu_global is not None else np.zeros(detector_shape)
            cy, cx = np.unravel_index(np.argmax(ndimage.gaussian_filter(mu, 3)), mu.shape)
            self.config.setdefault('calibration', {})['center_y'] = int(cy)
            self.config.setdefault('calibration', {})['center_x'] = int(cx)
            print(f"Auto-detected beam center: ({cy}, {cx})")

        # Support anisotropic pixel sizes with precedence hierarchy:
        # 1. Direct pixel_size (if specified, takes precedence)
        # 2. Anisotropic pixel_size_ky/pixel_size_kx 
        # 3. Fallback to pixel_size_k
        
        pixel_size_direct = cal.get('pixel_size')
        if pixel_size_direct is not None:
            # Convert pixel_size to k-space units (1/Å per pixel)
            pixel_size_units = cal.get('pixel_size_units', 'nm')
            pixel_size_k_converted = self._convert_pixel_size_to_k_space(pixel_size_direct, pixel_size_units)
            print(f"Using direct pixel_size from config: {pixel_size_direct} {pixel_size_units}/pixel")
            print(f"Converted to k-space: {pixel_size_k_converted:.6f} (1/Å)/pixel")
            dky = dkx = pixel_size_k_converted
        else:
            dky = cal.get('pixel_size_ky', cal.get('pixel_size_k', 1.0))
            dkx = cal.get('pixel_size_kx', cal.get('pixel_size_k', 1.0))
            
        # Log convergence angle if provided
        convergence_angle = cal.get('convergence_angle')
        if convergence_angle is not None:
            print(f"Convergence angle: {convergence_angle} mrad")
        
        # Ensure we have valid calibration values
        if dky is None:
            dky = 1.0
            print("Warning: pixel_size_ky is None, using default value 1.0")
        if dkx is None:
            dkx = 1.0
            print("Warning: pixel_size_kx is None, using default value 1.0")
        
        y = (np.arange(Ky) - cy) * dky
        x = (np.arange(Kx) - cx) * dkx
        ky_grid, kx_grid = np.meshgrid(y, x, indexing='ij')
        print(f"Created k-grids: center=({cy}, {cx}), dky={dky}, dkx={dkx}")
        return ky_grid, kx_grid

    def save_outputs(self, output_dir: Path, dpc_results: Dict, additional_outputs: Dict = None):
        """Save DPC outputs and any additional analysis results."""
        print("Saving outputs...")
        
        # Create directory structure
        output_dir = Path(output_dir)
        (output_dir / 'realspace').mkdir(parents=True, exist_ok=True)
        (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
        
        # Save DPC results
        for name, data in dpc_results.items():
            tifffile.imwrite(output_dir / 'realspace' / f'{name}.tif', data.astype(np.float32))
        
        # Save additional outputs if provided
        if additional_outputs:
            for name, data in additional_outputs.items():
                if isinstance(data, dict):
                    # Handle nested dictionaries (like contrast maps)
                    for sub_name, sub_data in data.items():
                        filename = f'{name}_{sub_name}.tif'
                        tifffile.imwrite(output_dir / 'realspace' / filename, sub_data.astype(np.float32))
                else:
                    tifffile.imwrite(output_dir / 'realspace' / f'{name}.tif', data.astype(np.float32))
        
        print(f"Saved outputs to {output_dir}")


class ClusterDPCAnalyzer(BaseDPCAnalyzer):
    """
    Data-driven virtual detectors for 4D-STEM analysis using cluster-based templates.
    
    This class implements the cluster-guided approach:
    1. Template generation from clustered diffraction data
    2. Virtual detector gate design based on cluster signatures  
    3. Matched-filter contrast mapping for cluster visualization
    4. Cluster-gated DPC/CoM analysis for polarization mapping
    
    The approach leverages unsupervised learning (clustering) to identify
    meaningful k-space signatures, then uses these to design optimal virtual
    detectors that enhance contrast for specific structural features.
    
    Key Methods:
    ------------
    build_templates() : Generate z-scored, L2-normalized templates for each cluster
    generate_gate() : Convert templates to binary detector gates with symmetrization
    compute_contrast_maps() : Apply matched filtering for cluster-specific contrast
    compute_dpc_com() : Perform cluster-gated differential phase contrast analysis
    
    The class supports both CPU and GPU computation (via CuPy) and includes
    comprehensive timing and validation metrics.
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        self.templates = {}
        self.gates = {}
        self.mu_clusters = {}

    def load_labels(self, labels_path: Path, scan_shape: Tuple[int, int]) -> np.ndarray:
        """Load cluster labels."""
        if labels_path.suffix == '.npy':
            labels = np.load(labels_path)
        elif labels_path.suffix == '.npz':
            data = np.load(labels_path)
            if 'cluster_labels' in data:
                labels = data['cluster_labels']
            elif 'labels' in data:
                labels = data['labels']
            else:
                keys = list(data.keys())
                labels = data[keys[0]]
                print(f"Warning: Using '{keys[0]}' as cluster labels")
        else:
            raise ValueError(f"Unsupported label format: {labels_path.suffix}")
        
        # Ensure labels match scan shape
        total_expected = scan_shape[0] * scan_shape[1]
        if len(labels) != total_expected:
            raise ValueError(f"Label count ({len(labels)}) != scan size ({total_expected})")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"Loaded {len(labels)} labels with {n_clusters} clusters")
        
        return labels
    
    def build_templates(self, dataset: H5Dataset, labels: np.ndarray, chunk_size: int = 1024):
        """
        Build cluster-guided templates in k-space.
        
        This is the core method that generates differential templates highlighting
        the k-space features that distinguish each cluster. The process:
        
        1. Computes cluster-wise mean patterns: μ_c(k) = mean_{i∈cluster_c} I_i(k)
        2. Computes global statistics: μ(k), σ(k) over all patterns
        3. Creates z-scored templates: T_c(k) = (μ_c(k) - μ(k)) / (σ(k) + ε)
        4. L2-normalizes templates to ensure fair comparison
        5. Generates detector gates G_c(k) from templates
        
        Templates are processed and stored in both self.templates and self.gates
        dictionaries, indexed by cluster ID.
        
        Parameters:
        -----------
        dataset : H5Dataset
            Lazy-loaded 4D-STEM dataset
        labels : np.ndarray
            Cluster labels for each diffraction pattern (shape: N,)
        chunk_size : int, optional
            Number of patterns to process at once (default: 1024)
            
        Notes:
        ------
        - Applies preprocessing (log transform, central masking) before template generation
        - Templates represent deviations from the global mean in units of global std
        - L2 normalization prevents templates with large support from dominating
        - Empty clusters (count=0) are skipped with a warning
        """
        unique_clusters = np.unique(labels)
        valid_clusters = [c for c in unique_clusters if c != -1]
        
        print(f"Building templates for {len(valid_clusters)} clusters...")
        
        # Beam center detection with proper priority: config > interactive > auto-detect
        cal = self.config.get('calibration', {})
        cy, cx = cal.get('center_y'), cal.get('center_x')
        interactive_mode = self.config.get('interactive_center', False)
        interactive_kspace_mode = self.config.get('interactive_kspace', False)
        known_spacing = self.config.get('known_spacing', None)
        
        # Load sample data if needed for detection or calibration
        sample_frames = None
        sample_mean = None
        if (cy is None or cx is None or interactive_mode or interactive_kspace_mode):
            sample_frames = dataset.load_chunk(0, min(100, dataset.total_frames))
            sample_mean = np.mean(sample_frames, axis=0).astype(np.float32)
        
        # Priority 1: Use config values if both are specified
        if cy is not None and cx is not None:
            print(f"Using beam center from config: ({cy}, {cx})")
        else:
            # Priority 2: Interactive mode if enabled and config incomplete
            if interactive_mode:
                print("Interactive beam center picking enabled...")
                cy_picked, cx_picked = interactive_beam_center_picker(sample_mean)
                cy, cx = cy_picked, cx_picked
                self.config.setdefault('calibration', {})['center_y'] = cy
                self.config.setdefault('calibration', {})['center_x'] = cx
                print(f"Interactively selected beam center: ({cy}, {cx})")
            else:
                # Priority 3: Auto-detect missing coordinates
                cy_det, cx_det = np.unravel_index(np.argmax(ndimage.gaussian_filter(sample_mean, 3)), sample_mean.shape)
                if cy is None:
                    cy = int(cy_det)
                    self.config.setdefault('calibration', {})['center_y'] = cy
                if cx is None:
                    cx = int(cx_det)
                    self.config.setdefault('calibration', {})['center_x'] = cx
                print(f"Auto-detected beam center: ({cy}, {cx})")
        
        center = (cy, cx) if cy is not None and cx is not None else None
        
        # Print final center coordinates being used for all calculations
        print(f"FINAL BEAM CENTER: ({cy}, {cx}) - This will be used for all calculations")
        
        # Interactive k-space calibration if requested
        if interactive_kspace_mode:
            print("Interactive k-space calibration enabled...")
            pixel_size_ky, pixel_size_kx = interactive_kspace_calibration(
                sample_mean, (cy, cx), known_spacing
            )
            # Update config with new calibration
            # Update both anisotropic and direct pixel size fields
            self.config.setdefault('calibration', {})['pixel_size_ky'] = pixel_size_ky
            self.config.setdefault('calibration', {})['pixel_size_kx'] = pixel_size_kx
            
            # If isotropic (ky == kx), also set the direct pixel_size for future precedence
            if abs(pixel_size_ky - pixel_size_kx) < 1e-10:
                self.config.setdefault('calibration', {})['pixel_size'] = pixel_size_ky
                print(f"K-space calibration updated: isotropic pixel_size={pixel_size_ky:.6f}")
            else:
                print(f"K-space calibration updated: anisotropic ky={pixel_size_ky:.6f}, kx={pixel_size_kx:.6f}")
        
        # Initialize preprocessor with detected center
        prep = Preprocessor(self.config.get('preprocess', {}), (dataset.Ky, dataset.Kx), center)
        
        # Initialize accumulators
        cluster_sums = {c: None for c in valid_clusters}
        cluster_counts = {c: 0 for c in valid_clusters}
        global_sum = None
        global_sum_sq = None
        total_count = 0
        
        # Process data in chunks
        n_chunks = (dataset.total_frames + chunk_size - 1) // chunk_size
        
        with Timer(f"Processing {n_chunks} chunks for template building"):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                frames = dataset.load_chunk(start_idx, chunk_size)
                
                # Apply preprocessing
                frames = prep(frames)
                
                # Promote to float64 for accurate accumulation
                if self.use_gpu:
                    frames = self.xp.asarray(frames, dtype=self.xp.float64)
                else:
                    frames = frames.astype(np.float64)
                
                # Update global statistics
                if global_sum is None:
                    global_sum = self.xp.sum(frames, axis=0)
                    global_sum_sq = self.xp.sum(frames**2, axis=0)
                else:
                    global_sum += self.xp.sum(frames, axis=0)
                    global_sum_sq += self.xp.sum(frames**2, axis=0)
                
                total_count += len(frames)
                
                # Update cluster-specific sums
                end_idx = min(start_idx + chunk_size, dataset.total_frames)
                chunk_labels = labels[start_idx:end_idx]
                
                for c in valid_clusters:
                    cluster_mask = chunk_labels == c
                    if np.any(cluster_mask):
                        cluster_frames = frames[cluster_mask]
                        cluster_sum = self.xp.sum(cluster_frames, axis=0)
                        
                        if cluster_sums[c] is None:
                            cluster_sums[c] = cluster_sum
                        else:
                            cluster_sums[c] += cluster_sum
                        
                        cluster_counts[c] += len(cluster_frames)
                
                # Show progress
                if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                    progress = (chunk_idx + 1) / n_chunks * 100
                    print(f"  Progress: {progress:.1f}%")
        
        # Compute global statistics with stable variance
        self.mu_global = global_sum / total_count
        global_mean_sq = self.mu_global ** 2
        global_ex2 = global_sum_sq / total_count
        global_var = self.xp.maximum(global_ex2 - global_mean_sq, 0.0)
        self.sigma_global = self.xp.sqrt(global_var + 1e-12)
        
        # Compute cluster templates
        for c in valid_clusters:
            if cluster_counts[c] == 0:
                print(f"Warning: Skipping empty cluster {c}")
                continue
            
            mu_c = cluster_sums[c] / cluster_counts[c]
            
            # Store true cluster mean
            self.mu_clusters[c] = self.xp.asnumpy(mu_c) if self.use_gpu else mu_c
            
            # Template: T_c = (mu_c - mu) / (sigma + eps) with scale-aware epsilon
            eps = 1e-6 * float(self.xp.median(self.sigma_global))
            template = (mu_c - self.mu_global) / (self.sigma_global + eps)
            
            # L2 normalize template
            template = template / (self.xp.linalg.norm(template) + 1e-12)
            
            # Generate gate from template
            gate = self.generate_gate(template, c)
            
            if self.use_gpu:
                template = self.xp.asnumpy(template)
                gate = self.xp.asnumpy(gate)
            
            self.templates[c] = template
            self.gates[c] = gate
            
            print(f"  Cluster {c}: {cluster_counts[c]} frames, template range [{template.min():.3f}, {template.max():.3f}]")
        
        if self.use_gpu:
            self.mu_global = self.xp.asnumpy(self.mu_global)
            self.sigma_global = self.xp.asnumpy(self.sigma_global)
        
        print(f"Built {len(self.templates)} templates")
        
        # Optional split-half validation
        if self.config.get('validation', {}).get('enable_split_half', False):
            self.validate_split_half(dataset, labels, valid_clusters)
    
    def validate_split_half(self, dataset: H5Dataset, labels: np.ndarray, valid_clusters: List[int]):
        """
        Perform split-half validation to check template stability.
        
        Randomly splits the data into two halves, rebuilds templates on each half,
        and computes SSIM between corresponding templates. Low SSIM indicates
        overfitting or noisy templates.
        """
        print("🔍 Performing split-half validation...")
        
        # Randomly split labels into two halves
        np.random.seed(42)  # For reproducible results
        n_frames = len(labels)
        split_mask = np.random.rand(n_frames) < 0.5
        
        labels_a = labels.copy()
        labels_b = labels.copy()
        labels_a[~split_mask] = -1  # Mark as invalid
        labels_b[split_mask] = -1   # Mark as invalid
        
        # Quick template building for both halves (using smaller chunks)
        chunk_size = 512
        prep = Preprocessor(self.config.get('preprocess', {}), (dataset.Ky, dataset.Kx))
        
        for split_name, split_labels in [('A', labels_a), ('B', labels_b)]:
            # Initialize accumulators for this split
            cluster_sums = {c: None for c in valid_clusters}
            cluster_counts = {c: 0 for c in valid_clusters}
            global_sum = None
            total_count = 0
            
            # Process data in chunks (limited for speed)
            n_chunks = min(10, (dataset.total_frames + chunk_size - 1) // chunk_size)
            
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                frames = dataset.load_chunk(start_idx, chunk_size)
                frames = prep(frames)
                
                if self.use_gpu:
                    frames = self.xp.asarray(frames)
                
                # Update global statistics
                if global_sum is None:
                    global_sum = self.xp.sum(frames, axis=0)
                else:
                    global_sum += self.xp.sum(frames, axis=0)
                total_count += len(frames)
                
                # Update cluster-specific sums
                end_idx = min(start_idx + chunk_size, dataset.total_frames)
                chunk_labels = split_labels[start_idx:end_idx]
                
                for c in valid_clusters:
                    cluster_mask = chunk_labels == c
                    if np.any(cluster_mask):
                        cluster_frames = frames[cluster_mask]
                        cluster_sum = self.xp.sum(cluster_frames, axis=0)
                        
                        if cluster_sums[c] is None:
                            cluster_sums[c] = cluster_sum
                        else:
                            cluster_sums[c] += cluster_sum
                        
                        cluster_counts[c] += len(cluster_frames)
            
            # Compute templates for this split
            mu_global = global_sum / total_count
            templates_split = {}
            
            for c in valid_clusters:
                if cluster_counts[c] > 0:
                    mu_c = cluster_sums[c] / cluster_counts[c]
                    template = (mu_c - mu_global) / (self.xp.sqrt((mu_c - mu_global)**2 + 1e-8))
                    template = template / (self.xp.linalg.norm(template) + 1e-12)
                    
                    if self.use_gpu:
                        template = self.xp.asnumpy(template)
                    
                    templates_split[c] = template
            
            # Store templates for this split
            if split_name == 'A':
                templates_a = templates_split
            else:
                templates_b = templates_split
        
        # Compare templates using SSIM
        print("  Split-half SSIM results:")
        total_ssim = 0
        valid_pairs = 0
        
        for c in valid_clusters:
            if c in templates_a and c in templates_b:
                ssim_val = ssim(templates_a[c], templates_b[c], data_range=1.0)
                print(f"    Cluster {c}: SSIM = {ssim_val:.3f}")
                total_ssim += ssim_val
                valid_pairs += 1
                
                if ssim_val < 0.5:
                    print(f"    Warning: Low SSIM for cluster {c} suggests unstable template")
        
        if valid_pairs > 0:
            mean_ssim = total_ssim / valid_pairs
            print(f"  Mean SSIM: {mean_ssim:.3f}")
            if mean_ssim < 0.5:
                print("  Warning: Low mean SSIM suggests template overfitting or noise")
            else:
                print("  Templates appear stable")
        
        print("Split-half validation completed")
    
    def generate_gate(self, template: np.ndarray, cluster_id: int) -> np.ndarray:
        """
        Generate detector gate from template using percentile thresholding.
        
        Gates define which k-space regions contribute to DPC and contrast analysis
        for each cluster. The process:
        
        1. Percentile thresholding: Keep top X% of template values
        2. Symmetrization: G = 0.5 * (G + G[::-1, ::-1]) to preserve CoM accuracy
        3. Optional radial band-limiting to exclude unwanted reflections
        4. Gaussian smoothing to reduce artifacts
        
        The symmetrization step is crucial for unbiased DPC analysis - it ensures
        that the gate doesn't introduce systematic deflections in the CoM calculation.
        
        Parameters:
        -----------
        template : np.ndarray
            Differential template T_c(k) for this cluster
        cluster_id : int
            Cluster identifier (for logging/debugging)
            
        Returns:
        --------
        gate : np.ndarray
            Binary detector gate in range [0, 1], same shape as template
            
        Configuration Parameters:
        -------------------------
        templates.top_percent : float (default: 10)
            Percentage of template values to keep (e.g., 10 = top 10%)
        templates.roi_r : List[float] (default: None)
            Radial ROI limits [r_min, r_max] in pixels from detector center
        templates.smooth_sigma : float (default: 1.0)
            Gaussian smoothing sigma for final gate
        """
        cfg = self.config.get('templates', {})
        top_percent = cfg.get('top_percent', 10)             # keep top 10% by default
        smooth_sigma = cfg.get('smooth_sigma', 1.0)
        rmin, rmax = cfg.get('roi_r', [None, None])          # radial ring in pixels
        gate_type = cfg.get('gate_type', 'hard')             # 'hard', 'soft', or 'smooth_first'

        if gate_type == 'soft':
            # Soft gates: use template directly as weights without thresholding
            gate = template.copy()
            # Normalize to [0,1] range
            gate = (gate - gate.min()) / (gate.max() - gate.min() + 1e-12)
        elif gate_type == 'smooth_first':
            # Smooth template first, then threshold
            smooth_template = ndimage.gaussian_filter(template, smooth_sigma) if smooth_sigma > 0 else template
            thr = np.percentile(smooth_template, 100 - top_percent)
            gate = (smooth_template >= thr).astype(np.float32)
        else:
            # Hard gates (original): threshold first, then smooth
            thr = np.percentile(template, 100 - top_percent)
            gate = (template >= thr).astype(np.float32)

        # symmetrize to preserve odd (kx, ky) moment behavior
        gate = 0.5 * (gate + gate[::-1, ::-1])

        # optional radial band-limit
        if rmin is not None or rmax is not None:
            Ky, Kx = gate.shape
            yy, xx = np.mgrid[:Ky, :Kx]
            cy = self.config.get('calibration', {}).get('center_y', Ky//2)
            cx = self.config.get('calibration', {}).get('center_x', Kx//2)
            
            # Use detector center if not specified (before auto-detection)
            if cy is None:
                cy = Ky // 2
            if cx is None:
                cx = Kx // 2
                
            rr = np.hypot(yy - cy, xx - cx)
            ring = np.ones_like(gate, dtype=np.float32)
            if rmin is not None: 
                ring *= (rr >= rmin)
            if rmax is not None: 
                ring *= (rr <= rmax)
            gate *= ring

        # Apply final smoothing only for hard gates (soft/smooth_first already handled)
        if gate_type == 'hard' and smooth_sigma > 0:
            gate = ndimage.gaussian_filter(gate, smooth_sigma)

        gate = np.clip(gate, 0, 1)
        
        # Report gate coverage
        coverage = float((gate > 0.1).mean())
        print(f"  Gate {cluster_id}: coverage={coverage:.1%}")
        
        return gate
    
    def compute_contrast_maps(self, dataset: H5Dataset, labels: np.ndarray, chunk_size: int = 1024) -> Dict[int, np.ndarray]:
        """Compute matched-filter contrast maps for each cluster."""
        print("Computing matched-filter contrast maps...")
        
        # Get center from config (should be set during template building)
        cal = self.config.get('calibration', {})
        center = (cal.get('center_y'), cal.get('center_x'))
        center = center if center[0] is not None and center[1] is not None else None
        
        # Initialize preprocessor with same center as template building
        prep = Preprocessor(self.config.get('preprocess', {}), (dataset.Ky, dataset.Kx), center)
        
        valid_clusters = list(self.templates.keys())
        Ny, Nx = dataset.scan_shape
        
        # Initialize contrast maps
        contrast_maps = {c: np.zeros((Ny, Nx), dtype=np.float32) for c in valid_clusters}
        
        n_chunks = (dataset.total_frames + chunk_size - 1) // chunk_size
        
        with Timer(f"Processing {n_chunks} chunks for contrast mapping"):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                frames = dataset.load_chunk(start_idx, chunk_size)
                end_idx = min(start_idx + chunk_size, dataset.total_frames)
                
                # Apply preprocessing
                frames = prep(frames)
                
                if self.use_gpu:
                    frames = self.xp.asarray(frames)
                    mu_global = self.xp.asarray(self.mu_global)
                    sigma_global = self.xp.asarray(self.sigma_global)
                else:
                    mu_global = self.mu_global
                    sigma_global = self.sigma_global
                
                # Z-score normalization with scale-aware epsilon and whitening options
                eps = 1e-6 * float(self.xp.median(sigma_global))
                whiten_mode = self.config.get('preprocess', {}).get('whiten', 'none')
                
                if whiten_mode == 'diag':
                    # Diagonal whitening (standard z-score per pixel)
                    z_frames = (frames - mu_global[None, :, :]) / (sigma_global[None, :, :] + eps)
                elif whiten_mode == 'none':
                    # Mean-centering only, no normalization
                    z_frames = frames - mu_global[None, :, :]
                else:
                    # Default: full z-score normalization 
                    z_frames = (frames - mu_global[None, :, :]) / (sigma_global[None, :, :] + eps)
                
                # Compute contrast for each cluster
                for c in valid_clusters:
                    template = self.xp.asarray(self.templates[c]) if self.use_gpu else self.templates[c]
                    
                    # S_c = sum_k T_c(k) * Z_i(k)
                    contrast_chunk = self.xp.sum(template[None, :, :] * z_frames, axis=(1, 2))
                    
                    if self.use_gpu:
                        contrast_chunk = self.xp.asnumpy(contrast_chunk)
                    
                    # Map back to 2D scan coordinates
                    chunk_indices = np.arange(start_idx, end_idx)
                    y_coords = chunk_indices // Nx
                    x_coords = chunk_indices % Nx
                    
                    contrast_maps[c][y_coords, x_coords] = contrast_chunk
                
                # Progress update
                if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                    progress = (chunk_idx + 1) / n_chunks * 100
                    print(f"  Progress: {progress:.1f}%")
        
        print(f"Computed contrast maps for {len(valid_clusters)} clusters")
        return contrast_maps
    
    def compute_argmax_confidence(self, contrast_maps: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute argmax map and confidence from contrast maps."""
        print("Computing argmax and confidence maps...")
        
        clusters = list(contrast_maps.keys())
        Ny, Nx = list(contrast_maps.values())[0].shape
        
        # Stack all contrast maps
        contrast_stack = np.stack([contrast_maps[c] for c in clusters], axis=0)
        
        # Softmax for confidence with temperature
        tau = self.config.get('matched_filter', {}).get('softmax_temp', 2.0)
        max_vals = np.max(contrast_stack, axis=0, keepdims=True)
        exp_vals = np.exp((contrast_stack - max_vals) / tau)
        softmax_vals = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
        
        # Argmax map
        argmax_indices = np.argmax(contrast_stack, axis=0)
        argmax_map = np.array([clusters[i] for i in argmax_indices.flat]).reshape(Ny, Nx)
        
        # Confidence (max softmax value)
        confidence_map = np.max(softmax_vals, axis=0)
        
        print(f"Computed argmax (range: {argmax_map.min()}-{argmax_map.max()}) and confidence (mean: {confidence_map.mean():.3f})")
        
        return argmax_map, confidence_map
    
    def make_k_grids(self, detector_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create k-space coordinate grids."""
        Ky, Kx = detector_shape
        cal = self.config.get('calibration', {})
        cy, cx = cal.get('center_y'), cal.get('center_x')

        if cy is None or cx is None:
            mu = self.mu_global if self.mu_global is not None else np.zeros(detector_shape)
            cy, cx = np.unravel_index(np.argmax(ndimage.gaussian_filter(mu, 3)), mu.shape)
            self.config.setdefault('calibration', {})['center_y'] = int(cy)
            self.config.setdefault('calibration', {})['center_x'] = int(cx)
            print(f"Auto-detected beam center: ({cy}, {cx})")

        # Support anisotropic pixel sizes with precedence hierarchy:
        # 1. Direct pixel_size (if specified, takes precedence)
        # 2. Anisotropic pixel_size_ky/pixel_size_kx 
        # 3. Fallback to pixel_size_k
        
        pixel_size_direct = cal.get('pixel_size')
        if pixel_size_direct is not None:
            # Convert pixel_size to k-space units (1/Å per pixel)
            pixel_size_units = cal.get('pixel_size_units', 'nm')
            pixel_size_k_converted = self._convert_pixel_size_to_k_space(pixel_size_direct, pixel_size_units)
            print(f"Using direct pixel_size from config: {pixel_size_direct} {pixel_size_units}/pixel")
            print(f"Converted to k-space: {pixel_size_k_converted:.6f} (1/Å)/pixel")
            dky = dkx = pixel_size_k_converted
        else:
            dky = cal.get('pixel_size_ky', cal.get('pixel_size_k', 1.0))
            dkx = cal.get('pixel_size_kx', cal.get('pixel_size_k', 1.0))
            
        # Log convergence angle if provided
        convergence_angle = cal.get('convergence_angle')
        if convergence_angle is not None:
            print(f"Convergence angle: {convergence_angle} mrad")
        
        # Ensure we have valid calibration values
        if dky is None:
            dky = 1.0
            print("Warning: pixel_size_ky is None, using default value 1.0")
        if dkx is None:
            dkx = 1.0
            print("Warning: pixel_size_kx is None, using default value 1.0")
        
        y = (np.arange(Ky) - cy) * dky
        x = (np.arange(Kx) - cx) * dkx
        ky_grid, kx_grid = np.meshgrid(y, x, indexing='ij')
        print(f"Created k-grids: center=({cy}, {cx}), dky={dky}, dkx={dkx}")
        return ky_grid, kx_grid
    
    def compute_dpc_com(self, dataset: H5Dataset, labels: np.ndarray, 
                       ky_grid: np.ndarray, kx_grid: np.ndarray, 
                       chunk_size: int = 1024) -> Dict[str, np.ndarray]:
        """
        Compute differential phase contrast (DPC) using cluster-gated center-of-mass.
        
        This method implements the core DPC analysis with cluster-specific gating
        to enhance sensitivity to particular structural features. The process:
        
        1. Precompute radial ROI masks and gate lookup tables for efficiency
        2. For each chunk, select appropriate gates (union or adaptive mode)
        3. Compute vectorized center-of-mass: C_x = Σ(k_x * G * I) / Σ(G * I)
        4. Apply baseline correction to remove systematic offsets
        5. Convert CoM shifts to projected electric field components
        
        Two gating modes are supported:
        - 'union': Use union of all gates (max coverage, vectorized processing)
        - 'adaptive': Use gate of assigned cluster for each probe position (vectorized)
        
        Parameters:
        -----------
        dataset : H5Dataset
            4D-STEM dataset (raw intensities used, not preprocessed)
        labels : np.ndarray
            Cluster assignments for adaptive gating mode
        ky_grid, kx_grid : np.ndarray
            K-space coordinate grids from make_k_grids()
        chunk_size : int, optional
            Processing chunk size for memory efficiency
            
        Returns:
        --------
        results : Dict[str, np.ndarray]
            Dictionary containing:
            - 'Ex', 'Ey': Electric field components (baseline corrected)
            - 'E_mag': Field magnitude |E| = sqrt(Ex² + Ey²)
            - 'E_angle': Field direction θ = arctan2(Ey, Ex)
            - 'com_x', 'com_y': Raw center-of-mass maps (before field conversion)
            
        Configuration Parameters:
        -------------------------
        dpc.gate_mode : str (default: 'union')
            Gating strategy: 'union' or 'adaptive'
        dpc.baseline : str (default: 'median')  
            Baseline correction: 'median', 'rowcol', or 'roi'
        dpc.roi : List[float] (default: None)
            Radial ROI for DPC analysis [r_min, r_max]
        dpc.field_scale : float (default: 1.0)
            Conversion factor from CoM shifts to electric field units
            
        Notes:
        ------
        - Uses raw (unprocessed) intensities to preserve CoM accuracy
        - Vectorized processing for significant speed improvements
        - Precomputed gate lookup and radial masks optimize adaptive mode
        - Baseline correction is essential to remove beam center uncertainties  
        - Row/column baseline correction suppresses scan drift and residual tilt
        - Field calibration depends on camera length and experimental geometry
        - Union gating provides maximum signal, adaptive gating maximum specificity
        """
        print("Computing DPC/CoM with cluster gating...")
        
        dpc_config = self.config.get('dpc', {})
        gate_mode = dpc_config.get('gate_mode', 'union')  # 'union' or 'adaptive'
        
        Ny, Nx = dataset.scan_shape
        
        # Initialize output maps
        com_x = np.zeros((Ny, Nx), dtype=np.float32)
        com_y = np.zeros((Ny, Nx), dtype=np.float32)
        
        # Precompute radial ring mask in pixel coordinates (not k-space units)
        Ky, Kx = ky_grid.shape
        cal = self.config.get('calibration', {})
        cy = cal.get('center_y', Ky//2)
        cx = cal.get('center_x', Kx//2)
        if cy is None: 
            cy = Ky//2
        if cx is None: 
            cx = Kx//2

        yy, xx = np.mgrid[:Ky, :Kx]
        rr_px = np.hypot(yy - cy, xx - cx)

        if 'roi' in dpc_config:
            rmin, rmax = dpc_config['roi']
            ring = ((rmin is None) | (rr_px >= rmin)) & ((rmax is None) | (rr_px <= rmax))
        else:
            ring = np.ones_like(rr_px, dtype=bool)
        
        # Safety net to catch empty ROI masks
        cov_ring = float((ring > 0).mean())
        if cov_ring == 0:
            print("Warning: DPC ROI ring is empty. Disabling ROI for this run.")
            ring = np.ones_like(ring, dtype=bool)

        # Precompute gate lookup for adaptive mode
        gate_ids = sorted(self.gates.keys())
        gate_stack = np.stack([self.gates[g] * ring for g in gate_ids], axis=0)  # (C, Ky, Kx)
        
        # Create union gate if needed
        if gate_mode == 'union':
            gate_union = np.zeros_like(list(self.gates.values())[0])
            for gate in self.gates.values():
                gate_union = np.maximum(gate_union, gate)
            gate_union = gate_union * ring
            
            # Safety check for union gate coverage
            cov_union = float((gate_union > 0.1).mean())
            print(f"  Using union gate (coverage: {cov_union:.1%})")
            if cov_union < 1e-4:
                print("Warning: Union gate coverage ≈ 0%. Check ROI units (pixels vs k-units) and gating thresholds.")
        
        n_chunks = (dataset.total_frames + chunk_size - 1) // chunk_size
        
        with Timer(f"Processing {n_chunks} chunks for DPC/CoM"):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                frames = dataset.load_chunk(start_idx, chunk_size)
                end_idx = min(start_idx + chunk_size, dataset.total_frames)
                
                # Promote to float64 for accurate CoM calculations
                if self.use_gpu:
                    frames = self.xp.asarray(frames, dtype=self.xp.float64)
                    ky_grid_gpu = self.xp.asarray(ky_grid, dtype=self.xp.float64)
                    kx_grid_gpu = self.xp.asarray(kx_grid, dtype=self.xp.float64)
                else:
                    frames = frames.astype(np.float64)
                    ky_grid_gpu = ky_grid.astype(np.float64)
                    kx_grid_gpu = kx_grid.astype(np.float64)
                
                chunk_indices = np.arange(start_idx, end_idx)
                chunk_labels = labels[chunk_indices]
                
                if gate_mode == 'union':
                    # Vectorized computation for union mode
                    gate = self.xp.asarray(gate_union) if self.use_gpu else gate_union
                    
                    # Vectorized CoM for the whole chunk
                    weighted = gate[None, :, :] * frames  # Broadcasting
                    den = self.xp.sum(weighted, axis=(1,2), keepdims=True) + 1e-8
                    com_x_vals = self.xp.sum(kx_grid_gpu[None, :, :] * weighted, axis=(1,2)) / den.squeeze()
                    com_y_vals = self.xp.sum(ky_grid_gpu[None, :, :] * weighted, axis=(1,2)) / den.squeeze()
                    
                    if self.use_gpu:
                        com_x_vals = self.xp.asnumpy(com_x_vals)
                        com_y_vals = self.xp.asnumpy(com_y_vals)
                    
                    # Map to scan coordinates
                    y_coords = chunk_indices // Nx
                    x_coords = chunk_indices % Nx
                    com_x[y_coords, x_coords] = com_x_vals
                    com_y[y_coords, x_coords] = com_y_vals
                    
                elif gate_mode == 'adaptive':
                    # Vectorized computation for adaptive mode with proper label mapping
                    id_to_idx = {cid: i for i, cid in enumerate(gate_ids)}
                    default_idx = 0  # Use first gate as fallback
                    lbl_idx = np.array([id_to_idx.get(int(lbl), default_idx) for lbl in chunk_labels], dtype=int)
                    
                    chunk_gates = gate_stack[lbl_idx]  # (Nchunk, Ky, Kx)
                    
                    if self.use_gpu:
                        chunk_gates = self.xp.asarray(chunk_gates)
                    
                    # Vectorized CoM for the whole chunk
                    weighted = chunk_gates * frames
                    den = self.xp.sum(weighted, axis=(1,2), keepdims=True) + 1e-8
                    com_x_vals = self.xp.sum(kx_grid_gpu[None, :, :] * weighted, axis=(1,2)) / den.squeeze()
                    com_y_vals = self.xp.sum(ky_grid_gpu[None, :, :] * weighted, axis=(1,2)) / den.squeeze()
                    
                    if self.use_gpu:
                        com_x_vals = self.xp.asnumpy(com_x_vals)
                        com_y_vals = self.xp.asnumpy(com_y_vals)
                    
                    # Map to scan coordinates
                    y_coords = chunk_indices // Nx
                    x_coords = chunk_indices % Nx
                    com_x[y_coords, x_coords] = com_x_vals
                    com_y[y_coords, x_coords] = com_y_vals
                
                # Progress update
                if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                    progress = (chunk_idx + 1) / n_chunks * 100
                    print(f"  Progress: {progress:.1f}%")
        
        # Subtract baseline after CoM
        mode = self.config.get('dpc', {}).get('baseline', 'median')
        if mode == 'median':
            com_x -= np.median(com_x)
            com_y -= np.median(com_y)
            print(f"  Applied median baseline correction")
        elif mode == 'rowcol':
            # Standard row/column median baseline
            com_x -= np.median(com_x, axis=1, keepdims=True)
            com_y -= np.median(com_y, axis=0, keepdims=True)
            
            # Add running median filter to remove scan drift waves
            k = dpc_config.get('running_median_kernel', 31)  # odd window size
            if k > 1:
                # Apply 1D running median along fast-scan direction
                com_x -= medfilt(com_x, kernel_size=(1, k))
                com_y -= medfilt(com_y, kernel_size=(k, 1))
                print(f"  Applied row/column + running median (k={k}) baseline correction")
            else:
                print(f"  Applied row/column baseline correction")
        elif mode == 'roi':
            x0, y0, w, h = self.config['dpc']['baseline_region']
            bx = np.mean(com_x[y0:y0+h, x0:x0+w])
            by = np.mean(com_y[y0:y0+h, x0:x0+w])
            com_x -= bx
            com_y -= by
            print(f"  Applied ROI baseline correction: ({bx:.4f}, {by:.4f})")
        
        # Convert to projected fields (simplified - assumes linear relationship)
        field_scale = dpc_config.get('field_scale', 1.0)
        Ex = com_x * field_scale
        Ey = com_y * field_scale
        
        # Compute magnitude and angle
        E_mag = np.sqrt(Ex**2 + Ey**2)
        E_angle = np.arctan2(Ey, Ex)
        
        print(f"Computed DPC fields: |E| range [{E_mag.min():.4f}, {E_mag.max():.4f}]")
        
        return {
            'Ex': Ex,
            'Ey': Ey, 
            'E_mag': E_mag,
            'E_angle': E_angle,
            'com_x': com_x,
            'com_y': com_y
        }
    
    def save_outputs(self, output_dir: Path, contrast_maps: Dict, argmax_map: np.ndarray, 
                    confidence_map: np.ndarray, dpc_results: Dict):
        """Save all outputs to structured directories."""
        print("Saving outputs...")
        
        # Create directory structure
        output_dir = Path(output_dir)
        (output_dir / 'kspace').mkdir(parents=True, exist_ok=True)
        (output_dir / 'realspace').mkdir(parents=True, exist_ok=True)
        (output_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (output_dir / 'logs').mkdir(parents=True, exist_ok=True)
        
        # Save k-space artifacts
        for cluster_id in self.templates.keys():
            # Templates
            np.save(output_dir / 'kspace' / f'template_T_{cluster_id}.npy', 
                   self.templates[cluster_id])
            # Gates  
            np.save(output_dir / 'kspace' / f'gate_G_{cluster_id}.npy', 
                   self.gates[cluster_id])
            
            # Save and visualize true cluster mean
            np.save(output_dir / 'kspace' / f'mu_cluster_{cluster_id}.npy', self.mu_clusters[cluster_id])
            plt.figure(figsize=(6, 6))
            plt.imshow(self.mu_clusters[cluster_id], cmap='viridis')
            plt.title(f'Mean Pattern - Cluster {cluster_id}')
            plt.colorbar()
            plt.savefig(output_dir / 'kspace' / f'mu_cluster_{cluster_id}.png', dpi=150)
            plt.close()
        
        # Save global statistics
        np.save(output_dir / 'kspace' / 'mu_global.npy', self.mu_global)
        np.save(output_dir / 'kspace' / 'sigma_global.npy', self.sigma_global)
        
        # Save real-space maps
        for cluster_id, contrast_map in contrast_maps.items():
            tifffile.imwrite(output_dir / 'realspace' / f'S_cluster_{cluster_id}.tif', 
                           contrast_map.astype(np.float32))
        
        tifffile.imwrite(output_dir / 'realspace' / 'S_argmax.tif', argmax_map)
        tifffile.imwrite(output_dir / 'realspace' / 'S_confidence.tif', confidence_map)
        
        # Save DPC results
        for name, data in dpc_results.items():
            tifffile.imwrite(output_dir / 'realspace' / f'{name}.tif', data.astype(np.float32))
        
        print(f"Saved outputs to {output_dir}")


class ConventionalDPCAnalyzer(BaseDPCAnalyzer):
    """
    Conventional DPC/CoM analysis for 4D-STEM data.
    
    This class implements standard differential phase contrast analysis:
    1. Beam center detection using multiple methods
    2. K-space calibration with interactive options
    3. Center-of-mass computation with various masking options
    4. Baseline correction and field calibration
    5. Polarization mapping and visualization
    
    Unlike the cluster-based approach, this uses conventional techniques:
    - Simple annular or disk detectors
    - Global beam center for all patterns
    - Standard CoM calculation without gating
    """
    
    def __init__(self, config: Dict):
        super().__init__(config)
        
    def optimize_annular_detector(self, dataset: H5Dataset, center: Tuple[int, int]) -> Dict[str, float]:
        """
        Optimize annular detector parameters using radial profile analysis.
        
        Parameters:
        -----------
        dataset : H5Dataset
            4D-STEM dataset
        center : Tuple[int, int]
            Beam center coordinates (cy, cx)
            
        Returns:
        --------
        detector_params : Dict[str, float]
            Optimized detector parameters
        """
        # Load sample data for optimization
        sample_frames = dataset.load_chunk(0, min(100, dataset.total_frames))
        sample_mean = np.mean(sample_frames, axis=0).astype(np.float32)
        
        # Interactive optimization
        detector_params = interactive_annular_detector_optimization(sample_mean, center)
        
        # Update config with optimized parameters  
        dpc_config = self.config.setdefault('dpc', {})
        dpc_config['detector_mode'] = detector_params['mode']
        dpc_config['r1_inner'] = detector_params['r1_inner']
        dpc_config['r1_outer'] = detector_params['r1_outer'] 
        
        if detector_params['mode'] == 'two_ring':
            dpc_config['r2_inner'] = detector_params['r2_inner']
            dpc_config['r2_outer'] = detector_params['r2_outer']
            dpc_config['ring2_weight'] = detector_params['ring2_weight']
            
        print(f"Optimized detector: {detector_params}")
        return detector_params
    
    def compute_conventional_dpc(self, dataset: H5Dataset, 
                               ky_grid: np.ndarray, kx_grid: np.ndarray, 
                               chunk_size: int = 1024) -> Dict[str, np.ndarray]:
        """
        Compute conventional DPC/CoM analysis.
        
        This implements standard DPC analysis similar to py4DSTEM's approach:
        1. Use all pixels or a simple annular detector
        2. Compute center-of-mass for each diffraction pattern
        3. Apply baseline correction
        4. Convert to projected electric field
        
        Parameters:
        -----------
        dataset : H5Dataset
            4D-STEM dataset (uses raw intensities)
        ky_grid, kx_grid : np.ndarray
            K-space coordinate grids
        chunk_size : int
            Processing chunk size
            
        Returns:
        --------
        results : Dict[str, np.ndarray]
            Dictionary with 'Ex', 'Ey', 'E_mag', 'E_angle', 'com_x', 'com_y'
        """
        print("Computing conventional DPC/CoM analysis...")
        
        dpc_config = self.config.get('dpc', {})
        
        # Set defaults for new probe compensation features
        dpc_config.setdefault('radial_whitening', True)
        dpc_config.setdefault('radial_weighting', True)
        dpc_config.setdefault('radial_notching', True)
        dpc_config.setdefault('detector_symmetrization', True)
        dpc_config.setdefault('curl_free_projection', True)
        dpc_config.setdefault('auto_orientation', True)
        dpc_config.setdefault('vector_smoothing', True)
        dpc_config.setdefault('vector_smoothing_sigma', 1.0)
        
        Ny, Nx = dataset.scan_shape
        
        # Initialize output maps
        com_x = np.zeros((Ny, Nx), dtype=np.float32)
        com_y = np.zeros((Ny, Nx), dtype=np.float32)
        
        # Create detector mask
        Ky, Kx = ky_grid.shape
        cal = self.config.get('calibration', {})
        cy = cal.get('center_y', Ky//2)
        cx = cal.get('center_x', Kx//2)
        
        # Create detector mask (single-ring, two-ring, or legacy fallback)
        yy, xx = np.mgrid[:Ky, :Kx]
        rr_px = np.hypot(yy - cy, xx - cx)
        
        def soft_ring(rr_px, r_in, r_out, edge=2.0):
            """
            Create a soft ring detector with raised-cosine edges.
            
            Parameters:
            -----------
            rr_px : np.ndarray
                Radial distance array in pixels
            r_in : float
                Inner radius
            r_out : float  
                Outer radius
            edge : float
                Edge softness width in pixels (default 2.0)
                
            Returns:
            --------
            weight : np.ndarray
                Soft ring weight (0 outside, 1 inside, smooth transitions)
            """
            # Smooth transitions from 0 to 1 at inner edge and 1 to 0 at outer edge
            w = np.clip((rr_px - (r_in - edge)) / edge, 0, 1) * \
                np.clip(((r_out + edge) - rr_px) / edge, 0, 1)
            # Apply raised-cosine window for smoother transitions
            return 0.5 - 0.5 * np.cos(np.pi * np.clip(w, 0, 1))
        
        detector_mode = dpc_config.get('detector_mode', 'single')
        
        if detector_mode == 'two_ring':
            # Two-ring detector with optional weighting
            r1_inner = dpc_config.get('r1_inner', 6)
            r1_outer = dpc_config.get('r1_outer', 48)
            r2_inner = dpc_config.get('r2_inner', 80)
            r2_outer = dpc_config.get('r2_outer', 92)
            ring2_weight = dpc_config.get('ring2_weight', 0.5)
            
            # Create detector masks (soft or hard binary)
            enable_soft_rings = dpc_config.get('enable_soft_rings', True)
            edge_width = dpc_config.get('soft_ring_edge', 1.0)
            
            if enable_soft_rings and edge_width > 0:
                # Soft weighted masks with raised-cosine edges
                ring1 = soft_ring(rr_px, r1_inner, r1_outer, edge=edge_width)
                ring2 = soft_ring(rr_px, r2_inner, r2_outer, edge=edge_width)
            else:
                # Hard binary masks (legacy behavior)
                ring1 = ((rr_px >= r1_inner) & (rr_px <= r1_outer)).astype(np.float64)
                ring2 = ((rr_px >= r2_inner) & (rr_px <= r2_outer)).astype(np.float64)
            
            # Weighted combination: ring1 has weight 1.0, ring2 has configurable weight
            detector_mask = ring1 + ring2 * ring2_weight
            
            coverage1 = float((ring1 > 0).mean())  # Fractional coverage
            coverage2 = float((ring2 > 0).mean())
            # For two-ring: report sum of weights / total area
            total_coverage = float(detector_mask.sum() / detector_mask.size)
            
            print(f"  Using two-ring detector:")
            print(f"    Ring1: [{r1_inner}, {r1_outer}] px (coverage: {coverage1:.1%}, weight: 1.0)")
            print(f"    Ring2: [{r2_inner}, {r2_outer}] px (coverage: {coverage2:.1%}, weight: {ring2_weight:.1f})")
            print(f"    Total coverage: {total_coverage:.1%}")
            if enable_soft_rings and edge_width > 0:
                print(f"    Soft edges: {edge_width:.1f} px")
            else:
                print(f"    Hard binary edges (legacy mode)")
        
        else:
            # Single ring or legacy fallback
            if detector_mode == 'single':
                r1_inner = dpc_config.get('r1_inner', dpc_config.get('mask_inner_radius', 0))
                r1_outer = dpc_config.get('r1_outer', dpc_config.get('mask_outer_radius', None))
            else:
                # Legacy fallback
                r1_inner = dpc_config.get('mask_inner_radius', 0)
                r1_outer = dpc_config.get('mask_outer_radius', None)
                
            if r1_outer is None:
                r1_outer = min(Ky, Kx) // 2
            
            # Create single annular mask (soft or hard binary)
            enable_soft_rings = dpc_config.get('enable_soft_rings', True)
            edge_width = dpc_config.get('soft_ring_edge', 1.0)
            
            if enable_soft_rings and edge_width > 0:
                # Soft weighted mask with raised-cosine edges
                detector_mask = soft_ring(rr_px, r1_inner, r1_outer, edge=edge_width)
            else:
                # Hard binary mask (legacy behavior)
                detector_mask = ((rr_px >= r1_inner) & (rr_px <= r1_outer)).astype(np.float64)
            coverage = float((detector_mask > 0).mean())
            print(f"  Using single-ring detector: [{r1_inner}, {r1_outer}] px (coverage: {coverage:.1%})")
            if enable_soft_rings and edge_width > 0:
                print(f"    Soft edges: {edge_width:.1f} px")
            else:
                print(f"    Hard binary edges (legacy mode)")
        
        # Save original mask before any modifications
        original_mask = detector_mask.copy()
        
        # Build azimuthal coordinate array (needed for debugging and pruning)
        theta = np.arctan2(yy - cy, xx - cx)
        
        # Apply azimuthal pruning to remove Bragg disk bias
        enable_azimuthal_pruning = dpc_config.get('azimuthal_pruning', True)
        
        # Save detector mask for debugging if requested
        debug_mode = dpc_config.get('debug_detector', False)
        if debug_mode:
            import matplotlib.pyplot as plt
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
            
            # Show detector mask before pruning
            im1 = ax1.imshow(detector_mask, cmap='viridis')
            ax1.set_title('Detector Mask (Before Pruning)')
            ax1.set_xlabel('X (pixels)')
            ax1.set_ylabel('Y (pixels)')
            plt.colorbar(im1, ax=ax1)
            
            # Mark center
            ax1.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
            
        if enable_azimuthal_pruning:
            print("  Applying azimuthal pruning to suppress Bragg disks...")
            
            # Load sample data for analysis
            sample_frames = dataset.load_chunk(0, min(100, dataset.total_frames))
            sample_mean = np.mean(sample_frames, axis=0).astype(np.float32)
            
            def prune_bright_sectors(weight, sample_pattern, theta, keep_frac=0.9, n_theta=72):
                """
                Remove brightest azimuthal sectors to suppress Bragg disk bias.
                
                Parameters:
                -----------
                weight : np.ndarray
                    Detector weight array
                sample_pattern : np.ndarray
                    Sample diffraction pattern for analysis
                theta : np.ndarray
                    Azimuthal angle array
                keep_frac : float
                    Fraction of sectors to keep (default 0.9, removes top 10%)
                n_theta : int
                    Number of azimuthal bins (default 72, 5° per bin)
                    
                Returns:
                --------
                pruned_weight : np.ndarray
                    Weight array with bright sectors removed
                """
                # Create azimuthal bins
                theta_edges = np.linspace(-np.pi, np.pi, n_theta + 1)
                theta_idx = np.digitize(theta, theta_edges) - 1
                theta_idx = np.clip(theta_idx, 0, n_theta - 1)  # Ensure valid indices
                
                # Compute mean intensity per sector within detector region
                I_ring = sample_pattern * (weight > 1e-3)
                sector_mean = np.zeros(n_theta)
                
                for i in range(n_theta):
                    sector_mask = (theta_idx == i) & (weight > 1e-3)
                    if np.any(sector_mask):
                        sector_mean[i] = I_ring[sector_mask].mean()
                
                # Find bright sectors to remove
                if np.any(sector_mean > 0):
                    cutoff = np.quantile(sector_mean[sector_mean > 0], keep_frac)
                    bright_sectors = np.where(sector_mean > cutoff)[0]
                    
                    # Create mask for bright sectors
                    bright_mask = np.isin(theta_idx, bright_sectors)
                    pruned_weight = np.where(bright_mask, 0.0, weight)
                    
                    n_removed = len(bright_sectors)
                    print(f"    Removed {n_removed}/{n_theta} bright sectors ({n_removed/n_theta:.1%})")
                    
                    return pruned_weight
                else:
                    return weight
            
            # Apply pruning to detector mask
            keep_frac = dpc_config.get('azimuthal_keep_fraction', 0.9)
            detector_mask = prune_bright_sectors(detector_mask, sample_mean, theta, keep_frac=keep_frac)
        
        # Apply radial whitening for bullseye probe compensation
        enable_radial_whitening = dpc_config.get('radial_whitening', True)
        if enable_radial_whitening:
            print("  Applying radial whitening to compensate for bullseye probe effects...")
            
            # Use sample data for radial profile computation
            if 'sample_mean' not in locals():
                sample_frames = dataset.load_chunk(0, min(100, dataset.total_frames))
                sample_mean = np.mean(sample_frames, axis=0).astype(np.float32)
            
            def compute_radial_whitening(mean_dp, center_y, center_x, detector_shape):
                """
                Compute radial whitening gain map to flatten bullseye probe structure.
                
                This divides by the radial profile to compensate for bright rings in
                bullseye/hollow-cone probes that can bias center-of-mass calculations.
                """
                Ky, Kx = detector_shape
                yy, xx = np.indices((Ky, Kx), dtype=np.float32)
                rr = np.hypot(yy - center_y, xx - center_x).astype(np.int32)
                
                # Compute radial mean intensity profile
                nb = rr.max() + 1
                rad_sum = np.bincount(rr.ravel(), mean_dp.ravel(), minlength=nb)
                rad_cnt = np.bincount(rr.ravel(), minlength=nb).astype(np.float32)
                rad_mean = rad_sum / np.maximum(rad_cnt, 1)
                
                # Avoid divide-by-zero & over-amplifying dark rings
                min_intensity = np.percentile(rad_mean[rad_mean > 0], 5) if np.any(rad_mean > 0) else 1.0
                rad_gain = 1.0 / np.maximum(rad_mean, min_intensity)
                
                # Clamp gain to prevent extreme amplification
                median_gain = np.median(rad_gain[rad_gain > 0]) if np.any(rad_gain > 0) else 1.0
                max_gain = min(10.0, median_gain * 3.0)  # ≤10x absolute cap, 3x relative
                rad_gain = np.minimum(rad_gain, max_gain)
                
                # Create 2D gain map
                gain_map = rad_gain[rr]
                
                return gain_map, rad_mean, rad_gain
            
            gain_map, rad_profile, rad_gain = compute_radial_whitening(sample_mean, cy, cx, (Ky, Kx))
            
            # Apply whitening to detector mask
            detector_mask = detector_mask * gain_map
            
            # Add radial notch pruning for bullseye peaks
            enable_radial_notching = dpc_config.get('radial_notching', True)
            if enable_radial_notching:
                # Find peaks in radial profile
                peaks = np.where((rad_profile[1:-1] > rad_profile[:-2]) &
                                (rad_profile[1:-1] > rad_profile[2:]))[0] + 1
                
                # Get detector ring radii for filtering peaks
                r1_inner = dpc_config.get('r1_inner', 6)
                r1_outer = dpc_config.get('r1_outer', 48) 
                r2_inner = dpc_config.get('r2_inner', 80)
                r2_outer = dpc_config.get('r2_outer', 92)
                
                # Keep peaks that fall inside active detector rings (with small margin)
                valid_peaks = []
                for p in peaks:
                    if (r1_inner + 2 <= p <= r1_outer - 2) or (r2_inner + 2 <= p <= r2_outer - 2):
                        valid_peaks.append(p)
                
                if valid_peaks:
                    # Create radial coordinate map
                    rr_px = np.hypot(yy - cy, xx - cx)
                    notch = np.ones_like(detector_mask)
                    
                    for p in valid_peaks:
                        # Create 2-3 pixel wide radial notch
                        notch[(rr_px >= p - 1) & (rr_px <= p + 1)] = 0.0
                    
                    detector_mask *= notch
                    print(f"    Applied radial notches at {len(valid_peaks)} bullseye peaks: {valid_peaks}")
                else:
                    print("    No significant bullseye peaks found in detector rings")
            
            # Optional: Add 1/r weighting to reduce lever arm effects
            enable_radial_weighting = dpc_config.get('radial_weighting', True)
            if enable_radial_weighting:
                rr_float = np.hypot(yy - cy, xx - cx)
                radial_weight = 1.0 / np.maximum(rr_float, 1.0)
                detector_mask = detector_mask * radial_weight
                print("    Applied 1/r radial weighting to reduce lever arm effects")
            
            # Optional: Enforce 180° symmetry around center to eliminate residual Ex/Ey bias
            enable_symmetrization = dpc_config.get('detector_symmetrization', True)
            if enable_symmetrization:
                detector_mask = 0.5 * (detector_mask + detector_mask[::-1, ::-1])
                print("    Applied 180° symmetrization to eliminate directional bias")
            
            if debug_mode:
                # Save radial analysis plots
                import matplotlib.pyplot as plt
                fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
                
                # Original mean DP
                im1 = ax1.imshow(np.log(sample_mean + 1), cmap='gray')
                ax1.set_title('Original Mean Diffraction Pattern (log scale)')
                ax1.plot(cx, cy, 'r+', markersize=10)
                plt.colorbar(im1, ax=ax1, shrink=0.8)
                
                # Gain map
                im2 = ax2.imshow(gain_map, cmap='viridis')
                ax2.set_title('Radial Whitening Gain Map')
                ax2.plot(cx, cy, 'r+', markersize=10)
                plt.colorbar(im2, ax=ax2, shrink=0.8)
                
                # Radial profiles
                r_values = np.arange(len(rad_profile))
                ax3.semilogy(r_values, rad_profile, 'b-', label='Original intensity')
                ax3.set_xlabel('Radius (pixels)')
                ax3.set_ylabel('Mean Intensity')
                ax3.set_title('Radial Intensity Profile')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
                
                # Gain profile
                ax4.plot(r_values, rad_gain, 'r-', label='Whitening gain')
                ax4.set_xlabel('Radius (pixels)')
                ax4.set_ylabel('Gain Factor')
                ax4.set_title('Radial Whitening Gain')
                ax4.legend()
                ax4.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig('debug_radial_whitening.png', dpi=150, bbox_inches='tight')
                print(f"    Debug: Radial whitening analysis saved to debug_radial_whitening.png")
                plt.close(fig)
            
            print(f"    Applied radial whitening with max gain factor: {rad_gain.max():.2f}")
            
            # Show debugging info for azimuthal pruning
            if debug_mode:
                # Show detector mask after pruning
                im2 = ax2.imshow(detector_mask, cmap='viridis')
                ax2.set_title('Detector Mask (After Pruning)')
                ax2.set_xlabel('X (pixels)')
                ax2.set_ylabel('Y (pixels)')
                plt.colorbar(im2, ax=ax2)
                ax2.plot(cx, cy, 'r+', markersize=10, markeredgewidth=2)
                
                plt.tight_layout()
                plt.savefig('debug_detector_mask.png', dpi=150, bbox_inches='tight')
                print(f"  Debug: Detector masks saved to debug_detector_mask.png")
                
                # Analyze directional bias
                pruned_pixels = (original_mask > 1e-3) & (detector_mask < 1e-3)
                if np.any(pruned_pixels):
                    # Compute angular distribution of pruned pixels
                    pruned_theta = theta[pruned_pixels]
                    print(f"  Debug: Pruned {np.sum(pruned_pixels)} pixels")
                    
                    # Check for directional bias in pruning
                    theta_bins = np.linspace(-np.pi, np.pi, 9)  # 8 directions
                    theta_centers = (theta_bins[:-1] + theta_bins[1:]) / 2
                    hist, _ = np.histogram(pruned_theta, bins=theta_bins)
                    
                    # Convert to degrees and direction names for readability
                    directions = ['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE']
                    print("  Debug: Directional pruning distribution:")
                    for i, (direction, count) in enumerate(zip(directions, hist)):
                        angle_deg = np.degrees(theta_centers[i])
                        print(f"    {direction} ({angle_deg:+.0f}°): {count} pixels")
                
                plt.close(fig)  # Clean up
        
        # Check detector mask symmetry to diagnose Ex/Ey bias
        if debug_mode:
            print("  Debug: Analyzing detector mask symmetry...")
            
            # Test mask symmetry around center
            def check_mask_symmetry(mask, center_y, center_x):
                """Check if detector mask is symmetric around center"""
                Ky, Kx = mask.shape
                
                # Create symmetric test points
                test_points = []
                for dr in [10, 20, 30]:  # Test at different radii
                    for angle in np.linspace(0, 2*np.pi, 8, endpoint=False):
                        dy = int(dr * np.sin(angle))
                        dx = int(dr * np.cos(angle))
                        y1, x1 = int(center_y + dy), int(center_x + dx)
                        y2, x2 = int(center_y - dy), int(center_x - dx)
                        
                        # Check bounds
                        if 0 <= y1 < Ky and 0 <= x1 < Kx and 0 <= y2 < Ky and 0 <= x2 < Kx:
                            test_points.append(((y1, x1), (y2, x2), f"r={dr}, θ={np.degrees(angle):.0f}°"))
                
                # Compare symmetric points
                asymmetries = []
                for (y1, x1), (y2, x2), label in test_points:
                    val1, val2 = mask[y1, x1], mask[y2, x2]
                    if val1 > 1e-6 or val2 > 1e-6:  # Only compare non-zero points
                        asymmetry = abs(val1 - val2) / max(val1 + val2, 1e-6)
                        asymmetries.append(asymmetry)
                        if asymmetry > 0.1:  # Report significant asymmetries
                            print(f"    Asymmetry at {label}: {val1:.3f} vs {val2:.3f} (diff: {asymmetry:.1%})")
                
                avg_asymmetry = np.mean(asymmetries) if asymmetries else 0
                print(f"    Average mask asymmetry: {avg_asymmetry:.1%}")
                return avg_asymmetry
            
            symmetry_error = check_mask_symmetry(detector_mask, cy, cx)
            if symmetry_error > 0.05:
                print(f"  WARNING: Detector mask shows significant asymmetry ({symmetry_error:.1%})")
                print(f"           This could cause Ex/Ey bias in DPC results")
        
        n_chunks = (dataset.total_frames + chunk_size - 1) // chunk_size
        
        with Timer(f"Processing {n_chunks} chunks for conventional DPC"):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                frames = dataset.load_chunk(start_idx, chunk_size)
                end_idx = min(start_idx + chunk_size, dataset.total_frames)
                
                # Convert to float64 for accurate CoM calculations
                if self.use_gpu:
                    frames = self.xp.asarray(frames, dtype=self.xp.float64)
                    ky_grid_gpu = self.xp.asarray(ky_grid, dtype=self.xp.float64)
                    kx_grid_gpu = self.xp.asarray(kx_grid, dtype=self.xp.float64)
                    mask_gpu = self.xp.asarray(detector_mask, dtype=self.xp.float64)
                else:
                    frames = frames.astype(np.float64)
                    ky_grid_gpu = ky_grid.astype(np.float64)
                    kx_grid_gpu = kx_grid.astype(np.float64)
                    mask_gpu = detector_mask.astype(np.float64)
                
                # Apply weighted detector mask and compute CoM
                weighted_frames = frames * mask_gpu[None, :, :]
                
                # Compute weighted center of mass
                total_weight = self.xp.sum(weighted_frames, axis=(1,2), keepdims=True) + 1e-8
                com_x_vals = self.xp.sum(kx_grid_gpu[None, :, :] * weighted_frames, axis=(1,2)) / total_weight.squeeze()
                com_y_vals = self.xp.sum(ky_grid_gpu[None, :, :] * weighted_frames, axis=(1,2)) / total_weight.squeeze()
                
                if self.use_gpu:
                    com_x_vals = self.xp.asnumpy(com_x_vals)
                    com_y_vals = self.xp.asnumpy(com_y_vals)
                
                # Map to scan coordinates
                chunk_indices = np.arange(start_idx, end_idx)
                y_coords = chunk_indices // Nx
                x_coords = chunk_indices % Nx
                com_x[y_coords, x_coords] = com_x_vals
                com_y[y_coords, x_coords] = com_y_vals
                
                # Progress update
                if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                    progress = (chunk_idx + 1) / n_chunks * 100
                    print(f"  Progress: {progress:.1f}%")
        
        # Apply baseline correction
        baseline_mode = dpc_config.get('baseline', 'median')
        if baseline_mode == 'median':
            com_x -= np.median(com_x)
            com_y -= np.median(com_y)
            print(f"  Applied median baseline correction")
        elif baseline_mode == 'rowcol':
            # Standard row/column median baseline
            com_x -= np.median(com_x, axis=1, keepdims=True)
            com_y -= np.median(com_y, axis=0, keepdims=True)
            
            # Add running median filter to remove scan drift waves
            k = dpc_config.get('running_median_kernel', 31)  # odd window size
            if k > 1:
                # Apply 1D running median along fast-scan direction
                com_x -= medfilt(com_x, kernel_size=(1, k))
                com_y -= medfilt(com_y, kernel_size=(k, 1))
                print(f"  Applied row/column + running median (k={k}) baseline correction")
            else:
                print(f"  Applied row/column baseline correction")
        elif baseline_mode == 'roi':
            x0, y0, w, h = dpc_config['baseline_region']
            bx = np.mean(com_x[y0:y0+h, x0:x0+w])
            by = np.mean(com_y[y0:y0+h, x0:x0+w])
            com_x -= bx
            com_y -= by
            print(f"  Applied ROI baseline correction: ({bx:.4f}, {by:.4f})")
        
        # Apply curl-free projection (project E onto gradient field using Poisson solver)
        enable_curl_free = dpc_config.get('curl_free_projection', True)
        if enable_curl_free:
            print("  Applying curl-free projection (Fourier Poisson solver)...")
            
            # Copy current field for curl-free projection: E' = ∇φ where ∇²φ = ∇·E
            Ey, Ex = com_y.copy(), com_x.copy()  # Note: Ey first as in task.md
            Ny, Nx = Ex.shape
            
            # Create k-space grids
            ky = np.fft.fftfreq(Ny)[:, None]
            kx = np.fft.fftfreq(Nx)[None, :]
            ikx, iky = 2j * np.pi * kx, 2j * np.pi * ky
            
            # Compute divergence in real space: ∇·E = ∂Ex/∂x + ∂Ey/∂y
            div = np.real(np.fft.ifft2(ikx * np.fft.fft2(Ex) + iky * np.fft.fft2(Ey)))
            
            # Solve Poisson equation ∇²φ = ∇·E in Fourier space
            den = ikx**2 + iky**2
            den[0, 0] = 1.0  # Avoid division by zero at DC component
            phi_hat = np.fft.fft2(div) / den
            
            # Compute curl-free field: E'= ∇φ
            Ex_cf = np.real(np.fft.ifft2(ikx * phi_hat))
            Ey_cf = np.real(np.fft.ifft2(iky * phi_hat))
            
            # Update CoM fields with curl-free components
            com_x[:] = Ex_cf
            com_y[:] = Ey_cf
            
            print("    Projected field onto curl-free (gradient) space")
        
        # Apply automatic field orientation correction to fix Ex/Ey asymmetries
        enable_auto_orientation = dpc_config.get('auto_orientation', True)
        if enable_auto_orientation:
            print("  Applying automatic field orientation correction...")
            
            def rot(vx, vy, phi):
                """Rotate vector field by angle phi (radians)."""
                c, s = np.cos(phi), np.sin(phi)
                return c*vx - s*vy, s*vx + c*vy
            
            def curl_div(vx, vy):
                """Compute curl and divergence of 2D vector field."""
                # Simple finite difference (assuming uniform grid)
                dy_vx, dx_vx = np.gradient(vx)
                dy_vy, dx_vy = np.gradient(vy)
                curl = dx_vy - dy_vx  # ∇ × v = ∂vy/∂x - ∂vx/∂y
                div = dx_vx + dy_vy   # ∇ · v = ∂vx/∂x + ∂vy/∂y
                return curl, div
            
            def auto_orient(vx, vy, mask=None, debug=False):
                """
                Find optimal rotation and sign to minimize curl relative to divergence.
                
                For electrostatic fields, we expect ∇×E ≈ 0, so we minimize
                the curl-to-divergence ratio: |curl| / (|div| + ε)
                """
                best_score = 1e9
                best_phi = 0.0
                best_sign = 1
                
                # Test both signs (inversion)
                for sign in [+1, -1]:
                    vx_test = sign * vx
                    vy_test = sign * vy
                    
                    # Test rotation angles from -30 to +30 degrees (restrict search)
                    angles = np.deg2rad(np.arange(-30, 31, 1))
                    lambda_phi = 0.05  # regularization parameter (radians^-1)
                    scores = []
                    
                    for phi in angles:
                        vx_rot, vy_rot = rot(vx_test, vy_test, phi)
                        curl, div = curl_div(vx_rot, vy_rot)
                        
                        # Apply mask if provided
                        if mask is not None:
                            curl_masked = curl[mask]
                            div_masked = div[mask]
                        else:
                            curl_masked = curl
                            div_masked = div
                        
                        # Compute score with regularization: |curl|/|div| + λ|φ|
                        curl_rms = np.sqrt(np.nanmean(curl_masked**2))
                        div_rms = np.sqrt(np.nanmean(div_masked**2))
                        score = curl_rms / (div_rms + 1e-8) + lambda_phi * abs(phi)
                        scores.append(score)
                        
                        if score < best_score:
                            best_score = score
                            best_phi = phi
                            best_sign = sign
                    
                    if debug:
                        print(f"    Sign {sign:+d}: best score {min(scores):.4f} at {np.degrees(angles[np.argmin(scores)]):.1f}°")
                
                return best_phi, best_sign, best_score
            
            # Create confidence mask for reliable regions
            # Use CoM magnitude as a proxy for measurement confidence
            confidence = np.sqrt(com_x**2 + com_y**2)
            confidence_threshold = np.percentile(confidence[confidence > 0], 10) if np.any(confidence > 0) else 0
            mask = confidence > confidence_threshold
            
            # Find optimal orientation
            phi_opt, sign_opt, final_score = auto_orient(com_x, com_y, mask=mask)
            
            # Apply optimal transformation
            com_x_corrected, com_y_corrected = rot(sign_opt * com_x, sign_opt * com_y, phi_opt)
            
            # Store original values for comparison if debugging
            if debug_mode:
                original_com_x, original_com_y = com_x.copy(), com_y.copy()
            
            # Update the CoM values
            com_x[:] = com_x_corrected
            com_y[:] = com_y_corrected
            
            print(f"    Optimal rotation: {np.degrees(phi_opt):.1f}°")
            print(f"    Optimal sign: {sign_opt:+d}")
            print(f"    Final curl/div score: {final_score:.4f}")
            
            # Fix global sign using mean divergence criterion (removes 180° ambiguity)
            curl, div = curl_div(com_x, com_y)
            mean_div = np.nanmean(div)
            if mean_div < 0:
                # Flip sign to make average divergence positive (convention: charge density > 0)
                com_x *= -1
                com_y *= -1
                print(f"    Flipped field sign (mean div: {mean_div:.6f} -> {-mean_div:.6f})")
            else:
                print(f"    Kept field sign (mean div: {mean_div:.6f})")
            
            # Optional: Apply mild smoothing after orientation correction
            enable_vector_smoothing = dpc_config.get('vector_smoothing', True)
            if enable_vector_smoothing:
                from scipy.ndimage import gaussian_filter
                sigma = dpc_config.get('vector_smoothing_sigma', 1.0)
                if sigma > 0:
                    com_x = gaussian_filter(com_x, sigma=sigma)
                    com_y = gaussian_filter(com_y, sigma=sigma)
                    print(f"    Applied vector field smoothing (σ={sigma:.1f} pixels)")
            
            if debug_mode:
                # Save orientation correction analysis
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                
                # Original field
                axes[0,0].quiver(original_com_x[::8, ::8], original_com_y[::8, ::8], 
                               scale=None, scale_units='xy', angles='xy')
                axes[0,0].set_title('Original CoM Field')
                axes[0,0].set_aspect('equal')
                
                # Corrected field
                axes[0,1].quiver(com_x[::8, ::8], com_y[::8, ::8], 
                               scale=None, scale_units='xy', angles='xy')
                axes[0,1].set_title(f'Corrected Field (φ={np.degrees(phi_opt):.1f}°, sign={sign_opt:+d})')
                axes[0,1].set_aspect('equal')
                
                # Confidence mask
                axes[0,2].imshow(mask, cmap='gray')
                axes[0,2].set_title('Confidence Mask')
                
                # Curl comparison
                curl_orig, _ = curl_div(original_com_x, original_com_y)
                curl_corr, _ = curl_div(com_x, com_y)
                
                im1 = axes[1,0].imshow(curl_orig, cmap='RdBu_r')
                axes[1,0].set_title('Original Curl')
                plt.colorbar(im1, ax=axes[1,0], shrink=0.8)
                
                im2 = axes[1,1].imshow(curl_corr, cmap='RdBu_r')
                axes[1,1].set_title('Corrected Curl')
                plt.colorbar(im2, ax=axes[1,1], shrink=0.8)
                
                # Curl reduction
                curl_reduction = np.abs(curl_orig) - np.abs(curl_corr)
                im3 = axes[1,2].imshow(curl_reduction, cmap='RdYlGn')
                axes[1,2].set_title('Curl Reduction (green=better)')
                plt.colorbar(im3, ax=axes[1,2], shrink=0.8)
                
                plt.tight_layout()
                plt.savefig('debug_orientation_correction.png', dpi=150, bbox_inches='tight')
                print(f"    Debug: Orientation correction analysis saved to debug_orientation_correction.png")
                plt.close(fig)
        
        # Convert to projected electric field
        field_scale = dpc_config.get('field_scale', 1.0)
        Ex = com_x * field_scale
        Ey = com_y * field_scale
        
        # Compute magnitude and angle
        E_mag = np.sqrt(Ex**2 + Ey**2)
        E_angle = np.arctan2(Ey, Ex)
        
        print(f"Computed conventional DPC fields: |E| range [{E_mag.min():.4f}, {E_mag.max():.4f}]")
        
        return {
            'Ex': Ex,
            'Ey': Ey, 
            'E_mag': E_mag,
            'E_angle': E_angle,
            'com_x': com_x,
            'com_y': com_y
        }


def create_visualizations(output_dir: Path, contrast_maps: Dict, argmax_map: np.ndarray, 
                         confidence_map: np.ndarray, dpc_results: Dict, templates: Dict, gates: Dict):
    """Create publication-ready visualizations."""
    print("Creating visualizations...")
    
    figures_dir = output_dir / 'figures'
    
    # Template montage
    clusters = list(templates.keys())
    n_clusters = len(clusters)
    cols = min(4, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    
    # Handle axis indexing for different grid configurations
    if rows == 1 and cols == 1:
        # Single subplot
        axes = [axes]
    elif rows == 1:
        # Single row, multiple columns
        axes = axes.flatten()
    elif cols == 1:
        # Single column, multiple rows  
        axes = axes.flatten()
    else:
        # Multiple rows and columns
        axes = axes.flatten()
    
    fig.suptitle('Cluster Templates and Gates', fontsize=16)
    
    for i, cluster_id in enumerate(clusters):
        ax = axes[i]
        
        # Show template with gate overlay
        template = templates[cluster_id]
        gate = gates[cluster_id]
        
        im = ax.imshow(template, cmap='RdBu_r')
        ax.contour(gate, levels=[0.5], colors='lime', linewidths=2)
        ax.set_title(f'Cluster {cluster_id}')
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    # Hide unused subplots
    for i in range(n_clusters, rows * cols):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'montage_templates.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # HSV polarization visualization
    Ex, Ey = dpc_results['Ex'], dpc_results['Ey']
    E_mag, E_angle = dpc_results['E_mag'], dpc_results['E_angle']
    
    # Create HSV image
    hue = (E_angle + np.pi) / (2 * np.pi)  # Map [-π, π] to [0, 1]
    saturation = np.ones_like(hue)
    value = E_mag / np.percentile(E_mag, 99)  # Scale to 99th percentile
    value = np.clip(value, 0, 1)
    
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # HSV visualization
    ax1.imshow(rgb)
    ax1.set_title('Polarization (HSV: hue=angle, brightness=magnitude)')
    ax1.axis('off')
    
    # Quiver plot
    stride = max(1, E_mag.shape[0] // 20)  # Reasonable arrow density
    Y, X = np.mgrid[0:E_mag.shape[0]:stride, 0:E_mag.shape[1]:stride]
    
    ax2.imshow(E_mag, cmap='gray')
    ax2.quiver(X, Y, Ex[::stride, ::stride], Ey[::stride, ::stride], 
              E_mag[::stride, ::stride], cmap='jet', scale_units='xy', scale=1)
    ax2.set_title('Electric Field Quiver Plot')
    ax2.axis('off')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'hsv_polarisation.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # DPC quiver standalone
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(E_mag, cmap='viridis')
    
    stride = max(1, E_mag.shape[0] // 30)
    Y, X = np.mgrid[0:E_mag.shape[0]:stride, 0:E_mag.shape[1]:stride]
    ax.quiver(X, Y, Ex[::stride, ::stride], Ey[::stride, ::stride], 
             color='white', scale_units='xy', alpha=0.8)
    
    ax.set_title('DPC Vector Field')
    ax.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='|E|')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'DPC_quiver.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created visualizations in {figures_dir}")


def create_conventional_visualizations(output_dir: Path, dpc_results: Dict):
    """Create simplified visualizations for conventional DPC analysis."""
    print("Creating conventional DPC visualizations...")
    
    figures_dir = output_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    Ex, Ey = dpc_results['Ex'], dpc_results['Ey']
    E_mag, E_angle = dpc_results['E_mag'], dpc_results['E_angle']
    
    # HSV polarization visualization
    hue = (E_angle + np.pi) / (2 * np.pi)  # Map [-π, π] to [0, 1]
    saturation = np.ones_like(hue)
    value = E_mag / np.percentile(E_mag, 99)  # Scale to 99th percentile
    value = np.clip(value, 0, 1)
    
    hsv = np.stack([hue, saturation, value], axis=-1)
    rgb = hsv_to_rgb(hsv)
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(12, 10))
    
    # HSV visualization
    ax1.imshow(rgb, origin='lower')
    ax1.set_title('Polarization (HSV: hue=angle, brightness=magnitude)')
    ax1.axis('off')
    
    # Magnitude map
    im2 = ax2.imshow(E_mag, cmap='viridis', origin='lower')
    ax2.set_title('Field Magnitude |E|')
    ax2.axis('off')
    plt.colorbar(im2, ax=ax2, shrink=0.8)
    
    # X component
    im3 = ax3.imshow(Ex, cmap='RdBu_r', origin='lower')
    ax3.set_title('X Component Ex')
    ax3.axis('off')
    plt.colorbar(im3, ax=ax3, shrink=0.8)
    
    # Y component  
    im4 = ax4.imshow(Ey, cmap='RdBu_r', origin='lower')
    ax4.set_title('Y Component Ey')
    ax4.axis('off')
    plt.colorbar(im4, ax=ax4, shrink=0.8)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'conventional_dpc_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Quiver plot
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(E_mag, cmap='viridis', origin='lower')
    
    stride = max(1, E_mag.shape[0] // 30)
    Y, X = np.mgrid[0:E_mag.shape[0]:stride, 0:E_mag.shape[1]:stride]
    ax.quiver(X, Y, Ex[::stride, ::stride], Ey[::stride, ::stride], 
             color='white', scale_units='xy', alpha=0.8)
    
    ax.set_title('Conventional DPC Vector Field')
    ax.axis('off')
    plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'), ax=ax, label='|E|')
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'conventional_dpc_quiver.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Created conventional DPC visualizations in {figures_dir}")


def save_provenance(output_dir: Path, args: argparse.Namespace, config: Dict, timings: Dict):
    """Save complete provenance information."""
    logs_dir = output_dir / 'logs'
    
    # Convert Path objects to strings for JSON serialization
    serializable_args = {}
    for key, value in vars(args).items():
        if isinstance(value, Path):
            serializable_args[key] = str(value)
        else:
            serializable_args[key] = value
    
    # Runtime configuration
    run_info = {
        'timestamp': datetime.now().isoformat(),
        'command_line': ' '.join(sys.argv),
        'arguments': serializable_args,
        'config': config,
        'timings': timings,
        'system_info': {
            'python_version': sys.version,
            'numpy_version': np.__version__,
            'cupy_available': CUPY_AVAILABLE,
            'dask_available': DASK_AVAILABLE
        }
    }
    
    # Save run info
    with open(logs_dir / 'run.json', 'w') as f:
        json.dump(run_info, f, indent=2)
    
    # Save config  
    with open(logs_dir / 'config.yaml', 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    
    # Create simple hash for reproducibility
    config_str = json.dumps(config, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
    
    with open(logs_dir / 'timings.txt', 'w') as f:
        f.write(f"Configuration hash: {config_hash}\n")
        f.write("Timing breakdown:\n")
        for operation, duration in timings.items():
            f.write(f"  {operation}: {duration:.2f}s\n")
    
    print(f"Saved provenance to {logs_dir}")

def load_config(config_path: Optional[Path]) -> Dict:
    """
    Load configuration from YAML file or return sensible defaults.
    
    The configuration system provides fine-grained control over all aspects
    of the analysis pipeline. If no config file is provided, scientifically
    reasonable defaults are used.
    
    Parameters:
    -----------
    config_path : Optional[Path]
        Path to YAML configuration file, or None for defaults
        
    Returns:
    --------
    config : Dict
        Complete configuration dictionary with all parameters
        
    Default Configuration:
    ----------------------
    The default configuration includes:
    
    - templates: Percentile-based gating (top 10%), moderate smoothing, no ROI limits
    - preprocess: Log transform enabled, background subtraction disabled, no whitening
    - matched_filter: Softmax temperature = 2.0 for non-saturated confidence
    - dpc: Adaptive gating mode, median baseline correction, no ROI limits
    - calibration: Auto-detect beam center, relative k-space units
    - validation: Split-half validation disabled by default
    
    See the script documentation for complete parameter descriptions.
    """
    default_config = {
        'device': 'cpu',
        'preprocess': {
            'background_subtract': False,
            'log_transform': True,
            'central_mask_px': None,
            'whiten': 'none'
        },
        'templates': {
            'top_percent': 10,
            'smooth_sigma': 1.0,
            'roi_r': [None, None],
            'gate_type': 'hard'  # 'hard', 'soft', or 'smooth_first'
        },
        'matched_filter': {
            'softmax_temp': 2.0
        },
        'dpc': {
            'gate_mode': 'adaptive',  # For cluster mode: 'adaptive' or 'union'
            'baseline': 'rowcol',     # 'median', 'rowcol', or 'roi'
            'running_median_kernel': 1,   # odd window size for running median (1 = disabled, try 15-51)
            'roi': [None, None],      # For cluster mode: radial ROI [r_min, r_max]
            'field_scale': 1.0,
            'baseline_region': [10, 10, 20, 20],  # [x0, y0, w, h] for ROI baseline mode
            # Azimuthal pruning for Bragg disk suppression (conservative defaults)
            'azimuthal_pruning': False,   # Disable by default - enable if needed
            'azimuthal_keep_fraction': 0.95,  # Keep 95% of sectors, remove only brightest 5%
            # Soft ring parameters
            'soft_ring_edge': 1.0,       # Edge softness in pixels (1.0 = conservative, 2.0 = smoother, 0 = hard binary)
            'enable_soft_rings': True,   # Set to False to revert to hard binary masks
            # Debugging options
            'debug_detector': False,     # Save detector mask visualizations for debugging
            # Conventional DPC parameters (legacy single-ring support)
            'mask_inner_radius': 0,   # Inner radius for annular detector (pixels) 
            'mask_outer_radius': None, # Outer radius for annular detector (pixels, None = half detector size)
            # Enhanced detector parameters
            'detector_mode': 'single', # 'single' or 'two_ring'
            'r1_inner': 6,            # Ring1 inner radius (pixels)
            'r1_outer': 48,           # Ring1 outer radius (pixels) 
            'r2_inner': 80,           # Ring2 inner radius (pixels, for two_ring mode)
            'r2_outer': 92,           # Ring2 outer radius (pixels, for two_ring mode)
            'ring2_weight': 0.5       # Weight for Ring2 (reduces outer ring noise)
        },
        'calibration': {
            'center_y': None,
            'center_x': None,
            'pixel_size': None,     # Direct pixel size (takes precedence)
            'pixel_size_units': 'nm',  # Units for pixel_size: "nm" or "angstrom"
            'pixel_size_k': 1.0,
            'pixel_size_ky': None,  # If None, falls back to pixel_size_k or converted pixel_size
            'pixel_size_kx': None,  # If None, falls back to pixel_size_k or converted pixel_size
            'convergence_angle': None  # Convergence angle in mrad
        },
        'validation': {
            'enable_split_half': False
        }
    }
    
    if config_path and config_path.exists():
        with open(config_path, 'r') as f:
            user_config = yaml.safe_load(f)
        
        # Deep merge configs
        def deep_merge(default, user):
            for key, value in user.items():
                if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                    deep_merge(default[key], value)
                else:
                    default[key] = value
        
        deep_merge(default_config, user_config)
        print(f"Loaded configuration from {config_path}")
    else:
        print("Using default configuration")
    
    return default_config

def main():
    parser = argparse.ArgumentParser(
        description="Data-driven Virtual Detectors for 4D-STEM Polarisation Mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/analysis/latent_cluster_polar_map.py \\
        --data patterns.h5 --dset /entry/data \\
        --labels clusters.npy --scan-shape 256 256 \\
        --out results --device cuda:0

    python scripts/analysis/latent_cluster_polar_map.py \\
        --data patterns.h5 --labels clustering_data.npz \\
        --scan-shape 128 128 --config config.yaml \\
        --chunks 1024 --out polarization_analysis

    python scripts/analysis/latent_cluster_polar_map.py \\
        --data patterns.h5 --labels clusters.npy \\
        --scan-shape 128 128 --interactive-center \\
        --out results

    python scripts/analysis/latent_cluster_polar_map.py \\
        --data patterns.h5 --labels clusters.npy \\
        --scan-shape 128 128 --interactive-kspace \\
        --known-spacing 3.14 --out results
        """
    )
    
    # Required arguments
    parser.add_argument('--data', type=Path, required=True,
                       help='Path to 4D-STEM HDF5 data file')
    parser.add_argument('--mode', type=str, choices=['cluster', 'conventional'], default='cluster',
                       help='Analysis mode: cluster-based or conventional DPC (default: cluster)')
    parser.add_argument('--labels', type=Path, 
                       help='Path to cluster labels (.npy or .npz) - required for cluster mode')
    parser.add_argument('--scan-shape', type=int, nargs=2, required=True,
                       metavar=('NY', 'NX'), help='Scan dimensions (height width)')
    parser.add_argument('--out', type=Path, required=True,
                       help='Output directory')
    
    # Optional arguments
    parser.add_argument('--dset', type=str, default='/entry/data',
                       help='HDF5 dataset path (default: /entry/data)')
    parser.add_argument('--config', type=Path,
                       help='Configuration YAML file')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Compute device (cpu, cuda:0, etc.)')
    parser.add_argument('--chunks', type=int, default=1024,
                       help='Chunk size for processing (default: 1024)')
    parser.add_argument('--interactive-center', action='store_true',
                       help='Enable interactive beam center picking (opens GUI window)')
    parser.add_argument('--interactive-kspace', action='store_true',
                       help='Enable interactive k-space calibration by clicking Bragg peaks')
    parser.add_argument('--interactive-detector', action='store_true',
                       help='Enable interactive annular detector optimization (conventional mode only)')
    parser.add_argument('--known-spacing', type=float,
                       help='Known d-spacing in Angstroms for absolute k-space calibration')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    
    # Labels are only required for cluster mode
    if args.mode == 'cluster':
        if not args.labels:
            raise ValueError("--labels is required for cluster mode")
        if not args.labels.exists():
            raise FileNotFoundError(f"Labels file not found: {args.labels}")
    
    # Warn if labels provided for conventional mode
    if args.mode == 'conventional' and args.labels:
        print("Warning: --labels provided but will be ignored in conventional mode")
        
    # Validate interactive detector option
    if args.interactive_detector and args.mode != 'conventional':
        print("Warning: --interactive-detector only works in conventional mode, ignoring")
    
    print("="*80)
    if args.mode == 'cluster':
        print("DATA-DRIVEN VIRTUAL DETECTORS FOR 4D-STEM POLARISATION MAPPING")
    else:
        print("CONVENTIONAL DPC/COM ANALYSIS FOR 4D-STEM POLARISATION MAPPING")
    print("="*80)
    print(f"Mode: {args.mode}")
    print(f"Data: {args.data}")
    if args.labels:
        print(f"Labels: {args.labels}")
    print(f"Scan shape: {args.scan_shape[0]} × {args.scan_shape[1]}")
    print(f"Output: {args.out}")
    print(f"Device: {args.device}")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    config['device'] = args.device  # Override device from CLI
    config['interactive_center'] = args.interactive_center  # Override interactive mode from CLI
    config['interactive_kspace'] = args.interactive_kspace  # Override k-space calibration from CLI
    config['interactive_detector'] = args.interactive_detector  # Override detector optimization from CLI
    if args.known_spacing:
        config['known_spacing'] = args.known_spacing  # Override known spacing from CLI
    
    # Initialize timing
    total_timer = Timer("Complete analysis", verbose=False)
    timings = {}
    
    with total_timer:
        # Initialize analyzer based on mode
        if args.mode == 'cluster':
            analyzer = ClusterDPCAnalyzer(config)
        else:
            analyzer = ConventionalDPCAnalyzer(config)
        
        # Load data (common for both modes)
        with Timer("Loading 4D-STEM data") as t:
            dataset = analyzer.load_data(args.data, args.dset, tuple(args.scan_shape))
        timings['data_loading'] = t.elapsed
        
        # Detect beam center and calibrate k-space (common for both modes)
        with Timer("Beam center detection and calibration") as t:
            center = analyzer.detect_beam_center(dataset)
            analyzer.calibrate_k_space(dataset, center)
        timings['calibration'] = t.elapsed
        
        # Create k-space grids (common for both modes)
        with Timer("Creating k-space grids") as t:
            ky_grid, kx_grid = analyzer.make_k_grids((dataset.Ky, dataset.Kx))
        timings['k_grids'] = t.elapsed
        
        if args.mode == 'cluster':
            # Cluster-based analysis
            with Timer("Loading cluster labels") as t:
                labels = analyzer.load_labels(args.labels, tuple(args.scan_shape))
            timings['label_loading'] = t.elapsed
            
            # Build templates
            with Timer("Building cluster templates") as t:
                analyzer.build_templates(dataset, labels, args.chunks)
            timings['template_building'] = t.elapsed
            
            # Compute contrast maps
            with Timer("Computing contrast maps") as t:
                contrast_maps = analyzer.compute_contrast_maps(dataset, labels, args.chunks)
            timings['contrast_maps'] = t.elapsed
            
            with Timer("Computing argmax and confidence") as t:
                argmax_map, confidence_map = analyzer.compute_argmax_confidence(contrast_maps)
            timings['argmax_confidence'] = t.elapsed
            
            # Cluster-gated DPC/CoM analysis
            with Timer("Computing cluster-gated DPC/CoM fields") as t:
                dpc_results = analyzer.compute_dpc_com(dataset, labels, ky_grid, kx_grid, args.chunks)
            timings['dpc_com'] = t.elapsed
            
            # Save outputs
            with Timer("Saving outputs") as t:
                additional_outputs = {
                    'contrast_maps': contrast_maps,
                    'argmax_map': argmax_map,
                    'confidence_map': confidence_map
                }
                analyzer.save_outputs(args.out, dpc_results, additional_outputs)
                
                # Save k-space artifacts
                output_dir = Path(args.out)
                (output_dir / 'kspace').mkdir(parents=True, exist_ok=True)
                for cluster_id in analyzer.templates.keys():
                    np.save(output_dir / 'kspace' / f'template_T_{cluster_id}.npy', analyzer.templates[cluster_id])
                    np.save(output_dir / 'kspace' / f'gate_G_{cluster_id}.npy', analyzer.gates[cluster_id])
                    np.save(output_dir / 'kspace' / f'mu_cluster_{cluster_id}.npy', analyzer.mu_clusters[cluster_id])
                np.save(output_dir / 'kspace' / 'mu_global.npy', analyzer.mu_global)
                np.save(output_dir / 'kspace' / 'sigma_global.npy', analyzer.sigma_global)
            timings['saving'] = t.elapsed
            
            # Create visualizations
            with Timer("Creating visualizations") as t:
                create_visualizations(args.out, contrast_maps, argmax_map, confidence_map, 
                                    dpc_results, analyzer.templates, analyzer.gates)
            timings['visualizations'] = t.elapsed
            
        else:
            # Conventional DPC analysis
            
            # Optional interactive detector optimization
            if config.get('interactive_detector', False):
                with Timer("Interactive detector optimization") as t:
                    analyzer.optimize_annular_detector(dataset, center)
                timings['detector_optimization'] = t.elapsed
            
            with Timer("Computing conventional DPC/CoM fields") as t:
                dpc_results = analyzer.compute_conventional_dpc(dataset, ky_grid, kx_grid, args.chunks)
            timings['dpc_com'] = t.elapsed
            
            # Save outputs
            with Timer("Saving outputs") as t:
                analyzer.save_outputs(args.out, dpc_results)
            timings['saving'] = t.elapsed
            
            # Create simplified visualizations for conventional mode
            with Timer("Creating visualizations") as t:
                create_conventional_visualizations(args.out, dpc_results)
            timings['visualizations'] = t.elapsed
        
        # Save provenance
        timings['total'] = total_timer.elapsed
        save_provenance(args.out, args, config, timings)
        
        # Clean up
        dataset.close()
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Total time: {total_timer.elapsed:.1f}s")
    print(f"Results saved to: {args.out}")
    print("\nGenerated outputs:")
    print("  kspace/     - Templates, gates, and mean patterns")
    print("  realspace/  - Contrast maps and DPC field components")  
    print("  figures/    - Publication-ready visualizations")
    print("  logs/       - Configuration and timing information")
    print("="*80)

if __name__ == "__main__":
    main()