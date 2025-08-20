#!/usr/bin/env python3
"""
Data-driven Virtual Detectors for 4D-STEM Polarisation Mapping

This script implements a novel approach to 4D-STEM analysis that combines autoencoder-based 
clustering with physics-informed virtual detector design for polarization mapping in 
ferroelectric and other functional materials.

METHODOLOGY OVERVIEW
====================

The approach consists of four main stages:

1. CLUSTER-GUIDED TEMPLATE GENERATION
   - Uses pre-computed cluster labels from autoencoder latent space analysis
   - Builds mean diffraction patterns μ_c(k) for each cluster c
   - Creates differential templates: T_c(k) = (μ_c(k) - μ(k)) / (σ(k) + ε)
   - L2-normalizes templates to ensure fair comparison across clusters
   - Templates highlight k-space features that distinguish each cluster

2. VIRTUAL DETECTOR DESIGN
   - Converts templates to detector gates G_c(k) via percentile thresholding
   - Keeps only top X% of template values (default: 10%) to focus on most distinctive features
   - Applies symmetrization: G_c = 0.5 * (G_c + G_c[::-1, ::-1]) to preserve CoM accuracy
   - Optional radial band-limiting to exclude unwanted reflections (HOLZ, etc.)
   - Gates define which k-space regions contribute to analysis for each cluster

3. MATCHED-FILTER CONTRAST MAPPING
   - For each probe position i: S_c(i) = Σ_k T_c(k) * [(I_i(k) - μ(k)) / (σ(k) + ε)]
   - This gives cluster-specific contrast maps S_c(x,y) in real space
   - Softmax with temperature creates confidence maps and argmax cluster assignment
   - Reveals spatial distribution of different diffraction behaviors

4. CLUSTER-GATED DPC/COM ANALYSIS
   - Computes center-of-mass (CoM) for each diffraction pattern using cluster-specific gates
   - Two modes: 'union' (use max of all gates) or 'adaptive' (use gate of assigned cluster)
   - CoM shifts C_x, C_y are converted to projected electric fields E_x, E_y
   - Baseline subtraction removes systematic offsets from beam center uncertainties
   - Generates polarization maps: |E|, angle θ, and HSV visualization

PHYSICS INTERPRETATION
======================

The cluster-guided approach assumes that different local structural environments 
(ferroelectric domains, defects, interfaces) produce characteristic diffraction signatures.
By learning these signatures via unsupervised clustering, we can:

- Design optimal virtual detectors that are sensitive to specific structural features
- Reduce noise by focusing analysis on informative k-space regions
- Map polarization textures with enhanced contrast and spatial resolution
- Distinguish between different types of domains or structural phases

The DPC analysis measures the deflection of the electron beam due to local electric fields,
which are related to ferroelectric polarization, space charge, and built-in fields.
Cluster gating enhances sensitivity by weighting the analysis toward k-space regions
that correlate with the structural features of interest.

MATHEMATICAL FORMULATION
========================

Template Generation:
    μ_c(k) = mean_{i∈cluster_c} I_i(k)                    # cluster mean
    μ(k) = mean_i I_i(k)                                   # global mean
    σ(k) = std_i I_i(k)                                    # global std
    T_c(k) = (μ_c(k) - μ(k)) / (σ(k) + ε)                # z-scored template
    T_c(k) = T_c(k) / ||T_c(k)||_2                        # L2 normalize

Gate Generation:
    threshold = percentile(T_c, 100 - top_percent)
    G_c(k) = T_c(k) >= threshold                           # binary mask
    G_c(k) = 0.5 * (G_c(k) + G_c(k)[::-1, ::-1])         # symmetrize
    G_c(k) = G_c(k) * radial_mask(k)                      # optional ROI

Matched Filtering:
    Z_i(k) = (I_i(k) - μ(k)) / (σ(k) + ε)                # z-score normalize
    # Optional diagonal whitening: Z_i(k) = (I_i(k) - μ(k)) / (σ(k) + ε)
    S_c(i) = Σ_k T_c(k) * Z_i(k)                         # template correlation
    
    # Softmax with temperature for confidence
    confidence_c(i) = exp(S_c(i)/τ) / Σ_j exp(S_j(i)/τ)

DPC/CoM with Gating:
    # Center of mass calculation
    C_x(i) = Σ_k k_x * G(k) * I_i(k) / Σ_k G(k) * I_i(k)
    C_y(i) = Σ_k k_y * G(k) * I_i(k) / Σ_k G(k) * I_i(k)
    
    # Baseline correction options
    C_x(i) = C_x(i) - median(C_x)                    # global median
    C_x(i) = C_x(i) - median(C_x, axis=1)[:, None]  # row-wise median
    C_x(i) = C_x(i) - mean(C_x[ROI])                # ROI-based
    
    # Convert to electric field (simplified)
    E_x(i) = α * C_x(i)
    E_y(i) = α * C_y(i)
    
    where α is a calibration factor depending on camera length and pixel size.

USAGE EXAMPLES
==============

Basic usage with auto-detection:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --dset patterns \
      --labels cluster_results.npz --scan-shape 128 128 \
      --out polarization_analysis

With custom configuration:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 256 256 --config config.yaml \
      --device cuda:0 --chunks 2048 --out results

With interactive beam center picking:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 128 128 --interactive-center \
      --out results

With interactive k-space calibration:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 128 128 --interactive-kspace \
      --known-spacing 3.14 --out results

With both interactive modes:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 128 128 --interactive-center --interactive-kspace \
      --known-spacing 2.45 --out results

Configuration options (config.yaml):
    templates:
      top_percent: 10        # Keep top 10% of template values
      smooth_sigma: 1.5      # Gaussian smoothing of gates
      roi_r: [8, 80]        # Radial ROI limits (inner, outer radius)
    
    preprocess:
      log_transform: true    # Apply log(I+1) transform
      background_subtract: true  # Subtract mean background
      central_mask_px: 8     # Mask central beam (radius in pixels)
      whiten: diag           # Diagonal whitening: 'none', 'diag'
    
    matched_filter:
      softmax_temp: 2.0      # Temperature for softmax confidence
    
    dpc:
      gate_mode: adaptive    # 'union' or 'adaptive' gating
      baseline: rowcol       # 'median', 'rowcol', or 'roi' baseline correction
      roi: [8, 80]          # Radial ROI for DPC analysis
      field_scale: 1.0       # Conversion factor to physical field units
    
    calibration:
      center_y: null         # Auto-detect beam center if null
      center_x: null
      pixel_size: null       # Direct pixel size in nm/pixel (takes precedence)
      pixel_size_units: "nm" # Units for pixel_size: "nm" or "angstrom" 
      pixel_size_k: 1.0      # k-space pixel size (1/Å or relative units)
      pixel_size_ky: null    # Anisotropic k-space calibration (y-axis, same units as pixel_size_k)
      pixel_size_kx: null    # Anisotropic k-space calibration (x-axis, same units as pixel_size_k)
      convergence_angle: null # Convergence angle in mrad (for advanced calculations)
    
    validation:
      enable_split_half: false  # Enable template stability validation

OUTPUTS
=======

The script generates a structured output directory:

    outdir/
      kspace/                    # K-space artifacts
        template_T_{c}.npy       # Differential templates for each cluster
        gate_G_{c}.npy           # Virtual detector gates
        mu_cluster_{c}.npy       # True cluster mean patterns (.npy)
        mu_cluster_{c}.png       # Mean patterns (visualization)
        mu_global.npy            # Global mean pattern
        sigma_global.npy         # Global standard deviation
      
      realspace/                 # Real-space maps
        S_cluster_{c}.tif        # Contrast maps for each cluster
        S_argmax.tif             # Cluster assignment map
        S_confidence.tif         # Assignment confidence
        Ex.tif, Ey.tif           # Electric field components
        E_mag.tif, E_angle.tif   # Field magnitude and angle
        com_x.tif, com_y.tif     # Raw center-of-mass maps
      
      figures/                   # Visualizations
        montage_templates.png    # Template and gate overview
        hsv_polarisation.png     # HSV polarization visualization
        DPC_quiver.png           # Vector field quiver plot
      
      logs/                      # Provenance and metadata
        run.json                 # Complete run parameters and timing
        config.yaml              # Configuration used
        timings.txt              # Performance breakdown

INTERACTIVE CALIBRATION FEATURES
=================================

The script provides advanced interactive calibration capabilities:

1. **Interactive Beam Center Picking** (--interactive-center):
   - Opens a GUI window displaying the mean diffraction pattern
   - Shows auto-detected center as red dashed crosshairs for reference
   - Click once to select the desired beam center
   - Window closes automatically after selection
   - Updates config and uses selected center for all calculations

2. **Interactive K-space Calibration** (--interactive-kspace):
   - Enables calibration of k-space pixel sizes by clicking Bragg peaks
   - Displays mean pattern with log scaling for better peak visibility
   - Shows guiding circles and beam center marker
   - Click two opposite Bragg peaks (180° apart) or ring boundaries
   - Automatically calculates pixel_size_ky and pixel_size_kx
   - If pixel_size is specified in config, it takes precedence over calculations
   - Supports both absolute calibration (with --known-spacing) and relative
   - Updates config without reloading data for efficient workflow

3. **Combined Interactive Mode**:
   - Both modes can be used together for complete interactive setup
   - Center picking happens first, then k-space calibration
   - All calibration results are saved to config and used immediately

Interactive Calibration Workflow:
- Use --interactive-center to precisely set beam center by eye
- Use --interactive-kspace to calibrate against known Bragg reflections
- Provide --known-spacing X.XX (in Angstroms) for absolute k-space units
- Without known spacing, relative calibration normalizes selected radius to 1.0
- Results update the config and are used immediately without data reload

Example with known silicon (220) reflection (d = 1.92 Å):
    python script.py --data si.h5 --labels clusters.npy \\
                     --interactive-kspace --known-spacing 1.92 \\
                     --scan-shape 256 256 --out results

Example polar_config.yaml with direct pixel size specification:
    calibration:
      pixel_size: 0.1         # Direct specification takes precedence
      pixel_size_units: "nm"  # Units: "nm" or "angstrom"
      convergence_angle: 20.0 # mrad (logged for calculations)

PERFORMANCE AND OPTIMIZATION
=============================

The implementation includes several performance optimizations:

1. Vectorized Operations: CoM computation uses vectorized numpy/cupy operations
   - Processes entire chunks at once rather than frame-by-frame
   - Precomputed gate lookup tables for adaptive mode efficiency
   - Significant speedup especially for large datasets

2. Memory Efficiency: Streaming data processing with configurable chunk sizes
   - No need to load entire dataset into memory
   - Cached HDF5 file handles reduce I/O overhead
   - Support for both CPU and GPU acceleration via CuPy

3. Early Center Detection: Beam center detected once and used consistently
   - Eliminates disagreement between preprocessor and DPC centering
   - Uses small data sample for fast auto-detection

VALIDATION AND QUALITY ASSURANCE
================================

The script includes several built-in validation mechanisms:

1. Gate Coverage Analysis: Reports the fraction of detector covered by each gate
   - Healthy gates typically cover 10-40% of the detector
   - 100% coverage indicates poor template discrimination

2. Auto-centering Validation: Uses smoothed mean pattern to find beam center
   - Reports detected center coordinates
   - Applied consistently across preprocessing and analysis

3. Confidence Analysis: Non-saturated confidence maps indicate good cluster separation
   - Mean confidence ~1.0 suggests over-fitting or poor templates
   - Well-separated clusters should show confidence ~0.3-0.8
   - Optional diagonal whitening reduces central beam dominance

4. Baseline Correction: Ensures DPC fields are properly centered
   - Median baseline correction removes global systematic offsets
   - Row/column baseline correction suppresses scan drift and residual tilt
   - ROI-based correction uses reference vacuum region

5. Template Stability: Split-half validation checks template reproducibility
   - Randomly splits data and compares template similarity using SSIM
   - Warns if SSIM < 0.5, indicating potential overfitting or noise
   - Enabled via validation.enable_split_half config option

6. Physics Consistency: DPC fields should show expected symmetries
   - Field magnitude should be low in uniform regions
   - Domain walls should show enhanced field gradients

REFERENCES
==========

This implementation draws on several key concepts:

1. 4D-STEM and DPC theory:
   - Müller-Caspary et al., Ultramicroscopy 178, 62 (2017)
   - Lazić et al., Ultramicroscopy 160, 265 (2016)

2. Virtual detector design:
   - Ophus, Microsc. Microanal. 25, 563 (2019)
   - Close et al., Ultramicroscopy 159, 124 (2015)

3. Machine learning for diffraction analysis:
   - Kalinin et al., npj Comput. Mater. 7, 78 (2021)
   - Spurgeon et al., npj Comput. Mater. 7, 200 (2021)

AUTHORS AND ACKNOWLEDGEMENTS
============================

Implementation: Claude AI (Anthropic)
Scientific guidance: User requirements and domain expertise
Framework: Built on numpy, scipy, matplotlib, scikit-image

For questions or issues, please refer to the project documentation.
"""

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


class ClusterVirtualDetectors:
    """
    Main class implementing data-driven virtual detectors for 4D-STEM analysis.
    
    This class orchestrates the complete workflow:
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
        
        self.templates = {}
        self.gates = {}
        self.mu_clusters = {}
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
            com_x -= np.median(com_x, axis=1, keepdims=True)
            com_y -= np.median(com_y, axis=0, keepdims=True)
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
            'gate_mode': 'adaptive',
            'baseline': 'median',
            'roi': [None, None],
            'field_scale': 1.0,
            'baseline_region': [10, 10, 20, 20]  # [x0, y0, w, h] for ROI baseline mode
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
    parser.add_argument('--labels', type=Path, required=True,
                       help='Path to cluster labels (.npy or .npz)')
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
    parser.add_argument('--known-spacing', type=float,
                       help='Known d-spacing in Angstroms for absolute k-space calibration')
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    
    print("="*80)
    print("DATA-DRIVEN VIRTUAL DETECTORS FOR 4D-STEM POLARISATION MAPPING")
    print("="*80)
    print(f"Data: {args.data}")
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
    if args.known_spacing:
        config['known_spacing'] = args.known_spacing  # Override known spacing from CLI
    
    # Initialize timing
    total_timer = Timer("Complete analysis", verbose=False)
    timings = {}
    
    with total_timer:
        # Initialize analyzer
        analyzer = ClusterVirtualDetectors(config)
        
        # Load data
        with Timer("Loading 4D-STEM data") as t:
            dataset = analyzer.load_data(args.data, args.dset, tuple(args.scan_shape))
        timings['data_loading'] = t.elapsed
        
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
        
        # DPC/CoM analysis
        with Timer("Creating k-space grids") as t:
            ky_grid, kx_grid = analyzer.make_k_grids((dataset.Ky, dataset.Kx))
        timings['k_grids'] = t.elapsed
        
        with Timer("Computing DPC/CoM fields") as t:
            dpc_results = analyzer.compute_dpc_com(dataset, labels, ky_grid, kx_grid, args.chunks)
        timings['dpc_com'] = t.elapsed
        
        # Save outputs
        with Timer("Saving outputs") as t:
            analyzer.save_outputs(args.out, contrast_maps, argmax_map, confidence_map, dpc_results)
        timings['saving'] = t.elapsed
        
        # Create visualizations
        with Timer("Creating visualizations") as t:
            create_visualizations(args.out, contrast_maps, argmax_map, confidence_map, 
                                dpc_results, analyzer.templates, analyzer.gates)
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