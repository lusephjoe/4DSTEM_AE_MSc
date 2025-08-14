#!/usr/bin/env python3
"""
Data-driven Virtual Detectors for 4D-STEM Polarisation Mapping

This script ingests 4D-STEM diffraction data and clustered latent outputs from an autoencoder
to derive cluster-guided virtual detector templates in k-space, compute real-space contrast maps,
and generate DPC/CoM-based projected field maps using cluster-gated weightings.

Usage:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data file.h5 --dset /entry/data \
      --labels clusters.npy --scan-shape 256 256 \
      --config config.yaml --out outdir --device cuda:0 --chunks 4096
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
from scipy.signal import convolve2d
from scipy.optimize import minimize_scalar
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
            print(f"⏱️  {self.description}...")
        self.start_time = time.time()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        if self.verbose:
            print(f"✓ {self.description} completed in {elapsed:.2f}s")
        return False
    
    @property
    def elapsed(self):
        if self.end_time is None:
            return time.time() - self.start_time if self.start_time else 0
        return self.end_time - self.start_time

class H5Dataset:
    """Lazy loader for 4D-STEM HDF5 datasets with chunked access."""
    
    def __init__(self, filepath: Path, dataset_name: str, scan_shape: Tuple[int, int]):
        self.filepath = filepath
        self.dataset_name = dataset_name
        self.scan_shape = scan_shape
        self.Ny, self.Nx = scan_shape
        self.total_frames = self.Ny * self.Nx
        
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
        
        print(f"📁 Loaded H5 dataset: {self.dataset_shape} -> scan {scan_shape}, detector {self.Ky}x{self.Kx}")
    
    def __getitem__(self, indices):
        """Load frames by linear indices."""
        with h5py.File(self.filepath, 'r') as f:
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
        
        with h5py.File(self.filepath, 'r') as f:
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
    """Handles data preprocessing operations."""
    
    def __init__(self, config: Dict, detector_mask: Optional[np.ndarray] = None):
        self.config = config
        self.detector_mask = detector_mask
        self.background = config.get('background_subtract', False)
        self.log_transform = config.get('log_transform', False)
        self.normalize = config.get('normalize', True)
    
    def __call__(self, frames: np.ndarray) -> np.ndarray:
        """Apply preprocessing to frames."""
        processed = frames.astype(np.float32)
        
        # Background subtraction
        if self.background:
            processed = processed - np.mean(processed, axis=0, keepdims=True)
        
        # Log transform
        if self.log_transform:
            processed = np.log(processed + 1)
        
        # Apply detector mask
        if self.detector_mask is not None:
            processed = processed * self.detector_mask[np.newaxis, :, :]
        
        return processed

class ClusterVirtualDetectors:
    """Main class for cluster-guided virtual detector analysis."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.device = config.get('device', 'cpu')
        self.use_gpu = self.device.startswith('cuda') and CUPY_AVAILABLE
        
        if self.use_gpu:
            print(f"🚀 Using GPU acceleration: {self.device}")
            self.xp = cp
        else:
            print("🖥️  Using CPU computation")
            self.xp = np
        
        self.templates = {}
        self.gates = {}
        self.mu_global = None
        self.sigma_global = None
        
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
                print(f"⚠️  Using '{keys[0]}' as cluster labels")
        else:
            raise ValueError(f"Unsupported label format: {labels_path.suffix}")
        
        # Ensure labels match scan shape
        total_expected = scan_shape[0] * scan_shape[1]
        if len(labels) != total_expected:
            raise ValueError(f"Label count ({len(labels)}) != scan size ({total_expected})")
        
        unique_labels = np.unique(labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        print(f"📊 Loaded {len(labels)} labels with {n_clusters} clusters")
        
        return labels
    
    def build_templates(self, dataset: H5Dataset, labels: np.ndarray, chunk_size: int = 1024):
        """Build cluster-guided templates in k-space."""
        unique_clusters = np.unique(labels)
        valid_clusters = [c for c in unique_clusters if c != -1]
        
        print(f"🔨 Building templates for {len(valid_clusters)} clusters...")
        
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
                
                if self.use_gpu:
                    frames = self.xp.asarray(frames)
                
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
        
        # Compute global statistics
        self.mu_global = global_sum / total_count
        global_var = (global_sum_sq / total_count) - (self.mu_global ** 2)
        self.sigma_global = self.xp.sqrt(global_var + 1e-8)
        
        # Compute cluster templates
        for c in valid_clusters:
            if cluster_counts[c] == 0:
                print(f"⚠️  Skipping empty cluster {c}")
                continue
            
            mu_c = cluster_sums[c] / cluster_counts[c]
            
            # Template: T_c = (mu_c - mu) / (sigma + eps)
            template = (mu_c - self.mu_global) / (self.sigma_global + 1e-8)
            
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
        
        print(f"✓ Built {len(self.templates)} templates")
    
    def generate_gate(self, template: np.ndarray, cluster_id: int) -> np.ndarray:
        """Generate detector gate from template."""
        gate_config = self.config.get('templates', {})
        gate_type = gate_config.get('gate_type', 'sigmoid')
        threshold = gate_config.get('threshold', 2.0)
        alpha = gate_config.get('alpha', 3.0)
        smooth_sigma = gate_config.get('smooth_sigma', 1.0)
        
        if gate_type == 'sigmoid':
            # G_c = clip(sigmoid(alpha * T_c) - tau, 0, 1)
            sigmoid_vals = 1 / (1 + self.xp.exp(-alpha * template))
            gate = self.xp.clip(sigmoid_vals - threshold/10, 0, 1)
        elif gate_type == 'threshold':
            # G_c = 1[T_c > threshold]
            gate = (template > threshold).astype(np.float32)
        else:
            raise ValueError(f"Unknown gate type: {gate_type}")
        
        # Optional smoothing
        if smooth_sigma > 0:
            if self.use_gpu:
                # Use scipy for smoothing (move to CPU temporarily)
                gate_cpu = self.xp.asnumpy(gate)
                gate_cpu = ndimage.gaussian_filter(gate_cpu, smooth_sigma)
                gate = self.xp.asarray(gate_cpu)
            else:
                gate = ndimage.gaussian_filter(gate, smooth_sigma)
        
        # Ensure gate is in [0, 1]
        gate = self.xp.clip(gate, 0, 1)
        
        return gate
    
    def compute_contrast_maps(self, dataset: H5Dataset, labels: np.ndarray, chunk_size: int = 1024) -> Dict[int, np.ndarray]:
        """Compute matched-filter contrast maps for each cluster."""
        print("🎯 Computing matched-filter contrast maps...")
        
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
                
                if self.use_gpu:
                    frames = self.xp.asarray(frames)
                    mu_global = self.xp.asarray(self.mu_global)
                    sigma_global = self.xp.asarray(self.sigma_global)
                else:
                    mu_global = self.mu_global
                    sigma_global = self.sigma_global
                
                # Z-score normalization
                z_frames = (frames - mu_global[None, :, :]) / (sigma_global[None, :, :] + 1e-8)
                
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
        
        print(f"✓ Computed contrast maps for {len(valid_clusters)} clusters")
        return contrast_maps
    
    def compute_argmax_confidence(self, contrast_maps: Dict[int, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Compute argmax map and confidence from contrast maps."""
        print("📊 Computing argmax and confidence maps...")
        
        clusters = list(contrast_maps.keys())
        Ny, Nx = list(contrast_maps.values())[0].shape
        
        # Stack all contrast maps
        contrast_stack = np.stack([contrast_maps[c] for c in clusters], axis=0)
        
        # Softmax for confidence
        # Subtract max for numerical stability
        max_vals = np.max(contrast_stack, axis=0, keepdims=True)
        exp_vals = np.exp(contrast_stack - max_vals)
        softmax_vals = exp_vals / np.sum(exp_vals, axis=0, keepdims=True)
        
        # Argmax map
        argmax_indices = np.argmax(contrast_stack, axis=0)
        argmax_map = np.array([clusters[i] for i in argmax_indices.flat]).reshape(Ny, Nx)
        
        # Confidence (max softmax value)
        confidence_map = np.max(softmax_vals, axis=0)
        
        print(f"✓ Computed argmax (range: {argmax_map.min()}-{argmax_map.max()}) and confidence (mean: {confidence_map.mean():.3f})")
        
        return argmax_map, confidence_map
    
    def make_k_grids(self, detector_shape: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray]:
        """Create k-space coordinate grids."""
        calibration = self.config.get('calibration', {})
        
        Ky, Kx = detector_shape
        center_y = calibration.get('center_y', None)
        center_x = calibration.get('center_x', None)
        
        # Use image center if not specified
        if center_y is None:
            center_y = Ky // 2
        if center_x is None:
            center_x = Kx // 2
        
        # Pixel size in reciprocal space (1/Å or 1/nm)
        dk = calibration.get('pixel_size_k', 1.0)  # Default to relative units
        
        # Create coordinate arrays
        y_coords = (np.arange(Ky) - center_y) * dk
        x_coords = (np.arange(Kx) - center_x) * dk
        
        ky_grid, kx_grid = np.meshgrid(y_coords, x_coords, indexing='ij')
        
        print(f"📐 Created k-grids: center=({center_y}, {center_x}), dk={dk}")
        
        return ky_grid, kx_grid
    
    def compute_dpc_com(self, dataset: H5Dataset, labels: np.ndarray, 
                       ky_grid: np.ndarray, kx_grid: np.ndarray, 
                       chunk_size: int = 1024) -> Dict[str, np.ndarray]:
        """Compute DPC/CoM with cluster gating."""
        print("🧭 Computing DPC/CoM with cluster gating...")
        
        dpc_config = self.config.get('dpc', {})
        gate_mode = dpc_config.get('gate_mode', 'union')  # 'union' or 'adaptive'
        
        Ny, Nx = dataset.scan_shape
        
        # Initialize output maps
        com_x = np.zeros((Ny, Nx), dtype=np.float32)
        com_y = np.zeros((Ny, Nx), dtype=np.float32)
        
        # Create union gate if needed
        if gate_mode == 'union':
            gate_union = np.zeros_like(list(self.gates.values())[0])
            for gate in self.gates.values():
                gate_union = np.maximum(gate_union, gate)
            print(f"  Using union gate (coverage: {np.mean(gate_union > 0.1):.1%})")
        
        n_chunks = (dataset.total_frames + chunk_size - 1) // chunk_size
        
        with Timer(f"Processing {n_chunks} chunks for DPC/CoM"):
            for chunk_idx in range(n_chunks):
                start_idx = chunk_idx * chunk_size
                frames = dataset.load_chunk(start_idx, chunk_size)
                end_idx = min(start_idx + chunk_size, dataset.total_frames)
                
                if self.use_gpu:
                    frames = self.xp.asarray(frames)
                    ky_grid_gpu = self.xp.asarray(ky_grid)
                    kx_grid_gpu = self.xp.asarray(kx_grid)
                else:
                    ky_grid_gpu = ky_grid
                    kx_grid_gpu = kx_grid
                
                # Process each frame in chunk
                chunk_indices = np.arange(start_idx, end_idx)
                chunk_labels = labels[chunk_indices]
                
                for i, (frame_idx, label) in enumerate(zip(chunk_indices, chunk_labels)):
                    frame = frames[i] if self.use_gpu else frames[i]
                    
                    # Select appropriate gate
                    if gate_mode == 'union':
                        gate = self.xp.asarray(gate_union) if self.use_gpu else gate_union
                    elif gate_mode == 'adaptive' and label in self.gates:
                        gate = self.xp.asarray(self.gates[label]) if self.use_gpu else self.gates[label]
                    else:
                        # Fallback to uniform weighting
                        gate = self.xp.ones_like(frame) if self.use_gpu else np.ones_like(frame)
                    
                    # Compute weighted center of mass
                    weighted_intensity = gate * frame
                    total_weight = self.xp.sum(weighted_intensity) + 1e-8
                    
                    com_x_val = self.xp.sum(kx_grid_gpu * weighted_intensity) / total_weight
                    com_y_val = self.xp.sum(ky_grid_gpu * weighted_intensity) / total_weight
                    
                    if self.use_gpu:
                        com_x_val = self.xp.asnumpy(com_x_val)
                        com_y_val = self.xp.asnumpy(com_y_val)
                    
                    # Map to scan coordinates
                    scan_y = frame_idx // Nx
                    scan_x = frame_idx % Nx
                    com_x[scan_y, scan_x] = com_x_val
                    com_y[scan_y, scan_x] = com_y_val
                
                # Progress update
                if (chunk_idx + 1) % max(1, n_chunks // 10) == 0:
                    progress = (chunk_idx + 1) / n_chunks * 100
                    print(f"  Progress: {progress:.1f}%")
        
        # Convert to projected fields (simplified - assumes linear relationship)
        field_scale = dpc_config.get('field_scale', 1.0)
        Ex = com_x * field_scale
        Ey = com_y * field_scale
        
        # Compute magnitude and angle
        E_mag = np.sqrt(Ex**2 + Ey**2)
        E_angle = np.arctan2(Ey, Ex)
        
        print(f"✓ Computed DPC fields: |E| range [{E_mag.min():.4f}, {E_mag.max():.4f}]")
        
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
        print("💾 Saving outputs...")
        
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
            
            # Mean pattern for visualization
            # This would be computed during template building - simplified here
            mu_c_approx = self.mu_global + self.templates[cluster_id] * self.sigma_global
            plt.figure(figsize=(6, 6))
            plt.imshow(mu_c_approx, cmap='viridis')
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
        
        print(f"✓ Saved outputs to {output_dir}")

def create_visualizations(output_dir: Path, contrast_maps: Dict, argmax_map: np.ndarray, 
                         confidence_map: np.ndarray, dpc_results: Dict, templates: Dict, gates: Dict):
    """Create publication-ready visualizations."""
    print("🎨 Creating visualizations...")
    
    figures_dir = output_dir / 'figures'
    
    # Template montage
    clusters = list(templates.keys())
    n_clusters = len(clusters)
    cols = min(4, n_clusters)
    rows = (n_clusters + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(4*cols, 4*rows))
    if rows == 1:
        axes = axes.reshape(1, -1)
    fig.suptitle('Cluster Templates and Gates', fontsize=16)
    
    for i, cluster_id in enumerate(clusters):
        row, col = i // cols, i % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
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
        row, col = i // cols, i % cols
        axes[row, col].set_visible(False)
    
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
    
    print(f"✓ Created visualizations in {figures_dir}")

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
    
    print(f"✓ Saved provenance to {logs_dir}")

def load_config(config_path: Optional[Path]) -> Dict:
    """Load configuration from YAML file or return defaults."""
    default_config = {
        'device': 'cpu',
        'preprocess': {
            'background_subtract': False,
            'log_transform': True,
            'normalize': True
        },
        'templates': {
            'gate_type': 'sigmoid',  # 'sigmoid' or 'threshold'
            'threshold': 2.0,
            'alpha': 3.0,
            'smooth_sigma': 1.0
        },
        'dpc': {
            'gate_mode': 'union',  # 'union' or 'adaptive'
            'field_scale': 1.0
        },
        'calibration': {
            'center_y': None,  # Will use image center if None
            'center_x': None,
            'pixel_size_k': 1.0  # Reciprocal space pixel size
        },
        'validation': {
            'enable_split_half': False,
            'enable_controls': False
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
        print(f"📋 Loaded configuration from {config_path}")
    else:
        print("📋 Using default configuration")
    
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
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.data.exists():
        raise FileNotFoundError(f"Data file not found: {args.data}")
    if not args.labels.exists():
        raise FileNotFoundError(f"Labels file not found: {args.labels}")
    
    print("="*80)
    print("DATA-DRIVEN VIRTUAL DETECTORS FOR 4D-STEM POLARISATION MAPPING")
    print("="*80)
    print(f"📁 Data: {args.data}")
    print(f"📊 Labels: {args.labels}")
    print(f"📐 Scan shape: {args.scan_shape[0]} × {args.scan_shape[1]}")
    print(f"💾 Output: {args.out}")
    print(f"🖥️  Device: {args.device}")
    print("="*80)
    
    # Load configuration
    config = load_config(args.config)
    config['device'] = args.device  # Override device from CLI
    
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
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"🕒 Total time: {total_timer.elapsed:.1f}s")
    print(f"📂 Results saved to: {args.out}")
    print("\nGenerated outputs:")
    print("  📁 kspace/     - Templates, gates, and mean patterns")
    print("  📁 realspace/  - Contrast maps and DPC field components")  
    print("  📁 figures/    - Publication-ready visualizations")
    print("  📁 logs/       - Configuration and timing information")
    print("="*80)

if __name__ == "__main__":
    main()