#!/usr/bin/env python3
"""
Cluster-Averaged Diffraction Pattern Analysis

This script takes a 4D-STEM dataset and cluster labels to compute and visualize
cluster-averaged diffraction patterns (PACBED - Position-Averaged CBED).

The analysis includes:
1. Fair comparison: same centering, normalization, and background subtraction
2. High-SNR averaging within each cluster
3. Comprehensive visualization of all cluster diffraction patterns
4. Statistical analysis of cluster differences

Usage:
    python cluster_diffraction_analysis.py <h5_file> <cluster_labels> [options]

Example:
    python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --output cluster_analysis.png
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt
import sys
import os
from pathlib import Path
from typing import Tuple, Dict, List, Optional, Any
from scipy import ndimage
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Circle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import seaborn as sns

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'visualization'))
from stem_visualization import STEMVisualizer


class ClusterDiffractionAnalyzer:
    """
    Analyzer for cluster-averaged diffraction patterns.
    
    Provides methods for:
    - Loading and preprocessing 4D-STEM data
    - Computing cluster-averaged diffraction patterns  
    - Fair comparison with consistent centering and normalization
    - Comprehensive visualization and analysis
    """
    
    def __init__(self, data: np.ndarray, scan_shape: Tuple[int, int], cluster_labels: np.ndarray):
        """
        Initialize the analyzer.
        
        Args:
            data: 4D-STEM data (N, H, W)
            scan_shape: Shape of the scan grid (scan_y, scan_x)
            cluster_labels: Cluster labels for each scan position (scan_y, scan_x)
        """
        self.data = data
        self.scan_shape = scan_shape
        self.pattern_shape = data.shape[-2:]
        self.cluster_labels = cluster_labels
        
        # Validate inputs
        if cluster_labels.shape != scan_shape:
            raise ValueError(f"Cluster labels shape {cluster_labels.shape} doesn't match scan shape {scan_shape}")
        if data.shape[0] != np.prod(scan_shape):
            raise ValueError(f"Data length {data.shape[0]} doesn't match scan grid size {np.prod(scan_shape)}")
        
        # Get unique clusters
        self.unique_clusters = np.unique(cluster_labels[cluster_labels >= 0])  # Exclude any negative labels
        self.n_clusters = len(self.unique_clusters)
        
        print(f"Initialized analyzer with {self.n_clusters} clusters")
        print(f"Cluster IDs: {self.unique_clusters}")
        print(f"Data shape: {data.shape}")
        print(f"Scan shape: {scan_shape}")
        
        # Initialize storage for results
        self.cluster_patterns = {}
        self.cluster_centers = {}
        self.reference_pattern = None
        self.reference_center = None
        
    def compute_reference_pattern(self, method='mean'):
        """
        Compute reference diffraction pattern for consistent indexing.
        
        Args:
            method: Method for computing reference ('mean', 'median')
        """
        print(f"Computing reference pattern using {method}...")
        
        if method == 'mean':
            self.reference_pattern = np.mean(self.data, axis=0).astype(np.float64)
        elif method == 'median':
            self.reference_pattern = np.median(self.data, axis=0).astype(np.float64)
        else:
            raise ValueError(f"Unknown reference method: {method}")
        
        # Find reference beam center
        self.reference_center = self._find_beam_center(self.reference_pattern)
        print(f"Reference beam center: {self.reference_center}")
    
    def interactive_beam_center_selection(self):
        """
        Interactive beam center selection by clicking on the diffraction pattern.
        
        Returns:
            Tuple of (center_y, center_x) coordinates
        """
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        
        print("Interactive beam center selection:")
        print("1. A window will show the reference diffraction pattern")
        print("2. Click on the center of the direct beam")
        print("3. Click 'Done' to confirm or 'Auto' to use automatic detection")
        print("4. Close the window when finished")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        
        # Display reference pattern
        pattern_display = np.log(self.reference_pattern + 1e-6)
        im = ax.imshow(pattern_display, cmap='viridis')
        ax.set_title('Click on Direct Beam Center\n(Reference Pattern - Log Scale)', 
                    fontsize=14, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8)
        
        # Show current automatic center
        auto_y, auto_x = self.reference_center
        current_marker = ax.plot(auto_x, auto_y, 'r+', markersize=15, markeredgewidth=3, 
                               label='Current Center')[0]
        ax.legend()
        
        # State variables
        selected_center = self.reference_center
        click_marker = None
        
        def onclick(event):
            nonlocal selected_center, click_marker
            
            if event.inaxes != ax or event.button != 1:
                return
            
            # Get click coordinates
            x, y = int(round(event.xdata)), int(round(event.ydata))
            selected_center = (y, x)
            
            # Update marker
            if click_marker is not None:
                click_marker.remove()
            
            click_marker = ax.plot(x, y, 'go', markersize=12, markeredgewidth=2, 
                                 label='Selected Center')[0]
            ax.legend()
            ax.set_title(f'Selected Center: ({y}, {x})\nClick Done to confirm', 
                        fontsize=14, fontweight='bold')
            fig.canvas.draw()
        
        # Connect click event
        fig.canvas.mpl_connect('button_press_event', onclick)
        
        # Add control buttons
        button_height = 0.04
        button_width = 0.1
        
        # Done button
        ax_done = plt.axes([0.7, 0.02, button_width, button_height])
        done_button = Button(ax_done, 'Done')
        
        # Auto button  
        ax_auto = plt.axes([0.82, 0.02, button_width, button_height])
        auto_button = Button(ax_auto, 'Auto')
        
        # Reset button
        ax_reset = plt.axes([0.58, 0.02, button_width, button_height])
        reset_button = Button(ax_reset, 'Reset')
        
        def on_done(event):
            plt.close(fig)
        
        def on_auto(event):
            nonlocal selected_center, click_marker
            selected_center = self._find_beam_center(self.reference_pattern)
            
            # Update display
            if click_marker is not None:
                click_marker.remove()
            
            y, x = selected_center
            click_marker = ax.plot(x, y, 'go', markersize=12, markeredgewidth=2,
                                 label='Auto Center')[0]
            ax.legend()
            ax.set_title(f'Auto Center: ({y}, {x})\nClick Done to confirm',
                        fontsize=14, fontweight='bold')
            fig.canvas.draw()
        
        def on_reset(event):
            nonlocal selected_center, click_marker
            selected_center = self.reference_center
            
            if click_marker is not None:
                click_marker.remove()
                click_marker = None
            
            ax.legend()
            ax.set_title('Click on Direct Beam Center\n(Reference Pattern - Log Scale)',
                        fontsize=14, fontweight='bold')
            fig.canvas.draw()
        
        done_button.on_clicked(on_done)
        auto_button.on_clicked(on_auto)
        reset_button.on_clicked(on_reset)
        
        plt.show()
        
        # Update the reference center
        self.reference_center = selected_center
        print(f"Updated beam center: {self.reference_center}")
        
        return selected_center
        
    def _find_beam_center(self, pattern: np.ndarray) -> Tuple[int, int]:
        """
        Find the direct beam center in a diffraction pattern.
        
        Args:
            pattern: Diffraction pattern
            
        Returns:
            (center_y, center_x) coordinates
        """
        # Use center of mass approach for robustness
        # Convert to float32 if needed (scipy doesn't support float16)
        if pattern.dtype == np.float16:
            pattern_work = pattern.astype(np.float32)
        else:
            pattern_work = pattern
        
        # Apply Gaussian smoothing to reduce noise
        smooth_pattern = ndimage.gaussian_filter(pattern_work, sigma=2.0)
        
        # Find the maximum intensity region
        max_pos = np.unravel_index(np.argmax(smooth_pattern), smooth_pattern.shape)
        
        # Refine using center of mass in a local region
        region_size = min(pattern_work.shape) // 10
        y_min = max(0, max_pos[0] - region_size)
        y_max = min(pattern_work.shape[0], max_pos[0] + region_size)
        x_min = max(0, max_pos[1] - region_size)
        x_max = min(pattern_work.shape[1], max_pos[1] + region_size)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[y_min:y_max, x_min:x_max]
        region = smooth_pattern[y_min:y_max, x_min:x_max]
        
        # Center of mass calculation
        total_intensity = np.sum(region)
        if total_intensity > 0:
            center_y = np.sum(y_coords * region) / total_intensity
            center_x = np.sum(x_coords * region) / total_intensity
        else:
            center_y, center_x = max_pos
        
        return int(round(center_y)), int(round(center_x))
    
    def compute_cluster_patterns(self, normalize=True, subtract_background=True, 
                                recenter=True, method='mean', recenter_each=False):
        """
        Compute cluster-averaged diffraction patterns with consistent processing.
        
        Args:
            normalize: Whether to normalize by total counts
            subtract_background: Whether to subtract global background
            recenter: Whether to recenter patterns to reference
            method: Averaging method ('mean', 'median')
            recenter_each: Whether to recenter each pattern before averaging (reduces blur)
        """
        print("Computing cluster-averaged diffraction patterns...")
        
        # Compute reference if not already done
        if self.reference_pattern is None:
            self.compute_reference_pattern(method='mean')
        
        # Flatten cluster labels to match data indexing
        flat_labels = self.cluster_labels.flatten()
        
        for cluster_id in self.unique_clusters:
            print(f"Processing cluster {cluster_id}...")
            
            # Get indices for this cluster
            cluster_indices = np.where(flat_labels == cluster_id)[0]
            n_pixels = len(cluster_indices)
            print(f"  {n_pixels} pixels in cluster {cluster_id}")
            
            if n_pixels == 0:
                continue
            
            # Extract patterns for this cluster
            cluster_data = self.data[cluster_indices]
            
            # Optionally recenter each pattern before averaging (only if recentering is enabled)
            if recenter_each and recenter and self.reference_center is not None:
                print(f"  Recentering {n_pixels} patterns before averaging...")
                aligned = []
                ref_y, ref_x = self.reference_center
                
                for pattern in cluster_data:
                    # Find center of this pattern
                    center_y, center_x = self._find_beam_center(pattern)
                    
                    # Calculate shifts
                    dy = ref_y - center_y
                    dx = ref_x - center_x
                    
                    # Apply integer shifts using np.roll (fast)
                    aligned_pattern = np.roll(np.roll(pattern, int(dy), axis=0), int(dx), axis=1)
                    
                    # Optional: apply subpixel shift for even better alignment
                    # (disabled by default for speed)
                    # fractional_dy = dy - int(dy)
                    # fractional_dx = dx - int(dx)
                    # if abs(fractional_dy) > 0.1 or abs(fractional_dx) > 0.1:
                    #     from scipy import ndimage
                    #     aligned_pattern = ndimage.shift(aligned_pattern, 
                    #                                    (fractional_dy, fractional_dx), 
                    #                                    order=1, cval=0)
                    
                    aligned.append(aligned_pattern)
                
                cluster_data = np.stack(aligned, axis=0)
            
            # Compute average pattern
            if method == 'mean':
                avg_pattern = np.mean(cluster_data, axis=0).astype(np.float32)
            elif method == 'median':
                avg_pattern = np.median(cluster_data, axis=0).astype(np.float32)
            else:
                raise ValueError(f"Unknown averaging method: {method}")
            
            # Apply preprocessing steps
            processed_pattern = self._preprocess_pattern(
                avg_pattern, normalize, subtract_background, recenter
            )
            
            self.cluster_patterns[cluster_id] = processed_pattern
            
        print(f"Computed patterns for {len(self.cluster_patterns)} clusters")
    
    def _preprocess_pattern(self, pattern: np.ndarray, normalize: bool, 
                          subtract_background: bool, recenter: bool) -> np.ndarray:
        """
        Apply consistent preprocessing to a diffraction pattern.
        
        Args:
            pattern: Input diffraction pattern
            normalize: Whether to normalize by total counts
            subtract_background: Whether to subtract background
            recenter: Whether to recenter relative to reference
            
        Returns:
            Processed pattern
        """
        # Convert to float32 if needed for processing
        if pattern.dtype == np.float16:
            processed = pattern.astype(np.float32)
        else:
            processed = pattern.copy().astype(np.float32)
        
        # 1. Normalize by total counts
        if normalize:
            total_counts = np.sum(processed)
            if total_counts > 0:
                processed = processed / total_counts
        
        # 2. Subtract global background (estimate from high-radius annulus)
        if subtract_background:
            # Use high-radius annulus for robust background estimation
            cy, cx = self.reference_center
            H, W = processed.shape
            yy, xx = np.mgrid[0:H, 0:W]
            r = np.hypot(xx - cx, yy - cy)
            r_in, r_out = 0.35 * min(H, W), 0.48 * min(H, W)  # thin outer ring
            annulus_mask = (r >= r_in) & (r <= r_out)
            
            if np.sum(annulus_mask) > 0:
                background = np.median(processed[annulus_mask])
                processed = np.clip(processed - background, 0, None)
            else:
                # Fallback to edge pixels if annulus is empty
                edge_width = 5
                edge_pixels = np.concatenate([
                    processed[:edge_width, :].flatten(),
                    processed[-edge_width:, :].flatten(),
                    processed[:, :edge_width].flatten(),
                    processed[:, -edge_width:].flatten()
                ])
                background = np.median(edge_pixels)
                processed = np.maximum(processed - background, 0)
        
        # 3. Recenter (shift to align with reference center)
        if recenter and self.reference_center is not None:
            pattern_center = self._find_beam_center(processed)
            dy = self.reference_center[0] - pattern_center[0]
            dx = self.reference_center[1] - pattern_center[1]
            
            # Apply shift using scipy (processed is already float32)
            processed = ndimage.shift(processed, (dy, dx), order=1, cval=0)
        
        return processed
    
    def create_cluster_comparison_plot(self, figsize=(20, 12), use_log=True, 
                                     colormap='viridis', include_direction_labels=False) -> plt.Figure:
        """
        Create comprehensive comparison plot of all cluster patterns.
        
        Args:
            figsize: Figure size
            use_log: Whether to use log scale for intensity
            colormap: Colormap for diffraction patterns
            include_direction_labels: Whether to include direction labels in titles
            
        Returns:
            Matplotlib figure
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        n_clusters = len(self.cluster_patterns)
        
        # Calculate layout
        if n_clusters <= 4:
            rows, cols = 2, 2
        elif n_clusters <= 6:
            rows, cols = 2, 3
        elif n_clusters <= 9:
            rows, cols = 3, 3
        else:
            rows, cols = 4, 4
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        # Ensure axes is 2D
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        # Add reference pattern in first position
        plot_idx = 0
        row, col = plot_idx // cols, plot_idx % cols
        
        ref_pattern = self.reference_pattern
        if use_log:
            ref_pattern = np.log(ref_pattern + 1e-6)
        
        im = axes[row, col].imshow(ref_pattern, cmap=colormap)
        axes[row, col].set_title('Reference\n(Scan Average)', fontweight='bold')
        axes[row, col].axis('off')
        plt.colorbar(im, ax=axes[row, col], shrink=0.6)
        
        # Add beam center marker
        ref_y, ref_x = self.reference_center
        axes[row, col].plot(ref_x, ref_y, 'r+', markersize=10, markeredgewidth=2)
        
        plot_idx += 1
        
        # Get direction labels if requested
        direction_labels = {}
        if include_direction_labels:
            direction_labels = self.compute_cluster_directions()
        
        # Plot cluster patterns
        for cluster_id in sorted(self.cluster_patterns.keys()):
            if plot_idx >= rows * cols:
                break
                
            row, col = plot_idx // cols, plot_idx % cols
            
            pattern = self.cluster_patterns[cluster_id]
            if use_log:
                pattern = np.log(pattern + 1e-6)
            
            im = axes[row, col].imshow(pattern, cmap=colormap)
            
            # Create title with optional direction label
            title = f'Cluster {cluster_id}'
            if include_direction_labels and cluster_id in direction_labels:
                title += f'\n({direction_labels[cluster_id]})'
            
            axes[row, col].set_title(title, fontweight='bold')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.6)
            
            # Add beam center marker
            axes[row, col].plot(ref_x, ref_y, 'r+', markersize=8, markeredgewidth=2)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        # Add overall title
        scale_text = "(Log Scale)" if use_log else "(Linear Scale)"
        fig.suptitle(f'Cluster-Averaged Diffraction Patterns {scale_text}', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        return fig
    
    def create_difference_analysis(self, reference_cluster=None, figsize=(20, 12)) -> plt.Figure:
        """
        Create difference analysis between clusters and reference.
        
        Args:
            reference_cluster: Cluster ID to use as reference. If None, uses scan average.
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        # Choose reference
        if reference_cluster is not None and reference_cluster in self.cluster_patterns:
            reference = self.cluster_patterns[reference_cluster]
            ref_title = f"Cluster {reference_cluster}"
        else:
            reference = self.reference_pattern
            # Normalize reference pattern the same way as cluster patterns
            reference = self._preprocess_pattern(reference, True, True, False)
            ref_title = "Scan Average"
        
        n_clusters = len(self.cluster_patterns)
        cols = min(4, n_clusters)
        rows = (n_clusters + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        plot_idx = 0
        for cluster_id in sorted(self.cluster_patterns.keys()):
            if plot_idx >= rows * cols:
                break
                
            row, col = plot_idx // cols, plot_idx % cols
            
            # Compute difference
            pattern = self.cluster_patterns[cluster_id]
            difference = pattern - reference
            
            # Plot difference with symmetric colormap
            vmax = np.percentile(np.abs(difference), 95)
            im = axes[row, col].imshow(difference, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            axes[row, col].set_title(f'Cluster {cluster_id} - {ref_title}')
            axes[row, col].axis('off')
            plt.colorbar(im, ax=axes[row, col], shrink=0.6)
            
            # Add beam center marker
            ref_y, ref_x = self.reference_center
            axes[row, col].plot(ref_x, ref_y, 'k+', markersize=8, markeredgewidth=2)
            
            plot_idx += 1
        
        # Hide unused subplots
        for i in range(plot_idx, rows * cols):
            row, col = i // cols, i % cols
            axes[row, col].axis('off')
        
        fig.suptitle('Cluster Difference Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        return fig
    
    def create_enhanced_difference_analysis(self, figsize=(24, 16)) -> plt.Figure:
        """
        Create enhanced difference analysis with multiple visualization approaches.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with comprehensive difference analysis
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        # Prepare reference (scan average with same preprocessing)
        reference = self._preprocess_pattern(self.reference_pattern, True, True, False)
        
        n_clusters = len(self.cluster_patterns)
        cluster_ids = sorted(self.cluster_patterns.keys())
        
        # Create 4-row layout: raw differences, percentage differences, statistical significance, cumulative analysis
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(4, n_clusters + 1, height_ratios=[1, 1, 1, 1], width_ratios=[1]*n_clusters + [0.3])
        
        # Row 1: Raw differences (enhanced contrast)
        differences = {}
        for i, cluster_id in enumerate(cluster_ids):
            ax = fig.add_subplot(gs[0, i])
            
            pattern = self.cluster_patterns[cluster_id]
            difference = pattern - reference
            differences[cluster_id] = difference
            
            # Enhanced contrast using adaptive scaling
            vmax = np.percentile(np.abs(difference), 98)  # More sensitive to outliers
            im = ax.imshow(difference, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Cluster {cluster_id}\nRaw Difference', fontweight='bold')
            ax.axis('off')
            
            # Add beam center marker
            ref_y, ref_x = self.reference_center
            ax.plot(ref_x, ref_y, 'k+', markersize=8, markeredgewidth=2)
        
        # Colorbar for raw differences
        cbar_ax1 = fig.add_subplot(gs[0, -1])
        plt.colorbar(im, cax=cbar_ax1, label='Intensity Difference')
        
        # Row 2: Percentage differences (relative to reference)
        for i, cluster_id in enumerate(cluster_ids):
            ax = fig.add_subplot(gs[1, i])
            
            pattern = self.cluster_patterns[cluster_id]
            # Avoid division by zero
            relative_diff = (pattern - reference) / (reference + 1e-8) * 100
            
            # Clip extreme values for better visualization
            vmax = np.percentile(np.abs(relative_diff), 95)
            vmax = min(vmax, 200)  # Cap at 200% for readability
            
            im2 = ax.imshow(relative_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Cluster {cluster_id}\nRelative Difference (%)', fontweight='bold')
            ax.axis('off')
            ax.plot(ref_x, ref_y, 'k+', markersize=8, markeredgewidth=2)
        
        # Colorbar for percentage differences  
        cbar_ax2 = fig.add_subplot(gs[1, -1])
        plt.colorbar(im2, cax=cbar_ax2, label='Relative Difference (%)')
        
        # Row 3: Statistical significance map (Z-score-like measure)
        # Estimate noise from edge regions of reference pattern
        edge_width = 10
        edge_pixels = np.concatenate([
            reference[:edge_width, :].flatten(),
            reference[-edge_width:, :].flatten(),
            reference[:, :edge_width].flatten(),
            reference[:, -edge_width:].flatten()
        ])
        noise_std = np.std(edge_pixels)
        
        for i, cluster_id in enumerate(cluster_ids):
            ax = fig.add_subplot(gs[2, i])
            
            difference = differences[cluster_id]
            # Statistical significance: difference relative to noise
            significance = np.abs(difference) / (noise_std + 1e-8)
            
            # Log scale for significance
            significance_log = np.log10(significance + 1)
            
            im3 = ax.imshow(significance_log, cmap='hot', vmin=0, vmax=np.percentile(significance_log, 95))
            ax.set_title(f'Cluster {cluster_id}\nSignificance Map', fontweight='bold')
            ax.axis('off')
            ax.plot(ref_x, ref_y, 'k+', markersize=8, markeredgewidth=2)
        
        # Colorbar for significance
        cbar_ax3 = fig.add_subplot(gs[2, -1])
        plt.colorbar(im3, cax=cbar_ax3, label='log₁₀(|Diff|/Noise + 1)')
        
        # Row 4: Cumulative difference analysis
        ax_cumulative = fig.add_subplot(gs[3, :n_clusters])
        
        # Compute cumulative differences in concentric rings
        center_y, center_x = self.reference_center
        max_radius = min(self.pattern_shape) // 2
        radii = np.linspace(0, max_radius, 50)
        
        y_coords, x_coords = np.mgrid[0:self.pattern_shape[0], 0:self.pattern_shape[1]]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(cluster_ids)))
        
        for i, cluster_id in enumerate(cluster_ids):
            difference = differences[cluster_id]
            cumulative_diff = np.zeros_like(radii)
            
            for j, r in enumerate(radii[1:], 1):
                ring_mask = distances <= r
                cumulative_diff[j] = np.sum(np.abs(difference[ring_mask]))
            
            # Normalize by area
            cumulative_diff = cumulative_diff / (np.pi * radii**2 + 1e-8)
            
            ax_cumulative.plot(radii, cumulative_diff, color=colors[i], linewidth=3,
                             label=f'Cluster {cluster_id}', marker='o', markersize=4)
        
        ax_cumulative.set_xlabel('Radius from Beam Center (pixels)')
        ax_cumulative.set_ylabel('Cumulative |Difference| / Area')
        ax_cumulative.set_title('Radial Cumulative Difference Analysis')
        ax_cumulative.legend()
        ax_cumulative.grid(True, alpha=0.3)
        
        # Add overall title
        fig.suptitle('Enhanced Cluster Difference Analysis', fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        return fig
    
    def create_feature_highlight_analysis(self, figsize=(20, 8)) -> plt.Figure:
        """
        Create analysis highlighting specific diffraction features.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure with feature-enhanced visualization
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        reference = self._preprocess_pattern(self.reference_pattern, True, True, False)
        cluster_ids = sorted(self.cluster_patterns.keys())
        
        # Create 2x(n_clusters) layout
        fig, axes = plt.subplots(2, len(cluster_ids), figsize=figsize)
        if len(cluster_ids) == 1:
            axes = axes.reshape(2, 1)
        
        # Top row: High-contrast difference (enhanced for weak features)
        for i, cluster_id in enumerate(cluster_ids):
            ax = axes[0, i]
            
            pattern = self.cluster_patterns[cluster_id]
            difference = pattern - reference
            
            # Apply edge enhancement to highlight structural differences
            from scipy import ndimage
            # Create a Laplacian filter to enhance edges/features
            laplacian_kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
            enhanced_diff = ndimage.convolve(difference, laplacian_kernel, mode='constant')
            
            # Normalize and enhance contrast
            enhanced_diff = enhanced_diff / (np.std(enhanced_diff) + 1e-8)
            vmax = np.percentile(np.abs(enhanced_diff), 97)
            
            im = ax.imshow(enhanced_diff, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Cluster {cluster_id}\nEdge-Enhanced Differences', fontweight='bold')
            ax.axis('off')
            
            # Mark beam center
            ref_y, ref_x = self.reference_center
            ax.plot(ref_x, ref_y, 'k+', markersize=8, markeredgewidth=2)
            
            plt.colorbar(im, ax=ax, shrink=0.6)
        
        # Bottom row: Ratio analysis (cluster/reference)
        for i, cluster_id in enumerate(cluster_ids):
            ax = axes[1, i]
            
            pattern = self.cluster_patterns[cluster_id]
            # Compute ratio with smoothing to avoid noise amplification
            smoothed_reference = ndimage.gaussian_filter(reference, sigma=1.0)
            smoothed_pattern = ndimage.gaussian_filter(pattern, sigma=1.0)
            
            ratio = smoothed_pattern / (smoothed_reference + 1e-6)
            
            # Log scale ratio for better visualization
            log_ratio = np.log2(ratio + 1e-6)
            
            # Symmetric limits around 1 (log2(1) = 0)
            vmax = np.percentile(np.abs(log_ratio), 95)
            vmax = min(vmax, 2)  # Cap at 4x difference
            
            im = ax.imshow(log_ratio, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
            ax.set_title(f'Cluster {cluster_id}\nIntensity Ratio (log₂)', fontweight='bold')
            ax.axis('off')
            
            ax.plot(ref_x, ref_y, 'k+', markersize=8, markeredgewidth=2)
            
            # Add ratio colorbar with interpretable labels
            cbar = plt.colorbar(im, ax=ax, shrink=0.6)
            
            # Add ratio interpretation labels
            tick_positions = [-2, -1, 0, 1, 2]
            tick_labels = ['¼×', '½×', '1×', '2×', '4×']
            cbar.set_ticks([t for t in tick_positions if -vmax <= t <= vmax])
            cbar.set_ticklabels([tick_labels[i] for i, t in enumerate(tick_positions) if -vmax <= t <= vmax])
        
        fig.suptitle('Feature-Enhanced Cluster Analysis', fontsize=16, fontweight='bold', y=0.95)
        plt.tight_layout()
        return fig
    
    def create_radial_profile_comparison(self, figsize=(12, 8)) -> plt.Figure:
        """
        Create radial profile comparison between clusters.
        
        Args:
            figsize: Figure size
            
        Returns:
            Matplotlib figure
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Compute radial profiles
        center_y, center_x = self.reference_center
        max_radius = min(self.pattern_shape) // 2
        radii = np.arange(1, max_radius)
        
        # Create coordinate grids
        y_coords, x_coords = np.mgrid[0:self.pattern_shape[0], 0:self.pattern_shape[1]]
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Plot 1: Individual radial profiles
        colors = plt.cm.tab10(np.linspace(0, 1, len(self.cluster_patterns) + 1))
        
        # Reference profile
        ref_profile = self._compute_radial_profile(self.reference_pattern, distances, radii)
        ax1.semilogy(radii, ref_profile, 'k-', linewidth=2, label='Reference', alpha=0.7)
        
        # Cluster profiles
        for i, cluster_id in enumerate(sorted(self.cluster_patterns.keys())):
            pattern = self.cluster_patterns[cluster_id]
            profile = self._compute_radial_profile(pattern, distances, radii)
            ax1.semilogy(radii, profile, color=colors[i+1], linewidth=2, 
                        label=f'Cluster {cluster_id}')
        
        ax1.set_xlabel('Radius (pixels)')
        ax1.set_ylabel('Intensity')
        ax1.set_title('Radial Profiles Comparison')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Difference profiles (normalized)
        for i, cluster_id in enumerate(sorted(self.cluster_patterns.keys())):
            pattern = self.cluster_patterns[cluster_id]
            profile = self._compute_radial_profile(pattern, distances, radii)
            
            # Normalize profiles for fair comparison
            ref_norm = ref_profile / np.sum(ref_profile)
            profile_norm = profile / np.sum(profile)
            
            difference = (profile_norm - ref_norm) / (ref_norm + 1e-10)
            ax2.plot(radii, difference * 100, color=colors[i+1], linewidth=2, 
                    label=f'Cluster {cluster_id}')
        
        ax2.set_xlabel('Radius (pixels)')
        ax2.set_ylabel('Relative Difference (%)')
        ax2.set_title('Radial Profile Differences vs Reference')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.axhline(0, color='k', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig
    
    def create_bias_corrected_visualization(self, figsize=(20, 12), use_log=True,
                                          colormap='viridis', contrast_percentile=95) -> plt.Figure:
        """
        Create visualization of cluster patterns minus scan-average to check for global bias.
        This highlights unique cluster features by removing common background and reveals
        ghost half-integer disks that might be overshadowed by fundamental reflections.
        
        Args:
            figsize: Figure size
            use_log: Whether to use log scale for display
            colormap: Colormap to use
            contrast_percentile: Percentile for contrast adjustment
            
        Returns:
            Matplotlib figure
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        n_clusters = len(self.cluster_patterns)
        n_cols = min(n_clusters, 4)
        n_rows = (n_clusters + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        # Ensure axes is always a flat array for consistent indexing
        if n_clusters == 1:
            axes = np.array([axes])
        else:
            axes = np.array(axes).flatten()
        
        # Get reference pattern (scan average) - preprocess to match cluster preprocessing
        reference = self._preprocess_pattern(self.reference_pattern, True, True, False)
        
        cluster_ids = sorted(self.cluster_patterns.keys())
        
        # Compute cluster counts
        flat_labels = self.cluster_labels.flatten()
        cluster_counts = {cluster_id: np.sum(flat_labels == cluster_id) 
                         for cluster_id in cluster_ids}
        
        # Compute all bias-corrected patterns first for global color scaling
        bias_corr_list = []
        for cluster_id in cluster_ids:
            cluster_pattern = self.cluster_patterns[cluster_id]
            bias_corr_list.append(cluster_pattern - reference)
        
        # Apply log scale if requested
        if use_log:
            disp_list = [np.sign(b) * np.log10(1 + np.abs(b)) for b in bias_corr_list]
        else:
            disp_list = bias_corr_list
        
        # Choose ONE symmetric range for all panels
        global_v = np.percentile(np.abs(np.concatenate([d.ravel() for d in disp_list])), contrast_percentile)
        vmin, vmax = -global_v, global_v
        
        for i, cluster_id in enumerate(cluster_ids):
            ax = axes[i]
            
            # Use diverging colormap for better difference visualization
            cmap_to_use = 'RdBu_r' if use_log else colormap
            
            im = ax.imshow(disp_list[i], cmap=cmap_to_use, vmin=vmin, vmax=vmax,
                          origin='lower', interpolation='nearest')
            
            # Add beam center marker
            if hasattr(self, 'reference_center') and self.reference_center is not None:
                center_y, center_x = self.reference_center
                ax.plot(center_x, center_y, 'r+', markersize=8, markeredgewidth=2)
            
            ax.set_title(f'Cluster {cluster_id} - Scan Average\n(n={cluster_counts[cluster_id]})', 
                        fontsize=10)
            ax.axis('off')
            
            # Add colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cbar = plt.colorbar(im, cax=cax)
            cax.tick_params(labelsize=8)
            if use_log:
                cbar.set_label('Log₁₀(Intensity Difference)', fontsize=8)
            else:
                cbar.set_label('Intensity Difference', fontsize=8)
        
        # Hide unused subplots
        for i in range(len(cluster_ids), len(axes)):
            axes[i].axis('off')
        
        # Main title
        log_text = " (Log Scale)" if use_log else ""
        fig.suptitle(f'Cluster Patterns - Scan Average{log_text}\n'
                    f'Highlights unique cluster features and removes global bias', 
                    fontsize=14, y=0.95)
        
        plt.tight_layout()
        return fig
    
    def _compute_radial_profile(self, pattern: np.ndarray, distances: np.ndarray, 
                               radii: np.ndarray) -> np.ndarray:
        """
        Compute radial profile of a diffraction pattern.
        
        Args:
            pattern: Diffraction pattern
            distances: Distance array from beam center
            radii: Radii to evaluate
            
        Returns:
            Radial profile array
        """
        profile = np.zeros_like(radii, dtype=np.float64)
        
        for i, r in enumerate(radii):
            ring_mask = (distances >= r - 0.5) & (distances < r + 0.5)
            if np.sum(ring_mask) > 0:
                profile[i] = np.mean(pattern[ring_mask])
        
        return profile
    
    def compute_cluster_statistics(self) -> Dict[str, Any]:
        """
        Compute statistical measures for cluster comparison.
        
        Returns:
            Dictionary of statistics
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        stats = {}
        
        # Basic statistics per cluster
        for cluster_id in self.cluster_patterns:
            pattern = self.cluster_patterns[cluster_id]
            stats[f'cluster_{cluster_id}'] = {
                'mean_intensity': np.mean(pattern),
                'std_intensity': np.std(pattern),
                'max_intensity': np.max(pattern),
                'total_intensity': np.sum(pattern)
            }
        
        # Cross-correlation matrix
        cluster_ids = sorted(self.cluster_patterns.keys())
        n_clusters = len(cluster_ids)
        correlation_matrix = np.zeros((n_clusters, n_clusters))
        
        for i, id1 in enumerate(cluster_ids):
            for j, id2 in enumerate(cluster_ids):
                pattern1 = self.cluster_patterns[id1].flatten()
                pattern2 = self.cluster_patterns[id2].flatten()
                
                # Pearson correlation
                correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                correlation_matrix[i, j] = correlation
        
        stats['correlation_matrix'] = correlation_matrix
        stats['cluster_ids'] = cluster_ids
        
        return stats
    
    def save_analysis_results(self, output_dir: str):
        """
        Save analysis results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"Saving analysis results to {output_path}")
        
        # Save cluster patterns
        patterns_file = output_path / "cluster_patterns.npz"
        np.savez(patterns_file, **{f'cluster_{k}': v for k, v in self.cluster_patterns.items()})
        
        # Save reference pattern
        if self.reference_pattern is not None:
            np.save(output_path / "reference_pattern.npy", self.reference_pattern)
        
        # Save statistics in portable NPZ format
        stats = self.compute_cluster_statistics()
        
        # Extract arrays for NPZ format
        arrays_to_save = {
            'correlation_matrix': stats['correlation_matrix'],
            'cluster_ids': np.array(stats['cluster_ids'], dtype=int)
        }
        
        # Add per-cluster statistics as arrays
        for cluster_id in stats['cluster_ids']:
            cluster_stats = stats[f'cluster_{cluster_id}']
            for stat_name, value in cluster_stats.items():
                arrays_to_save[f'cluster_{cluster_id}_{stat_name}'] = np.array(value)
        
        np.savez(output_path / "cluster_statistics.npz", **arrays_to_save)
        
        # Optionally save metadata as JSON for complete portability
        import json
        metadata = {
            'cluster_ids': [int(x) for x in stats['cluster_ids']],
            'n_clusters': len(stats['cluster_ids']),
            'analysis_type': 'cluster_averaged_diffraction'
        }
        
        with open(output_path / "analysis_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print("Analysis results saved successfully")
    
    def guess_direction(self, pattern: np.ndarray, center: Tuple[int, int], 
                       r_in_frac: float = 0.15, r_out_frac: float = 0.45) -> str:
        """
        Guess crystal orientation direction based on intensity asymmetry.
        
        Args:
            pattern: Diffraction pattern
            center: Beam center coordinates (y, x)
            r_in_frac: Inner radius fraction of pattern size
            r_out_frac: Outer radius fraction of pattern size
            
        Returns:
            Direction label: "horizontal", "vertical", or "out-of-plane"
        """
        H, W = pattern.shape
        cy, cx = center
        yy, xx = np.mgrid[0:H, 0:W]
        rr = np.hypot(xx - cx, yy - cy)
        
        r_in = r_in_frac * min(H, W)
        r_out = r_out_frac * min(H, W)
        ring = (rr >= r_in) & (rr <= r_out)
        
        # Calculate average intensities in different quadrants
        left = pattern[ring & (xx < cx)].mean()
        right = pattern[ring & (xx > cx)].mean()
        top = pattern[ring & (yy < cy)].mean()
        bottom = pattern[ring & (yy > cy)].mean()
        
        # Calculate asymmetry measures
        lr_asymmetry = abs(left - right)
        tb_asymmetry = abs(top - bottom)
        
        # Classify based on dominant asymmetry
        if lr_asymmetry > 1.2 * tb_asymmetry:
            return "horizontal"
        elif tb_asymmetry > 1.2 * lr_asymmetry:
            return "vertical"
        else:
            return "out-of-plane"
    
    def compute_cluster_directions(self) -> Dict[int, str]:
        """
        Compute qualitative direction labels for all clusters.
        
        Returns:
            Dictionary mapping cluster_id to direction string
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        directions = {}
        print("Computing qualitative direction labels...")
        
        for cluster_id, pattern in self.cluster_patterns.items():
            direction = self.guess_direction(pattern, self.reference_center)
            directions[cluster_id] = direction
            print(f"  Cluster {cluster_id}: {direction}")
        
        return directions
    
    def diagnose_label_alignment(self) -> Dict[str, float]:
        """
        Diagnose potential label misalignment by checking cluster pattern similarity.
        
        Returns:
            Dictionary with diagnostic metrics
        """
        if not self.cluster_patterns:
            raise ValueError("No cluster patterns computed. Run compute_cluster_patterns() first.")
        
        # Compute cross-correlations between cluster patterns
        cluster_ids = sorted(self.cluster_patterns.keys())
        n_clusters = len(cluster_ids)
        
        correlations = []
        for i, id1 in enumerate(cluster_ids):
            for j, id2 in enumerate(cluster_ids):
                if i < j:  # Only upper triangle
                    pattern1 = self.cluster_patterns[id1].flatten()
                    pattern2 = self.cluster_patterns[id2].flatten()
                    correlation = np.corrcoef(pattern1, pattern2)[0, 1]
                    correlations.append(correlation)
        
        avg_correlation = np.mean(correlations)
        min_correlation = np.min(correlations)
        
        diagnostics = {
            'average_cross_correlation': avg_correlation,
            'minimum_cross_correlation': min_correlation,
            'n_clusters': n_clusters,
            'alignment_quality': 'GOOD' if avg_correlation < 0.9 else 'SUSPICIOUS'
        }
        
        print("\n" + "="*50)
        print("LABEL ALIGNMENT DIAGNOSTICS")
        print("="*50)
        print(f"Average cross-correlation: {avg_correlation:.3f}")
        print(f"Minimum cross-correlation: {min_correlation:.3f}")
        print(f"Alignment quality: {diagnostics['alignment_quality']}")
        
        if avg_correlation > 0.9:
            print("⚠️  WARNING: High cross-correlation suggests misaligned labels!")
            print("   All cluster patterns look too similar (like global average)")
            print("   Try running with --flip-labels to test coordinate alignment")
        else:
            print("✓ Good cluster distinctiveness - labels appear correctly aligned")
        
        return diagnostics


def load_h5_data(h5_path: str) -> Tuple[np.ndarray, Tuple[int, int], Dict[str, Any]]:
    """Load 4D-STEM data from H5 file."""
    print(f"Loading data from {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Load data
        if 'data' in f:
            data = f['data'][:]
        elif 'patterns' in f:
            data = f['patterns'][:]
        else:
            keys = list(f.keys())
            data = f[keys[0]][:]
        
        # Load metadata
        metadata = {}
        for key, value in f.attrs.items():
            metadata[key] = value
    
    print(f"Loaded data shape: {data.shape}")
    
    # Infer scan shape
    if len(data.shape) == 4:
        scan_shape = data.shape[:2]
        data = data.reshape(-1, *data.shape[2:])
    else:
        # Try to infer from metadata or make square assumption
        n_patterns = data.shape[0]
        scan_side = int(np.sqrt(n_patterns))
        if scan_side * scan_side == n_patterns:
            scan_shape = (scan_side, scan_side)
        else:
            # Find best rectangular fit
            factors = []
            for i in range(1, int(np.sqrt(n_patterns)) + 1):
                if n_patterns % i == 0:
                    factors.append((i, n_patterns // i))
            scan_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
    
    print(f"Inferred scan shape: {scan_shape}")
    return data, scan_shape, metadata


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Cluster-averaged diffraction pattern analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic analysis
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy
  
  # Specify output directory
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --output results/
  
  # Use median averaging and custom normalization
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --method median --no-normalize
  
  # Interactive beam center selection
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --interactive-center
  
  # With qualitative direction labeling and robust centering
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --label-guess --recenter-each
  
  # Troubleshoot scan coordinate alignment (if clusters look too similar)
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --flip-labels
  
  # Manual scan shape override (if auto-inference is wrong)
  python cluster_diffraction_analysis.py data.h5 cluster_labels.npy --scan-shape 209 194
        """
    )
    
    parser.add_argument('h5_file', help='Path to 4D-STEM H5 file')
    parser.add_argument('cluster_labels', help='Path to cluster labels (.npy file)')
    parser.add_argument('--output', '-o', default='cluster_analysis_results', 
                       help='Output directory (default: cluster_analysis_results)')
    parser.add_argument('--method', choices=['mean', 'median'], default='mean',
                       help='Averaging method (default: mean)')
    parser.add_argument('--no-normalize', action='store_true',
                       help='Skip normalization by total counts')
    parser.add_argument('--no-background', action='store_true', 
                       help='Skip background subtraction')
    parser.add_argument('--no-recenter', action='store_true',
                       help='Skip all recentering operations (both individual and final)')
    parser.add_argument('--colormap', default='viridis',
                       help='Colormap for diffraction patterns (default: viridis)')
    parser.add_argument('--interactive-center', action='store_true',
                       help='Interactive beam center selection')
    parser.add_argument('--recenter-each', action='store_true',
                       help='Recenter each pattern before averaging (requires recentering enabled, reduces blur)')
    parser.add_argument('--label-guess', action='store_true',
                       help='Include qualitative orientation labels (horizontal/vertical/out-of-plane)')
    parser.add_argument('--flip-labels', action='store_true',
                       help='Transpose cluster labels to test scan_y/scan_x alignment (troubleshooting)')
    parser.add_argument('--scan-shape', nargs=2, type=int, default=None,
                       help='Manual scan shape (scan_y scan_x) to override automatic inference')
    
    args = parser.parse_args()
    
    # Validate argument combinations
    if args.recenter_each and args.no_recenter:
        print("Warning: --recenter-each has no effect when --no-recenter is used")
        print("Either remove --no-recenter to enable recentering, or remove --recenter-each")
    
    try:
        # Load data
        data, auto_scan_shape, metadata = load_h5_data(args.h5_file)
        
        # Use manual scan shape if provided
        if args.scan_shape:
            scan_shape = tuple(args.scan_shape)
            print(f"Using manual scan shape: {scan_shape} (auto-inferred was: {auto_scan_shape})")
            # Validate that total patterns match
            expected_patterns = scan_shape[0] * scan_shape[1]
            if expected_patterns != data.shape[0]:
                print(f"Error: Manual scan shape {scan_shape} implies {expected_patterns} patterns, but data has {data.shape[0]}")
                return 1
        else:
            scan_shape = auto_scan_shape
            print(f"Using auto-inferred scan shape: {scan_shape}")
        
        # Load cluster labels
        print(f"Loading cluster labels from {args.cluster_labels}")
        cluster_labels = np.load(args.cluster_labels)
        
        # Validate cluster labels shape
        if cluster_labels.shape != scan_shape:
            print(f"Reshaping cluster labels from {cluster_labels.shape} to {scan_shape}")
            cluster_labels = cluster_labels.reshape(scan_shape)
        
        # Test label alignment with flip option
        if args.flip_labels:
            print("TROUBLESHOOTING: Flipping cluster labels (scan_y ↔ scan_x)")
            print(f"Original label shape: {cluster_labels.shape}")
            cluster_labels = cluster_labels.T  # Transpose to swap dimensions
            print(f"Flipped label shape: {cluster_labels.shape}")
            print("If cluster patterns now look more distinct, there was a scan coordinate mismatch!")
        
        # Create analyzer
        analyzer = ClusterDiffractionAnalyzer(data, scan_shape, cluster_labels)
        
        # Interactive beam center selection if requested
        if args.interactive_center:
            print("\nStarting interactive beam center selection...")
            analyzer.compute_reference_pattern(method='mean')  # Ensure reference is computed
            analyzer.interactive_beam_center_selection()
        
        # Compute cluster patterns
        print("\nComputing cluster-averaged diffraction patterns...")
        analyzer.compute_cluster_patterns(
            normalize=not args.no_normalize,
            subtract_background=not args.no_background,
            recenter=not args.no_recenter,
            method=args.method,
            recenter_each=args.recenter_each
        )
        
        # Run label alignment diagnostics
        alignment_diagnostics = analyzer.diagnose_label_alignment()
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        print("\nCreating cluster comparison plot...")
        fig1 = analyzer.create_cluster_comparison_plot(
            colormap=args.colormap, 
            include_direction_labels=args.label_guess
        )
        fig1.savefig(output_path / "cluster_patterns_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print("Creating difference analysis...")
        fig2 = analyzer.create_difference_analysis()
        fig2.savefig(output_path / "cluster_differences.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print("Creating enhanced difference analysis...")
        fig4 = analyzer.create_enhanced_difference_analysis()
        fig4.savefig(output_path / "enhanced_cluster_differences.png", dpi=300, bbox_inches='tight')
        plt.close(fig4)
        
        print("Creating feature-enhanced analysis...")
        fig5 = analyzer.create_feature_highlight_analysis()
        fig5.savefig(output_path / "feature_enhanced_differences.png", dpi=300, bbox_inches='tight')
        plt.close(fig5)
        
        print("Creating radial profile comparison...")
        fig3 = analyzer.create_radial_profile_comparison()
        fig3.savefig(output_path / "radial_profiles.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        print("Creating bias-corrected visualization (cluster - scan average)...")
        fig6 = analyzer.create_bias_corrected_visualization(colormap=args.colormap)
        fig6.savefig(output_path / "bias_corrected_patterns.png", dpi=300, bbox_inches='tight')
        plt.close(fig6)
        
        # Save analysis results
        analyzer.save_analysis_results(str(output_path))
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        stats = analyzer.compute_cluster_statistics()
        
        print(f"Number of clusters: {len(analyzer.unique_clusters)}")
        print(f"Cluster IDs: {analyzer.unique_clusters}")
        
        # Print direction labels if requested
        if args.label_guess:
            print("\nCluster Direction Labels:")
            direction_labels = analyzer.compute_cluster_directions()
            for cluster_id in sorted(direction_labels.keys()):
                print(f"  Cluster {cluster_id}: {direction_labels[cluster_id]}")
        
        # Print cluster statistics
        for cluster_id in sorted(analyzer.cluster_patterns.keys()):
            cluster_stats = stats[f'cluster_{cluster_id}']
            print(f"\nCluster {cluster_id}:")
            print(f"  Mean intensity: {cluster_stats['mean_intensity']:.4f}")
            print(f"  Intensity std: {cluster_stats['std_intensity']:.4f}")
            print(f"  Max intensity: {cluster_stats['max_intensity']:.4f}")
            print(f"  Total intensity: {cluster_stats['total_intensity']:.4f}")
        
        print(f"\nResults saved to: {output_path}")
        print("Generated files:")
        print("  - cluster_patterns_comparison.png: Side-by-side cluster patterns (with optional direction labels)")
        print("  - cluster_differences.png: Basic difference maps vs reference")
        print("  - enhanced_cluster_differences.png: Multi-level enhanced difference analysis")
        print("  - feature_enhanced_differences.png: Edge-enhanced and ratio analysis")
        print("  - radial_profiles.png: Radial profile analysis")
        print("  - cluster_patterns.npz: Computed cluster patterns")
        print("  - cluster_statistics.npz: Statistical analysis (NPZ format)")
        print("  - analysis_metadata.json: Portable metadata")
        print("  - reference_pattern.npy: Reference diffraction pattern")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())