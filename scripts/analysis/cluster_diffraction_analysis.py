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
        
    def _find_beam_center(self, pattern: np.ndarray) -> Tuple[int, int]:
        """
        Find the direct beam center in a diffraction pattern.
        
        Args:
            pattern: Diffraction pattern
            
        Returns:
            (center_y, center_x) coordinates
        """
        # Use center of mass approach for robustness
        # Apply Gaussian smoothing to reduce noise
        smooth_pattern = ndimage.gaussian_filter(pattern, sigma=2.0)
        
        # Find the maximum intensity region
        max_pos = np.unravel_index(np.argmax(smooth_pattern), smooth_pattern.shape)
        
        # Refine using center of mass in a local region
        region_size = min(pattern.shape) // 10
        y_min = max(0, max_pos[0] - region_size)
        y_max = min(pattern.shape[0], max_pos[0] + region_size)
        x_min = max(0, max_pos[1] - region_size)
        x_max = min(pattern.shape[1], max_pos[1] + region_size)
        
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
                                recenter=True, method='mean'):
        """
        Compute cluster-averaged diffraction patterns with consistent processing.
        
        Args:
            normalize: Whether to normalize by total counts
            subtract_background: Whether to subtract global background
            recenter: Whether to recenter patterns to reference
            method: Averaging method ('mean', 'median')
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
            
            # Compute average pattern
            if method == 'mean':
                avg_pattern = np.mean(cluster_data, axis=0).astype(np.float64)
            elif method == 'median':
                avg_pattern = np.median(cluster_data, axis=0).astype(np.float64)
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
        processed = pattern.copy()
        
        # 1. Normalize by total counts
        if normalize:
            total_counts = np.sum(processed)
            if total_counts > 0:
                processed = processed / total_counts
        
        # 2. Subtract global background (estimate from pattern edges)
        if subtract_background:
            # Use edge pixels as background estimate
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
            
            # Apply shift using scipy
            processed = ndimage.shift(processed, (dy, dx), order=1, cval=0)
        
        return processed
    
    def create_cluster_comparison_plot(self, figsize=(20, 12), use_log=True, 
                                     colormap='viridis') -> plt.Figure:
        """
        Create comprehensive comparison plot of all cluster patterns.
        
        Args:
            figsize: Figure size
            use_log: Whether to use log scale for intensity
            colormap: Colormap for diffraction patterns
            
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
        
        # Plot cluster patterns
        for cluster_id in sorted(self.cluster_patterns.keys()):
            if plot_idx >= rows * cols:
                break
                
            row, col = plot_idx // cols, plot_idx % cols
            
            pattern = self.cluster_patterns[cluster_id]
            if use_log:
                pattern = np.log(pattern + 1e-6)
            
            im = axes[row, col].imshow(pattern, cmap=colormap)
            axes[row, col].set_title(f'Cluster {cluster_id}', fontweight='bold')
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
    
    def create_difference_analysis(self, reference_cluster=None, figsize=(15, 10)) -> plt.Figure:
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
        
        # Save statistics
        stats = self.compute_cluster_statistics()
        np.save(output_path / "cluster_statistics.npy", stats)
        
        print("Analysis results saved successfully")


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
                       help='Skip recentering')
    parser.add_argument('--colormap', default='viridis',
                       help='Colormap for diffraction patterns (default: viridis)')
    
    args = parser.parse_args()
    
    try:
        # Load data
        data, scan_shape, metadata = load_h5_data(args.h5_file)
        
        # Load cluster labels
        print(f"Loading cluster labels from {args.cluster_labels}")
        cluster_labels = np.load(args.cluster_labels)
        
        # Validate cluster labels shape
        if cluster_labels.shape != scan_shape:
            print(f"Reshaping cluster labels from {cluster_labels.shape} to {scan_shape}")
            cluster_labels = cluster_labels.reshape(scan_shape)
        
        # Create analyzer
        analyzer = ClusterDiffractionAnalyzer(data, scan_shape, cluster_labels)
        
        # Compute cluster patterns
        print("\nComputing cluster-averaged diffraction patterns...")
        analyzer.compute_cluster_patterns(
            normalize=not args.no_normalize,
            subtract_background=not args.no_background,
            recenter=not args.no_recenter,
            method=args.method
        )
        
        # Create output directory
        output_path = Path(args.output)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create visualizations
        print("\nCreating cluster comparison plot...")
        fig1 = analyzer.create_cluster_comparison_plot(colormap=args.colormap)
        fig1.savefig(output_path / "cluster_patterns_comparison.png", dpi=300, bbox_inches='tight')
        plt.close(fig1)
        
        print("Creating difference analysis...")
        fig2 = analyzer.create_difference_analysis()
        fig2.savefig(output_path / "cluster_differences.png", dpi=300, bbox_inches='tight')
        plt.close(fig2)
        
        print("Creating radial profile comparison...")
        fig3 = analyzer.create_radial_profile_comparison()
        fig3.savefig(output_path / "radial_profiles.png", dpi=300, bbox_inches='tight')
        plt.close(fig3)
        
        # Save analysis results
        analyzer.save_analysis_results(str(output_path))
        
        # Print summary statistics
        print("\n" + "="*60)
        print("ANALYSIS SUMMARY")
        print("="*60)
        stats = analyzer.compute_cluster_statistics()
        
        print(f"Number of clusters: {len(analyzer.unique_clusters)}")
        print(f"Cluster IDs: {analyzer.unique_clusters}")
        
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
        print("  - cluster_patterns_comparison.png: Side-by-side cluster patterns")
        print("  - cluster_differences.png: Difference maps vs reference")
        print("  - radial_profiles.png: Radial profile analysis")
        print("  - cluster_patterns.npz: Computed cluster patterns")
        print("  - cluster_statistics.npy: Statistical analysis")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())