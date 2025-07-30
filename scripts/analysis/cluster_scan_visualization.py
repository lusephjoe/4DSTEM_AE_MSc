#!/usr/bin/env python3
"""
Cluster Scan Visualization for 4D-STEM Analysis

Overlays UMAP clustering results on STEM images to visualize domains and topologies.
Takes the output from umap_latent_visualization.py and shows clusters as colored overlays
on STEM virtual detector images.

Usage:
    python scripts/analysis/cluster_scan_visualization.py \
        --umap_results umap_analysis/umap_data.npz \
        --raw_data data/patterns.h5 \
        --output_dir cluster_visualization \
        --scan_shape 64 64 \
        --virtual bf
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Tuple, Optional, Dict
import sys

# Add path for reusing existing modules
sys.path.append(str(Path(__file__).parent.parent / "visualization"))
from scan_util import factorise_scan, coords_to_sparse_image, validate_coords, print_coord_summary, raster_coords
from stem_visualization import STEMVisualizer


class ClusterScanVisualizer:
    """Simple cluster visualization on STEM images."""
    
    def __init__(self, umap_results_path: Path, raw_data_path: Path):
        """Initialize with paths to UMAP results and raw data."""
        self.umap_results_path = umap_results_path
        self.raw_data_path = raw_data_path
        self.cluster_labels = None
        self.spatial_coords = None
        self.raw_data = None
        self.metadata = None
        
    def load_tensor(self, path: Path) -> np.ndarray:
        """Load .pt, .npy, .npz, or .h5 → numpy array (reused from visualise_scan_latents.py)"""
        if path.suffix in {".pt", ".pth"}:
            import torch
            return torch.load(path, map_location="cpu").numpy()
        elif path.suffix == ".npy":
            return np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)
            if 'embeddings' in data:
                return data['embeddings']
            elif 'data' in data:
                return data['data']
            else:
                keys = list(data.keys())
                if keys:
                    print(f"Warning: Using first array '{keys[0]}' from .npz file")
                    return data[keys[0]]
                else:
                    raise ValueError(f"No arrays found in .npz file: {path}")
        elif path.suffix in {".h5", ".hdf5"}:
            import h5py
            with h5py.File(path, 'r') as f:
                if 'data' in f:
                    data = f['data'][:]
                elif 'patterns' in f:
                    data = f['patterns'][:]
                elif 'array' in f:
                    data = f['array'][:]
                else:
                    keys = list(f.keys())
                    if keys:
                        dataset_name = keys[0]
                        print(f"Warning: Using first dataset '{dataset_name}' from .h5 file")
                        data = f[dataset_name][:]
                    else:
                        raise ValueError(f"No datasets found in .h5 file: {path}")
            
            data = data.astype(np.float32)
            if len(data.shape) == 3:
                data = data[:, np.newaxis, :, :]
            return data
        else:
            raise ValueError(f"Unknown file type {path.suffix}")
    
    def load_umap_results(self):
        """Load UMAP clustering results."""
        print(f"Loading UMAP results from: {self.umap_results_path}")
        
        if self.umap_results_path.suffix == '.npz':
            data = np.load(self.umap_results_path, allow_pickle=True)
            self.cluster_labels = data['cluster_labels']
            self.spatial_coords = data.get('spatial_coordinates', None)
            
        elif self.umap_results_path.suffix == '.json':
            with open(self.umap_results_path, 'r') as f:
                data = json.load(f)
            self.cluster_labels = np.array(data['cluster_labels'])
            self.spatial_coords = np.array(data['spatial_coordinates']) if data['spatial_coordinates'] else None
            self.metadata = data.get('metadata', {})
        
        else:
            raise ValueError(f"Unsupported UMAP results format: {self.umap_results_path.suffix}")
        
        if self.cluster_labels is None:
            raise ValueError("No cluster labels found in UMAP results")
        
        # Get cluster statistics
        unique_labels = np.unique(self.cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        n_noise = np.sum(self.cluster_labels == -1)
        
        print(f"✓ Loaded clustering results:")
        print(f"  Patterns: {len(self.cluster_labels)}")
        print(f"  Clusters: {n_clusters}")
        print(f"  Noise points: {n_noise}")
        
        if self.spatial_coords is not None:
            print(f"  Spatial coordinates: {self.spatial_coords.shape}")
        
        return self.cluster_labels, self.spatial_coords
    
    def load_raw_data(self):
        """Load raw 4D-STEM data."""
        print(f"Loading raw data from: {self.raw_data_path}")
        self.raw_data = self.load_tensor(self.raw_data_path)
        print(f"✓ Loaded raw data: {self.raw_data.shape}")
        return self.raw_data
    
    def setup_coordinates(self, scan_shape):
        """Setup coordinate system for visualization with manual scan shape."""
        N = len(self.cluster_labels)
        Ny, Nx = scan_shape
        
        print(f"Setting up coordinates for {N} patterns with manual scan shape {Ny}x{Nx}")
        
        # Validate that scan shape matches pattern count
        expected_patterns = Ny * Nx
        if N != expected_patterns:
            print(f"WARNING: Pattern count ({N}) != scan dimensions ({Ny}x{Nx} = {expected_patterns})")
            print("This may indicate missing patterns or incorrect scan shape")
        
        # Always generate raster coordinates for the manual scan shape
        # This ensures coordinates match the raw data, not the UMAP-generated coordinates
        print(f"Generating raster coordinates for {Ny}x{Nx} scan...")
        coords = raster_coords(Ny, Nx)
        
        # Truncate coordinates if we have fewer patterns than expected
        if len(coords) > N:
            print(f"Truncating coordinates from {len(coords)} to {N} to match pattern count")
            coords = coords[:N]
        
        # Validate final coordinates
        if len(coords) != N:
            print(f"ERROR: Generated {len(coords)} coordinates but need {N}")
            return None
        
        print(f"Final scan configuration:")
        print(f"  Scan dimensions: {Ny} x {Nx} (manual)")
        print(f"  Generated coordinates: {len(coords)}")
        print(f"  Coordinate range: Y=[{coords[:, 0].min()}, {coords[:, 0].max()}], X=[{coords[:, 1].min()}, {coords[:, 1].max()}]")
        
        return coords
    
    def create_cluster_overlay(self, coords, scan_shape, virtual_type='bf', output_dir=Path("cluster_analysis")):
        """Create the main cluster overlay visualization."""
        Ny, Nx = scan_shape
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create background images
        print("Creating STEM background images...")
        
        # Virtual detector images using STEMVisualizer
        stem_viz = STEMVisualizer(self.raw_data[:, 0], scan_shape=(Ny, Nx))
        print(f"Direct beam detected at: {stem_viz.direct_beam_position} (y, x)")
        
        # Create bright field image
        pattern_size = min(self.raw_data.shape[-2:])
        bf_radius_pixels = int(0.1 * pattern_size // 2)  # Default BF radius
        bf_image = stem_viz.create_bright_field_image(radius=bf_radius_pixels)
        
        # Create dark field image
        center_y, center_x = stem_viz.direct_beam_position
        # Create annular dark field region (exclude central bright field)
        inner_radius = int(0.2 * pattern_size // 2)
        outer_radius = int(0.8 * pattern_size // 2)
        
        # Create mask for annular region
        df_mask = np.zeros(self.raw_data.shape[-2:], dtype=bool)
        y_indices, x_indices = np.ogrid[:self.raw_data.shape[-2], :self.raw_data.shape[-1]]
        distances = np.sqrt((y_indices - center_y)**2 + (x_indices - center_x)**2)
        df_mask = (distances >= inner_radius) & (distances <= outer_radius)
        
        # Apply mask and sum over the annular region
        df_values = np.sum(self.raw_data[:, 0] * df_mask, axis=(1, 2))
        df_image = coords_to_sparse_image(coords, df_values, (Ny, Nx))
        
        # Use the requested virtual type for the right panel
        if virtual_type == "bf":
            virt_image = bf_image
            virt_title = "Bright Field"
        else:
            virt_image = df_image
            virt_title = "Dark Field"
        
        # Create cluster overlay
        print("Creating cluster overlay visualization...")
        
        # Get unique clusters (excluding noise)
        unique_labels = np.unique(self.cluster_labels)
        valid_clusters = [label for label in unique_labels if label != -1]
        n_clusters = len(valid_clusters)
        
        # Setup colors for clusters
        if n_clusters <= 10:
            colors = plt.cm.tab10(np.linspace(0, 1, 10))[:n_clusters]
        elif n_clusters <= 20:
            colors = plt.cm.tab20(np.linspace(0, 1, 20))[:n_clusters]
        else:
            colors = plt.cm.hsv(np.linspace(0, 1, n_clusters))
        
        # Create figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 12))
        fig.suptitle('Cluster Overlay Analysis', fontsize=16, fontweight='bold')
        
        print(f"Data shapes - BF: {bf_image.shape}, DF: {df_image.shape}")
        print(f"Coordinate range: Y=[{coords[:, 0].min()}, {coords[:, 0].max()}], X=[{coords[:, 1].min()}, {coords[:, 1].max()}]")
        
        # Panel 1: Virtual Dark Field
        ax = axes[0, 0]
        im1 = ax.imshow(df_image, cmap="gray", origin='upper')
        ax.set_title("Virtual Dark Field")
        ax.axis("off")
        
        # Panel 2: Virtual Bright Field
        ax = axes[0, 1]
        im2 = ax.imshow(bf_image, cmap="gray", origin='upper')
        ax.set_title(f"Virtual Bright Field")
        ax.axis("off")
        
        # Panel 3: Clusters overlaid on virtual dark field
        ax = axes[1, 0]
        
        # Create cluster color overlay image
        cluster_overlay = np.zeros((Ny, Nx, 4))  # RGBA
        
        # Fill cluster overlay with colors
        for i, cluster_id in enumerate(valid_clusters):
            mask = self.cluster_labels == cluster_id
            cluster_coords = coords[mask]
            for coord in cluster_coords:
                y, x = coord[0], coord[1]
                if 0 <= y < Ny and 0 <= x < Nx:
                    cluster_overlay[y, x] = [*colors[i][:3], 0.3]  # RGB + alpha (more transparent)
        
        # Add noise points
        if -1 in unique_labels:
            noise_mask = self.cluster_labels == -1
            noise_coords = coords[noise_mask]
            for coord in noise_coords:
                y, x = coord[0], coord[1]
                if 0 <= y < Ny and 0 <= x < Nx:
                    cluster_overlay[y, x] = [0.5, 0.5, 0.5, 0.15]  # Gray with very low alpha
        
        # Display STEM image first
        ax.imshow(df_image, cmap="gray", origin='upper')
        # Then overlay the clusters
        ax.imshow(cluster_overlay, origin='upper')
        
        ax.set_title(f'Clusters on Virtual Dark Field ({n_clusters} clusters)')
        ax.axis("off")
        
        # Create manual legend for the overlay
        legend_elements = []
        for i, cluster_id in enumerate(valid_clusters[:12]):  # Limit to first 12 for space
            legend_elements.append(plt.Rectangle((0, 0), 1, 1, facecolor=colors[i], alpha=0.3, label=f'C{cluster_id}'))
        
        if n_clusters <= 12:
            ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        else:
            ax.text(0.02, 0.98, f'{n_clusters} clusters\n(showing top 12)', 
                   transform=ax.transAxes, ha='left', va='top',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9))
        
        # Panel 4: Clusters overlaid on virtual bright field
        ax = axes[1, 1]
        
        # Display STEM image first
        ax.imshow(bf_image, cmap="gray", origin='upper')
        # Then overlay the same cluster colors
        ax.imshow(cluster_overlay, origin='upper')
        
        ax.set_title(f'Clusters on Virtual Bright Field')
        ax.axis("off")
        
        plt.tight_layout()
        
        # Save the main visualization
        overview_path = output_dir / 'cluster_overview.png'
        plt.savefig(overview_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved cluster overview: {overview_path}")
        
        plt.close()
        
        return valid_clusters, colors
    
    def create_domain_analysis(self, coords, scan_shape, valid_clusters, colors, output_dir):
        """Create domain size and distribution analysis."""
        print("Creating domain analysis...")
        
        # Calculate cluster statistics
        cluster_stats = []
        for cluster_id in valid_clusters:
            mask = self.cluster_labels == cluster_id
            size = np.sum(mask)
            cluster_stats.append({'id': cluster_id, 'size': size, 'fraction': size / len(self.cluster_labels)})
        
        # Sort by size
        cluster_stats.sort(key=lambda x: x['size'], reverse=True)
        
        # Create analysis figure
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Cluster Domain Analysis', fontsize=16, fontweight='bold')
        
        # Panel 1: Cluster size distribution
        ax = axes[0, 0]
        cluster_ids = [stat['id'] for stat in cluster_stats]
        cluster_sizes = [stat['size'] for stat in cluster_stats]
        
        bars = ax.bar(range(len(cluster_sizes)), cluster_sizes, 
                     color=[colors[valid_clusters.index(cid)] for cid in cluster_ids], alpha=0.7)
        ax.set_title('Cluster Sizes')
        ax.set_ylabel('Number of Points')
        ax.set_xlabel('Clusters (ordered by size)')
        
        # Label only the top 10% largest clusters
        n_top = max(1, len(cluster_sizes) // 10)
        for i in range(n_top):
            bar = bars[i]
            cluster_id = cluster_ids[i]
            ax.text(bar.get_x() + bar.get_width()/2, 
                   bar.get_height() + max(cluster_sizes)*0.01,
                   f'C{cluster_id}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xticks([])
        
        # Panel 2: Size distribution histogram
        ax = axes[0, 1]
        ax.hist(cluster_sizes, bins=min(20, len(cluster_sizes)), alpha=0.7, color='steelblue')
        ax.set_title('Cluster Size Distribution')
        ax.set_xlabel('Cluster Size (points)')
        ax.set_ylabel('Frequency')
        ax.axvline(np.mean(cluster_sizes), color='red', linestyle='--', label=f'Mean: {np.mean(cluster_sizes):.0f}')
        ax.axvline(np.median(cluster_sizes), color='orange', linestyle='--', label=f'Median: {np.median(cluster_sizes):.0f}')
        ax.legend()
        
        # Panel 3: Top clusters (largest 8)
        ax = axes[1, 0]
        Ny, Nx = scan_shape
        
        # Create empty image
        cluster_map = np.full((Ny, Nx), -1, dtype=int)
        
        # Fill with cluster labels
        for i, coord in enumerate(coords):
            cluster_map[coord[0], coord[1]] = self.cluster_labels[i]
        
        # Show only top 8 clusters
        display_map = np.full((Ny, Nx), -1, dtype=int)
        top_8_clusters = cluster_ids[:8]
        
        for cluster_id in top_8_clusters:
            display_map[cluster_map == cluster_id] = cluster_id
        
        # Create custom colormap - fix bounds issue
        if len(top_8_clusters) > 0:
            # Create colormap with proper bounds
            cluster_colors = [colors[valid_clusters.index(cid)] for cid in top_8_clusters]
            cmap = plt.cm.colors.ListedColormap(['lightgray'] + cluster_colors)
            
            # Create monotonically increasing bounds
            sorted_clusters = sorted(top_8_clusters)
            bounds = [-1.5] + [cid - 0.5 for cid in sorted_clusters] + [max(sorted_clusters) + 0.5]
            norm = plt.cm.colors.BoundaryNorm(bounds, cmap.N)
            
            im = ax.imshow(display_map, cmap=cmap, norm=norm)
        else:
            # Fallback if no clusters
            im = ax.imshow(display_map, cmap='gray')
        ax.set_title(f'Top {len(top_8_clusters)} Largest Clusters')
        ax.axis('off')
        
        # Panel 4: Statistics table
        ax = axes[1, 1]
        ax.axis('off')
        
        # Create statistics text
        stats_text = f"Clustering Statistics:\n\n"
        stats_text += f"Total patterns: {len(self.cluster_labels)}\n"
        stats_text += f"Number of clusters: {len(valid_clusters)}\n"
        stats_text += f"Noise points: {np.sum(self.cluster_labels == -1)}\n\n"
        
        stats_text += "Top 5 Largest Clusters:\n"
        for i, stat in enumerate(cluster_stats[:5]):
            stats_text += f"  C{stat['id']}: {stat['size']} pts ({stat['fraction']:.1%})\n"
        
        stats_text += f"\nSize Statistics:\n"
        stats_text += f"  Mean size: {np.mean(cluster_sizes):.1f}\n"
        stats_text += f"  Median size: {np.median(cluster_sizes):.1f}\n"
        stats_text += f"  Std dev: {np.std(cluster_sizes):.1f}\n"
        stats_text += f"  Min size: {np.min(cluster_sizes)}\n"
        stats_text += f"  Max size: {np.max(cluster_sizes)}\n"
        
        ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontfamily='monospace', fontsize=10)
        
        plt.tight_layout()
        
        # Save domain analysis
        domain_path = output_dir / 'domain_analysis.png'
        plt.savefig(domain_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved domain analysis: {domain_path}")
        
        plt.close()
        
        # Save statistics as JSON (fix int64 serialization)
        stats_path = output_dir / 'cluster_statistics.json'
        
        # Convert cluster_stats to JSON-serializable format
        json_cluster_stats = []
        for stat in cluster_stats:
            json_cluster_stats.append({
                'id': int(stat['id']),  # Convert numpy int to Python int
                'size': int(stat['size']),
                'fraction': float(stat['fraction'])
            })
        
        stats_data = {
            'total_patterns': int(len(self.cluster_labels)),
            'n_clusters': int(len(valid_clusters)),
            'n_noise': int(np.sum(self.cluster_labels == -1)),
            'cluster_details': json_cluster_stats,
            'size_statistics': {
                'mean': float(np.mean(cluster_sizes)),
                'median': float(np.median(cluster_sizes)),
                'std': float(np.std(cluster_sizes)),
                'min': int(np.min(cluster_sizes)),
                'max': int(np.max(cluster_sizes))
            }
        }
        
        with open(stats_path, 'w') as f:
            json.dump(stats_data, f, indent=2)
        print(f"✓ Saved statistics: {stats_path}")


def main():
    parser = argparse.ArgumentParser(description="Cluster Scan Visualization for 4D-STEM Analysis")
    
    # Required arguments
    parser.add_argument("--umap_results", type=Path, required=True,
                       help="Path to UMAP results (.npz or .json)")
    parser.add_argument("--raw_data", type=Path, required=True,
                       help="Path to raw 4D-STEM data (.h5, .pt, .npz)")
    parser.add_argument("--output_dir", type=Path, default="cluster_visualization",
                       help="Output directory for visualizations")
    
    # Optional parameters
    parser.add_argument("--virtual", choices=["bf", "df"], default="bf",
                       help="Virtual detector type for background")
    parser.add_argument("--scan_shape", type=int, nargs=2, required=True,
                       metavar=("NY", "NX"), help="Scan dimensions (height width)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.umap_results.exists():
        raise FileNotFoundError(f"UMAP results not found: {args.umap_results}")
    if not args.raw_data.exists():
        raise FileNotFoundError(f"Raw data not found: {args.raw_data}")
    
    print("="*80)
    print("CLUSTER SCAN VISUALIZATION")
    print("="*80)
    print(f"UMAP results: {args.umap_results}")
    print(f"Raw data: {args.raw_data}")
    print(f"Output: {args.output_dir}")
    print(f"Virtual detector: {args.virtual}")
    print(f"Scan shape: {args.scan_shape[0]} x {args.scan_shape[1]}")
    print("="*80)
    
    # Initialize visualizer
    visualizer = ClusterScanVisualizer(args.umap_results, args.raw_data)
    
    # Load data
    cluster_labels, spatial_coords = visualizer.load_umap_results()
    raw_data = visualizer.load_raw_data()
    
    # Setup coordinates with manual scan shape
    coords = visualizer.setup_coordinates(args.scan_shape)
    if coords is None:
        print("ERROR: Could not setup coordinates")
        return
    Ny, Nx = args.scan_shape
    
    # Create visualizations
    valid_clusters, colors = visualizer.create_cluster_overlay(
        coords, (Ny, Nx), args.virtual, args.output_dir)
    
    visualizer.create_domain_analysis(
        coords, (Ny, Nx), valid_clusters, colors, args.output_dir)
    
    print("\n" + "="*80)
    print("CLUSTER VISUALIZATION COMPLETED")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Generated files:")
    print(f"  - cluster_overview.png: Main cluster overlay visualization")
    print(f"  - domain_analysis.png: Cluster size and distribution analysis")
    print(f"  - cluster_statistics.json: Quantitative cluster statistics")
    print("="*80)


if __name__ == "__main__":
    main()