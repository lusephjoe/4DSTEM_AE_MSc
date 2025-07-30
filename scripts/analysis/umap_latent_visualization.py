#!/usr/bin/env python3
"""
UMAP Latent Space Visualization for 4D-STEM Autoencoder Analysis

This script performs comprehensive latent space analysis using UMAP (Uniform Manifold 
Approximation and Projection) to visualize how diffraction patterns are organized in 
the autoencoder's latent space. UMAP preserves local neighborhood relationships while 
projecting high-dimensional latent vectors into 2D/3D space, revealing clustering 
structure that may correspond to different domains or orientations.

IMPORTANT: This script now works with pre-generated embeddings. First generate 
embeddings using scripts/visualization/generate_embeddings.py, then use this script 
for UMAP analysis.

Usage:
    # Step 1: Generate embeddings
    python scripts/visualization/generate_embeddings.py \
        --checkpoint results/ae_model.ckpt \
        --data data/patterns.h5 \
        --output embeddings/patterns_embeddings.npz \
        --save_spatial_coords

    # Step 2: UMAP analysis
    python scripts/analysis/umap_latent_visualization.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/umap_analysis \
        --n_neighbors 30 \
        --min_dist 0.1
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
import json
from typing import Tuple, Optional, Dict, List
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Scientific computing
import umap
from sklearn.cluster import HDBSCAN, KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter

class LatentSpaceAnalyzer:
    """Comprehensive latent space analysis using UMAP and clustering on pre-generated embeddings."""
    
    def __init__(self, embeddings_path: Path):
        """Initialize analyzer with pre-generated embeddings."""
        self.embeddings_path = embeddings_path
        self.embeddings = None
        self.spatial_coords = None
        self.metadata = None
        
    def load_embeddings(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load pre-generated embeddings from file."""
        print(f"Loading embeddings from: {self.embeddings_path}")
        
        if self.embeddings_path.suffix == '.npz':
            # NumPy format
            data = np.load(self.embeddings_path, allow_pickle=True)
            self.embeddings = data['embeddings']
            self.spatial_coords = data.get('spatial_coordinates', None)
            
            # Load metadata if available
            if 'metadata' in data:
                try:
                    metadata_str = str(data['metadata'].item())
                    self.metadata = json.loads(metadata_str)
                except:
                    self.metadata = None
            else:
                self.metadata = None
                
        elif self.embeddings_path.suffix == '.h5':
            # HDF5 format
            with h5py.File(self.embeddings_path, 'r') as f:
                self.embeddings = f['embeddings'][:]
                self.spatial_coords = f.get('spatial_coordinates', None)
                if self.spatial_coords is not None:
                    self.spatial_coords = self.spatial_coords[:]
                
                # Load metadata from attributes
                self.metadata = {}
                for key, value in f.attrs.items():
                    try:
                        # Try to parse as JSON first
                        self.metadata[key] = json.loads(value)
                    except:
                        # Fall back to direct value
                        self.metadata[key] = value
                        
        elif self.embeddings_path.suffix == '.pt':
            # PyTorch format (legacy)
            import torch
            self.embeddings = torch.load(self.embeddings_path, map_location='cpu').numpy()
            
            # Try to load spatial coordinates
            coord_path = self.embeddings_path.with_name(self.embeddings_path.stem + '_coords.npy')
            if coord_path.exists():
                self.spatial_coords = np.load(coord_path)
            
            # Try to load metadata
            meta_path = self.embeddings_path.with_name(self.embeddings_path.stem + '_metadata.json') 
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    self.metadata = json.load(f)
        else:
            raise ValueError(f"Unsupported embedding format: {self.embeddings_path.suffix}")
        
        print(f"✓ Loaded embeddings: {self.embeddings.shape}")
        print(f"  Latent dimension: {self.embeddings.shape[1]}")
        print(f"  Embedding range: [{self.embeddings.min():.3f}, {self.embeddings.max():.3f}]")
        
        if self.spatial_coords is not None:
            print(f"  Spatial coordinates: {self.spatial_coords.shape}")
        
        if self.metadata is not None:
            print(f"  Metadata keys: {list(self.metadata.keys())}")
        
        return self.embeddings, self.spatial_coords, self.metadata
    
    def compute_umap_embedding(self, n_neighbors: int = 15, min_dist: float = 0.1, 
                             n_components: int = 2, metric: str = 'euclidean',
                             random_state: int = 42) -> np.ndarray:
        """Compute UMAP embedding of latent space."""
        print(f"Computing UMAP embedding...")
        print(f"  n_neighbors: {n_neighbors}")
        print(f"  min_dist: {min_dist}")
        print(f"  n_components: {n_components}")
        print(f"  metric: {metric}")
        
        # Initialize UMAP
        reducer = umap.UMAP(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=n_components,
            metric=metric,
            random_state=random_state,
            verbose=True
        )
        
        # Fit and transform
        umap_embedding = reducer.fit_transform(self.embeddings)
        
        print(f"✓ UMAP embedding computed: {umap_embedding.shape}")
        print(f"  UMAP range: X=[{umap_embedding[:, 0].min():.2f}, {umap_embedding[:, 0].max():.2f}]")
        print(f"               Y=[{umap_embedding[:, 1].min():.2f}, {umap_embedding[:, 1].max():.2f}]")
        
        return umap_embedding, reducer
    
    def perform_clustering(self, umap_embedding: np.ndarray, 
                          method: str = 'hdbscan', **kwargs) -> np.ndarray:
        """Perform clustering on UMAP embedding."""
        print(f"Performing clustering using {method}...")
        
        if method.lower() == 'hdbscan':
            min_cluster_size = kwargs.get('min_cluster_size', 50)
            min_samples = kwargs.get('min_samples', 5)
            
            clusterer = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                cluster_selection_epsilon=kwargs.get('cluster_selection_epsilon', 0.0)
            )
            cluster_labels = clusterer.fit_predict(umap_embedding)
            
        elif method.lower() == 'kmeans':
            n_clusters = kwargs.get('n_clusters', 5)
            clusterer = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = clusterer.fit_predict(umap_embedding)
            
        else:
            raise ValueError(f"Unsupported clustering method: {method}")
        
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        print(f"✓ Clustering completed:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Silhouette score: {silhouette_score(umap_embedding, cluster_labels):.3f}")
        
        return cluster_labels
    
    def create_visualizations(self, umap_embedding: np.ndarray, 
                            cluster_labels: Optional[np.ndarray] = None,
                            output_dir: Path = Path("umap_analysis")) -> None:
        """Create comprehensive UMAP visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # 1. Basic UMAP scatter plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('UMAP Latent Space Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Basic UMAP embedding
        ax = axes[0, 0]
        scatter = ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                           c='steelblue', s=1, alpha=0.6)
        ax.set_title('UMAP Embedding of Latent Space')
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Density plot
        ax = axes[0, 1]
        try:
            # Create density plot using hexbin
            hb = ax.hexbin(umap_embedding[:, 0], umap_embedding[:, 1], 
                          gridsize=50, cmap='Blues', mincnt=1)
            ax.set_title('UMAP Density Distribution')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            plt.colorbar(hb, ax=ax, label='Point Density')
        except:
            # Fallback to regular scatter if hexbin fails
            ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                      c='steelblue', s=1, alpha=0.6)
            ax.set_title('UMAP Embedding (Density Fallback)')
        
        # Plot 3: Clustering results
        ax = axes[1, 0]
        if cluster_labels is not None:
            unique_labels = np.unique(cluster_labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            valid_cluster_labels = [label for label in unique_labels if label != -1]
            n_valid_clusters = len(valid_cluster_labels)
            
            for label, color in zip(unique_labels, colors):
                mask = cluster_labels == label
                if label == -1:
                    # Noise points in gray
                    ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1], 
                             c='gray', s=1, alpha=0.3, label='Noise')
                else:
                    ax.scatter(umap_embedding[mask, 0], umap_embedding[mask, 1], 
                             c=[color], s=2, alpha=0.7, label=f'C{label}')
            
            ax.set_title('UMAP with Clustering')
            
            # Smart legend handling based on number of clusters
            if n_valid_clusters <= 8:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, 
                         markerscale=0.8, handletextpad=0.1)
            elif n_valid_clusters <= 15:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, 
                         markerscale=0.6, handletextpad=0.1, ncol=2, columnspacing=0.5)
            else:
                # Too many clusters - show summary text instead
                ax.text(0.98, 0.98, f'{n_valid_clusters} clusters\n(legend omitted)', 
                       transform=ax.transAxes, ha='right', va='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=8)
        else:
            ax.scatter(umap_embedding[:, 0], umap_embedding[:, 1], 
                      c='steelblue', s=1, alpha=0.6)
            ax.set_title('UMAP Embedding (No Clustering)')
        
        ax.set_xlabel('UMAP 1')
        ax.set_ylabel('UMAP 2')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Spatial mapping (if spatial coordinates available)
        ax = axes[1, 1]
        if self.spatial_coords is not None and cluster_labels is not None:
            # Create spatial map colored by cluster
            scatter = ax.scatter(self.spatial_coords[:, 0], self.spatial_coords[:, 1], 
                               c=cluster_labels, cmap='Set1', s=4, alpha=0.8)
            ax.set_title('Spatial Distribution of Clusters')
            ax.set_xlabel('Scan X Position')
            ax.set_ylabel('Scan Y Position')
            plt.colorbar(scatter, ax=ax, label='Cluster ID')
        else:
            # Fallback: show spatial coordinates colored by UMAP position
            if self.spatial_coords is not None:
                scatter = ax.scatter(self.spatial_coords[:, 0], self.spatial_coords[:, 1], 
                                   c=umap_embedding[:, 0], cmap='viridis', s=4, alpha=0.8)
                ax.set_title('Spatial Coordinates (UMAP 1 color)')
                ax.set_xlabel('Scan X Position')
                ax.set_ylabel('Scan Y Position')
                plt.colorbar(scatter, ax=ax, label='UMAP 1')
            else:
                ax.text(0.5, 0.5, 'No spatial coordinates\navailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Spatial Information N/A')
        
        plt.tight_layout()
        plt.savefig(output_dir / 'umap_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / 'umap_overview.pdf', bbox_inches='tight')
        plt.close()
        
        # 2. Detailed cluster analysis plots
        if cluster_labels is not None:
            self._create_cluster_analysis_plots(umap_embedding, cluster_labels, output_dir)
        
        # 3. Parameter sensitivity analysis
        self._create_parameter_analysis(output_dir)
        
        print(f"✓ Visualizations saved to: {output_dir}")
    
    def _create_cluster_analysis_plots(self, umap_embedding: np.ndarray, 
                                     cluster_labels: np.ndarray, output_dir: Path) -> None:
        """Create detailed cluster analysis visualizations."""
        
        # Cluster statistics
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters > 1:
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle('Detailed Cluster Analysis', fontsize=16, fontweight='bold')
            
            # Plot cluster sizes
            ax = axes[0, 0]
            valid_labels = [label for label in unique_labels if label != -1]
            cluster_sizes = [np.sum(cluster_labels == label) for label in valid_labels]
            cluster_names = [f'Cluster {label}' for label in valid_labels]
            
            bars = ax.bar(cluster_names, cluster_sizes, alpha=0.7)
            ax.set_title('Cluster Sizes')
            ax.set_ylabel('Number of Points')
            ax.tick_params(axis='x', rotation=45)
            
            # Only label the top 10% of clusters by size to avoid overlap
            if len(cluster_sizes) > 0:
                # Calculate threshold for top 10% (minimum 1 cluster)
                n_top_clusters = max(1, int(np.ceil(len(cluster_sizes) * 0.1)))
                # Get indices of top clusters by size
                top_indices = np.argsort(cluster_sizes)[-n_top_clusters:]
                
                for i, (bar, size, label) in enumerate(zip(bars, cluster_sizes, valid_labels)):
                    if i in top_indices:
                        # Only label the largest clusters
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + max(cluster_sizes)*0.01,
                               f'C{label}', ha='center', va='bottom', fontweight='bold')
            
            # Plot cluster separation (silhouette analysis)
            ax = axes[0, 1]
            if len(unique_labels) > 1:
                from sklearn.metrics import silhouette_samples
                silhouette_avg = silhouette_score(umap_embedding, cluster_labels)
                sample_silhouette_values = silhouette_samples(umap_embedding, cluster_labels)
                
                y_lower = 10
                for i, label in enumerate(unique_labels):
                    if label == -1:
                        continue
                    
                    cluster_silhouette_values = sample_silhouette_values[cluster_labels == label]
                    cluster_silhouette_values.sort()
                    
                    size_cluster_i = cluster_silhouette_values.shape[0]
                    y_upper = y_lower + size_cluster_i
                    
                    color = plt.cm.Set1(i / len(unique_labels))
                    ax.fill_betweenx(np.arange(y_lower, y_upper),
                                    0, cluster_silhouette_values,
                                    facecolor=color, edgecolor=color, alpha=0.7)
                    
                    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(label))
                    y_lower = y_upper + 10
                
                ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                          label=f'Average Score: {silhouette_avg:.3f}')
                ax.set_title('Silhouette Analysis')
                ax.set_xlabel('Silhouette Coefficient Values')
                ax.set_ylabel('Cluster Label')
                ax.legend()
            
            # Plot cluster centers and boundaries
            ax = axes[1, 0]
            valid_cluster_labels = [label for label in unique_labels if label != -1]
            n_valid_clusters = len(valid_cluster_labels)
            
            for label in valid_cluster_labels:
                mask = cluster_labels == label
                cluster_points = umap_embedding[mask]
                
                # Plot cluster points
                ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                          alpha=0.6, s=2, label=f'C{label}')
                
                # Plot cluster center with smaller crosses
                center = np.mean(cluster_points, axis=0)
                ax.scatter(center[0], center[1], marker='x', s=30, 
                          color='black', linewidth=2)
            
            ax.set_title('Cluster Centers and Boundaries')
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            
            # Improve legend handling based on number of clusters
            if n_valid_clusters <= 10:
                # For few clusters, use normal legend
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, 
                         markerscale=0.8, handletextpad=0.1)
            elif n_valid_clusters <= 20:
                # For medium number of clusters, use smaller legend with two columns
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=6, 
                         markerscale=0.6, handletextpad=0.1, ncol=2, columnspacing=0.5)
            else:
                # For many clusters, no legend (too cluttered)
                ax.text(0.98, 0.98, f'{n_valid_clusters} clusters\n(legend omitted)', 
                       transform=ax.transAxes, ha='right', va='top', 
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                       fontsize=8)
            
            ax.grid(True, alpha=0.3)
            
            # Plot distance matrix between clusters
            ax = axes[1, 1]
            if n_clusters > 1:
                # Compute cluster centers
                centers = []
                valid_labels = []
                for label in unique_labels:
                    if label != -1:
                        mask = cluster_labels == label
                        center = np.mean(umap_embedding[mask], axis=0)
                        centers.append(center)
                        valid_labels.append(label)
                
                centers = np.array(centers)
                distances = squareform(pdist(centers))
                
                im = ax.imshow(distances, cmap='viridis')
                ax.set_title('Inter-cluster Distances')
                ax.set_xticks(range(len(valid_labels)))
                ax.set_yticks(range(len(valid_labels)))
                ax.set_xticklabels([f'C{label}' for label in valid_labels])
                ax.set_yticklabels([f'C{label}' for label in valid_labels])
                plt.colorbar(im, ax=ax, label='Distance')
                
                # Add distance values to cells
                for i in range(len(valid_labels)):
                    for j in range(len(valid_labels)):
                        ax.text(j, i, f'{distances[i, j]:.2f}', 
                               ha='center', va='center', color='white' if distances[i, j] > distances.max()/2 else 'black')
            
            plt.tight_layout()
            plt.savefig(output_dir / 'cluster_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_parameter_analysis(self, output_dir: Path) -> None:
        """Create parameter sensitivity analysis (if embeddings available)."""
        if self.embeddings is None:
            return
        
        print("Creating parameter sensitivity analysis...")
        
        # Test different UMAP parameters
        n_neighbors_list = [5, 15, 30, 50]
        min_dist_list = [0.01, 0.1, 0.5, 1.0]
        
        fig, axes = plt.subplots(len(n_neighbors_list), len(min_dist_list), 
                               figsize=(16, 12))
        fig.suptitle('UMAP Parameter Sensitivity Analysis', fontsize=16, fontweight='bold')
        
        for i, n_neighbors in enumerate(n_neighbors_list):
            for j, min_dist in enumerate(min_dist_list):
                ax = axes[i, j]
                
                # Compute UMAP with current parameters
                reducer = umap.UMAP(
                    n_neighbors=n_neighbors,
                    min_dist=min_dist,
                    n_components=2,
                    random_state=42,
                    verbose=False
                )
                
                # Sample data if too large (for speed)
                if len(self.embeddings) > 5000:
                    indices = np.random.choice(len(self.embeddings), 5000, replace=False)
                    sample_embeddings = self.embeddings[indices]
                else:
                    sample_embeddings = self.embeddings
                
                embedding = reducer.fit_transform(sample_embeddings)
                
                ax.scatter(embedding[:, 0], embedding[:, 1], s=1, alpha=0.6)
                ax.set_title(f'n_neighbors={n_neighbors}\nmin_dist={min_dist}', fontsize=10)
                ax.set_xticks([])
                ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig(output_dir / 'parameter_sensitivity.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def save_results(self, umap_embedding: np.ndarray, cluster_labels: Optional[np.ndarray],
                    output_dir: Path, metadata: Dict) -> None:
        """Save analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings and results
        results = {
            'original_embeddings': self.embeddings.tolist(),
            'umap_embedding': umap_embedding.tolist(),
            'spatial_coordinates': self.spatial_coords.tolist() if self.spatial_coords is not None else None,
            'cluster_labels': cluster_labels.tolist() if cluster_labels is not None else None,
            'metadata': metadata
        }
        
        with open(output_dir / 'umap_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as numpy arrays for easy loading
        np.savez(output_dir / 'umap_data.npz',
                original_embeddings=self.embeddings,
                umap_embedding=umap_embedding,
                spatial_coordinates=self.spatial_coords,
                cluster_labels=cluster_labels)
        
        print(f"✓ Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="UMAP Latent Space Visualization for 4D-STEM Autoencoder")
    
    # Required arguments - now uses pre-generated embeddings
    parser.add_argument("--embeddings", type=Path, required=True,
                       help="Path to pre-generated embeddings (.npz, .h5, .pt)")
    parser.add_argument("--output_dir", type=Path, default="umap_analysis",
                       help="Output directory for results and plots")
    
    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=30,
                       help="UMAP n_neighbors parameter (15-50 recommended)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter (0.01-0.5 recommended)")
    parser.add_argument("--n_components", type=int, default=2,
                       help="Number of UMAP dimensions (2 or 3)")
    parser.add_argument("--metric", type=str, default="euclidean",
                       help="Distance metric for UMAP")
    
    # Clustering parameters
    parser.add_argument("--clustering", type=str, default="hdbscan",
                       choices=["hdbscan", "kmeans", "none"],
                       help="Clustering method to apply")
    parser.add_argument("--min_cluster_size", type=int, default=50,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--n_clusters", type=int, default=5,
                       help="Number of clusters for K-means")
    
    # Analysis parameters
    parser.add_argument("--subsample", type=int, default=None,
                       help="Randomly subsample N embeddings for faster analysis")
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize embeddings before UMAP")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
    
    print("="*80)
    print("UMAP LATENT SPACE ANALYSIS")
    print("="*80)
    print(f"Embeddings: {args.embeddings}")
    print(f"Output: {args.output_dir}")
    print(f"UMAP parameters: n_neighbors={args.n_neighbors}, min_dist={args.min_dist}")
    print(f"Clustering: {args.clustering}")
    print("="*80)
    
    # Initialize analyzer
    analyzer = LatentSpaceAnalyzer(args.embeddings)
    
    # Load embeddings
    embeddings, spatial_coords, metadata = analyzer.load_embeddings()
    
    # Optional subsampling for large datasets
    if args.subsample and args.subsample < len(embeddings):
        print(f"Subsampling {args.subsample} embeddings from {len(embeddings)} total...")
        indices = np.random.choice(len(embeddings), args.subsample, replace=False)
        embeddings = embeddings[indices]
        if spatial_coords is not None:
            spatial_coords = spatial_coords[indices]
        analyzer.embeddings = embeddings
        analyzer.spatial_coords = spatial_coords
    
    # Optional standardization
    if args.standardize:
        print("Standardizing embeddings...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        analyzer.embeddings = embeddings
    
    # Compute UMAP embedding
    umap_embedding, umap_reducer = analyzer.compute_umap_embedding(
        n_neighbors=args.n_neighbors,
        min_dist=args.min_dist,
        n_components=args.n_components,
        metric=args.metric,
        random_state=args.random_state
    )
    
    # Perform clustering
    cluster_labels = None
    if args.clustering != "none":
        if args.clustering == "hdbscan":
            cluster_labels = analyzer.perform_clustering(
                umap_embedding, "hdbscan",
                min_cluster_size=args.min_cluster_size
            )
        elif args.clustering == "kmeans":
            cluster_labels = analyzer.perform_clustering(
                umap_embedding, "kmeans",
                n_clusters=args.n_clusters
            )
    
    # Create visualizations
    analyzer.create_visualizations(umap_embedding, cluster_labels, args.output_dir)
    
    # Save results
    result_metadata = {
        'embeddings_path': str(args.embeddings),
        'n_patterns': len(embeddings),
        'latent_dimension': embeddings.shape[1],
        'umap_parameters': {
            'n_neighbors': args.n_neighbors,
            'min_dist': args.min_dist,
            'n_components': args.n_components,
            'metric': args.metric
        },
        'clustering_method': args.clustering,
        'standardized': args.standardize,
        'subsampled': args.subsample,
        'original_metadata': metadata
    }
    
    analyzer.save_results(umap_embedding, cluster_labels, args.output_dir, result_metadata)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    
    if cluster_labels is not None:
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        print(f"Clusters found: {n_clusters}")
        print(f"Noise points: {n_noise}")
    
    print("\nGenerated files:")
    print("  - umap_overview.png/pdf: Main visualization")
    print("  - cluster_analysis.png: Detailed cluster analysis") 
    print("  - parameter_sensitivity.png: Parameter comparison")
    print("  - umap_results.json: Full results")
    print("  - umap_data.npz: Numpy data arrays")
    print("="*80)

if __name__ == "__main__":
    main()