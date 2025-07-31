#!/usr/bin/env python3
"""
Dimension Reduction for 4D-STEM Autoencoder Latent Space Analysis

This script performs dimension reduction on high-dimensional latent embeddings using
UMAP (Uniform Manifold Approximation and Projection) or PCA (Principal Component Analysis).
It includes parameter optimization capabilities to find the best settings for your data.

IMPORTANT: This script works with pre-generated embeddings. First generate embeddings 
using scripts/visualization/generate_embeddings.py, then use this script for dimension reduction.

Usage:
    # Basic UMAP reduction
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/dimension_reduction \
        --method umap \
        --n_neighbors 30 \
        --min_dist 0.1

    # PCA reduction
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/dimension_reduction \
        --method pca \
        --n_components 2

    # Parameter optimization
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/dimension_reduction \
        --method umap \
        --optimize_parameters \
        --subsample 5000
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

class DimensionReducer:
    """Comprehensive dimension reduction using UMAP, PCA with parameter optimization."""
    
    def __init__(self, embeddings_path: Path):
        """Initialize reducer with pre-generated embeddings."""
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
                             random_state: int = 42) -> Tuple[np.ndarray, object]:
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
        reduced_embedding = reducer.fit_transform(self.embeddings)
        
        print(f"✓ UMAP embedding computed: {reduced_embedding.shape}")
        print(f"  UMAP range: X=[{reduced_embedding[:, 0].min():.2f}, {reduced_embedding[:, 0].max():.2f}]")
        print(f"               Y=[{reduced_embedding[:, 1].min():.2f}, {reduced_embedding[:, 1].max():.2f}]")
        
        return reduced_embedding, reducer
    
    def compute_pca_embedding(self, n_components: int = 2, 
                            random_state: int = 42) -> Tuple[np.ndarray, object]:
        """Compute PCA embedding of latent space."""
        print(f"Computing PCA embedding...")
        print(f"  n_components: {n_components}")
        
        # Initialize PCA
        reducer = PCA(n_components=n_components, random_state=random_state)
        
        # Fit and transform
        reduced_embedding = reducer.fit_transform(self.embeddings)
        
        print(f"✓ PCA embedding computed: {reduced_embedding.shape}")
        print(f"  Explained variance ratio: {reducer.explained_variance_ratio_}")
        print(f"  Total variance explained: {reducer.explained_variance_ratio_.sum():.3f}")
        print(f"  PCA range: X=[{reduced_embedding[:, 0].min():.2f}, {reduced_embedding[:, 0].max():.2f}]")
        if reduced_embedding.shape[1] > 1:
            print(f"             Y=[{reduced_embedding[:, 1].min():.2f}, {reduced_embedding[:, 1].max():.2f}]")
        
        return reduced_embedding, reducer
    
    def optimize_umap_parameters(self, output_dir: Path, subsample: Optional[int] = None) -> Dict:
        """Optimize UMAP parameters using grid search."""
        print("Optimizing UMAP parameters...")
        
        # Use subset for optimization if specified
        embeddings_to_use = self.embeddings
        if subsample and subsample < len(self.embeddings):
            print(f"Subsampling {subsample} embeddings for optimization...")
            indices = np.random.choice(len(self.embeddings), subsample, replace=False)
            embeddings_to_use = self.embeddings[indices]
        
        # Parameter grid
        n_neighbors_list = [5, 15, 30, 50, 100]
        min_dist_list = [0.01, 0.1, 0.5, 1.0]
        
        results = []
        best_score = -1
        best_params = {}
        
        print(f"Testing {len(n_neighbors_list) * len(min_dist_list)} parameter combinations...")
        
        for n_neighbors in tqdm(n_neighbors_list, desc="n_neighbors"):
            for min_dist in min_dist_list:
                try:
                    # Compute UMAP
                    reducer = umap.UMAP(
                        n_neighbors=n_neighbors,
                        min_dist=min_dist,
                        n_components=2,
                        random_state=42,
                        verbose=False
                    )
                    embedding = reducer.fit_transform(embeddings_to_use)
                    
                    # Evaluate using clustering quality (silhouette score with K-means)
                    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(embedding)
                    
                    silhouette = silhouette_score(embedding, cluster_labels)
                    
                    # Store results
                    result = {
                        'n_neighbors': n_neighbors,
                        'min_dist': min_dist,
                        'silhouette_score': silhouette,
                        'n_clusters': len(set(cluster_labels))
                    }
                    results.append(result)
                    
                    # Track best parameters
                    if silhouette > best_score:
                        best_score = silhouette
                        best_params = {
                            'n_neighbors': n_neighbors,
                            'min_dist': min_dist,
                            'silhouette_score': silhouette
                        }
                        
                except Exception as e:
                    print(f"Failed for n_neighbors={n_neighbors}, min_dist={min_dist}: {e}")
                    continue
        
        # Save results
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / 'umap_optimization_results.csv', index=False)
        
        # Create optimization visualization
        self._create_optimization_plots(results_df, output_dir, 'umap')
        
        print(f"✓ Best UMAP parameters found:")
        print(f"  n_neighbors: {best_params['n_neighbors']}")
        print(f"  min_dist: {best_params['min_dist']}")
        print(f"  silhouette_score: {best_params['silhouette_score']:.3f}")
        
        return best_params
    
    def optimize_pca_components(self, output_dir: Path, max_components: int = 10) -> Dict:
        """Analyze PCA components and find optimal number."""
        print("Analyzing PCA components...")
        
        # Compute PCA with maximum components
        pca = PCA(n_components=min(max_components, self.embeddings.shape[1]))
        pca.fit(self.embeddings)
        
        # Analyze explained variance
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Find optimal number of components (e.g., 95% variance explained)
        optimal_components = np.argmax(cumulative_variance >= 0.95) + 1
        
        results = {
            'optimal_components': int(optimal_components),
            'explained_variance_ratio': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'variance_at_optimal': float(cumulative_variance[optimal_components - 1])
        }
        
        # Create PCA analysis plots
        self._create_pca_analysis_plots(explained_variance, cumulative_variance, 
                                      optimal_components, output_dir)
        
        # Save results
        with open(output_dir / 'pca_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"✓ PCA analysis completed:")
        print(f"  Optimal components: {optimal_components}")
        print(f"  Variance explained: {results['variance_at_optimal']:.3f}")
        
        return results
    
    def _create_optimization_plots(self, results_df: pd.DataFrame, output_dir: Path, method: str):
        """Create parameter optimization visualization plots."""
        if method == 'umap':
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('UMAP Parameter Optimization Results', fontsize=16, fontweight='bold')
            
            # Heatmap of silhouette scores
            ax = axes[0]
            pivot_table = results_df.pivot('n_neighbors', 'min_dist', 'silhouette_score')
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', ax=ax)
            ax.set_title('Silhouette Score Heatmap')
            ax.set_xlabel('min_dist')
            ax.set_ylabel('n_neighbors')
            
            # Parameter vs score scatter
            ax = axes[1]
            scatter = ax.scatter(results_df['n_neighbors'], results_df['min_dist'], 
                               c=results_df['silhouette_score'], s=100, cmap='viridis')
            ax.set_xlabel('n_neighbors')
            ax.set_ylabel('min_dist')
            ax.set_title('Parameter Space Exploration')
            plt.colorbar(scatter, ax=ax, label='Silhouette Score')
            
            # Highlight best point
            best_idx = results_df['silhouette_score'].idxmax()
            best_row = results_df.loc[best_idx]
            ax.scatter(best_row['n_neighbors'], best_row['min_dist'], 
                      s=200, facecolors='none', edgecolors='red', linewidth=3)
            ax.annotate(f'Best: {best_row["silhouette_score"]:.3f}', 
                       (best_row['n_neighbors'], best_row['min_dist']),
                       xytext=(10, 10), textcoords='offset points',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
            
            plt.tight_layout()
            plt.savefig(output_dir / 'umap_parameter_optimization.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_pca_analysis_plots(self, explained_variance: np.ndarray, 
                                 cumulative_variance: np.ndarray, 
                                 optimal_components: int, output_dir: Path):
        """Create PCA analysis visualization plots."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PCA Component Analysis', fontsize=16, fontweight='bold')
        
        # Individual explained variance
        ax = axes[0]
        components = np.arange(1, len(explained_variance) + 1)
        ax.bar(components, explained_variance, alpha=0.7)
        ax.set_xlabel('Principal Component')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Individual Component Variance')
        ax.grid(True, alpha=0.3)
        
        # Cumulative explained variance
        ax = axes[1]
        ax.plot(components, cumulative_variance, 'bo-', linewidth=2, markersize=4)
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        ax.axvline(x=optimal_components, color='r', linestyle='--', alpha=0.7, 
                  label=f'Optimal: {optimal_components} components')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('Cumulative Variance Explained')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_component_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_basic_visualization(self, reduced_embedding: np.ndarray, 
                                 method: str, output_dir: Path) -> None:
        """Create basic visualization of reduced embeddings."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('default')
        
        # Check if we have 1D or 2D+ embeddings
        is_1d = reduced_embedding.shape[1] == 1
        
        if is_1d:
            # Special handling for 1D embeddings
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{method.upper()} Dimension Reduction Results (1D)', fontsize=16, fontweight='bold')
            
            # Plot 1: 1D scatter with random y-jitter
            ax = axes[0, 0]
            y_jitter = np.random.normal(0, 0.1, len(reduced_embedding))
            scatter = ax.scatter(reduced_embedding[:, 0], y_jitter, 
                               c='steelblue', s=1, alpha=0.6)
            ax.set_title(f'{method.upper()} Embedding (1D with jitter)')
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel('Random Jitter')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Histogram
            ax = axes[0, 1]
            ax.hist(reduced_embedding[:, 0], bins=50, alpha=0.7, color='steelblue')
            ax.set_title(f'{method.upper()} Distribution')
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Spatial mapping (if available)
            ax = axes[1, 0]
            if self.spatial_coords is not None:
                scatter = ax.scatter(self.spatial_coords[:, 0], self.spatial_coords[:, 1], 
                                   c=reduced_embedding[:, 0], cmap='viridis', s=4, alpha=0.8)
                ax.set_title(f'Spatial Distribution ({method.upper()} 1 color)')
                ax.set_xlabel('Scan X Position')
                ax.set_ylabel('Scan Y Position')
                plt.colorbar(scatter, ax=ax, label=f'{method.upper()} 1')
            else:
                ax.text(0.5, 0.5, 'No spatial coordinates\navailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Spatial Information N/A')
            
            # Plot 4: Box plot
            ax = axes[1, 1]
            ax.boxplot(reduced_embedding[:, 0], vert=True)
            ax.set_title(f'{method.upper()} Component Statistics')
            ax.set_ylabel(f'{method.upper()} 1')
            ax.grid(True, alpha=0.3)
            
        else:
            # Standard 2D+ visualization
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            fig.suptitle(f'{method.upper()} Dimension Reduction Results', fontsize=16, fontweight='bold')
            
            # Plot 1: Basic scatter plot
            ax = axes[0, 0]
            scatter = ax.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                               c='steelblue', s=1, alpha=0.6)
            ax.set_title(f'{method.upper()} Embedding')
            ax.set_xlabel(f'{method.upper()} 1')
            ax.set_ylabel(f'{method.upper()} 2')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Density plot
            ax = axes[0, 1]
            try:
                hb = ax.hexbin(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                              gridsize=50, cmap='Blues', mincnt=1)
                ax.set_title(f'{method.upper()} Density Distribution')
                ax.set_xlabel(f'{method.upper()} 1')
                ax.set_ylabel(f'{method.upper()} 2')
                plt.colorbar(hb, ax=ax, label='Point Density')
            except:
                ax.scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                          c='steelblue', s=1, alpha=0.6)
                ax.set_title(f'{method.upper()} Embedding (Density Fallback)')
            
            # Plot 3: Spatial mapping (if available)
            ax = axes[1, 0]
            if self.spatial_coords is not None:
                scatter = ax.scatter(self.spatial_coords[:, 0], self.spatial_coords[:, 1], 
                                   c=reduced_embedding[:, 0], cmap='viridis', s=4, alpha=0.8)
                ax.set_title(f'Spatial Distribution ({method.upper()} 1 color)')
                ax.set_xlabel('Scan X Position')
                ax.set_ylabel('Scan Y Position')
                plt.colorbar(scatter, ax=ax, label=f'{method.upper()} 1')
            else:
                ax.text(0.5, 0.5, 'No spatial coordinates\navailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Spatial Information N/A')
            
            # Plot 4: Component correlation (if spatial coords available)
            ax = axes[1, 1]
            if self.spatial_coords is not None:
                scatter = ax.scatter(self.spatial_coords[:, 0], self.spatial_coords[:, 1], 
                                   c=reduced_embedding[:, 1], cmap='plasma', s=4, alpha=0.8)
                ax.set_title(f'Spatial Distribution ({method.upper()} 2 color)')
                ax.set_xlabel('Scan X Position')
                ax.set_ylabel('Scan Y Position')
                plt.colorbar(scatter, ax=ax, label=f'{method.upper()} 2')
            else:
                # Show histogram of component values instead
                ax.hist2d(reduced_embedding[:, 0], reduced_embedding[:, 1], bins=50, cmap='Blues')
                ax.set_title(f'{method.upper()} Component Correlation')
                ax.set_xlabel(f'{method.upper()} 1')
                ax.set_ylabel(f'{method.upper()} 2')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{method}_reduction_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{method}_reduction_overview.pdf', bbox_inches='tight')
        plt.close()
        
        print(f"✓ Basic visualization saved to: {output_dir}")
    
    def save_results(self, reduced_embedding: np.ndarray, reducer: object,
                    method: str, output_dir: Path, metadata: Dict) -> None:
        """Save dimension reduction results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save embeddings and results
        results = {
            'original_embeddings': self.embeddings.tolist(),
            'reduced_embedding': reduced_embedding.tolist(),
            'spatial_coordinates': self.spatial_coords.tolist() if self.spatial_coords is not None else None,
            'method': method,
            'metadata': metadata
        }
        
        with open(output_dir / f'{method}_reduction_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as numpy arrays for easy loading
        save_dict = {
            'original_embeddings': self.embeddings,
            'reduced_embedding': reduced_embedding,
            'spatial_coordinates': self.spatial_coords,
            'method': method
        }
        
        # Add method-specific reducer parameters
        if hasattr(reducer, 'get_params'):
            save_dict['reducer_params'] = reducer.get_params()
        
        np.savez(output_dir / f'{method}_reduction_data.npz', **save_dict)
        
        print(f"✓ Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Dimension Reduction for 4D-STEM Autoencoder Latent Space")
    
    # Required arguments
    parser.add_argument("--embeddings", type=Path, required=True,
                       help="Path to pre-generated embeddings (.npz, .h5, .pt)")
    parser.add_argument("--output_dir", type=Path, default="dimension_reduction",
                       help="Output directory for results and plots")
    
    # Method selection
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "pca"],
                       help="Dimension reduction method")
    
    # UMAP parameters
    parser.add_argument("--n_neighbors", type=int, default=30,
                       help="UMAP n_neighbors parameter (15-50 recommended)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter (0.01-0.5 recommended)")
    parser.add_argument("--metric", type=str, default="euclidean",
                       help="Distance metric for UMAP")
    
    # PCA parameters
    parser.add_argument("--n_components", type=int, default=2,
                       help="Number of components for dimension reduction")
    
    # Optimization parameters
    parser.add_argument("--optimize_parameters", action="store_true",
                       help="Perform parameter optimization (slow)")
    parser.add_argument("--subsample", type=int, default=None,
                       help="Randomly subsample N embeddings for faster analysis")
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize embeddings before reduction")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
    
    print("="*80)
    print("DIMENSION REDUCTION ANALYSIS")
    print("="*80)
    print(f"Embeddings: {args.embeddings}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method.upper()}")
    if args.optimize_parameters:
        print("Parameter optimization: ENABLED")
    print("="*80)
    
    # Initialize reducer
    reducer = DimensionReducer(args.embeddings)
    
    # Load embeddings
    embeddings, spatial_coords, metadata = reducer.load_embeddings()
    
    # Optional subsampling for large datasets
    if args.subsample and args.subsample < len(embeddings):
        print(f"Subsampling {args.subsample} embeddings from {len(embeddings)} total...")
        indices = np.random.choice(len(embeddings), args.subsample, replace=False)
        embeddings = embeddings[indices]
        if spatial_coords is not None:
            spatial_coords = spatial_coords[indices]
        reducer.embeddings = embeddings
        reducer.spatial_coords = spatial_coords
    
    # Optional standardization
    if args.standardize:
        print("Standardizing embeddings...")
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)
        reducer.embeddings = embeddings
    
    # Parameter optimization
    best_params = {}
    if args.optimize_parameters:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        if args.method == "umap":
            best_params = reducer.optimize_umap_parameters(args.output_dir, args.subsample)
        elif args.method == "pca":
            best_params = reducer.optimize_pca_components(args.output_dir, args.n_components)
    
    # Use optimized parameters if available
    if args.method == "umap":
        n_neighbors = best_params.get('n_neighbors', args.n_neighbors)
        min_dist = best_params.get('min_dist', args.min_dist)
        
        reduced_embedding, fitted_reducer = reducer.compute_umap_embedding(
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            n_components=args.n_components,
            metric=args.metric,
            random_state=args.random_state
        )
        
    elif args.method == "pca":
        n_components = best_params.get('optimal_components', args.n_components)
        
        reduced_embedding, fitted_reducer = reducer.compute_pca_embedding(
            n_components=n_components,
            random_state=args.random_state
        )
    
    # Create visualizations
    reducer.create_basic_visualization(reduced_embedding, args.method, args.output_dir)
    
    # Save results
    result_metadata = {
        'embeddings_path': str(args.embeddings),
        'n_patterns': len(embeddings),
        'latent_dimension': embeddings.shape[1],
        'method': args.method,
        'parameters': best_params if best_params else (
            {'n_neighbors': args.n_neighbors, 'min_dist': args.min_dist, 'metric': args.metric} 
            if args.method == 'umap' else {'n_components': args.n_components}
        ),
        'standardized': args.standardize,
        'subsampled': args.subsample,
        'original_metadata': metadata
    }
    
    reducer.save_results(reduced_embedding, fitted_reducer, args.method, 
                        args.output_dir, result_metadata)
    
    print("\n" + "="*80)
    print("DIMENSION REDUCTION COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Method: {args.method.upper()}")
    print(f"Output shape: {reduced_embedding.shape}")
    
    print("\nGenerated files:")
    print(f"  - {args.method}_reduction_overview.png/pdf: Main visualization")
    if args.optimize_parameters:
        if args.method == "umap":
            print("  - umap_parameter_optimization.png: Parameter optimization results")
            print("  - umap_optimization_results.csv: Detailed optimization data")
        else:
            print("  - pca_component_analysis.png: Component analysis")
            print("  - pca_analysis_results.json: PCA analysis data")
    print(f"  - {args.method}_reduction_results.json: Full results")
    print(f"  - {args.method}_reduction_data.npz: Numpy data arrays")
    print("="*80)

if __name__ == "__main__":
    main()