#!/usr/bin/env python3
"""
Dimension Reduction for 4D-STEM Autoencoder Latent Space Analysis

This refactored script provides a modular, extensible framework for performing dimension 
reduction on high-dimensional latent embeddings using UMAP (Uniform Manifold Approximation 
and Projection) or PCA (Principal Component Analysis). It features comprehensive parameter 
optimization, hyperparameter sweeping, and advanced visualization capabilities.

Key Features:
    - Modular architecture with separated concerns (loading, reduction, optimization, visualization)
    - Efficient PCA hyperparameter sweeping with custom dimension lists
    - Multi-criteria optimization (variance, clustering quality, reconstruction error)
    - Comprehensive visualization suite with detailed analysis plots
    - Support for multiple file formats (.npz, .h5, .pt)
    - Robust error handling and progress tracking

IMPORTANT: This script works with pre-generated embeddings. First generate embeddings 
using scripts/visualization/generate_embeddings.py, then use this script for dimension reduction.

Architecture:
    - DimensionReductionConfig: Configuration management
    - EmbeddingLoader: Multi-format data loading
    - BaseReducer: Abstract interface for reduction methods
    - UMAPReducer/PCAReducer: Method-specific implementations
    - ParameterOptimizer: Advanced hyperparameter optimization
    - DimensionReductionVisualizer: Comprehensive plotting
    - ResultsSaver: Multi-format result persistence
    - DimensionReductionPipeline: Main orchestrator

Usage Examples:

    # Basic UMAP reduction
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/dimension_reduction \
        --method umap \
        --n_neighbors 30 \
        --min_dist 0.1

    # PCA with component sweep (automatically optimizes across specified values)
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/pca_sweep \
        --method pca \
        --pca_component_sweep "30,50,80,100"

    # PCA with single component count (no optimization)
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/pca_fixed \
        --method pca \
        --n_components 50

    # Advanced PCA sweep with preprocessing
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/pca_analysis \
        --method pca \
        --pca_component_sweep "10,20,30,40,50,75,100,150" \
        --standardize \
        --subsample 10000

    # UMAP parameter optimization
    python scripts/analysis/dimension_reduction.py \
        --embeddings embeddings/patterns_embeddings.npz \
        --output_dir results/umap_optimization \
        --method umap \
        --optimize_parameters \
        --subsample 5000

Output Files:
    Basic Analysis:
        - {method}_reduction_overview.png/pdf: Main visualization
        - {method}_reduction_results.json: Complete results with metadata
        - {method}_reduction_data.npz: Numpy arrays for further analysis
    
    UMAP Optimization:
        - umap_parameter_optimization.png: Parameter space exploration
        - umap_optimization_results.csv: Detailed optimization metrics
    
    PCA Hyperparameter Analysis:
        - pca_hyperparameter_analysis.png: 6-panel comprehensive analysis
        - pca_hyperparameter_sweep.csv: Detailed metrics for all dimensions
        - pca_hyperparameter_analysis.json: Complete analysis metadata
        - pca_explained_variance_detailed.png: Focused variance analysis

Performance Notes:
    - Use --subsample for large datasets to speed up optimization
    - PCA component sweep is highly efficient with vectorized operations
    - UMAP optimization can be slow; consider subsampling for initial exploration
    - --standardize recommended for datasets with varying scales
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import h5py
import json
from typing import Tuple, Optional, Dict, List, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Add path for scan utilities
import sys
sys.path.append(str(Path(__file__).parent.parent / "visualization"))

# Scientific computing
import umap
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import pandas as pd

@dataclass
class DimensionReductionConfig:
    """
    Configuration for dimension reduction analysis.
    
    This dataclass centralizes all configuration parameters for the dimension reduction
    pipeline, making it easy to modify parameters programmatically and ensuring
    consistent parameter passing throughout the system.
    
    Attributes:
        embeddings_path: Path to pre-generated embeddings file (.npz, .h5, .pt)
        output_dir: Directory for saving results and visualizations
        method: Reduction method ('umap' or 'pca')
        n_components: Target number of dimensions for reduction
        standardize: Whether to standardize embeddings before reduction
        subsample: Optional number of samples to use (for performance)
        optimize_parameters: Whether to perform hyperparameter optimization
        random_state: Random seed for reproducibility
        pca_component_sweep: Custom list of PCA dimensions to test during optimization
        scan_shape: Optional scan dimensions (Ny, Nx) to override spatial coordinates
    """
    embeddings_path: Path
    output_dir: Path
    method: str = "umap"
    n_components: int = 2
    standardize: bool = False
    subsample: Optional[int] = None
    optimize_parameters: bool = False
    random_state: int = 42
    pca_component_sweep: Optional[List[int]] = None
    scan_shape: Optional[Tuple[int, int]] = None

@dataclass 
class UMAPConfig:
    """Configuration for UMAP reduction."""
    n_neighbors: int = 30
    min_dist: float = 0.1
    metric: str = "euclidean"
    n_components: int = 2
    random_state: int = 42

@dataclass
class PCAConfig:
    """Configuration for PCA reduction."""
    n_components: int = 2
    random_state: int = 42

class BaseReducer(ABC):
    """
    Abstract base class for dimension reduction methods.
    
    This abstract class defines the interface that all dimension reduction
    implementations must follow, enabling easy extension with new methods
    while maintaining consistent behavior throughout the pipeline.
    
    The interface separates the concerns of fitting/transforming data from
    method-specific configuration, making it easy to swap different algorithms
    or compare their performance.
    """
    
    @abstractmethod
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, object]:
        """
        Fit the reducer to embeddings and transform them.
        
        Args:
            embeddings: High-dimensional input embeddings to reduce
            
        Returns:
            Tuple of (reduced_embeddings, fitted_reducer_object)
        """
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """
        Get the string identifier for this reduction method.
        
        Returns:
            Method name (e.g., 'umap', 'pca') for file naming and logging
        """
        pass

class UMAPReducer(BaseReducer):
    """UMAP dimension reduction implementation."""
    
    def __init__(self, config: UMAPConfig):
        self.config = config
        self.reducer = None
    
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, object]:
        """Compute UMAP embedding of latent space."""
        print(f"Computing UMAP embedding...")
        print(f"  n_neighbors: {self.config.n_neighbors}")
        print(f"  min_dist: {self.config.min_dist}")
        print(f"  n_components: {self.config.n_components}")
        print(f"  metric: {self.config.metric}")
        
        self.reducer = umap.UMAP(
            n_neighbors=self.config.n_neighbors,
            min_dist=self.config.min_dist,
            n_components=self.config.n_components,
            metric=self.config.metric,
            random_state=self.config.random_state,
            verbose=True
        )
        
        reduced_embedding = self.reducer.fit_transform(embeddings)
        
        print(f"✓ UMAP embedding computed: {reduced_embedding.shape}")
        print(f"  UMAP range: X=[{reduced_embedding[:, 0].min():.2f}, {reduced_embedding[:, 0].max():.2f}]")
        print(f"               Y=[{reduced_embedding[:, 1].min():.2f}, {reduced_embedding[:, 1].max():.2f}]")
        
        return reduced_embedding, self.reducer
    
    def get_method_name(self) -> str:
        return "umap"

class PCAReducer(BaseReducer):
    """PCA dimension reduction implementation."""
    
    def __init__(self, config: PCAConfig):
        self.config = config
        self.reducer = None
    
    def fit_transform(self, embeddings: np.ndarray) -> Tuple[np.ndarray, object]:
        """Compute PCA embedding of latent space."""
        print(f"Computing PCA embedding...")
        print(f"  n_components: {self.config.n_components}")
        
        self.reducer = PCA(
            n_components=self.config.n_components, 
            random_state=self.config.random_state
        )
        
        reduced_embedding = self.reducer.fit_transform(embeddings)
        
        print(f"✓ PCA embedding computed: {reduced_embedding.shape}")
        print(f"  Explained variance ratio: {self.reducer.explained_variance_ratio_}")
        print(f"  Total variance explained: {self.reducer.explained_variance_ratio_.sum():.3f}")
        print(f"  PCA range: X=[{reduced_embedding[:, 0].min():.2f}, {reduced_embedding[:, 0].max():.2f}]")
        if reduced_embedding.shape[1] > 1:
            print(f"             Y=[{reduced_embedding[:, 1].min():.2f}, {reduced_embedding[:, 1].max():.2f}]")
        
        return reduced_embedding, self.reducer
    
    def get_method_name(self) -> str:
        return "pca"

class EmbeddingLoader:
    """Handles loading embeddings from various file formats."""
    
    def __init__(self, embeddings_path: Path):
        self.embeddings_path = embeddings_path
        
    def load(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load pre-generated embeddings from file."""
        print(f"Loading embeddings from: {self.embeddings_path}")
        
        if self.embeddings_path.suffix == '.npz':
            return self._load_npz()
        elif self.embeddings_path.suffix == '.h5':
            return self._load_h5()
        elif self.embeddings_path.suffix == '.pt':
            return self._load_pytorch()
        else:
            raise ValueError(f"Unsupported embedding format: {self.embeddings_path.suffix}")
    
    def _load_npz(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load from NumPy format."""
        data = np.load(self.embeddings_path, allow_pickle=True)
        embeddings = data['embeddings']
        spatial_coords = data.get('spatial_coordinates', None)
        
        metadata = None
        if 'metadata' in data:
            try:
                metadata_str = str(data['metadata'].item())
                metadata = json.loads(metadata_str)
            except:
                metadata = None
        
        return embeddings, spatial_coords, metadata
    
    def _load_h5(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load from HDF5 format."""
        with h5py.File(self.embeddings_path, 'r') as f:
            embeddings = f['embeddings'][:]
            spatial_coords = f.get('spatial_coordinates', None)
            if spatial_coords is not None:
                spatial_coords = spatial_coords[:]
            
            metadata = {}
            for key, value in f.attrs.items():
                try:
                    metadata[key] = json.loads(value)
                except:
                    metadata[key] = value
        
        return embeddings, spatial_coords, metadata
    
    def _load_pytorch(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load from PyTorch format (legacy)."""
        import torch
        embeddings = torch.load(self.embeddings_path, map_location='cpu').numpy()
        
        coord_path = self.embeddings_path.with_name(self.embeddings_path.stem + '_coords.npy')
        spatial_coords = np.load(coord_path) if coord_path.exists() else None
        
        meta_path = self.embeddings_path.with_name(self.embeddings_path.stem + '_metadata.json') 
        metadata = None
        if meta_path.exists():
            with open(meta_path, 'r') as f:
                metadata = json.load(f)
        
        return embeddings, spatial_coords, metadata

class ParameterOptimizer:
    """
    Advanced parameter optimization for dimension reduction methods.
    
    This class provides sophisticated hyperparameter optimization capabilities
    including grid search, multi-criteria evaluation, and comprehensive
    analysis of parameter spaces. It supports both UMAP and PCA optimization
    with different strategies appropriate to each method.
    
    For PCA, it offers:
        - Hyperparameter sweeping across custom dimension lists
        - Multi-criteria optimization (variance, clustering, reconstruction)
        - Elbow detection and variance thresholding
        - Comprehensive visualization of the optimization landscape
    
    For UMAP, it provides:
        - Grid search over n_neighbors and min_dist parameters
        - Silhouette score-based evaluation
        - Parameter space visualization
    """
    
    def optimize_umap_parameters(self, embeddings: np.ndarray, 
                               output_dir: Path, subsample: Optional[int] = None) -> Dict:
        """Optimize UMAP parameters using grid search."""
        print("Optimizing UMAP parameters...")
        
        embeddings_to_use = self._prepare_embeddings_for_optimization(embeddings, subsample)
        
        n_neighbors_list = [5, 15, 30, 50, 100]
        min_dist_list = [0.01, 0.1, 0.5, 1.0]
        
        results = []
        best_score = -1
        best_params = {}
        
        print(f"Testing {len(n_neighbors_list) * len(min_dist_list)} parameter combinations...")
        
        for n_neighbors in tqdm(n_neighbors_list, desc="n_neighbors"):
            for min_dist in min_dist_list:
                result = self._evaluate_umap_params(embeddings_to_use, n_neighbors, min_dist)
                if result is not None:
                    results.append(result)
                    if result['silhouette_score'] > best_score:
                        best_score = result['silhouette_score']
                        best_params = {
                            'n_neighbors': n_neighbors,
                            'min_dist': min_dist,
                            'silhouette_score': best_score
                        }
        
        self._save_optimization_results(results, output_dir, 'umap', best_params)
        return best_params
    
    def optimize_pca_components(self, embeddings: np.ndarray, 
                              output_dir: Path, max_components: int = 10, 
                              component_sweep: Optional[List[int]] = None) -> Dict:
        """
        Comprehensive PCA hyperparameter optimization with multi-criteria analysis.
        
        This method performs an efficient sweep across PCA dimensions, evaluating
        each configuration using multiple metrics to provide comprehensive guidance
        for optimal dimensionality selection.
        
        Args:
            embeddings: Input high-dimensional embeddings
            output_dir: Directory for saving optimization results
            max_components: Maximum components to test (if component_sweep not provided)
            component_sweep: Custom list of component counts to test (e.g., [30,50,80,100])
            
        Returns:
            Dictionary containing optimization results with keys:
                - optimal_components_95_variance: Components for 95% variance threshold
                - optimal_components_elbow: Components from elbow method
                - optimal_components_clustering: Components with best clustering performance
                - recommended_components: Overall recommended component count
                - hyperparameter_sweep_results: Detailed metrics for all tested dimensions
                
        The method evaluates each dimension count using:
            - Explained variance ratio (reconstruction quality)
            - Silhouette score (clustering separability)
            - Reconstruction error (information preservation)
            - Dimensionality reduction ratio (efficiency metric)
        """
        print("Analyzing PCA components...")
        
        # If specific dimensions provided for sweep, use those; otherwise use range
        if component_sweep is not None:
            components_to_test = [c for c in component_sweep if c <= embeddings.shape[1]]
            max_comp_for_analysis = max(components_to_test)
        else:
            components_to_test = list(range(1, min(max_components, embeddings.shape[1]) + 1))
            max_comp_for_analysis = max(components_to_test)
        
        print(f"Testing PCA dimensions: {components_to_test}")
        
        # Fit PCA with maximum components needed for analysis
        pca_full = PCA(n_components=max_comp_for_analysis)
        pca_full.fit(embeddings)
        
        explained_variance = pca_full.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)
        
        # Perform hyperparameter sweep
        sweep_results = self._perform_pca_hyperparameter_sweep(
            embeddings, components_to_test, output_dir
        )
        
        # Find optimal based on various criteria
        optimal_95_variance = np.argmax(cumulative_variance >= 0.95) + 1
        optimal_elbow = self._find_elbow_point(explained_variance)
        
        # Get best from sweep (highest silhouette score)
        best_from_sweep = max(sweep_results, key=lambda x: x['silhouette_score'])
        
        results = {
            'optimal_components_95_variance': int(optimal_95_variance),
            'optimal_components_elbow': int(optimal_elbow),
            'optimal_components_clustering': best_from_sweep['n_components'],
            'best_silhouette_score': best_from_sweep['silhouette_score'],
            'explained_variance_ratio': explained_variance.tolist(),
            'cumulative_variance': cumulative_variance.tolist(),
            'variance_at_95': float(cumulative_variance[min(optimal_95_variance - 1, len(cumulative_variance) - 1)]),
            'hyperparameter_sweep_results': sweep_results,
            'recommended_components': best_from_sweep['n_components']
        }
        
        # Create comprehensive analysis plots
        self._create_pca_hyperparameter_plots(results, components_to_test, output_dir)
        
        # Save detailed results
        with open(output_dir / 'pca_hyperparameter_analysis.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save sweep results as CSV for easy analysis
        sweep_df = pd.DataFrame(sweep_results)
        sweep_df.to_csv(output_dir / 'pca_hyperparameter_sweep.csv', index=False)
        
        print(f"✓ PCA hyperparameter analysis completed:")
        print(f"  Optimal (95% variance): {optimal_95_variance} components")
        print(f"  Optimal (elbow method): {optimal_elbow} components")
        print(f"  Optimal (clustering): {best_from_sweep['n_components']} components (silhouette: {best_from_sweep['silhouette_score']:.3f})")
        print(f"  Recommended: {results['recommended_components']} components")
        
        return results
    
    def _perform_pca_hyperparameter_sweep(self, embeddings: np.ndarray, 
                                        components_to_test: List[int], 
                                        output_dir: Path) -> List[Dict]:
        """Perform hyperparameter sweep for PCA dimensions."""
        results = []
        
        print(f"Performing hyperparameter sweep for {len(components_to_test)} PCA dimensions...")
        
        for n_components in tqdm(components_to_test, desc="PCA dimensions"):
            try:
                # Fit PCA with current number of components
                pca = PCA(n_components=n_components, random_state=42)
                reduced_embedding = pca.fit_transform(embeddings)
                
                # Calculate metrics
                explained_variance_ratio = pca.explained_variance_ratio_.sum()
                
                # Evaluate clustering quality if we have enough dimensions
                if reduced_embedding.shape[1] >= 2:
                    # Use adaptive number of clusters based on data size
                    n_clusters = min(8, max(3, len(embeddings) // 1000))
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                    cluster_labels = kmeans.fit_predict(reduced_embedding)
                    silhouette = silhouette_score(reduced_embedding, cluster_labels)
                else:
                    # For 1D, create artificial clusters based on quantiles
                    quantiles = np.quantile(reduced_embedding.ravel(), [0.33, 0.66])
                    cluster_labels = np.digitize(reduced_embedding.ravel(), quantiles)
                    # Use a simple dispersion metric for 1D
                    silhouette = -np.std(reduced_embedding) / np.mean(np.abs(reduced_embedding))
                
                # Calculate reconstruction error (relative to original dimensionality)
                reconstruction = pca.inverse_transform(reduced_embedding)
                reconstruction_error = np.mean(np.sum((embeddings - reconstruction) ** 2, axis=1))
                relative_error = reconstruction_error / np.mean(np.sum(embeddings ** 2, axis=1))
                
                result = {
                    'n_components': n_components,
                    'explained_variance_ratio': float(explained_variance_ratio),
                    'silhouette_score': float(silhouette),
                    'reconstruction_error': float(reconstruction_error),
                    'relative_reconstruction_error': float(relative_error),
                    'n_clusters_used': len(set(cluster_labels)),
                    'dimensionality_reduction_ratio': float(n_components / embeddings.shape[1])
                }
                
                results.append(result)
                
            except Exception as e:
                print(f"Failed for n_components={n_components}: {e}")
                continue
        
        return results
    
    def _find_elbow_point(self, explained_variance: np.ndarray) -> int:
        """Find elbow point in explained variance using the kneedle algorithm approximation."""
        if len(explained_variance) < 3:
            return 1
        
        # Simple elbow detection: find point with maximum curvature
        y = explained_variance
        
        # Calculate second derivative (discrete approximation)
        if len(y) >= 3:
            second_derivative = np.gradient(np.gradient(y))
            # Find the point with maximum second derivative (most curved)
            elbow_idx = np.argmax(np.abs(second_derivative[1:-1])) + 1
            return int(elbow_idx + 1)  # Convert to 1-indexed
        else:
            return 2
    
    def _prepare_embeddings_for_optimization(self, embeddings: np.ndarray, 
                                           subsample: Optional[int]) -> np.ndarray:
        """Prepare embeddings for optimization (subsampling if needed)."""
        if subsample and subsample < len(embeddings):
            print(f"Subsampling {subsample} embeddings for optimization...")
            indices = np.random.choice(len(embeddings), subsample, replace=False)
            return embeddings[indices]
        return embeddings
    
    def _evaluate_umap_params(self, embeddings: np.ndarray, 
                            n_neighbors: int, min_dist: float) -> Optional[Dict]:
        """Evaluate UMAP parameters using silhouette score."""
        try:
            reducer = umap.UMAP(
                n_neighbors=n_neighbors,
                min_dist=min_dist,
                n_components=2,
                random_state=42,
                verbose=False
            )
            embedding = reducer.fit_transform(embeddings)
            
            kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(embedding)
            
            silhouette = silhouette_score(embedding, cluster_labels)
            
            return {
                'n_neighbors': n_neighbors,
                'min_dist': min_dist,
                'silhouette_score': silhouette,
                'n_clusters': len(set(cluster_labels))
            }
        except Exception as e:
            print(f"Failed for n_neighbors={n_neighbors}, min_dist={min_dist}: {e}")
            return None
    
    def _save_optimization_results(self, results: List[Dict], output_dir: Path, 
                                 method: str, best_params: Dict):
        """Save optimization results and create plots."""
        results_df = pd.DataFrame(results)
        results_df.to_csv(output_dir / f'{method}_optimization_results.csv', index=False)
        
        self._create_optimization_plots(results_df, output_dir, method, best_params)
        
        print(f"✓ Best {method.upper()} parameters found:")
        for key, value in best_params.items():
            print(f"  {key}: {value}")
    
    def _create_optimization_plots(self, results_df: pd.DataFrame, output_dir: Path, 
                                 method: str, best_params: Dict):
        """Create parameter optimization visualization plots."""
        if method == 'umap':
            fig, axes = plt.subplots(1, 2, figsize=(15, 6))
            fig.suptitle('UMAP Parameter Optimization Results', fontsize=16, fontweight='bold')
            
            pivot_table = results_df.pivot(index='n_neighbors', columns='min_dist', values='silhouette_score')
            sns.heatmap(pivot_table, annot=True, fmt='.3f', cmap='viridis', ax=axes[0])
            axes[0].set_title('Silhouette Score Heatmap')
            
            scatter = axes[1].scatter(results_df['n_neighbors'], results_df['min_dist'], 
                                   c=results_df['silhouette_score'], s=100, cmap='viridis')
            axes[1].set_xlabel('n_neighbors')
            axes[1].set_ylabel('min_dist')
            axes[1].set_title('Parameter Space Exploration')
            plt.colorbar(scatter, ax=axes[1], label='Silhouette Score')
            
            best_idx = results_df['silhouette_score'].idxmax()
            best_row = results_df.loc[best_idx]
            axes[1].scatter(best_row['n_neighbors'], best_row['min_dist'], 
                          s=200, facecolors='none', edgecolors='red', linewidth=3)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'umap_parameter_optimization.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    def _create_pca_hyperparameter_plots(self, results: Dict, 
                                       components_tested: List[int], 
                                       output_dir: Path):
        """Create comprehensive PCA hyperparameter analysis plots."""
        sweep_results = results['hyperparameter_sweep_results']
        sweep_df = pd.DataFrame(sweep_results)
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('PCA Hyperparameter Analysis', fontsize=16, fontweight='bold')
        
        # Plot 1: Explained variance vs components
        ax = axes[0, 0]
        ax.plot(sweep_df['n_components'], sweep_df['explained_variance_ratio'], 'bo-', linewidth=2)
        ax.axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        ax.axvline(x=results['optimal_components_95_variance'], color='r', linestyle='--', alpha=0.7)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Explained Variance Ratio')
        ax.set_title('Explained Variance vs Components')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 2: Silhouette score vs components
        ax = axes[0, 1]
        ax.plot(sweep_df['n_components'], sweep_df['silhouette_score'], 'go-', linewidth=2)
        best_idx = sweep_df['silhouette_score'].idxmax()
        best_comp = sweep_df.loc[best_idx, 'n_components']
        best_score = sweep_df.loc[best_idx, 'silhouette_score']
        ax.axvline(x=best_comp, color='g', linestyle='--', alpha=0.7)
        ax.scatter(best_comp, best_score, color='red', s=100, zorder=5)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Silhouette Score')
        ax.set_title('Clustering Quality vs Components')
        ax.grid(True, alpha=0.3)
        
        # Plot 3: Reconstruction error vs components
        ax = axes[0, 2]
        ax.semilogy(sweep_df['n_components'], sweep_df['relative_reconstruction_error'], 'mo-', linewidth=2)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Relative Reconstruction Error (log scale)')
        ax.set_title('Reconstruction Error vs Components')
        ax.grid(True, alpha=0.3)
        
        # Plot 4: Multi-criteria comparison
        ax = axes[1, 0]
        # Normalize metrics for comparison
        norm_variance = sweep_df['explained_variance_ratio'] / sweep_df['explained_variance_ratio'].max()
        norm_silhouette = (sweep_df['silhouette_score'] - sweep_df['silhouette_score'].min()) / \
                         (sweep_df['silhouette_score'].max() - sweep_df['silhouette_score'].min())
        norm_error = 1 - (sweep_df['relative_reconstruction_error'] / sweep_df['relative_reconstruction_error'].max())
        
        ax.plot(sweep_df['n_components'], norm_variance, 'b-', label='Normalized Explained Variance', linewidth=2)
        ax.plot(sweep_df['n_components'], norm_silhouette, 'g-', label='Normalized Silhouette Score', linewidth=2)
        ax.plot(sweep_df['n_components'], norm_error, 'm-', label='Normalized Inv. Recon. Error', linewidth=2)
        
        # Mark optimal points
        ax.axvline(x=results['optimal_components_95_variance'], color='b', linestyle=':', alpha=0.7, label='95% Variance')
        ax.axvline(x=results['optimal_components_elbow'], color='orange', linestyle=':', alpha=0.7, label='Elbow Point')
        ax.axvline(x=results['optimal_components_clustering'], color='g', linestyle=':', alpha=0.7, label='Best Clustering')
        
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Normalized Metric Value')
        ax.set_title('Multi-Criteria Comparison')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Plot 5: Dimensionality reduction ratio
        ax = axes[1, 1]
        ax.plot(sweep_df['n_components'], sweep_df['dimensionality_reduction_ratio'], 'co-', linewidth=2)
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Reduction Ratio (output/input dims)')
        ax.set_title('Dimensionality Reduction Ratio')
        ax.grid(True, alpha=0.3)
        
        # Plot 6: Summary recommendations
        ax = axes[1, 2]
        ax.axis('off')
        
        summary_text = f"""
PCA Hyperparameter Recommendations:

• 95% Variance: {results['optimal_components_95_variance']} components
  └─ Variance explained: {results['variance_at_95']:.3f}

• Elbow Method: {results['optimal_components_elbow']} components
  └─ Balance between complexity and performance

• Best Clustering: {results['optimal_components_clustering']} components
  └─ Silhouette score: {results['best_silhouette_score']:.3f}

• Recommended: {results['recommended_components']} components
  └─ Best overall clustering performance

Components tested: {min(components_tested)} - {max(components_tested)}
Total combinations: {len(components_tested)}
"""
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
               verticalalignment='top', bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.8))
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_hyperparameter_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create separate detailed plot for explained variance
        self._create_detailed_variance_plot(results, output_dir)
    
    def _create_detailed_variance_plot(self, results: Dict, output_dir: Path):
        """Create detailed explained variance analysis plot."""
        explained_variance = np.array(results['explained_variance_ratio'])
        cumulative_variance = np.array(results['cumulative_variance'])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        fig.suptitle('PCA Explained Variance Analysis', fontsize=16, fontweight='bold')
        
        components = np.arange(1, len(explained_variance) + 1)
        axes[0].bar(components, explained_variance, alpha=0.7, color='steelblue')
        axes[0].axvline(x=results['optimal_components_elbow'], color='orange', 
                       linestyle='--', alpha=0.7, label=f'Elbow: {results["optimal_components_elbow"]}')
        axes[0].set_xlabel('Principal Component')
        axes[0].set_ylabel('Explained Variance Ratio')
        axes[0].set_title('Individual Component Variance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(components, cumulative_variance, 'bo-', linewidth=2, markersize=4)
        axes[1].axhline(y=0.95, color='r', linestyle='--', alpha=0.7, label='95% threshold')
        axes[1].axvline(x=results['optimal_components_95_variance'], color='r', linestyle='--', alpha=0.7, 
                      label=f'95% Variance: {results["optimal_components_95_variance"]} components')
        axes[1].axvline(x=results['optimal_components_clustering'], color='g', linestyle='--', alpha=0.7, 
                      label=f'Best Clustering: {results["optimal_components_clustering"]} components')
        axes[1].set_xlabel('Number of Components')
        axes[1].set_ylabel('Cumulative Explained Variance')
        axes[1].set_title('Cumulative Variance Explained')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'pca_explained_variance_detailed.png', dpi=300, bbox_inches='tight')
        plt.close()

class DimensionReductionVisualizer:
    """Handles visualization of dimension reduction results."""
    
    def __init__(self, spatial_coords: Optional[np.ndarray] = None):
        self.spatial_coords = spatial_coords
    
    def create_basic_visualization(self, reduced_embedding: np.ndarray, 
                                 method: str, output_dir: Path) -> None:
        """Create basic visualization of reduced embeddings."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        plt.style.use('default')
        
        n_components = reduced_embedding.shape[1]
        
        if n_components == 1:
            self._create_1d_visualization(reduced_embedding, method, output_dir)
        elif n_components == 2:
            self._create_2d_visualization(reduced_embedding, method, output_dir)
        else:
            # For >2 components, create multi-component visualization
            self._create_multicomponent_visualization(reduced_embedding, method, output_dir)
        
        print(f"✓ Basic visualization saved to: {output_dir}")
    
    def _create_1d_visualization(self, reduced_embedding: np.ndarray, 
                               method: str, output_dir: Path):
        """Create visualization for 1D embeddings."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{method.upper()} Dimension Reduction Results (1D)', fontsize=16, fontweight='bold')
        
        y_jitter = np.random.normal(0, 0.1, len(reduced_embedding))
        axes[0, 0].scatter(reduced_embedding[:, 0], y_jitter, c='steelblue', s=1, alpha=0.6)
        axes[0, 0].set_title(f'{method.upper()} Embedding (1D with jitter)')
        axes[0, 0].set_xlabel(f'{method.upper()} 1')
        axes[0, 0].set_ylabel('Random Jitter')
        axes[0, 0].grid(True, alpha=0.3)
        
        axes[0, 1].hist(reduced_embedding[:, 0], bins=50, alpha=0.7, color='steelblue')
        axes[0, 1].set_title(f'{method.upper()} Distribution')
        axes[0, 1].set_xlabel(f'{method.upper()} 1')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        self._create_spatial_plot(axes[1, 0], reduced_embedding[:, 0], f'{method.upper()} 1')
        
        axes[1, 1].boxplot(reduced_embedding[:, 0], vert=True)
        axes[1, 1].set_title(f'{method.upper()} Component Statistics')
        axes[1, 1].set_ylabel(f'{method.upper()} 1')
        axes[1, 1].grid(True, alpha=0.3)
        
        self._save_visualization(fig, method, output_dir)
    
    def _create_2d_visualization(self, reduced_embedding: np.ndarray, 
                               method: str, output_dir: Path):
        """Create visualization for 2D+ embeddings."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle(f'{method.upper()} Dimension Reduction Results', fontsize=16, fontweight='bold')
        
        axes[0, 0].scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                          c='steelblue', s=1, alpha=0.6)
        axes[0, 0].set_title(f'{method.upper()} Embedding')
        axes[0, 0].set_xlabel(f'{method.upper()} 1')
        axes[0, 0].set_ylabel(f'{method.upper()} 2')
        axes[0, 0].grid(True, alpha=0.3)
        
        try:
            hb = axes[0, 1].hexbin(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                                  gridsize=50, cmap='Blues', mincnt=1)
            axes[0, 1].set_title(f'{method.upper()} Density Distribution')
            axes[0, 1].set_xlabel(f'{method.upper()} 1')
            axes[0, 1].set_ylabel(f'{method.upper()} 2')
            plt.colorbar(hb, ax=axes[0, 1], label='Point Density')
        except:
            axes[0, 1].scatter(reduced_embedding[:, 0], reduced_embedding[:, 1], 
                              c='steelblue', s=1, alpha=0.6)
            axes[0, 1].set_title(f'{method.upper()} Embedding (Density Fallback)')
        
        self._create_spatial_plot(axes[1, 0], reduced_embedding[:, 0], f'{method.upper()} 1')
        
        if self.spatial_coords is not None:
            self._create_spatial_plot(axes[1, 1], reduced_embedding[:, 1], f'{method.upper()} 2')
        else:
            axes[1, 1].hist2d(reduced_embedding[:, 0], reduced_embedding[:, 1], bins=50, cmap='Blues')
            axes[1, 1].set_title(f'{method.upper()} Component Correlation')
            axes[1, 1].set_xlabel(f'{method.upper()} 1')
            axes[1, 1].set_ylabel(f'{method.upper()} 2')
        
        self._save_visualization(fig, method, output_dir)
    
    def _create_multicomponent_visualization(self, reduced_embedding: np.ndarray, 
                                           method: str, output_dir: Path):
        """Create visualization for >2 component embeddings."""
        n_components = reduced_embedding.shape[1]
        n_cols = min(6, n_components)  # Max 6 columns for readability
        n_rows = int(np.ceil(n_components / n_cols))
        
        # Create figure with appropriate size
        fig_w = 4 * n_cols
        fig_h = 4 * n_rows
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h))
        fig.suptitle(f'{method.upper()} Components ({n_components} total)', fontsize=16, fontweight='bold')
        
        # Handle case where we have only one row
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        
        # Plot each component as a spatial map
        for k in range(n_components):
            row = k // n_cols
            col = k % n_cols
            ax = axes[row, col]
            
            component_values = reduced_embedding[:, k]
            self._create_spatial_plot(ax, component_values, f'{method.upper()} {k+1}')
        
        # Hide unused subplots
        for k in range(n_components, n_rows * n_cols):
            row = k // n_cols
            col = k % n_cols
            axes[row, col].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{method}_reduction_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{method}_reduction_overview.pdf', bbox_inches='tight')
        plt.close()
    
    def _create_spatial_plot(self, ax, values, label):
        """Create spatial distribution plot if coordinates are available."""
        if self.spatial_coords is not None:
            # Follow cluster_scan_visualization.py approach: create 2D array and use imshow
            # Determine scan shape from coordinates
            y_coords = self.spatial_coords[:, 0]
            x_coords = self.spatial_coords[:, 1]
            Ny = int(y_coords.max()) + 1
            Nx = int(x_coords.max()) + 1
            
            # Create 2D array for spatial mapping
            spatial_map = np.full((Ny, Nx), np.nan)
            
            # Fill the array with values at coordinate positions
            for i, (y, x) in enumerate(self.spatial_coords):
                if 0 <= y < Ny and 0 <= x < Nx:
                    spatial_map[int(y), int(x)] = values[i]
            
            # Use imshow with proper origin like cluster_scan_visualization.py
            im = ax.imshow(spatial_map, cmap='viridis', origin='upper', interpolation='nearest')
            ax.set_title(f'Spatial Distribution ({label})')
            ax.set_xlabel('Scan X Position')
            ax.set_ylabel('Scan Y Position')
            plt.colorbar(im, ax=ax, label=label)
        else:
            ax.text(0.5, 0.5, 'No spatial coordinates\navailable', 
                   ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Spatial Information N/A')
    
    def _save_visualization(self, fig, method, output_dir):
        """Save visualization to files."""
        plt.tight_layout()
        plt.savefig(output_dir / f'{method}_reduction_overview.png', dpi=300, bbox_inches='tight')
        plt.savefig(output_dir / f'{method}_reduction_overview.pdf', bbox_inches='tight')
        plt.close()

class ResultsSaver:
    """Handles saving dimension reduction results."""
    
    def save_results(self, embeddings: np.ndarray, reduced_embedding: np.ndarray, 
                    reducer: object, spatial_coords: Optional[np.ndarray],
                    method: str, output_dir: Path, metadata: Dict) -> None:
        """Save dimension reduction results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        results = {
            'original_embeddings': embeddings.tolist(),
            'reduced_embedding': reduced_embedding.tolist(),
            'spatial_coordinates': spatial_coords.tolist() if spatial_coords is not None else None,
            'method': method,
            'metadata': metadata
        }
        
        with open(output_dir / f'{method}_reduction_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        save_dict = {
            'original_embeddings': embeddings,
            'reduced_embedding': reduced_embedding,
            'spatial_coordinates': spatial_coords,
            'method': method
        }
        
        if hasattr(reducer, 'get_params'):
            save_dict['reducer_params'] = reducer.get_params()
        
        np.savez(output_dir / f'{method}_reduction_data.npz', **save_dict)
        
        print(f"✓ Results saved to: {output_dir}")

class DimensionReductionPipeline:
    """
    Main orchestrator for the complete dimension reduction analysis pipeline.
    
    This class coordinates all aspects of the dimension reduction workflow,
    from data loading through final result saving. It demonstrates the benefits
    of the modular architecture by cleanly separating concerns while providing
    a simple interface for end-users.
    
    The pipeline automatically:
        1. Loads embeddings from various file formats
        2. Preprocesses data (subsampling, standardization)
        3. Performs hyperparameter optimization if requested
        4. Applies the selected reduction method with optimal parameters
        5. Creates comprehensive visualizations
        6. Saves results in multiple formats
        
    This design makes it easy to extend functionality, modify individual
    components, or use the pipeline programmatically from other scripts.
    """
    
    def __init__(self, config: DimensionReductionConfig):
        self.config = config
        self.loader = EmbeddingLoader(config.embeddings_path)
        self.optimizer = ParameterOptimizer()
        self.saver = ResultsSaver()
        
    def run(self) -> None:
        """Execute the complete dimension reduction pipeline."""
        print("="*80)
        print("DIMENSION REDUCTION ANALYSIS")
        print("="*80)
        print(f"Embeddings: {self.config.embeddings_path}")
        print(f"Output: {self.config.output_dir}")
        print(f"Method: {self.config.method.upper()}")
        if self.config.optimize_parameters:
            print("Parameter optimization: ENABLED")
        if self.config.pca_component_sweep:
            print(f"PCA component sweep: {self.config.pca_component_sweep}")
        print("="*80)
        
        embeddings, spatial_coords, metadata = self.loader.load()
        self._print_loading_summary(embeddings, spatial_coords, metadata)
        
        embeddings, spatial_coords = self._preprocess_data(embeddings, spatial_coords)
        
        best_params = self._optimize_parameters_if_requested(embeddings)
        
        reducer = self._create_reducer(best_params)
        reduced_embedding, fitted_reducer = reducer.fit_transform(embeddings)
        
        visualizer = DimensionReductionVisualizer(spatial_coords)
        visualizer.create_basic_visualization(reduced_embedding, self.config.method, self.config.output_dir)
        
        result_metadata = self._create_result_metadata(embeddings, metadata, best_params)
        self.saver.save_results(embeddings, reduced_embedding, fitted_reducer, spatial_coords,
                              self.config.method, self.config.output_dir, result_metadata)
        
        self._print_completion_summary(reduced_embedding)
    
    def _print_loading_summary(self, embeddings: np.ndarray, 
                             spatial_coords: Optional[np.ndarray], metadata: Optional[Dict]):
        """Print summary of loaded data."""
        print(f"✓ Loaded embeddings: {embeddings.shape}")
        print(f"  Latent dimension: {embeddings.shape[1]}")
        print(f"  Embedding range: [{embeddings.min():.3f}, {embeddings.max():.3f}]")
        
        if spatial_coords is not None:
            print(f"  Spatial coordinates: {spatial_coords.shape}")
        
        if metadata is not None:
            print(f"  Metadata keys: {list(metadata.keys())}")
    
    def _preprocess_data(self, embeddings: np.ndarray, 
                        spatial_coords: Optional[np.ndarray]) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Preprocess embeddings (subsampling and standardization)."""
        # Handle scan shape override
        if self.config.scan_shape is not None:
            from scan_util import raster_coords
            Ny, Nx = self.config.scan_shape
            print(f"Using provided scan shape: {Ny} x {Nx}")
            print(f"Generating raster coordinates for {len(embeddings)} patterns...")
            
            # Generate proper raster coordinates
            expected_patterns = Ny * Nx
            if len(embeddings) != expected_patterns:
                print(f"WARNING: Pattern count ({len(embeddings)}) != scan dimensions ({Ny}x{Nx} = {expected_patterns})")
                if len(embeddings) < expected_patterns:
                    print("Using available patterns with raster coordinates")
                    spatial_coords = raster_coords(Ny, Nx)[:len(embeddings)]
                else:
                    print("Truncating patterns to match scan dimensions")
                    embeddings = embeddings[:expected_patterns]
                    spatial_coords = raster_coords(Ny, Nx)
            else:
                spatial_coords = raster_coords(Ny, Nx)
            print(f"Generated {len(spatial_coords)} spatial coordinates")
        
        if self.config.subsample and self.config.subsample < len(embeddings):
            print(f"Subsampling {self.config.subsample} embeddings from {len(embeddings)} total...")
            indices = np.random.choice(len(embeddings), self.config.subsample, replace=False)
            embeddings = embeddings[indices]
            if spatial_coords is not None:
                spatial_coords = spatial_coords[indices]
        
        if self.config.standardize:
            print("Standardizing embeddings...")
            scaler = StandardScaler()
            embeddings = scaler.fit_transform(embeddings)
        
        return embeddings, spatial_coords
    
    def _optimize_parameters_if_requested(self, embeddings: np.ndarray) -> Dict:
        """Optimize parameters if requested or if PCA component sweep is specified."""
        best_params = {}
        
        # For PCA, automatically optimize if component sweep is specified
        should_optimize_pca = (self.config.method == "pca" and 
                              self.config.pca_component_sweep is not None)
        
        if self.config.optimize_parameters or should_optimize_pca:
            self.config.output_dir.mkdir(parents=True, exist_ok=True)
            
            if self.config.method == "umap":
                if not self.config.optimize_parameters:
                    print("Note: UMAP optimization requires --optimize_parameters flag")
                    return best_params
                best_params = self.optimizer.optimize_umap_parameters(
                    embeddings, self.config.output_dir, self.config.subsample)
                    
            elif self.config.method == "pca":
                # For PCA, component sweep is the primary way to specify components
                component_sweep = self.config.pca_component_sweep
                
                if component_sweep is not None:
                    print(f"PCA component sweep specified: {component_sweep}")
                    print("Automatically running optimization across specified components...")
                elif self.config.optimize_parameters:
                    # Only fall back to single component if explicit optimization requested
                    component_sweep = [self.config.n_components]
                    print(f"No component sweep specified, optimizing single value: {component_sweep}")
                    print("Note: Use --pca_component_sweep to test multiple values efficiently")
                else:
                    return best_params
                
                best_params = self.optimizer.optimize_pca_components(
                    embeddings, self.config.output_dir, max(component_sweep), component_sweep)
        return best_params
    
    def _create_reducer(self, best_params: Dict) -> BaseReducer:
        """Create appropriate reducer based on method and parameters."""
        if self.config.method == "umap":
            umap_config = UMAPConfig(
                n_neighbors=best_params.get('n_neighbors', 30),
                min_dist=best_params.get('min_dist', 0.1),
                n_components=self.config.n_components,
                random_state=self.config.random_state
            )
            return UMAPReducer(umap_config)
        elif self.config.method == "pca":
            # For PCA: use optimized components if available, otherwise fall back to n_components
            if best_params and 'recommended_components' in best_params:
                n_components = best_params['recommended_components']
                print(f"Using optimized PCA components: {n_components}")
            else:
                n_components = self.config.n_components
                print(f"Using specified n_components: {n_components}")
            
            pca_config = PCAConfig(
                n_components=n_components,
                random_state=self.config.random_state
            )
            return PCAReducer(pca_config)
        else:
            raise ValueError(f"Unsupported method: {self.config.method}")
    
    def _create_result_metadata(self, embeddings: np.ndarray, 
                              metadata: Optional[Dict], best_params: Dict) -> Dict:
        """Create metadata for results."""
        return {
            'embeddings_path': str(self.config.embeddings_path),
            'n_patterns': len(embeddings),
            'latent_dimension': embeddings.shape[1],
            'method': self.config.method,
            'parameters': best_params if best_params else self._get_default_params(),
            'standardized': self.config.standardize,
            'subsampled': self.config.subsample,
            'original_metadata': metadata
        }
    
    def _get_default_params(self) -> Dict:
        """Get default parameters for the method."""
        if self.config.method == 'umap':
            return {'n_neighbors': 30, 'min_dist': 0.1, 'metric': 'euclidean'}
        else:
            return {'n_components': self.config.n_components}
    
    def _print_completion_summary(self, reduced_embedding: np.ndarray):
        """Print completion summary."""
        print("\n" + "="*80)
        print("DIMENSION REDUCTION COMPLETED SUCCESSFULLY")
        print("="*80)
        print(f"Results saved to: {self.config.output_dir}")
        print(f"Method: {self.config.method.upper()}")
        print(f"Output shape: {reduced_embedding.shape}")
        
        print("\nGenerated files:")
        print(f"  - {self.config.method}_reduction_overview.png/pdf: Main visualization")
        if self.config.optimize_parameters:
            if self.config.method == "umap":
                print("  - umap_parameter_optimization.png: Parameter optimization results")
                print("  - umap_optimization_results.csv: Detailed optimization data")
            else:
                print("  - pca_hyperparameter_analysis.png: Comprehensive analysis")
                print("  - pca_hyperparameter_sweep.csv: Detailed sweep results")
                print("  - pca_hyperparameter_analysis.json: Analysis metadata")
        print(f"  - {self.config.method}_reduction_results.json: Full results")
        print(f"  - {self.config.method}_reduction_data.npz: Numpy data arrays")
        print("="*80)

def create_config_from_args(args) -> DimensionReductionConfig:
    """
    Create configuration object from command line arguments.
    
    This function bridges the gap between CLI argument parsing and the
    internal configuration system. It handles special parsing for complex
    arguments like PCA component sweeps and provides validation.
    
    Args:
        args: Parsed command line arguments from argparse
        
    Returns:
        DimensionReductionConfig object ready for pipeline initialization
        
    Note:
        PCA component sweep accepts comma-separated integers like "30,50,80,100"
        or space-separated integers. Invalid formats are gracefully handled
        with warning messages.
    """
    # Parse PCA component sweep if provided
    pca_component_sweep = None
    if hasattr(args, 'pca_component_sweep') and args.pca_component_sweep:
        try:
            # Support comma-separated values or space-separated
            if ',' in args.pca_component_sweep:
                pca_component_sweep = [int(x.strip()) for x in args.pca_component_sweep.split(',')]
            else:
                pca_component_sweep = [int(x) for x in args.pca_component_sweep.split()]
            print(f"PCA component sweep: {pca_component_sweep}")
        except ValueError:
            print(f"Warning: Invalid PCA component sweep format: {args.pca_component_sweep}")
            pca_component_sweep = None
    
    # Convert scan_shape to tuple if provided
    scan_shape = tuple(args.scan_shape) if args.scan_shape else None
    
    return DimensionReductionConfig(
        embeddings_path=args.embeddings,
        output_dir=args.output_dir,
        method=args.method,
        n_components=args.n_components,
        standardize=args.standardize,
        subsample=args.subsample,
        optimize_parameters=args.optimize_parameters,
        random_state=args.random_state,
        pca_component_sweep=pca_component_sweep,
        scan_shape=scan_shape
    )

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Dimension Reduction for 4D-STEM Autoencoder Latent Space")
    
    parser.add_argument("--embeddings", type=Path, required=True,
                       help="Path to pre-generated embeddings (.npz, .h5, .pt)")
    parser.add_argument("--output_dir", type=Path, default="dimension_reduction",
                       help="Output directory for results and plots")
    
    parser.add_argument("--method", type=str, default="umap", choices=["umap", "pca"],
                       help="Dimension reduction method")
    
    parser.add_argument("--n_neighbors", type=int, default=30,
                       help="UMAP n_neighbors parameter (15-50 recommended)")
    parser.add_argument("--min_dist", type=float, default=0.1,
                       help="UMAP min_dist parameter (0.01-0.5 recommended)")
    parser.add_argument("--metric", type=str, default="euclidean",
                       help="Distance metric for UMAP")
    
    parser.add_argument("--n_components", type=int, default=2,
                       help="Number of components for dimension reduction (for PCA, prefer --pca_component_sweep)")
    parser.add_argument("--pca_component_sweep", type=str, default=None,
                       help="Comma-separated list of PCA components to test and optimize (e.g., '30,50,80,100'). Automatically runs optimization when specified.")
    
    parser.add_argument("--optimize_parameters", action="store_true",
                       help="Perform parameter optimization (slow)")
    parser.add_argument("--subsample", type=int, default=None,
                       help="Randomly subsample N embeddings for faster analysis")
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize embeddings before reduction")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--scan_shape", type=int, nargs=2, default=None,
                       metavar=("NY", "NX"), help="Scan dimensions (height width) - overrides spatial coordinates from file")
    
    return parser.parse_args()

def main():
    """
    Main entry point for dimension reduction analysis.
    
    This function provides the command-line interface for the dimension
    reduction pipeline. It handles argument parsing, configuration creation,
    and pipeline execution with proper error handling.
    
    The function demonstrates the clean separation of concerns in the
    refactored architecture - argument parsing, configuration, and execution
    are handled by separate, focused components.
    
    Raises:
        FileNotFoundError: If the specified embeddings file doesn't exist
        Various exceptions: From pipeline execution, propagated with context
    """
    args = parse_arguments()
    
    if not args.embeddings.exists():
        raise FileNotFoundError(f"Embeddings not found: {args.embeddings}")
    
    config = create_config_from_args(args)
    pipeline = DimensionReductionPipeline(config)
    pipeline.run()

if __name__ == "__main__":
    main()