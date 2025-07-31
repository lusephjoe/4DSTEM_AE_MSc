#!/usr/bin/env python3
"""
Clustering Analysis for Embeddings

This script performs comprehensive clustering analysis on both dimension-reduced embeddings
(from UMAP, PCA, etc.) and original latent embeddings from autoencoders.
It supports multiple clustering algorithms and provides extensive cluster quality analysis.

Usage:
    # Clustering analysis on reduced embeddings
    python scripts/analysis/clustering_analysis.py \
        --reduced_embeddings results/dimension_reduction/umap_reduction_data.npz \
        --output_dir results/clustering_analysis \
        --method hdbscan \
        --min_cluster_size 50

    # Clustering analysis on original latent embeddings
    python scripts/analysis/clustering_analysis.py \
        --latent_embeddings workspace/proj_data_ds4_l32/outputs/embeddings/ds4_4epoch_embeddings.npz \
        --output_dir results/clustering_analysis \
        --method kmeans \
        --n_clusters 8 \
        --standardize

    # Compare multiple clustering methods
    python scripts/analysis/clustering_analysis.py \
        --reduced_embeddings results/dimension_reduction/umap_reduction_data.npz \
        --output_dir results/clustering_analysis \
        --method all \
        --compare_methods
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
from sklearn.cluster import HDBSCAN, KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from scipy.ndimage import gaussian_filter
import pandas as pd

class ClusteringAnalyzer:
    """Comprehensive clustering analysis for embeddings (reduced or original latent)."""
    
    def __init__(self, embeddings_path: Path, use_latent: bool = False, standardize: bool = False):
        """Initialize analyzer with embeddings."""
        self.embeddings_path = embeddings_path
        self.use_latent = use_latent
        self.standardize = standardize
        self.embeddings = None
        self.original_embeddings = None  # For storing original when using reduced
        self.spatial_coords = None
        self.metadata = None
        self.data_type = None  # 'latent' or 'reduced'
        self.reduction_method = None
        
    def load_embeddings(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load embeddings from file (either reduced or original latent)."""
        print(f"Loading embeddings from: {self.embeddings_path}")
        
        if self.use_latent:
            # Load original latent embeddings
            self.embeddings, self.spatial_coords, self.metadata = self._load_latent_embeddings()
            self.data_type = "latent"
        else:
            # Load reduced embeddings (UMAP/PCA)
            self.embeddings, self.spatial_coords, self.metadata = self._load_reduced_embeddings()
            self.data_type = "reduced"
        
        print(f"✓ Loaded {self.data_type} embeddings: {self.embeddings.shape}")
        print(f"  Embedding range: [{self.embeddings.min():.3f}, {self.embeddings.max():.3f}]")
        
        # Optional standardization (especially useful for high-dimensional latent embeddings)
        if self.standardize:
            print("Standardizing embeddings...")
            scaler = StandardScaler()
            self.embeddings = scaler.fit_transform(self.embeddings)
            print(f"✓ Embeddings standardized")
        
        if self.reduction_method:
            print(f"  Reduction method: {self.reduction_method}")
        
        if self.original_embeddings is not None:
            print(f"  Original embeddings: {self.original_embeddings.shape}")
        
        if self.spatial_coords is not None:
            print(f"  Spatial coordinates: {self.spatial_coords.shape}")
        
        if self.metadata is not None:
            print(f"  Metadata keys: {list(self.metadata.keys())}")
        
        return self.embeddings, self.spatial_coords, self.metadata
    
    def _load_latent_embeddings(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load original latent embeddings from autoencoder output."""
        if self.embeddings_path.suffix == '.npz':
            # NumPy format
            data = np.load(self.embeddings_path, allow_pickle=True)
            embeddings = data['embeddings']
            spatial_coords = data.get('spatial_coordinates', None)
            
            # Load metadata if available
            if 'metadata' in data:
                try:
                    metadata_str = str(data['metadata'].item())
                    metadata = json.loads(metadata_str)
                except:
                    metadata = None
            else:
                metadata = None
                
        elif self.embeddings_path.suffix == '.h5':
            # HDF5 format
            with h5py.File(self.embeddings_path, 'r') as f:
                embeddings = f['embeddings'][:]
                spatial_coords = f.get('spatial_coordinates', None)
                if spatial_coords is not None:
                    spatial_coords = spatial_coords[:]
                
                # Load metadata from attributes
                metadata = {}
                for key, value in f.attrs.items():
                    try:
                        metadata[key] = json.loads(value)
                    except:
                        metadata[key] = value
                        
        elif self.embeddings_path.suffix == '.pt':
            # PyTorch format (legacy)
            import torch
            embeddings = torch.load(self.embeddings_path, map_location='cpu').numpy()
            
            # Try to load spatial coordinates
            coord_path = self.embeddings_path.with_name(self.embeddings_path.stem + '_coords.npy')
            if coord_path.exists():
                spatial_coords = np.load(coord_path)
            else:
                spatial_coords = None
            
            # Try to load metadata
            meta_path = self.embeddings_path.with_name(self.embeddings_path.stem + '_metadata.json') 
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = None
        else:
            raise ValueError(f"Unsupported embedding format: {self.embeddings_path.suffix}")
        
        return embeddings, spatial_coords, metadata
    
    def _load_reduced_embeddings(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load dimension-reduced embeddings (UMAP/PCA)."""
        if self.embeddings_path.suffix == '.npz':
            data = np.load(self.embeddings_path, allow_pickle=True)
            
            # Try different possible keys for reduced embeddings
            if 'reduced_embedding' in data:
                embeddings = data['reduced_embedding']
            elif 'umap_embedding' in data:
                embeddings = data['umap_embedding']
            elif 'pca_embedding' in data:
                embeddings = data['pca_embedding']
            else:
                raise ValueError("No recognized reduced embedding found in file")
            
            # Store original embeddings if available
            self.original_embeddings = data.get('original_embeddings', None)
            spatial_coords = data.get('spatial_coordinates', None)
            self.reduction_method = str(data.get('method', 'unknown'))
            
            # Load metadata if available
            if 'metadata' in data:
                try:
                    metadata = data['metadata'].item()
                except:
                    metadata = None
            else:
                metadata = None
                
        elif self.embeddings_path.suffix == '.h5':
            # HDF5 format
            with h5py.File(self.embeddings_path, 'r') as f:
                embeddings = f['reduced_embedding'][:]
                self.original_embeddings = f.get('original_embeddings', None)
                if self.original_embeddings is not None:
                    self.original_embeddings = self.original_embeddings[:]
                spatial_coords = f.get('spatial_coordinates', None)
                if spatial_coords is not None:
                    spatial_coords = spatial_coords[:]
                
                # Load metadata from attributes
                metadata = {}
                for key, value in f.attrs.items():
                    try:
                        metadata[key] = json.loads(value)
                    except:
                        metadata[key] = value
        else:
            raise ValueError(f"Unsupported reduced embedding format: {self.embeddings_path.suffix}")
        
        return embeddings, spatial_coords, metadata
    
    def perform_hdbscan_clustering(self, min_cluster_size: int = 50, min_samples: int = 5,
                                 cluster_selection_epsilon: float = 0.0) -> Tuple[np.ndarray, Dict]:
        """Perform HDBSCAN clustering."""
        print(f"Performing HDBSCAN clustering...")
        print(f"  min_cluster_size: {min_cluster_size}")
        print(f"  min_samples: {min_samples}")
        print(f"  cluster_selection_epsilon: {cluster_selection_epsilon}")
        
        clusterer = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon
        )
        cluster_labels = clusterer.fit_predict(self.embeddings)
        
        # Compute metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        metrics = self._compute_clustering_metrics(cluster_labels)
        
        results = {
            'method': 'hdbscan',
            'parameters': {
                'min_cluster_size': min_cluster_size,
                'min_samples': min_samples,
                'cluster_selection_epsilon': cluster_selection_epsilon
            },
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'metrics': metrics,
            'clusterer': clusterer
        }
        
        print(f"✓ HDBSCAN clustering completed:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
        
        return cluster_labels, results
    
    def perform_kmeans_clustering(self, n_clusters: int = 5, random_state: int = 42,
                                n_init: int = 10) -> Tuple[np.ndarray, Dict]:
        """Perform K-means clustering."""
        print(f"Performing K-means clustering...")
        print(f"  n_clusters: {n_clusters}")
        print(f"  n_init: {n_init}")
        
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        cluster_labels = clusterer.fit_predict(self.embeddings)
        
        # Compute metrics
        metrics = self._compute_clustering_metrics(cluster_labels)
        
        results = {
            'method': 'kmeans',
            'parameters': {
                'n_clusters': n_clusters,
                'random_state': random_state,
                'n_init': n_init
            },
            'n_clusters': n_clusters,
            'n_noise': 0,
            'metrics': metrics,
            'clusterer': clusterer
        }
        
        print(f"✓ K-means clustering completed:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
        print(f"  Inertia: {clusterer.inertia_:.2f}")
        
        return cluster_labels, results
    
    def perform_dbscan_clustering(self, eps: float = 0.5, min_samples: int = 5) -> Tuple[np.ndarray, Dict]:
        """Perform DBSCAN clustering."""
        print(f"Performing DBSCAN clustering...")
        print(f"  eps: {eps}")
        print(f"  min_samples: {min_samples}")
        
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = clusterer.fit_predict(self.embeddings)
        
        # Compute metrics
        n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
        n_noise = list(cluster_labels).count(-1)
        
        metrics = self._compute_clustering_metrics(cluster_labels)
        
        results = {
            'method': 'dbscan',
            'parameters': {
                'eps': eps,
                'min_samples': min_samples
            },
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'metrics': metrics,
            'clusterer': clusterer
        }
        
        print(f"✓ DBSCAN clustering completed:")
        print(f"  Number of clusters: {n_clusters}")
        print(f"  Number of noise points: {n_noise}")
        print(f"  Silhouette score: {metrics.get('silhouette_score', 'N/A')}")
        
        return cluster_labels, results
    
    def perform_gaussian_mixture_clustering(self, n_components: int = 5, 
                                          random_state: int = 42) -> Tuple[np.ndarray, Dict]:
        """Perform Gaussian Mixture Model clustering."""
        print(f"Performing Gaussian Mixture clustering...")
        print(f"  n_components: {n_components}")
        
        clusterer = GaussianMixture(n_components=n_components, random_state=random_state)
        clusterer.fit(self.embeddings)
        cluster_labels = clusterer.predict(self.embeddings)
        
        # Compute metrics
        metrics = self._compute_clustering_metrics(cluster_labels)
        
        results = {
            'method': 'gaussian_mixture',
            'parameters': {
                'n_components': n_components,
                'random_state': random_state
            },
            'n_clusters': n_components,
            'n_noise': 0,
            'metrics': metrics,
            'clusterer': clusterer,
            'bic': clusterer.bic(self.embeddings),
            'aic': clusterer.aic(self.embeddings)
        }
        
        print(f"✓ Gaussian Mixture clustering completed:")
        print(f"  Number of components: {n_components}")
        print(f"  Silhouette score: {metrics['silhouette_score']:.3f}")
        print(f"  BIC: {results['bic']:.2f}")
        
        return cluster_labels, results
    
    def _compute_clustering_metrics(self, cluster_labels: np.ndarray) -> Dict:
        """Compute comprehensive clustering evaluation metrics."""
        metrics = {}
        
        # Only compute metrics if we have valid clusters
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters > 1:
            # Remove noise points for silhouette score calculation
            valid_mask = cluster_labels != -1
            if np.sum(valid_mask) > 1:
                try:
                    metrics['silhouette_score'] = silhouette_score(
                        self.embeddings[valid_mask], cluster_labels[valid_mask])
                except:
                    metrics['silhouette_score'] = None
                    
                try:
                    metrics['calinski_harabasz_score'] = calinski_harabasz_score(
                        self.embeddings[valid_mask], cluster_labels[valid_mask])
                except:
                    metrics['calinski_harabasz_score'] = None
                    
                try:
                    metrics['davies_bouldin_score'] = davies_bouldin_score(
                        self.embeddings[valid_mask], cluster_labels[valid_mask])
                except:
                    metrics['davies_bouldin_score'] = None
        
        return metrics
    
    def compare_clustering_methods(self, output_dir: Path) -> Dict:
        """Compare multiple clustering methods and find the best one."""
        print("Comparing clustering methods...")
        
        methods_to_test = [
            ('hdbscan', {'min_cluster_size': 50}),
            ('kmeans', {'n_clusters': 5}),
            ('kmeans', {'n_clusters': 8}),
            ('dbscan', {'eps': 0.5}),
            ('gaussian_mixture', {'n_components': 5})
        ]
        
        comparison_results = []
        all_results = {}
        
        for method, params in tqdm(methods_to_test, desc="Testing methods"):
            try:
                if method == 'hdbscan':
                    labels, results = self.perform_hdbscan_clustering(**params)
                elif method == 'kmeans':
                    labels, results = self.perform_kmeans_clustering(**params)
                elif method == 'dbscan':
                    labels, results = self.perform_dbscan_clustering(**params)
                elif method == 'gaussian_mixture':
                    labels, results = self.perform_gaussian_mixture_clustering(**params)
                
                # Store results
                method_key = f"{method}_{hash(str(params))}"
                all_results[method_key] = {
                    'labels': labels,
                    'results': results
                }
                
                # Add to comparison
                comparison_entry = {
                    'method': method,
                    'parameters': str(params),
                    'n_clusters': results['n_clusters'],
                    'n_noise': results['n_noise'],
                    'silhouette_score': results['metrics'].get('silhouette_score', None),
                    'calinski_harabasz_score': results['metrics'].get('calinski_harabasz_score', None),
                    'davies_bouldin_score': results['metrics'].get('davies_bouldin_score', None)
                }
                
                # Add method-specific metrics
                if 'bic' in results:
                    comparison_entry['bic'] = results['bic']
                if 'aic' in results:
                    comparison_entry['aic'] = results['aic']
                
                comparison_results.append(comparison_entry)
                
            except Exception as e:
                print(f"Failed for {method} with {params}: {e}")
                continue
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(comparison_results)
        comparison_df.to_csv(output_dir / 'clustering_method_comparison.csv', index=False)
        
        # Find best method based on silhouette score
        valid_scores = comparison_df.dropna(subset=['silhouette_score'])
        if not valid_scores.empty:
            best_idx = valid_scores['silhouette_score'].idxmax()
            best_method = valid_scores.loc[best_idx]
            
            print(f"✓ Best clustering method:")
            print(f"  Method: {best_method['method']}")
            print(f"  Parameters: {best_method['parameters']}")
            print(f"  Silhouette score: {best_method['silhouette_score']:.3f}")
            print(f"  Number of clusters: {best_method['n_clusters']}")
        
        # Create comparison visualization
        self._create_method_comparison_plots(comparison_df, all_results, output_dir)
        
        return all_results
    
    def _create_method_comparison_plots(self, comparison_df: pd.DataFrame, 
                                      all_results: Dict, output_dir: Path):
        """Create visualization comparing different clustering methods."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Clustering Method Comparison', fontsize=16, fontweight='bold')
        
        # Plot 1: Silhouette scores
        ax = axes[0, 0]
        valid_scores = comparison_df.dropna(subset=['silhouette_score'])
        if not valid_scores.empty:
            bars = ax.bar(range(len(valid_scores)), valid_scores['silhouette_score'])
            ax.set_title('Silhouette Scores')
            ax.set_ylabel('Silhouette Score')
            ax.set_xticks(range(len(valid_scores)))
            ax.set_xticklabels([f"{row['method']}\n{row['parameters'][:20]}..." 
                               for _, row in valid_scores.iterrows()], 
                              rotation=45, ha='right', fontsize=8)
            
            # Highlight best method
            best_idx = valid_scores['silhouette_score'].idxmax()
            bars[best_idx].set_color('red')
        
        # Plot 2: Number of clusters
        ax = axes[0, 1]
        ax.bar(range(len(comparison_df)), comparison_df['n_clusters'])
        ax.set_title('Number of Clusters')
        ax.set_ylabel('Number of Clusters')
        ax.set_xticks(range(len(comparison_df)))
        ax.set_xticklabels([f"{row['method']}" for _, row in comparison_df.iterrows()], 
                          rotation=45, ha='right', fontsize=8)
        
        # Plot 3: Davies-Bouldin scores (lower is better)
        ax = axes[0, 2]
        valid_db = comparison_df.dropna(subset=['davies_bouldin_score'])
        if not valid_db.empty:
            ax.bar(range(len(valid_db)), valid_db['davies_bouldin_score'])
            ax.set_title('Davies-Bouldin Scores (lower = better)')
            ax.set_ylabel('Davies-Bouldin Score')
            ax.set_xticks(range(len(valid_db)))
            ax.set_xticklabels([f"{row['method']}" for _, row in valid_db.iterrows()], 
                              rotation=45, ha='right', fontsize=8)
        
        # Plot 4-6: Show best clustering results
        best_methods = comparison_df.nlargest(3, 'silhouette_score').dropna(subset=['silhouette_score'])
        
        for i, (_, method_info) in enumerate(best_methods.iterrows()):
            if i >= 3:
                break
                
            ax = axes[1, i]
            
            # Find corresponding results
            method_key = None
            for key, result in all_results.items():
                if (result['results']['method'] == method_info['method'] and
                    str(result['results']['parameters']) == method_info['parameters']):
                    method_key = key
                    break
            
            if method_key:
                labels = all_results[method_key]['labels']
                unique_labels = np.unique(labels)
                
                # Plot clustering result
                for label in unique_labels:
                    mask = labels == label
                    if label == -1:
                        ax.scatter(self.reduced_embeddings[mask, 0], 
                                 self.reduced_embeddings[mask, 1], 
                                 c='gray', s=1, alpha=0.3, label='Noise')
                    else:
                        ax.scatter(self.reduced_embeddings[mask, 0], 
                                 self.reduced_embeddings[mask, 1], 
                                 s=2, alpha=0.7, label=f'C{label}')
                
                ax.set_title(f'{method_info["method"].title()}\n'
                           f'Silhouette: {method_info["silhouette_score"]:.3f}')
                ax.set_xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.title()} 1')
                ax.set_ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.title()} 2')
                
                # Only show legend if few clusters
                if len(unique_labels) <= 8:
                    ax.legend(fontsize=6, markerscale=0.5)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'clustering_method_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def create_detailed_cluster_analysis(self, cluster_labels: np.ndarray, 
                                       clustering_results: Dict, output_dir: Path) -> None:
        """Create detailed cluster analysis visualizations."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # For high-dimensional data, create a 2D representation for visualization
        embeddings_2d = self.embeddings
        if self.embeddings.shape[1] > 2:
            print("Creating 2D representation for visualization using PCA...")
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings)
            print(f"✓ PCA for visualization: explained variance = {pca.explained_variance_ratio_.sum():.3f}")
        
        # Set up plotting style
        plt.style.use('default')
        sns.set_palette("husl")
        
        unique_labels = np.unique(cluster_labels)
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        
        if n_clusters > 1:
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Detailed Cluster Analysis - {clustering_results["method"].title()}', 
                        fontsize=16, fontweight='bold')
            
            # Plot 1: Main clustering result
            ax = axes[0, 0]
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = cluster_labels == label
                if label == -1:
                    ax.scatter(embeddings_2d[mask, 0], 
                             embeddings_2d[mask, 1], 
                             c='gray', s=1, alpha=0.3, label='Noise')
                else:
                    ax.scatter(embeddings_2d[mask, 0], 
                             embeddings_2d[mask, 1], 
                             c=[color], s=2, alpha=0.7, label=f'C{label}')
            
            ax.set_title('Clustering Results')
            ax.set_xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.title()} 1')
            ax.set_ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.title()} 2')
            
            # Smart legend handling
            if n_clusters <= 8:
                ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
            
            # Plot 2: Cluster sizes
            ax = axes[0, 1]
            valid_labels = [label for label in unique_labels if label != -1]
            cluster_sizes = [np.sum(cluster_labels == label) for label in valid_labels]
            
            bars = ax.bar(range(len(cluster_sizes)), cluster_sizes, alpha=0.7)
            ax.set_title('Cluster Sizes')
            ax.set_ylabel('Number of Points')
            ax.set_xlabel('Cluster')
            ax.set_xticks([])
            
            # Label only the largest clusters
            if cluster_sizes:
                max_size = max(cluster_sizes)
                for i, (bar, size, label) in enumerate(zip(bars, cluster_sizes, valid_labels)):
                    if size > max_size * 0.1:  # Only label clusters with >10% of max size
                        ax.text(bar.get_x() + bar.get_width()/2, 
                               bar.get_height() + max_size*0.01,
                               f'C{label}', ha='center', va='bottom', fontweight='bold')
            
            # Plot 3: Silhouette analysis
            ax = axes[0, 2]
            if clustering_results['metrics'].get('silhouette_score') is not None:
                from sklearn.metrics import silhouette_samples
                silhouette_avg = clustering_results['metrics']['silhouette_score']
                
                valid_mask = cluster_labels != -1
                if np.sum(valid_mask) > 1:
                    sample_silhouette_values = silhouette_samples(
                        self.embeddings[valid_mask], cluster_labels[valid_mask])
                    
                    y_lower = 10
                    valid_labels_for_silhouette = [label for label in unique_labels if label != -1]
                    
                    for i, label in enumerate(valid_labels_for_silhouette):
                        cluster_silhouette_values = sample_silhouette_values[
                            cluster_labels[valid_mask] == label]
                        cluster_silhouette_values.sort()
                        
                        size_cluster_i = cluster_silhouette_values.shape[0]
                        y_upper = y_lower + size_cluster_i
                        
                        color = plt.cm.Set1(i / len(valid_labels_for_silhouette))
                        ax.fill_betweenx(np.arange(y_lower, y_upper),
                                        0, cluster_silhouette_values,
                                        facecolor=color, edgecolor=color, alpha=0.7)
                        
                        y_lower = y_upper + 10
                    
                    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
                              label=f'Average: {silhouette_avg:.3f}')
                    ax.set_title('Silhouette Analysis')
                    ax.set_xlabel('Silhouette Coefficient')
                    ax.legend()
            
            # Plot 4: Spatial distribution (if available)
            ax = axes[1, 0]
            if self.spatial_coords is not None:
                scatter = ax.scatter(self.spatial_coords[:, 0], self.spatial_coords[:, 1], 
                                   c=cluster_labels, cmap='Set1', s=4, alpha=0.8)
                ax.set_title('Spatial Distribution of Clusters')
                ax.set_xlabel('Scan X Position')
                ax.set_ylabel('Scan Y Position')
                plt.colorbar(scatter, ax=ax, label='Cluster ID')
            else:
                ax.text(0.5, 0.5, 'No spatial coordinates\navailable', 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title('Spatial Information N/A')
            
            # Plot 5: Cluster centers and distances
            ax = axes[1, 1]
            if n_clusters > 1:
                # Compute cluster centers
                centers = []
                valid_labels = []
                for label in unique_labels:
                    if label != -1:
                        mask = cluster_labels == label
                        center = np.mean(embeddings_2d[mask], axis=0)
                        centers.append(center)
                        valid_labels.append(label)
                
                centers = np.array(centers)
                
                # Plot cluster points and centers
                for i, label in enumerate(valid_labels):
                    mask = cluster_labels == label
                    cluster_points = embeddings_2d[mask]
                    
                    ax.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                              alpha=0.6, s=2, label=f'C{label}')
                    ax.scatter(centers[i, 0], centers[i, 1], marker='x', s=100, 
                              color='black', linewidth=3)
                
                ax.set_title('Cluster Centers and Boundaries')
                ax.set_xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.title()} 1')
                ax.set_ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.title()} 2')
                
                if len(valid_labels) <= 8:
                    ax.legend(fontsize=8, markerscale=0.5)
            
            # Plot 6: Inter-cluster distance matrix
            ax = axes[1, 2]
            if n_clusters > 1 and n_clusters <= 20:  # Only for reasonable number of clusters
                # Compute cluster centers
                centers = []
                valid_labels = []
                for label in unique_labels:
                    if label != -1:
                        mask = cluster_labels == label
                        center = np.mean(embeddings_2d[mask], axis=0)
                        centers.append(center)
                        valid_labels.append(label)
                
                if len(centers) > 1:
                    centers = np.array(centers)
                    distances = squareform(pdist(centers))
                    
                    im = ax.imshow(distances, cmap='viridis')
                    ax.set_title('Inter-cluster Distances')
                    ax.set_xticks(range(len(valid_labels)))
                    ax.set_yticks(range(len(valid_labels)))
                    ax.set_xticklabels([f'C{label}' for label in valid_labels])
                    ax.set_yticklabels([f'C{label}' for label in valid_labels])
                    plt.colorbar(im, ax=ax, label='Distance')
                    
                    # Add distance values for small matrices
                    if len(valid_labels) <= 8:
                        for i in range(len(valid_labels)):
                            for j in range(len(valid_labels)):
                                color = 'white' if distances[i, j] > distances.max()/2 else 'black'
                                ax.text(j, i, f'{distances[i, j]:.2f}', 
                                       ha='center', va='center', color=color, fontsize=8)
            
            plt.tight_layout()
            plt.savefig(output_dir / 'detailed_cluster_analysis.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        print(f"✓ Detailed cluster analysis saved to: {output_dir}")
    
    def save_results(self, cluster_labels: np.ndarray, clustering_results: Dict,
                    output_dir: Path) -> None:
        """Save clustering analysis results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create JSON-serializable version of clustering results
        serializable_clustering_results = {}
        for key, value in clustering_results.items():
            if key == 'clusterer':
                # Skip the clusterer object as it's not JSON serializable
                continue
            elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                serializable_clustering_results[key] = value
            elif hasattr(value, 'tolist'):
                # Convert numpy arrays to lists
                serializable_clustering_results[key] = value.tolist()
            else:
                # Convert other objects to string representation
                serializable_clustering_results[key] = str(value)
        
        # Save clustering results
        results = {
            'embeddings': self.embeddings.tolist(),
            'original_embeddings': self.original_embeddings.tolist() if self.original_embeddings is not None else None,
            'spatial_coordinates': self.spatial_coords.tolist() if self.spatial_coords is not None else None,
            'cluster_labels': cluster_labels.tolist(),
            'clustering_results': serializable_clustering_results,
            'data_type': self.data_type,
            'reduction_method': self.reduction_method,
            'metadata': self.metadata
        }
        
        with open(output_dir / 'clustering_analysis_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Save as numpy arrays for easy loading
        np.savez(output_dir / 'clustering_analysis_data.npz',
                embeddings=self.embeddings,
                original_embeddings=self.original_embeddings,
                spatial_coordinates=self.spatial_coords,
                cluster_labels=cluster_labels)
        
        print(f"✓ Results saved to: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Clustering Analysis for Embeddings (reduced or latent)")
    
    # Data input (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--reduced_embeddings", type=Path,
                           help="Path to dimension-reduced embeddings (.npz, .h5)")
    data_group.add_argument("--latent_embeddings", type=Path,
                           help="Path to original latent embeddings (.npz, .h5, .pt)")
    
    parser.add_argument("--output_dir", type=Path, default="clustering_analysis",
                       help="Output directory for results and plots")
    
    # Clustering method selection
    parser.add_argument("--method", type=str, default="hdbscan", 
                       choices=["hdbscan", "kmeans", "dbscan", "gaussian_mixture", "all"],
                       help="Clustering method to use")
    
    # HDBSCAN parameters
    parser.add_argument("--min_cluster_size", type=int, default=50,
                       help="Minimum cluster size for HDBSCAN")
    parser.add_argument("--min_samples", type=int, default=5,
                       help="Minimum samples for HDBSCAN")
    
    # K-means parameters
    parser.add_argument("--n_clusters", type=int, default=5,
                       help="Number of clusters for K-means")
    
    # DBSCAN parameters
    parser.add_argument("--eps", type=float, default=0.5,
                       help="Epsilon parameter for DBSCAN")
    
    # Gaussian Mixture parameters
    parser.add_argument("--n_components", type=int, default=5,
                       help="Number of components for Gaussian Mixture")
    
    # Analysis options
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize embeddings before clustering (recommended for latent embeddings)")
    parser.add_argument("--compare_methods", action="store_true",
                       help="Compare multiple clustering methods")
    parser.add_argument("--random_state", type=int, default=42,
                       help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    # Handle input arguments and determine data type
    if args.latent_embeddings:
        data_path = args.latent_embeddings
        use_latent = True
        data_type_str = "Latent embeddings"
    elif args.reduced_embeddings:
        data_path = args.reduced_embeddings
        use_latent = False
        data_type_str = "Reduced embeddings"
    else:
        raise ValueError("Must specify either --latent_embeddings or --reduced_embeddings")
    
    # Validate inputs
    if not data_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {data_path}")
    
    print("="*80)
    print("CLUSTERING ANALYSIS")
    print("="*80)
    print(f"{data_type_str}: {data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Method: {args.method}")
    if args.standardize:
        print("Standardization: ENABLED")
    if args.compare_methods:
        print("Method comparison: ENABLED")
    print("="*80)
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize analyzer
    analyzer = ClusteringAnalyzer(data_path, use_latent, args.standardize)
    
    # Load embeddings
    embeddings, spatial_coords, metadata = analyzer.load_embeddings()
    
    # Perform clustering analysis
    if args.method == "all" or args.compare_methods:
        # Compare multiple methods
        all_results = analyzer.compare_clustering_methods(args.output_dir)
        
        # Use best method for detailed analysis
        best_method_key = max(all_results.keys(), 
                             key=lambda k: all_results[k]['results']['metrics'].get('silhouette_score', -1))
        cluster_labels = all_results[best_method_key]['labels']
        clustering_results = all_results[best_method_key]['results']
        
    else:
        # Single method analysis
        if args.method == "hdbscan":
            cluster_labels, clustering_results = analyzer.perform_hdbscan_clustering(
                min_cluster_size=args.min_cluster_size,
                min_samples=args.min_samples
            )
        elif args.method == "kmeans":
            cluster_labels, clustering_results = analyzer.perform_kmeans_clustering(
                n_clusters=args.n_clusters,
                random_state=args.random_state
            )
        elif args.method == "dbscan":
            cluster_labels, clustering_results = analyzer.perform_dbscan_clustering(
                eps=args.eps,
                min_samples=args.min_samples
            )
        elif args.method == "gaussian_mixture":
            cluster_labels, clustering_results = analyzer.perform_gaussian_mixture_clustering(
                n_components=args.n_components,
                random_state=args.random_state
            )
    
    # Create detailed cluster analysis
    analyzer.create_detailed_cluster_analysis(cluster_labels, clustering_results, args.output_dir)
    
    # Save results
    analyzer.save_results(cluster_labels, clustering_results, args.output_dir)
    
    print("\n" + "="*80)
    print("CLUSTERING ANALYSIS COMPLETED SUCCESSFULLY")
    print("="*80)
    print(f"Results saved to: {args.output_dir}")
    print(f"Data type: {data_type_str}")
    print(f"Method used: {clustering_results['method']}")
    print(f"Number of clusters: {clustering_results['n_clusters']}")
    if clustering_results['n_noise'] > 0:
        print(f"Noise points: {clustering_results['n_noise']}")
    
    silhouette = clustering_results['metrics'].get('silhouette_score')
    if silhouette is not None:
        print(f"Silhouette score: {silhouette:.3f}")
    
    print("\nGenerated files:")
    print("  - detailed_cluster_analysis.png: Main cluster visualization")
    if args.compare_methods or args.method == "all":
        print("  - clustering_method_comparison.png: Method comparison")
        print("  - clustering_method_comparison.csv: Comparison data")
    print("  - clustering_analysis_results.json: Full results")
    print("  - clustering_analysis_data.npz: Numpy data arrays")
    print("="*80)

if __name__ == "__main__":
    main()