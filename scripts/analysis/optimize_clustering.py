#!/usr/bin/env python3
"""
Clustering Parameter Optimization for Embeddings

Systematically tests clustering parameters for K-means, Spectral clustering, and HDBSCAN to find
optimal settings based on multiple metrics (silhouette score, calinski-harabasz, etc.).
Works with both dimension-reduced embeddings (UMAP/PCA) and original latent embeddings.

Usage:
    # For UMAP/PCA reduced embeddings
    python scripts/analysis/optimize_clustering.py \
        --reduced_data workspace/proj_data_ds4_l32/visualisations/umap_analysis_nn_50_md_pointzero1/umap_data.npz \
        --output_dir workspace/proj_data_ds4_l32/visualisations/clustering_optimization \
        --max_clusters 20 \
        --subsample 10000

    # For original latent embeddings (high-dimensional)
    python scripts/analysis/optimize_clustering.py \
        --latent_embeddings workspace/proj_data_ds4_l32/outputs/embeddings/ds4_4epoch_embeddings.npz \
        --output_dir workspace/proj_data_ds4_l32/visualisations/clustering_optimization \
        --max_clusters 20 \
        --subsample 10000 \
        --standardize
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
import h5py
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Clustering algorithms
from sklearn.cluster import KMeans, HDBSCAN, SpectralClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

class ClusteringOptimizer:
    """Optimize clustering parameters for embeddings (reduced or original latent)."""
    
    def __init__(self, data_path: Path, subsample: Optional[int] = None, 
                 use_latent: bool = False, standardize: bool = False):
        """Initialize with embedding data."""
        self.data_path = data_path
        self.subsample = subsample
        self.use_latent = use_latent
        self.standardize = standardize
        self.embeddings = None
        self.spatial_coords = None
        self.metadata = None
        self.data_type = None
        self.results = {}
        
    def load_embeddings(self) -> np.ndarray:
        """Load embeddings from file (either reduced or original latent)."""
        print(f"Loading embeddings from: {self.data_path}")
        
        if self.use_latent:
            # Load original latent embeddings
            self.embeddings, self.spatial_coords, self.metadata = self._load_latent_embeddings()
            self.data_type = "latent"
        else:
            # Load reduced embeddings (UMAP/PCA)
            self.embeddings, self.spatial_coords, self.metadata = self._load_reduced_embeddings()
            self.data_type = "reduced"
        
        print(f"✓ Loaded {self.data_type} embeddings: {self.embeddings.shape}")
        
        # Optional standardization (especially useful for high-dimensional latent embeddings)
        if self.standardize:
            print("Standardizing embeddings...")
            scaler = StandardScaler()
            self.embeddings = scaler.fit_transform(self.embeddings)
            print(f"✓ Embeddings standardized")
        
        # Optional subsampling
        if self.subsample and self.subsample < len(self.embeddings):
            print(f"Subsampling {self.subsample} points from {len(self.embeddings)} total...")
            indices = np.random.choice(len(self.embeddings), self.subsample, replace=False)
            self.embeddings = self.embeddings[indices]
            if self.spatial_coords is not None:
                self.spatial_coords = self.spatial_coords[indices]
            print(f"✓ Using {len(self.embeddings)} subsampled points")
        
        return self.embeddings
    
    def _load_latent_embeddings(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load original latent embeddings from autoencoder output."""
        if self.data_path.suffix == '.npz':
            # NumPy format
            data = np.load(self.data_path, allow_pickle=True)
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
                
        elif self.data_path.suffix == '.h5':
            # HDF5 format
            with h5py.File(self.data_path, 'r') as f:
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
                        
        elif self.data_path.suffix == '.pt':
            # PyTorch format (legacy)
            import torch
            embeddings = torch.load(self.data_path, map_location='cpu').numpy()
            
            # Try to load spatial coordinates
            coord_path = self.data_path.with_name(self.data_path.stem + '_coords.npy')
            if coord_path.exists():
                spatial_coords = np.load(coord_path)
            else:
                spatial_coords = None
            
            # Try to load metadata
            meta_path = self.data_path.with_name(self.data_path.stem + '_metadata.json') 
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            else:
                metadata = None
        else:
            raise ValueError(f"Unsupported embedding format: {self.data_path.suffix}")
        
        return embeddings, spatial_coords, metadata
    
    def _load_reduced_embeddings(self) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[Dict]]:
        """Load dimension-reduced embeddings (UMAP/PCA)."""
        if self.data_path.suffix == '.npz':
            data = np.load(self.data_path, allow_pickle=True)
            
            # Try different possible keys for reduced embeddings
            if 'umap_embedding' in data:
                embeddings = data['umap_embedding']
            elif 'reduced_embedding' in data:
                embeddings = data['reduced_embedding']
            elif 'pca_embedding' in data:
                embeddings = data['pca_embedding']
            else:
                raise ValueError("No recognized reduced embedding found in file")
            
            spatial_coords = data.get('spatial_coordinates', None)
            
            # Load metadata if available
            if 'metadata' in data:
                try:
                    metadata = data['metadata'].item()
                except:
                    metadata = None
            else:
                metadata = None
        else:
            raise ValueError(f"Unsupported reduced embedding format: {self.data_path.suffix}")
        
        return embeddings, spatial_coords, metadata
    
    def optimize_kmeans(self, max_clusters: int = 20, clusters: List[int] = None, save_individual: bool = True, output_dir: Path = None) -> Dict:
        """Test K-means with different numbers of clusters."""
        if clusters is not None:
            cluster_list = clusters
            print(f"Optimizing K-means clustering for specified clusters: {cluster_list}...")
        else:
            cluster_list = list(range(2, max_clusters + 1))
            print(f"Optimizing K-means clustering (k=2 to {max_clusters})...")
        
        kmeans_results = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'inertia': []
        }
        
        # Create subdirectory for K-means results
        if save_individual and output_dir:
            kmeans_dir = output_dir / 'kmeans_individual_results'
            kmeans_dir.mkdir(parents=True, exist_ok=True)
        
        for k in tqdm(cluster_list, desc="K-means"):
            # Fit K-means
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            # Calculate metrics
            sil_score = silhouette_score(self.embeddings, labels)
            ch_score = calinski_harabasz_score(self.embeddings, labels)
            db_score = davies_bouldin_score(self.embeddings, labels)
            
            kmeans_results['n_clusters'].append(k)
            kmeans_results['silhouette'].append(sil_score)
            kmeans_results['calinski_harabasz'].append(ch_score)
            kmeans_results['davies_bouldin'].append(db_score)
            kmeans_results['inertia'].append(kmeans.inertia_)
            
            # Save individual results
            if save_individual and output_dir:
                self._save_individual_result('kmeans', k, labels, kmeans, 
                                           sil_score, ch_score, db_score, 
                                           kmeans_dir, extra_info={'inertia': kmeans.inertia_})
        
        # Find best parameters
        best_sil_idx = np.argmax(kmeans_results['silhouette'])
        best_ch_idx = np.argmax(kmeans_results['calinski_harabasz'])
        best_db_idx = np.argmin(kmeans_results['davies_bouldin'])  # Lower is better
        
        kmeans_summary = {
            'best_silhouette': {
                'n_clusters': kmeans_results['n_clusters'][best_sil_idx],
                'score': kmeans_results['silhouette'][best_sil_idx]
            },
            'best_calinski_harabasz': {
                'n_clusters': kmeans_results['n_clusters'][best_ch_idx],
                'score': kmeans_results['calinski_harabasz'][best_ch_idx]
            },
            'best_davies_bouldin': {
                'n_clusters': kmeans_results['n_clusters'][best_db_idx],
                'score': kmeans_results['davies_bouldin'][best_db_idx]
            }
        }
        
        self.results['kmeans'] = {
            'data': kmeans_results,
            'summary': kmeans_summary
        }
        
        print(f"✓ K-means optimization complete")
        print(f"  Best silhouette: k={kmeans_summary['best_silhouette']['n_clusters']} (score={kmeans_summary['best_silhouette']['score']:.3f})")
        
        return self.results['kmeans']
    
    def optimize_hdbscan(self, save_individual: bool = True, output_dir: Path = None) -> Dict:
        """Test HDBSCAN with different parameters."""
        print("Optimizing HDBSCAN clustering...")
        
        # Parameter grid
        min_cluster_sizes = [10, 20, 50, 100, 200, 500]
        min_samples_list = [5, 10, 20, 50]
        
        hdbscan_results = {
            'min_cluster_size': [],
            'min_samples': [],
            'n_clusters': [],
            'n_noise': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        # Create subdirectory for HDBSCAN results
        if save_individual and output_dir:
            hdbscan_dir = output_dir / 'hdbscan_individual_results'
            hdbscan_dir.mkdir(parents=True, exist_ok=True)
        
        best_silhouette = -1
        best_params = None
        
        total_combinations = len(min_cluster_sizes) * len(min_samples_list)
        
        with tqdm(total=total_combinations, desc="HDBSCAN") as pbar:
            for min_cluster_size in min_cluster_sizes:
                for min_samples in min_samples_list:
                    # Skip invalid combinations
                    if min_samples > min_cluster_size:
                        pbar.update(1)
                        continue
                    
                    # Fit HDBSCAN
                    clusterer = HDBSCAN(
                        min_cluster_size=min_cluster_size,
                        min_samples=min_samples,
                        cluster_selection_epsilon=0.0
                    )
                    labels = clusterer.fit_predict(self.embeddings)
                    
                    # Count clusters and noise
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    n_noise = list(labels).count(-1)
                    
                    # Skip if too few clusters or too much noise
                    if n_clusters < 2 or n_noise > len(labels) * 0.5:
                        hdbscan_results['min_cluster_size'].append(min_cluster_size)
                        hdbscan_results['min_samples'].append(min_samples)
                        hdbscan_results['n_clusters'].append(n_clusters)
                        hdbscan_results['n_noise'].append(n_noise)
                        hdbscan_results['silhouette'].append(-1)
                        hdbscan_results['calinski_harabasz'].append(0)
                        hdbscan_results['davies_bouldin'].append(999)
                        pbar.update(1)
                        continue
                    
                    # Calculate metrics
                    try:
                        sil_score = silhouette_score(self.embeddings, labels)
                        ch_score = calinski_harabasz_score(self.embeddings, labels)
                        db_score = davies_bouldin_score(self.embeddings, labels)
                    except:
                        sil_score = -1
                        ch_score = 0
                        db_score = 999
                    
                    hdbscan_results['min_cluster_size'].append(min_cluster_size)
                    hdbscan_results['min_samples'].append(min_samples)
                    hdbscan_results['n_clusters'].append(n_clusters)
                    hdbscan_results['n_noise'].append(n_noise)
                    hdbscan_results['silhouette'].append(sil_score)
                    hdbscan_results['calinski_harabasz'].append(ch_score)
                    hdbscan_results['davies_bouldin'].append(db_score)
                    
                    # Save individual results
                    if save_individual and output_dir:
                        param_name = f"mcs{min_cluster_size}_ms{min_samples}"
                        self._save_individual_result('hdbscan', param_name, labels, clusterer, 
                                                   sil_score, ch_score, db_score, hdbscan_dir,
                                                   extra_info={'n_clusters': n_clusters, 'n_noise': n_noise,
                                                             'min_cluster_size': min_cluster_size, 'min_samples': min_samples})
                    
                    # Track best silhouette
                    if sil_score > best_silhouette:
                        best_silhouette = sil_score
                        best_params = {
                            'min_cluster_size': min_cluster_size,
                            'min_samples': min_samples,
                            'n_clusters': n_clusters,
                            'silhouette': sil_score
                        }
                    
                    pbar.update(1)
        
        # Find best parameters for each metric
        valid_indices = [i for i, s in enumerate(hdbscan_results['silhouette']) if s > -1]
        
        if valid_indices:
            valid_sil = [hdbscan_results['silhouette'][i] for i in valid_indices]
            valid_ch = [hdbscan_results['calinski_harabasz'][i] for i in valid_indices]
            valid_db = [hdbscan_results['davies_bouldin'][i] for i in valid_indices]
            
            best_sil_idx = valid_indices[np.argmax(valid_sil)]
            best_ch_idx = valid_indices[np.argmax(valid_ch)]
            best_db_idx = valid_indices[np.argmin(valid_db)]
            
            hdbscan_summary = {
                'best_silhouette': {
                    'min_cluster_size': hdbscan_results['min_cluster_size'][best_sil_idx],
                    'min_samples': hdbscan_results['min_samples'][best_sil_idx],
                    'n_clusters': hdbscan_results['n_clusters'][best_sil_idx],
                    'score': hdbscan_results['silhouette'][best_sil_idx]
                },
                'best_calinski_harabasz': {
                    'min_cluster_size': hdbscan_results['min_cluster_size'][best_ch_idx],
                    'min_samples': hdbscan_results['min_samples'][best_ch_idx],
                    'n_clusters': hdbscan_results['n_clusters'][best_ch_idx],
                    'score': hdbscan_results['calinski_harabasz'][best_ch_idx]
                },
                'best_davies_bouldin': {
                    'min_cluster_size': hdbscan_results['min_cluster_size'][best_db_idx],
                    'min_samples': hdbscan_results['min_samples'][best_db_idx],
                    'n_clusters': hdbscan_results['n_clusters'][best_db_idx],
                    'score': hdbscan_results['davies_bouldin'][best_db_idx]
                }
            }
        else:
            hdbscan_summary = {'error': 'No valid clustering results found'}
        
        self.results['hdbscan'] = {
            'data': hdbscan_results,
            'summary': hdbscan_summary
        }
        
        print(f"✓ HDBSCAN optimization complete")
        if 'error' not in hdbscan_summary:
            best = hdbscan_summary['best_silhouette']
            print(f"  Best silhouette: min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']} (score={best['score']:.3f})")
        
        return self.results['hdbscan']
    
    def optimize_spectral(self, max_clusters: int = 20, clusters: List[int] = None, save_individual: bool = True, output_dir: Path = None) -> Dict:
        """Test Spectral clustering with different numbers of clusters."""
        if clusters is not None:
            cluster_list = clusters
            print(f"Optimizing Spectral clustering for specified clusters: {cluster_list}...")
        else:
            cluster_list = list(range(2, max_clusters + 1))
            print(f"Optimizing Spectral clustering (k=2 to {max_clusters})...")
        
        spectral_results = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': []
        }
        
        # Create subdirectory for Spectral results
        if save_individual and output_dir:
            spectral_dir = output_dir / 'spectral_individual_results'
            spectral_dir.mkdir(parents=True, exist_ok=True)
        
        for k in tqdm(cluster_list, desc="Spectral"):
            # Fit Spectral clustering
            spectral = SpectralClustering(n_clusters=k, random_state=42, n_init=10)
            labels = spectral.fit_predict(self.embeddings)
            
            # Calculate metrics
            sil_score = silhouette_score(self.embeddings, labels)
            ch_score = calinski_harabasz_score(self.embeddings, labels)
            db_score = davies_bouldin_score(self.embeddings, labels)
            
            spectral_results['n_clusters'].append(k)
            spectral_results['silhouette'].append(sil_score)
            spectral_results['calinski_harabasz'].append(ch_score)
            spectral_results['davies_bouldin'].append(db_score)
            
            # Save individual results
            if save_individual and output_dir:
                self._save_individual_result('spectral', k, labels, spectral, 
                                           sil_score, ch_score, db_score, spectral_dir)
        
        # Find best parameters
        best_sil_idx = np.argmax(spectral_results['silhouette'])
        best_ch_idx = np.argmax(spectral_results['calinski_harabasz'])
        best_db_idx = np.argmin(spectral_results['davies_bouldin'])  # Lower is better
        
        spectral_summary = {
            'best_silhouette': {
                'n_clusters': spectral_results['n_clusters'][best_sil_idx],
                'score': spectral_results['silhouette'][best_sil_idx]
            },
            'best_calinski_harabasz': {
                'n_clusters': spectral_results['n_clusters'][best_ch_idx],
                'score': spectral_results['calinski_harabasz'][best_ch_idx]
            },
            'best_davies_bouldin': {
                'n_clusters': spectral_results['n_clusters'][best_db_idx],
                'score': spectral_results['davies_bouldin'][best_db_idx]
            }
        }
        
        self.results['spectral'] = {
            'data': spectral_results,
            'summary': spectral_summary
        }
        
        print(f"✓ Spectral clustering optimization complete")
        print(f"  Best silhouette: k={spectral_summary['best_silhouette']['n_clusters']} (score={spectral_summary['best_silhouette']['score']:.3f})")
        
        return self.results['spectral']
    
    def _save_individual_result(self, method: str, param_identifier: str, labels: np.ndarray, 
                               clusterer, sil_score: float, ch_score: float, db_score: float,
                               save_dir: Path, extra_info: Dict = None) -> None:
        """Save individual clustering result with labels, metrics, and visualization."""
        
        # Create subdirectory for this specific parameter combination
        result_dir = save_dir / f"{method}_{param_identifier}"
        result_dir.mkdir(parents=True, exist_ok=True)
        
        # Save cluster labels
        labels_path = result_dir / 'cluster_labels.npy'
        np.save(labels_path, labels)
        
        # Save metrics
        metrics = {
            'silhouette_score': float(sil_score),
            'calinski_harabasz_score': float(ch_score), 
            'davies_bouldin_score': float(db_score),
            'n_points': len(labels),
            'unique_labels': labels.tolist() if len(np.unique(labels)) < 50 else f"{len(np.unique(labels))} unique labels"
        }
        
        if extra_info:
            metrics.update(extra_info)
            
        metrics_path = result_dir / 'metrics.json'
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Create visualization if embeddings are 2D or can be reduced to 2D
        self._create_individual_visualization(labels, result_dir, method, param_identifier, metrics)
        
        # Save spatial coordinates if available
        if self.spatial_coords is not None:
            spatial_path = result_dir / 'spatial_coordinates.npy'
            np.save(spatial_path, self.spatial_coords)
    
    def _create_individual_visualization(self, labels: np.ndarray, result_dir: Path, 
                                        method: str, param_identifier: str, metrics: Dict) -> None:
        """Create visualization for individual clustering result."""
        
        # For high-dimensional data, create a 2D representation
        embeddings_2d = self.embeddings
        if self.embeddings.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings)
            
        plt.figure(figsize=(10, 8))
        
        # Handle noise points for HDBSCAN
        unique_labels = np.unique(labels)
        
        if -1 in unique_labels:  # HDBSCAN with noise
            # Plot noise points first (in gray)
            noise_mask = labels == -1
            plt.scatter(embeddings_2d[noise_mask, 0], embeddings_2d[noise_mask, 1], 
                       c='gray', s=1, alpha=0.3, label='Noise')
            
            # Plot clusters
            cluster_labels = unique_labels[unique_labels != -1]
            colors = plt.cm.Set1(np.linspace(0, 1, len(cluster_labels)))
            
            for i, (label, color) in enumerate(zip(cluster_labels, colors)):
                mask = labels == label
                plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                           c=[color], s=2, alpha=0.7, label=f'C{label}')
        else:
            # Regular clustering without noise
            plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, 
                       cmap='Set1', s=2, alpha=0.7)
        
        plt.title(f'{method.title()} Clustering: {param_identifier}\\n'
                 f'Silhouette: {metrics["silhouette_score"]:.3f}, '
                 f'CH: {metrics["calinski_harabasz_score"]:.1f}, '
                 f'DB: {metrics["davies_bouldin_score"]:.3f}')
        
        plt.xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 1')
        plt.ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 2')
        
        # Only show legend if reasonable number of clusters
        n_clusters = len(unique_labels) - (1 if -1 in unique_labels else 0)
        if n_clusters <= 10:
            plt.legend(fontsize=8, markerscale=2)
            
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = result_dir / f'{method}_{param_identifier}_clustering.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def create_visualization(self, output_dir: Path) -> None:
        """Create comprehensive visualization of clustering optimization results."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # For high-dimensional data, create a 2D representation for visualization
        embeddings_2d = self.embeddings
        if self.embeddings.shape[1] > 2:
            print("Creating 2D representation for visualization using PCA...")
            pca = PCA(n_components=2, random_state=42)
            embeddings_2d = pca.fit_transform(self.embeddings)
            print(f"✓ PCA for visualization: explained variance = {pca.explained_variance_ratio_.sum():.3f}")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        # Create main figure - adjust size based on number of methods
        n_methods = sum([1 for method in ['kmeans', 'spectral', 'hdbscan'] if method in self.results])
        if n_methods == 3:
            fig = plt.figure(figsize=(18, 16))
        else:
            fig = plt.figure(figsize=(16, 12))
        fig.suptitle('Clustering Parameter Optimization Results', fontsize=16, fontweight='bold')
        
        # K-means results
        if 'kmeans' in self.results:
            kmeans_data = self.results['kmeans']['data']
            
            # Panel 1: K-means metrics vs number of clusters
            ax1 = plt.subplot(2, 3, 1)
            ax1.plot(kmeans_data['n_clusters'], kmeans_data['silhouette'], 'o-', label='Silhouette', linewidth=2)
            ax1.set_xlabel('Number of Clusters')
            ax1.set_ylabel('Silhouette Score')
            ax1.set_title('K-means: Silhouette Score')
            ax1.grid(True, alpha=0.3)
            
            # Highlight best
            best_k = self.results['kmeans']['summary']['best_silhouette']['n_clusters']
            best_score = self.results['kmeans']['summary']['best_silhouette']['score']
            ax1.axvline(best_k, color='red', linestyle='--', alpha=0.7)
            ax1.text(best_k, best_score, f'  Best: k={best_k}\n  Score={best_score:.3f}', 
                    verticalalignment='bottom', fontweight='bold')
            
            # Panel 2: K-means elbow method
            ax2 = plt.subplot(2, 3, 2)
            ax2.plot(kmeans_data['n_clusters'], kmeans_data['inertia'], 'o-', color='orange', linewidth=2)
            ax2.set_xlabel('Number of Clusters')
            ax2.set_ylabel('Inertia')
            ax2.set_title('K-means: Elbow Method')
            ax2.grid(True, alpha=0.3)
            
            # Panel 3: K-means multiple metrics
            ax3 = plt.subplot(2, 3, 3)
            # Normalize metrics to 0-1 for comparison
            sil_norm = np.array(kmeans_data['silhouette'])
            ch_norm = np.array(kmeans_data['calinski_harabasz']) / np.max(kmeans_data['calinski_harabasz'])
            db_norm = 1 - (np.array(kmeans_data['davies_bouldin']) / np.max(kmeans_data['davies_bouldin']))  # Invert DB (lower is better)
            
            ax3.plot(kmeans_data['n_clusters'], sil_norm, 'o-', label='Silhouette', linewidth=2)
            ax3.plot(kmeans_data['n_clusters'], ch_norm, 's-', label='Calinski-Harabasz', linewidth=2)
            ax3.plot(kmeans_data['n_clusters'], db_norm, '^-', label='Davies-Bouldin (inv)', linewidth=2)
            ax3.set_xlabel('Number of Clusters')
            ax3.set_ylabel('Normalized Score')
            ax3.set_title('K-means: All Metrics')
            ax3.legend(loc='best', fontsize=8)
            ax3.grid(True, alpha=0.3)
        
        # Spectral results
        if 'spectral' in self.results:
            spectral_data = self.results['spectral']['data']
            
            # Determine panel positions based on available methods
            base_row = 1 if 'kmeans' in self.results else 0
            
            # Panel 4: Spectral metrics vs number of clusters
            ax4 = plt.subplot(3 if 'spectral' in self.results else 2, 3, 4)
            ax4.plot(spectral_data['n_clusters'], spectral_data['silhouette'], 'o-', label='Silhouette', linewidth=2, color='green')
            ax4.set_xlabel('Number of Clusters')
            ax4.set_ylabel('Silhouette Score')
            ax4.set_title('Spectral: Silhouette Score')
            ax4.grid(True, alpha=0.3)
            
            # Highlight best
            best_k = self.results['spectral']['summary']['best_silhouette']['n_clusters']
            best_score = self.results['spectral']['summary']['best_silhouette']['score']
            ax4.axvline(best_k, color='red', linestyle='--', alpha=0.7)
            ax4.text(best_k, best_score, f'  Best: k={best_k}\n  Score={best_score:.3f}', 
                    verticalalignment='bottom', fontweight='bold')
            
            # Panel 5: Spectral multiple metrics
            ax5 = plt.subplot(3 if 'spectral' in self.results else 2, 3, 5)
            # Normalize metrics to 0-1 for comparison
            sil_norm = np.array(spectral_data['silhouette'])
            ch_norm = np.array(spectral_data['calinski_harabasz']) / np.max(spectral_data['calinski_harabasz'])
            db_norm = 1 - (np.array(spectral_data['davies_bouldin']) / np.max(spectral_data['davies_bouldin']))  # Invert DB (lower is better)
            
            ax5.plot(spectral_data['n_clusters'], sil_norm, 'o-', label='Silhouette', linewidth=2, color='green')
            ax5.plot(spectral_data['n_clusters'], ch_norm, 's-', label='Calinski-Harabasz', linewidth=2, color='purple')
            ax5.plot(spectral_data['n_clusters'], db_norm, '^-', label='Davies-Bouldin (inv)', linewidth=2, color='brown')
            ax5.set_xlabel('Number of Clusters')
            ax5.set_ylabel('Normalized Score')
            ax5.set_title('Spectral: All Metrics')
            ax5.legend(loc='best', fontsize=8)
            ax5.grid(True, alpha=0.3)
        
        # HDBSCAN results
        if 'hdbscan' in self.results and 'error' not in self.results['hdbscan']['summary']:
            hdbscan_data = self.results['hdbscan']['data']
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame(hdbscan_data)
            df_valid = df[df['silhouette'] > -1]  # Only valid results
            
            if len(df_valid) > 0:
                # Determine panel positions based on available methods
                panel_offset = 3 if 'spectral' in self.results else 0
                n_rows = 3 if 'spectral' in self.results else 2
                
                # HDBSCAN heatmap
                ax_hdb1 = plt.subplot(n_rows, 3, 4 + panel_offset)
                
                # Create pivot table for heatmap
                pivot_data = df_valid.groupby(['min_cluster_size', 'min_samples'])['silhouette'].mean().unstack()
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax_hdb1, cbar_kws={'label': 'Silhouette Score'})
                ax_hdb1.set_title('HDBSCAN: Silhouette Score Heatmap')
                ax_hdb1.set_xlabel('Min Samples')
                ax_hdb1.set_ylabel('Min Cluster Size')
                
                # HDBSCAN clusters vs parameters
                ax_hdb2 = plt.subplot(n_rows, 3, 5 + panel_offset)
                scatter = ax_hdb2.scatter(df_valid['min_cluster_size'], df_valid['min_samples'], 
                                    c=df_valid['n_clusters'], s=60, cmap='plasma', alpha=0.7)
                plt.colorbar(scatter, ax=ax_hdb2, label='Number of Clusters')
                ax_hdb2.set_xlabel('Min Cluster Size')
                ax_hdb2.set_ylabel('Min Samples')
                ax_hdb2.set_title('HDBSCAN: Number of Clusters')
                ax_hdb2.set_xscale('log')
                
                # Highlight best
                best_params = self.results['hdbscan']['summary']['best_silhouette']
                ax_hdb2.scatter(best_params['min_cluster_size'], best_params['min_samples'], 
                          s=200, c='red', marker='x', linewidths=3, label='Best')
                ax_hdb2.legend()
                
                # HDBSCAN noise vs parameters
                ax_hdb3 = plt.subplot(n_rows, 3, 6 + panel_offset)
                scatter = ax_hdb3.scatter(df_valid['min_cluster_size'], df_valid['min_samples'], 
                                    c=df_valid['n_noise'], s=60, cmap='Reds', alpha=0.7)
                plt.colorbar(scatter, ax=ax_hdb3, label='Noise Points')
                ax_hdb3.set_xlabel('Min Cluster Size')
                ax_hdb3.set_ylabel('Min Samples')
                ax_hdb3.set_title('HDBSCAN: Noise Points')
                ax_hdb3.set_xscale('log')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'clustering_optimization.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved optimization plot: {plot_path}")
        
        plt.close()
        
        # Create additional visualization showing best clustering results
        self._create_best_clustering_visualization(embeddings_2d, output_dir)
    
    def _create_best_clustering_visualization(self, embeddings_2d: np.ndarray, output_dir: Path) -> None:
        """Create visualization showing the best clustering results."""
        if not self.results:
            return
        
        # Count available methods to determine subplot layout
        n_methods = sum([1 for method in ['kmeans', 'spectral', 'hdbscan'] if method in self.results and 'error' not in self.results.get(method, {})])
        
        if n_methods == 0:
            return
        
        fig, axes = plt.subplots(1, n_methods, figsize=(n_methods * 7, 6))
        if n_methods == 1:
            axes = [axes]  # Make it iterable for consistency
        fig.suptitle(f'Best Clustering Results ({self.data_type.title()} Embeddings)', 
                    fontsize=16, fontweight='bold')
        
        plot_count = 0
        
        # K-means best result
        if 'kmeans' in self.results and 'error' not in self.results['kmeans']:
            ax = axes[plot_count]
            best_k = self.results['kmeans']['summary']['best_silhouette']['n_clusters']
            
            # Re-run best K-means for visualization
            kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(self.embeddings)
            
            # Plot with 2D representation
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, 
                      cmap='Set1', s=2, alpha=0.7)
            ax.set_title(f'Best K-means (k={best_k})\n'
                        f'Silhouette: {self.results["kmeans"]["summary"]["best_silhouette"]["score"]:.3f}')
            ax.set_xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 1')
            ax.set_ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 2')
            ax.grid(True, alpha=0.3)
            
            # Add cluster centers if using 2D embeddings directly
            if self.embeddings.shape[1] == 2:
                centers = kmeans.cluster_centers_
                ax.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, linewidths=3)
            
            plot_count += 1
        
        # Spectral best result
        if 'spectral' in self.results and 'error' not in self.results['spectral']:
            ax = axes[plot_count]
            best_k = self.results['spectral']['summary']['best_silhouette']['n_clusters']
            
            # Re-run best Spectral for visualization
            spectral = SpectralClustering(n_clusters=best_k, random_state=42, n_init=10)
            labels = spectral.fit_predict(self.embeddings)
            
            # Plot with 2D representation
            ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, 
                      cmap='Set2', s=2, alpha=0.7)
            ax.set_title(f'Best Spectral (k={best_k})\n'
                        f'Silhouette: {self.results["spectral"]["summary"]["best_silhouette"]["score"]:.3f}')
            ax.set_xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 1')
            ax.set_ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 2')
            ax.grid(True, alpha=0.3)
            
            plot_count += 1
        
        # HDBSCAN best result
        if 'hdbscan' in self.results and 'error' not in self.results['hdbscan']:
            ax = axes[plot_count]
            best_params = self.results['hdbscan']['summary']['best_silhouette']
            
            # Re-run best HDBSCAN for visualization
            clusterer = HDBSCAN(
                min_cluster_size=best_params['min_cluster_size'],
                min_samples=best_params['min_samples'],
                cluster_selection_epsilon=0.0
            )
            labels = clusterer.fit_predict(self.embeddings)
            
            # Plot with 2D representation
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            
            for label, color in zip(unique_labels, colors):
                mask = labels == label
                if label == -1:
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                             c='gray', s=1, alpha=0.3, label='Noise')
                else:
                    ax.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                             c=[color], s=2, alpha=0.7, label=f'C{label}')
            
            ax.set_title(f'Best HDBSCAN ({best_params["n_clusters"]} clusters)\n'
                        f'Silhouette: {best_params["score"]:.3f}')
            ax.set_xlabel('Component 1' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 1')
            ax.set_ylabel('Component 2' if self.embeddings.shape[1] > 2 else f'{self.data_type.split()[0]} 2')
            ax.grid(True, alpha=0.3)
            
            # Only show legend if reasonable number of clusters
            if best_params["n_clusters"] <= 10:
                ax.legend(fontsize=8, markerscale=2)
        
        # Hide unused subplots if fewer methods were run
        for i in range(plot_count, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = output_dir / 'best_clustering_results.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✓ Saved best clustering visualization: {plot_path}")
        
        plt.close()
    
    def save_results(self, output_dir: Path) -> None:
        """Save detailed results to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON summary
        summary_path = output_dir / 'clustering_optimization_summary.json'
        summary = {}
        
        if 'kmeans' in self.results:
            summary['kmeans'] = self.results['kmeans']['summary']
        
        if 'spectral' in self.results:
            summary['spectral'] = self.results['spectral']['summary']
        
        if 'hdbscan' in self.results:
            summary['hdbscan'] = self.results['hdbscan']['summary']
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"✓ Saved summary: {summary_path}")
        
        # Save detailed CSV results
        if 'kmeans' in self.results:
            kmeans_df = pd.DataFrame(self.results['kmeans']['data'])
            kmeans_path = output_dir / 'kmeans_optimization.csv'
            kmeans_df.to_csv(kmeans_path, index=False)
            print(f"✓ Saved K-means results: {kmeans_path}")
        
        if 'spectral' in self.results:
            spectral_df = pd.DataFrame(self.results['spectral']['data'])
            spectral_path = output_dir / 'spectral_optimization.csv'
            spectral_df.to_csv(spectral_path, index=False)
            print(f"✓ Saved Spectral results: {spectral_path}")
        
        if 'hdbscan' in self.results:
            hdbscan_df = pd.DataFrame(self.results['hdbscan']['data'])
            hdbscan_path = output_dir / 'hdbscan_optimization.csv'
            hdbscan_df.to_csv(hdbscan_path, index=False)
            print(f"✓ Saved HDBSCAN results: {hdbscan_path}")
        
        # Print summary to console
        print("\n" + "="*70)
        print("CLUSTERING OPTIMIZATION SUMMARY")
        print("="*70)
        
        if 'kmeans' in summary:
            print("K-MEANS RESULTS:")
            best = summary['kmeans']['best_silhouette']
            print(f"  Best silhouette score: {best['score']:.3f} at k={best['n_clusters']}")
            
            best_ch = summary['kmeans']['best_calinski_harabasz']
            print(f"  Best Calinski-Harabasz: {best_ch['score']:.1f} at k={best_ch['n_clusters']}")
            
            best_db = summary['kmeans']['best_davies_bouldin']
            print(f"  Best Davies-Bouldin: {best_db['score']:.3f} at k={best_db['n_clusters']}")
        
        if 'spectral' in summary:
            print("\nSPECTRAL CLUSTERING RESULTS:")
            best = summary['spectral']['best_silhouette']
            print(f"  Best silhouette score: {best['score']:.3f} at k={best['n_clusters']}")
            
            best_ch = summary['spectral']['best_calinski_harabasz']
            print(f"  Best Calinski-Harabasz: {best_ch['score']:.1f} at k={best_ch['n_clusters']}")
            
            best_db = summary['spectral']['best_davies_bouldin']
            print(f"  Best Davies-Bouldin: {best_db['score']:.3f} at k={best_db['n_clusters']}")
        
        if 'hdbscan' in summary and 'error' not in summary['hdbscan']:
            print("\nHDBSCAN RESULTS:")
            best = summary['hdbscan']['best_silhouette']
            print(f"  Best silhouette score: {best['score']:.3f}")
            print(f"    Parameters: min_cluster_size={best['min_cluster_size']}, min_samples={best['min_samples']}")
            print(f"    Resulted in {best['n_clusters']} clusters")
            
            best_ch = summary['hdbscan']['best_calinski_harabasz']
            print(f"  Best Calinski-Harabasz: {best_ch['score']:.1f}")
            print(f"    Parameters: min_cluster_size={best_ch['min_cluster_size']}, min_samples={best_ch['min_samples']}")
        
        print("="*70)


def parse_cluster_specification(cluster_spec: str) -> List[int]:
    """Parse cluster specification string into list of cluster numbers.
    
    Supports formats like:
    - '2,3,5,8' -> [2, 3, 5, 8]
    - '2-10' -> [2, 3, 4, 5, 6, 7, 8, 9, 10]
    - '2,5-8,12' -> [2, 5, 6, 7, 8, 12]
    """
    clusters = []
    
    for part in cluster_spec.split(','):
        part = part.strip()
        
        if '-' in part:
            # Handle range specification
            try:
                start, end = part.split('-')
                start, end = int(start.strip()), int(end.strip())
                clusters.extend(range(start, end + 1))
            except ValueError:
                raise ValueError(f"Invalid range specification: '{part}'")
        else:
            # Handle single number
            try:
                clusters.append(int(part))
            except ValueError:
                raise ValueError(f"Invalid cluster number: '{part}'")
    
    # Remove duplicates and sort
    clusters = sorted(set(clusters))
    
    # Validate cluster numbers
    if any(c < 2 for c in clusters):
        raise ValueError("Cluster numbers must be >= 2")
    
    return clusters


def main():
    parser = argparse.ArgumentParser(description="Optimize clustering parameters for embeddings (reduced or latent)")
    
    # Data input (mutually exclusive)
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--reduced_data", type=Path,
                           help="Path to reduced embeddings file (UMAP/PCA) (.npz)")
    data_group.add_argument("--latent_embeddings", type=Path,
                           help="Path to original latent embeddings file (.npz, .h5, .pt)")
    
    # Legacy support for old argument name
    parser.add_argument("--umap_data", type=Path,
                       help="DEPRECATED: Use --reduced_data instead")
    
    parser.add_argument("--output_dir", type=Path, default="clustering_optimization",
                       help="Output directory for results")
    parser.add_argument("--max_clusters", type=int, default=20,
                       help="Maximum number of clusters to test for K-means")
    parser.add_argument("--clusters", type=str, default=None,
                       help="Specific clusters to test (e.g., '2,3,5,8' or '2-10' or '2,5-8,12')")
    parser.add_argument("--subsample", type=int, default=None,
                       help="Subsample points for faster testing (None = use all)")
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize embeddings before clustering (recommended for latent embeddings)")
    parser.add_argument("--skip_kmeans", action="store_true",
                       help="Skip K-means optimization")
    parser.add_argument("--skip_spectral", action="store_true",
                       help="Skip Spectral clustering optimization")
    parser.add_argument("--skip_hdbscan", action="store_true",
                       help="Skip HDBSCAN optimization")
    parser.add_argument("--save_individual", action="store_true", default=True,
                       help="Save individual clustering results for each parameter combination (default: True)")
    parser.add_argument("--no_save_individual", dest="save_individual", action="store_false",
                       help="Don't save individual clustering results")
    
    args = parser.parse_args()
    
    # Handle legacy argument and determine data path and type
    if args.umap_data:
        print("WARNING: --umap_data is deprecated. Use --reduced_data instead.")
        data_path = args.umap_data
        use_latent = False
    elif args.latent_embeddings:
        data_path = args.latent_embeddings
        use_latent = True
    elif args.reduced_data:
        data_path = args.reduced_data
        use_latent = False
    else:
        raise ValueError("Must specify either --latent_embeddings or --reduced_data")
    
    # Validate inputs
    if not data_path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    # Parse cluster specification
    clusters_to_test = None
    if args.clusters:
        try:
            clusters_to_test = parse_cluster_specification(args.clusters)
            print(f"Custom clusters specified: {clusters_to_test}")
        except ValueError as e:
            raise ValueError(f"Invalid cluster specification: {e}")
    
    print("="*70)
    print("CLUSTERING PARAMETER OPTIMIZATION")
    print("="*70)
    data_type = "Latent embeddings" if use_latent else "Reduced embeddings" 
    print(f"{data_type}: {data_path}")
    print(f"Output: {args.output_dir}")
    if clusters_to_test:
        print(f"Testing specific clusters: {clusters_to_test}")
    else:
        print(f"Max K-means clusters: {args.max_clusters}")
    if args.subsample:
        print(f"Subsampling: {args.subsample} points")
    if args.standardize:
        print("Standardization: ENABLED")
    print("="*70)
    
    # Initialize optimizer
    optimizer = ClusteringOptimizer(data_path, args.subsample, use_latent, args.standardize)
    
    # Load data
    optimizer.load_embeddings()
    
    # Run optimizations
    if not args.skip_kmeans:
        optimizer.optimize_kmeans(args.max_clusters, clusters=clusters_to_test, save_individual=args.save_individual, output_dir=args.output_dir)
    
    if not args.skip_spectral:
        optimizer.optimize_spectral(args.max_clusters, clusters=clusters_to_test, save_individual=args.save_individual, output_dir=args.output_dir)
    
    if not args.skip_hdbscan:
        optimizer.optimize_hdbscan(save_individual=args.save_individual, output_dir=args.output_dir)
    
    # Create visualizations and save results
    optimizer.create_visualization(args.output_dir)
    optimizer.save_results(args.output_dir)
    
    print(f"\n✓ Clustering optimization completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()