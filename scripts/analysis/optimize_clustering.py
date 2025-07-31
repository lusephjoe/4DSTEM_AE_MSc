#!/usr/bin/env python3
"""
Clustering Parameter Optimization for Embeddings

Systematically tests clustering parameters for both K-means and HDBSCAN to find
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
from sklearn.cluster import KMeans, HDBSCAN
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
    
    def optimize_kmeans(self, max_clusters: int = 20) -> Dict:
        """Test K-means with different numbers of clusters."""
        print(f"Optimizing K-means clustering (k=2 to {max_clusters})...")
        
        kmeans_results = {
            'n_clusters': [],
            'silhouette': [],
            'calinski_harabasz': [],
            'davies_bouldin': [],
            'inertia': []
        }
        
        for k in tqdm(range(2, max_clusters + 1), desc="K-means"):
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
    
    def optimize_hdbscan(self) -> Dict:
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
        
        # Create main figure
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
        
        # HDBSCAN results
        if 'hdbscan' in self.results and 'error' not in self.results['hdbscan']['summary']:
            hdbscan_data = self.results['hdbscan']['data']
            
            # Create DataFrame for easier plotting
            df = pd.DataFrame(hdbscan_data)
            df_valid = df[df['silhouette'] > -1]  # Only valid results
            
            if len(df_valid) > 0:
                # Panel 4: HDBSCAN heatmap
                ax4 = plt.subplot(2, 3, 4)
                
                # Create pivot table for heatmap
                pivot_data = df_valid.groupby(['min_cluster_size', 'min_samples'])['silhouette'].mean().unstack()
                
                sns.heatmap(pivot_data, annot=True, fmt='.3f', cmap='viridis', ax=ax4, cbar_kws={'label': 'Silhouette Score'})
                ax4.set_title('HDBSCAN: Silhouette Score Heatmap')
                ax4.set_xlabel('Min Samples')
                ax4.set_ylabel('Min Cluster Size')
                
                # Panel 5: HDBSCAN clusters vs parameters
                ax5 = plt.subplot(2, 3, 5)
                scatter = ax5.scatter(df_valid['min_cluster_size'], df_valid['min_samples'], 
                                    c=df_valid['n_clusters'], s=60, cmap='plasma', alpha=0.7)
                plt.colorbar(scatter, ax=ax5, label='Number of Clusters')
                ax5.set_xlabel('Min Cluster Size')
                ax5.set_ylabel('Min Samples')
                ax5.set_title('HDBSCAN: Number of Clusters')
                ax5.set_xscale('log')
                
                # Highlight best
                best_params = self.results['hdbscan']['summary']['best_silhouette']
                ax5.scatter(best_params['min_cluster_size'], best_params['min_samples'], 
                          s=200, c='red', marker='x', linewidths=3, label='Best')
                ax5.legend()
                
                # Panel 6: HDBSCAN noise vs parameters
                ax6 = plt.subplot(2, 3, 6)
                scatter = ax6.scatter(df_valid['min_cluster_size'], df_valid['min_samples'], 
                                    c=df_valid['n_noise'], s=60, cmap='Reds', alpha=0.7)
                plt.colorbar(scatter, ax=ax6, label='Noise Points')
                ax6.set_xlabel('Min Cluster Size')
                ax6.set_ylabel('Min Samples')
                ax6.set_title('HDBSCAN: Noise Points')
                ax6.set_xscale('log')
        
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
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
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
            scatter = ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, 
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
        
        # Hide unused subplot if only one method was run
        if plot_count == 1:
            axes[1].set_visible(False)
        
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
    parser.add_argument("--subsample", type=int, default=None,
                       help="Subsample points for faster testing (None = use all)")
    parser.add_argument("--standardize", action="store_true",
                       help="Standardize embeddings before clustering (recommended for latent embeddings)")
    parser.add_argument("--skip_kmeans", action="store_true",
                       help="Skip K-means optimization")
    parser.add_argument("--skip_hdbscan", action="store_true",
                       help="Skip HDBSCAN optimization")
    
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
    
    print("="*70)
    print("CLUSTERING PARAMETER OPTIMIZATION")
    print("="*70)
    data_type = "Latent embeddings" if use_latent else "Reduced embeddings" 
    print(f"{data_type}: {data_path}")
    print(f"Output: {args.output_dir}")
    print(f"Max K-means clusters: {args.max_clusters}")
    if args.subsample:
        print(f"Subsampling: {args.subsample} points")
    if args.standardize:
        print("Standardization: ENABLED")
    print("="*70)
    
    # Initialize optimizer
    optimizer = ClusteringOptimizer(data_path, args.subsample, use_latent, args.standardize)
    
    # Load data
    embeddings = optimizer.load_embeddings()
    
    # Run optimizations
    if not args.skip_kmeans:
        optimizer.optimize_kmeans(args.max_clusters)
    
    if not args.skip_hdbscan:
        optimizer.optimize_hdbscan()
    
    # Create visualizations and save results
    optimizer.create_visualization(args.output_dir)
    optimizer.save_results(args.output_dir)
    
    print(f"\n✓ Clustering optimization completed!")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()