#!/usr/bin/env python3
"""
Clustering Parameter Optimization for UMAP Embeddings

Systematically tests clustering parameters for both K-means and HDBSCAN to find
optimal settings based on multiple metrics (silhouette score, calinski-harabasz, etc.).

Usage:
    python scripts/analysis/optimize_clustering.py \
        --umap_data workspace/proj_data_ds4_l32/visualisations/umap_analysis_nn_50_md_pointzero1/umap_data.npz \
        --output_dir workspace/proj_data_ds4_l32/visualisations/clustering_optimization \
        --max_clusters 20 \
        --subsample 10000
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Clustering algorithms
from sklearn.cluster import KMeans, HDBSCAN
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
from sklearn.preprocessing import StandardScaler

class ClusteringOptimizer:
    """Optimize clustering parameters for UMAP embeddings."""
    
    def __init__(self, umap_data_path: Path, subsample: Optional[int] = None):
        """Initialize with UMAP data."""
        self.umap_data_path = umap_data_path
        self.subsample = subsample
        self.umap_embedding = None
        self.results = {}
        
    def load_umap_data(self) -> np.ndarray:
        """Load UMAP embeddings from file."""
        print(f"Loading UMAP data from: {self.umap_data_path}")
        
        data = np.load(self.umap_data_path, allow_pickle=True)
        self.umap_embedding = data['umap_embedding']
        
        print(f"✓ Loaded UMAP embeddings: {self.umap_embedding.shape}")
        
        # Subsample if requested (for faster testing)
        if self.subsample and self.subsample < len(self.umap_embedding):
            print(f"Subsampling {self.subsample} points from {len(self.umap_embedding)} total...")
            indices = np.random.choice(len(self.umap_embedding), self.subsample, replace=False)
            self.umap_embedding = self.umap_embedding[indices]
            print(f"✓ Using subsampled data: {self.umap_embedding.shape}")
        
        return self.umap_embedding
    
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
            labels = kmeans.fit_predict(self.umap_embedding)
            
            # Calculate metrics
            sil_score = silhouette_score(self.umap_embedding, labels)
            ch_score = calinski_harabasz_score(self.umap_embedding, labels)
            db_score = davies_bouldin_score(self.umap_embedding, labels)
            
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
                    labels = clusterer.fit_predict(self.umap_embedding)
                    
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
                        sil_score = silhouette_score(self.umap_embedding, labels)
                        ch_score = calinski_harabasz_score(self.umap_embedding, labels)
                        db_score = davies_bouldin_score(self.umap_embedding, labels)
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
    parser = argparse.ArgumentParser(description="Optimize clustering parameters for UMAP embeddings")
    
    parser.add_argument("--umap_data", type=Path, required=True,
                       help="Path to UMAP data file (.npz)")
    parser.add_argument("--output_dir", type=Path, default="clustering_optimization",
                       help="Output directory for results")
    parser.add_argument("--max_clusters", type=int, default=20,
                       help="Maximum number of clusters to test for K-means")
    parser.add_argument("--subsample", type=int, default=None,
                       help="Subsample points for faster testing (None = use all)")
    parser.add_argument("--skip_kmeans", action="store_true",
                       help="Skip K-means optimization")
    parser.add_argument("--skip_hdbscan", action="store_true",
                       help="Skip HDBSCAN optimization")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not args.umap_data.exists():
        raise FileNotFoundError(f"UMAP data not found: {args.umap_data}")
    
    print("="*70)
    print("CLUSTERING PARAMETER OPTIMIZATION")
    print("="*70)
    print(f"UMAP data: {args.umap_data}")
    print(f"Output: {args.output_dir}")
    print(f"Max K-means clusters: {args.max_clusters}")
    if args.subsample:
        print(f"Subsampling: {args.subsample} points")
    print("="*70)
    
    # Initialize optimizer
    optimizer = ClusteringOptimizer(args.umap_data, args.subsample)
    
    # Load data
    umap_embedding = optimizer.load_umap_data()
    
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