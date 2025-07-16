#!/usr/bin/env python3
"""
Comprehensive Autoencoder Evaluation Script

This script provides a thorough evaluation of the autoencoder's reconstruction quality
and ensures the metrics accurately reflect the model's true performance.
"""

import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Dict, Tuple
import seaborn as sns
from sklearn.metrics import r2_score
from scipy import stats
from models.autoencoder import Autoencoder
from models.summary import calculate_metrics
from scripts.stem_visualization import STEMVisualizer

class LitAE(pl.LightningModule):
    """Lightning module wrapper for loading trained models."""
    
    def __init__(self, latent_dim: int, lr: float = 1e-3,
                 lambda_act: float = 1e-4, lambda_sim: float = 5e-5, lambda_div: float = 2e-4,
                 out_shape: tuple = (256, 256)):
        super().__init__()
        self.save_hyperparameters()
        self.model = Autoencoder(latent_dim, out_shape)
        self.train_losses = []
        
        # Regularization parameters
        self.lambda_act = lambda_act
        self.lambda_sim = lambda_sim
        self.lambda_div = lambda_div

    def forward(self, x):
        return self.model(x)


def load_model_from_checkpoint(checkpoint_path: str, device: str = "cpu") -> LitAE:
    """Load trained model from checkpoint."""
    print(f"Loading model from {checkpoint_path}")
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    latent_dim = hparams.get('latent_dim', 32)
    lr = hparams.get('lr', 1e-3)
    lambda_act = hparams.get('lambda_act', 1e-4)
    lambda_sim = hparams.get('lambda_sim', 5e-5)
    lambda_div = hparams.get('lambda_div', 2e-4)
    out_shape = hparams.get('out_shape', (256, 256))
    
    # Create model
    model = LitAE(latent_dim, lr, lambda_act, lambda_sim, lambda_div, out_shape)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    
    return model


def comprehensive_reconstruction_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> Dict:
    """Calculate comprehensive reconstruction metrics."""
    # Convert to numpy for analysis
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if orig_np.ndim == 4:
        orig_np = orig_np.squeeze(1)
        recon_np = recon_np.squeeze(1)
    
    metrics = {}
    
    # Basic intensity statistics
    metrics['orig_mean'] = np.mean(orig_np)
    metrics['orig_std'] = np.std(orig_np)
    metrics['orig_min'] = np.min(orig_np)
    metrics['orig_max'] = np.max(orig_np)
    
    metrics['recon_mean'] = np.mean(recon_np)
    metrics['recon_std'] = np.std(recon_np)
    metrics['recon_min'] = np.min(recon_np)
    metrics['recon_max'] = np.max(recon_np)
    
    # Intensity preservation
    metrics['intensity_ratio'] = metrics['recon_mean'] / max(metrics['orig_mean'], 1e-10)
    metrics['std_ratio'] = metrics['recon_std'] / max(metrics['orig_std'], 1e-10)
    
    # Non-zero pixel analysis
    orig_nonzero = np.count_nonzero(orig_np)
    recon_nonzero = np.count_nonzero(recon_np)
    metrics['nonzero_ratio'] = recon_nonzero / max(orig_nonzero, 1)
    
    # Reconstruction quality metrics
    metrics['mse'] = np.mean((orig_np - recon_np) ** 2)
    metrics['rmse'] = np.sqrt(metrics['mse'])
    metrics['mae'] = np.mean(np.abs(orig_np - recon_np))
    
    # PSNR calculation
    if metrics['mse'] > 0:
        max_val = max(np.max(orig_np), np.max(recon_np))
        metrics['psnr'] = 20 * np.log10(max_val / np.sqrt(metrics['mse']))
    else:
        metrics['psnr'] = float('inf')
    
    # Correlation analysis
    orig_flat = orig_np.flatten()
    recon_flat = recon_np.flatten()
    
    # Filter out zero values for meaningful correlation
    nonzero_mask = (orig_flat > 0) & (recon_flat > 0)
    if np.any(nonzero_mask):
        orig_nz = orig_flat[nonzero_mask]
        recon_nz = recon_flat[nonzero_mask]
        
        metrics['correlation'] = np.corrcoef(orig_nz, recon_nz)[0, 1]
        metrics['r2_score'] = r2_score(orig_nz, recon_nz)
        
        # Log-space correlation (important for STEM data)
        orig_log = np.log(orig_nz + 1)
        recon_log = np.log(recon_nz + 1)
        metrics['log_correlation'] = np.corrcoef(orig_log, recon_log)[0, 1]
    else:
        metrics['correlation'] = 0.0
        metrics['r2_score'] = 0.0
        metrics['log_correlation'] = 0.0
    
    # High-intensity region analysis (important features)
    high_intensity_thresh = np.percentile(orig_np, 95)
    high_mask = orig_np > high_intensity_thresh
    
    if np.any(high_mask):
        orig_high = orig_np[high_mask]
        recon_high = recon_np[high_mask]
        
        metrics['high_intensity_mse'] = np.mean((orig_high - recon_high) ** 2)
        metrics['high_intensity_correlation'] = np.corrcoef(orig_high, recon_high)[0, 1]
        metrics['high_intensity_preservation'] = np.mean(recon_high) / np.mean(orig_high)
    else:
        metrics['high_intensity_mse'] = metrics['mse']
        metrics['high_intensity_correlation'] = metrics['correlation']
        metrics['high_intensity_preservation'] = metrics['intensity_ratio']
    
    # Pattern-wise analysis
    pattern_mses = []
    pattern_correlations = []
    
    for i in range(orig_np.shape[0]):
        pattern_mse = np.mean((orig_np[i] - recon_np[i]) ** 2)
        pattern_mses.append(pattern_mse)
        
        # Correlation for individual patterns
        orig_pattern = orig_np[i].flatten()
        recon_pattern = recon_np[i].flatten()
        if np.var(orig_pattern) > 0 and np.var(recon_pattern) > 0:
            pattern_corr = np.corrcoef(orig_pattern, recon_pattern)[0, 1]
            pattern_correlations.append(pattern_corr)
    
    metrics['pattern_mse_std'] = np.std(pattern_mses)
    metrics['pattern_correlation_mean'] = np.mean(pattern_correlations) if pattern_correlations else 0.0
    metrics['pattern_correlation_std'] = np.std(pattern_correlations) if pattern_correlations else 0.0
    
    # Assessment of reconstruction quality
    metrics['reconstruction_quality'] = assess_reconstruction_quality(metrics)
    
    return metrics


def assess_reconstruction_quality(metrics: Dict) -> str:
    """Assess overall reconstruction quality based on metrics."""
    correlation = metrics.get('correlation', 0)
    intensity_ratio = metrics.get('intensity_ratio', 0)
    nonzero_ratio = metrics.get('nonzero_ratio', 0)
    
    # Define quality thresholds
    if correlation > 0.9 and 0.8 < intensity_ratio < 1.2 and nonzero_ratio > 0.9:
        return "EXCELLENT"
    elif correlation > 0.7 and 0.5 < intensity_ratio < 2.0 and nonzero_ratio > 0.7:
        return "GOOD"
    elif correlation > 0.5 and 0.1 < intensity_ratio < 5.0 and nonzero_ratio > 0.5:
        return "FAIR"
    elif correlation > 0.1 and intensity_ratio > 0.01 and nonzero_ratio > 0.1:
        return "POOR"
    else:
        return "FAILED"


def create_comprehensive_visualization(original: torch.Tensor, reconstructed: torch.Tensor, 
                                     metrics: Dict, output_path: str, num_samples: int = 6):
    """Create comprehensive visualization of reconstruction quality."""
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if orig_np.ndim == 4:
        orig_np = orig_np.squeeze(1)
        recon_np = recon_np.squeeze(1)
    
    num_samples = min(num_samples, orig_np.shape[0])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 15))
    gs = fig.add_gridspec(4, num_samples, height_ratios=[1, 1, 1, 0.8], hspace=0.3, wspace=0.3)
    
    # Row 1: Original patterns
    for i in range(num_samples):
        ax = fig.add_subplot(gs[0, i])
        im = ax.imshow(np.log(orig_np[i] + 1), cmap='viridis')
        ax.set_title(f'Original {i+1} (log)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Row 2: Reconstructed patterns
    for i in range(num_samples):
        ax = fig.add_subplot(gs[1, i])
        im = ax.imshow(np.log(recon_np[i] + 1), cmap='viridis')
        ax.set_title(f'Reconstructed {i+1} (log)', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Row 3: Difference maps
    for i in range(num_samples):
        ax = fig.add_subplot(gs[2, i])
        diff = orig_np[i] - recon_np[i]
        im = ax.imshow(diff, cmap='RdBu_r', vmin=-np.max(np.abs(diff)), vmax=np.max(np.abs(diff)))
        ax.set_title(f'Difference {i+1}', fontsize=12)
        ax.axis('off')
        plt.colorbar(im, ax=ax, shrink=0.8)
    
    # Row 4: Metrics and analysis
    ax_metrics = fig.add_subplot(gs[3, :3])
    ax_metrics.axis('off')
    
    # Create metrics table
    metrics_text = f"""RECONSTRUCTION EVALUATION REPORT
    
Quality Assessment: {metrics['reconstruction_quality']}

INTENSITY STATISTICS:
Original    - Mean: {metrics['orig_mean']:.6f}, Std: {metrics['orig_std']:.6f}, Range: [{metrics['orig_min']:.6f}, {metrics['orig_max']:.6f}]
Reconstructed - Mean: {metrics['recon_mean']:.6f}, Std: {metrics['recon_std']:.6f}, Range: [{metrics['recon_min']:.6f}, {metrics['recon_max']:.6f}]

RECONSTRUCTION METRICS:
MSE: {metrics['mse']:.6f}
RMSE: {metrics['rmse']:.6f}  
MAE: {metrics['mae']:.6f}
PSNR: {metrics['psnr']:.2f} dB

CORRELATION ANALYSIS:
Correlation: {metrics['correlation']:.4f}
R² Score: {metrics['r2_score']:.4f}
Log Correlation: {metrics['log_correlation']:.4f}

PRESERVATION RATIOS:
Intensity Ratio: {metrics['intensity_ratio']:.4f}
Std Ratio: {metrics['std_ratio']:.4f}
Non-zero Ratio: {metrics['nonzero_ratio']:.4f}

HIGH-INTENSITY FEATURES:
High-Int MSE: {metrics['high_intensity_mse']:.6f}
High-Int Correlation: {metrics['high_intensity_correlation']:.4f}
High-Int Preservation: {metrics['high_intensity_preservation']:.4f}

PATTERN VARIABILITY:
Pattern MSE Std: {metrics['pattern_mse_std']:.6f}
Pattern Corr Mean: {metrics['pattern_correlation_mean']:.4f} ± {metrics['pattern_correlation_std']:.4f}"""
    
    ax_metrics.text(0.05, 0.95, metrics_text, transform=ax_metrics.transAxes, fontsize=10,
                   verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    # Scatter plot of original vs reconstructed
    ax_scatter = fig.add_subplot(gs[3, 3:])
    orig_flat = orig_np.flatten()
    recon_flat = recon_np.flatten()
    
    # Sample points for visualization (too many points slow down plotting)
    sample_size = min(10000, len(orig_flat))
    indices = np.random.choice(len(orig_flat), sample_size, replace=False)
    
    ax_scatter.scatter(orig_flat[indices], recon_flat[indices], alpha=0.6, s=1)
    ax_scatter.plot([orig_flat.min(), orig_flat.max()], [orig_flat.min(), orig_flat.max()], 'r--', alpha=0.8)
    ax_scatter.set_xlabel('Original Intensity')
    ax_scatter.set_ylabel('Reconstructed Intensity')
    ax_scatter.set_title('Original vs Reconstructed\n(Perfect reconstruction = diagonal line)')
    ax_scatter.grid(True, alpha=0.3)
    
    # Add correlation to scatter plot
    ax_scatter.text(0.05, 0.95, f'Correlation: {metrics["correlation"]:.4f}', 
                   transform=ax_scatter.transAxes, fontsize=12,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    
    plt.suptitle('Autoencoder Reconstruction Evaluation', fontsize=16, fontweight='bold')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main function for comprehensive autoencoder evaluation."""
    parser = argparse.ArgumentParser(description='Comprehensive Autoencoder Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True, help='Path to test data')
    parser.add_argument('--output_path', type=str, default='autoencoder_evaluation.png', 
                       help='Output path for evaluation report')
    parser.add_argument('--num_samples', type=int, default=100, help='Number of test samples')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for inference')
    parser.add_argument('--device', type=str, default='cpu', help='Device to run on')
    
    args = parser.parse_args()
    
    # Load model
    model = load_model_from_checkpoint(args.checkpoint, args.device)
    model.to(args.device)
    
    # Load test data
    print(f"Loading test data from {args.data_path}")
    data = torch.load(args.data_path, map_location=args.device)
    
    # Limit to requested number of samples
    data = data[:args.num_samples]
    print(f"Evaluating on {len(data)} samples")
    
    # Generate reconstructions
    print("Generating reconstructions...")
    with torch.no_grad():
        reconstructed = model(data)
    
    # Comprehensive evaluation
    print("Computing comprehensive metrics...")
    metrics = comprehensive_reconstruction_metrics(data, reconstructed)
    
    # Print summary
    print("\n" + "="*60)
    print("AUTOENCODER EVALUATION SUMMARY")
    print("="*60)
    print(f"Quality Assessment: {metrics['reconstruction_quality']}")
    print(f"Correlation: {metrics['correlation']:.4f}")
    print(f"Intensity Ratio: {metrics['intensity_ratio']:.4f}")
    print(f"Non-zero Ratio: {metrics['nonzero_ratio']:.4f}")
    print(f"PSNR: {metrics['psnr']:.2f} dB")
    print(f"MSE: {metrics['mse']:.6f}")
    print("="*60)
    
    # Create comprehensive visualization
    print(f"Creating comprehensive visualization...")
    create_comprehensive_visualization(data, reconstructed, metrics, args.output_path)
    print(f"Evaluation report saved to: {args.output_path}")
    
    # Additional STEM-specific analysis
    stem_output = args.output_path.replace('.png', '_stem_analysis.png')
    print(f"Creating STEM-specific analysis...")
    
    # Use STEM visualizer for additional analysis
    try:
        stem_viz = STEMVisualizer(data.detach().cpu().numpy())
        stem_viz.save_complete_visualization(stem_output, reconstructed.detach().cpu().numpy())
        print(f"STEM analysis saved to: {stem_output}")
    except Exception as e:
        print(f"STEM analysis failed: {e}")


if __name__ == "__main__":
    main()