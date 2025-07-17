#!/usr/bin/env python3
"""
Standalone Reconstruction Script

Generate reconstruction comparisons using trained autoencoder models.
Can be run independently of training to evaluate model performance.
"""

import argparse
import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.autoencoder import Autoencoder
from models.summary import calculate_metrics, save_comparison_images
from .stem_visualization import STEMVisualizer


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
    print(f"Hyperparameters: {hparams}")
    
    # Create model with same architecture
    model = LitAE(
        latent_dim=hparams.get('latent_dim', 32),
        lr=hparams.get('lr', 1e-3),
        lambda_act=hparams.get('lambda_act', 1e-4),
        lambda_sim=hparams.get('lambda_sim', 5e-5),
        lambda_div=hparams.get('lambda_div', 2e-4),
        out_shape=hparams.get('out_shape', (256, 256))
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    
    print(f"Model loaded successfully. Latent dim: {model.hparams.latent_dim}")
    
    # Quick test to see if model produces meaningful output
    with torch.no_grad():
        test_input = torch.randn(1, 1, 256, 256).to(device)
        test_output = model(test_input)
        print(f"Test output stats: mean={test_output.mean():.6f}, std={test_output.std():.6f}, max={test_output.max():.6f}")
        
        if test_output.std() < 1e-6:
            print("WARNING: Model outputs appear to be constant (not trained properly)")
    
    return model


def generate_reconstructions(model: LitAE, data: torch.Tensor, device: str = "cpu") -> torch.Tensor:
    """Generate reconstructions from input data."""
    model.eval()
    data = data.to(device)
    
    with torch.no_grad():
        reconstructed = model(data)
    
    return reconstructed


def evaluate_reconstruction_quality(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Evaluate reconstruction quality with comprehensive metrics."""
    # Basic metrics
    metrics = calculate_metrics(original, reconstructed)
    
    # Additional analysis
    original_np = original.detach().cpu().numpy()
    reconstructed_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if original_np.ndim == 4:
        original_np = original_np.squeeze(1)
        reconstructed_np = reconstructed_np.squeeze(1)
    
    # Check for empty reconstructions
    recon_nonzero = np.count_nonzero(reconstructed_np)
    orig_nonzero = np.count_nonzero(original_np)
    
    print(f"Original non-zero pixels: {orig_nonzero}")
    print(f"Reconstruction non-zero pixels: {recon_nonzero}")
    
    # Check intensity ranges
    orig_mean = np.mean(original_np)
    orig_std = np.std(original_np)
    recon_mean = np.mean(reconstructed_np)
    recon_std = np.std(reconstructed_np)
    
    print(f"Original: mean={orig_mean:.6f}, std={orig_std:.6f}")
    print(f"Reconstruction: mean={recon_mean:.6f}, std={recon_std:.6f}")
    
    # Overall correlation (key metric for reconstruction quality)
    orig_flat = original_np.flatten()
    recon_flat = reconstructed_np.flatten()
    
    # Filter out zero regions for meaningful correlation
    nonzero_mask = (orig_flat > 0) & (recon_flat > 0)
    if np.any(nonzero_mask):
        correlation = np.corrcoef(orig_flat[nonzero_mask], recon_flat[nonzero_mask])[0, 1]
        # Log-space correlation (important for STEM data)
        orig_log = np.log(orig_flat[nonzero_mask] + 1)
        recon_log = np.log(recon_flat[nonzero_mask] + 1)
        log_correlation = np.corrcoef(orig_log, recon_log)[0, 1]
    else:
        correlation = 0.0
        log_correlation = 0.0
    
    # Calculate metrics only on regions with significant intensity
    threshold = np.percentile(original_np, 90)  # Top 10% of intensities
    mask = original_np > threshold
    
    if np.any(mask):
        masked_orig = original_np[mask]
        masked_recon = reconstructed_np[mask]
        
        masked_mse = np.mean((masked_orig - masked_recon) ** 2)
        masked_correlation = np.corrcoef(masked_orig.flatten(), masked_recon.flatten())[0, 1]
        
        metrics.update({
            'masked_mse': masked_mse,
            'masked_correlation': masked_correlation,
            'correlation': correlation,
            'log_correlation': log_correlation,
            'nonzero_ratio': recon_nonzero / max(orig_nonzero, 1),
            'intensity_ratio': recon_mean / max(orig_mean, 1e-10),
            'std_ratio': recon_std / max(orig_std, 1e-10)
        })
    
    # Assessment of reconstruction quality
    if correlation > 0.9 and 0.8 < metrics['intensity_ratio'] < 1.2:
        quality = "EXCELLENT"
    elif correlation > 0.7 and 0.5 < metrics['intensity_ratio'] < 2.0:
        quality = "GOOD"
    elif correlation > 0.5 and 0.1 < metrics['intensity_ratio'] < 5.0:
        quality = "FAIR"
    elif correlation > 0.1 and metrics['intensity_ratio'] > 0.01:
        quality = "POOR"
    else:
        quality = "FAILED"
    
    metrics['reconstruction_quality'] = quality
    
    print(f"Reconstruction Quality Assessment: {quality}")
    print(f"Overall Correlation: {correlation:.4f}")
    print(f"Log-space Correlation: {log_correlation:.4f}")
    
    return metrics


def save_detailed_comparison(original: torch.Tensor, reconstructed: torch.Tensor, 
                           output_path: str, metrics: dict, num_samples: int = 8):
    """Save detailed comparison with focus on meaningful regions."""
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if orig_np.ndim == 4:
        orig_np = orig_np.squeeze(1)
        recon_np = recon_np.squeeze(1)
    
    num_samples = min(num_samples, orig_np.shape[0])
    
    # Create figure
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 4, 8))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Original (log scale for better visibility)
        orig_log = np.log(orig_np[i] + 1)
        im1 = axes[0, i].imshow(orig_log, cmap='viridis')
        axes[0, i].set_title(f'Original {i+1} (log scale)')
        axes[0, i].axis('off')
        
        # Reconstructed (log scale)
        recon_log = np.log(recon_np[i] + 1)
        im2 = axes[1, i].imshow(recon_log, cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1} (log scale)')
        axes[1, i].axis('off')
        
        # Show intensity stats for each pattern
        orig_max = np.max(orig_np[i])
        recon_max = np.max(recon_np[i])
        axes[0, i].text(10, 10, f'Max: {orig_max:.3f}', color='white', fontsize=8, 
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
        axes[1, i].text(10, 10, f'Max: {recon_max:.3f}', color='white', fontsize=8,
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="black", alpha=0.7))
    
    # Add comprehensive metrics with quality assessment
    metrics_text = f"""Reconstruction Analysis:
Quality: {metrics.get('reconstruction_quality', 'UNKNOWN')}

Basic Metrics:
MSE: {metrics.get('mse', 0):.6f} ± {metrics.get('mse_std', 0):.6f}
PSNR: {metrics.get('psnr', 0):.2f} ± {metrics.get('psnr_std', 0):.2f} dB
SSIM: {metrics.get('ssim', 0):.4f} ± {metrics.get('ssim_std', 0):.4f}

Correlation Analysis:
Overall Correlation: {metrics.get('correlation', 0):.4f}
Log-space Correlation: {metrics.get('log_correlation', 0):.4f}
Masked Correlation: {metrics.get('masked_correlation', 0):.4f}

Intensity Preservation:
Intensity Ratio: {metrics.get('intensity_ratio', 0):.4f}
Std Ratio: {metrics.get('std_ratio', 0):.4f}
Non-zero Ratio: {metrics.get('nonzero_ratio', 0):.4f}

High-Intensity Regions:
Masked MSE: {metrics.get('masked_mse', 0):.6f}"""
    
    plt.figtext(0.02, 0.02, metrics_text, fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    """Main reconstruction function."""
    parser = argparse.ArgumentParser(description='Autoencoder Reconstruction Evaluation')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to test data (.pt)')
    parser.add_argument('--output_dir', type=str, default='outputs/reconstruction_results',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (auto, cpu, cuda, mps)')
    parser.add_argument('--num_samples', type=int, default=64,
                       help='Number of samples to reconstruct')
    parser.add_argument('--comparison_samples', type=int, default=8,
                       help='Number of samples to show in comparison')
    parser.add_argument('--scan_shape', nargs=2, type=int, default=None,
                       help='Scan shape for STEM visualization (scan_y scan_x)')
    
    args = parser.parse_args()
    
    # Handle device selection
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Load model
    try:
        model = load_model_from_checkpoint(args.checkpoint, device)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Load data
    print(f"Loading data from {args.data}")
    data = torch.load(args.data, map_location=device)
    
    # Take subset of data
    if len(data) > args.num_samples:
        indices = torch.randperm(len(data))[:args.num_samples]
        data = data[indices]
    
    print(f"Data shape: {data.shape}")
    
    # Generate reconstructions
    print("Generating reconstructions...")
    reconstructed = generate_reconstructions(model, data, device)
    
    # Evaluate quality
    print("Evaluating reconstruction quality...")
    metrics = evaluate_reconstruction_quality(data, reconstructed)
    
    # Print results
    print("\n" + "="*60)
    print("RECONSTRUCTION QUALITY RESULTS")
    print("="*60)
    print(f"{'Samples processed':<20}{len(data)}")
    print(f"{'MSE':<20}{metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
    print(f"{'PSNR (dB)':<20}{metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f}")
    print(f"{'SSIM':<20}{metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
    
    # Additional meaningful metrics
    if 'masked_mse' in metrics:
        print(f"{'Masked MSE':<20}{metrics['masked_mse']:.6f}")
    if 'correlation' in metrics:
        print(f"{'Correlation':<20}{metrics['correlation']:.4f}")
    if 'nonzero_ratio' in metrics:
        print(f"{'Non-zero ratio':<20}{metrics['nonzero_ratio']:.4f}")
    if 'intensity_ratio' in metrics:
        print(f"{'Intensity ratio':<20}{metrics['intensity_ratio']:.4f}")
    
    # Quality assessment
    if metrics.get('nonzero_ratio', 0) < 0.1:
        print("⚠️  WARNING: Very few non-zero pixels in reconstruction!")
    if metrics.get('intensity_ratio', 0) < 0.1:
        print("⚠️  WARNING: Reconstruction intensities much lower than original!")
    if metrics.get('correlation', 0) < 0.5:
        print("⚠️  WARNING: Low correlation between original and reconstruction!")
    
    print("="*60)
    
    # Save detailed comparison
    comparison_path = output_dir / "reconstruction_comparison.png"
    save_detailed_comparison(data, reconstructed, comparison_path, metrics, args.comparison_samples)
    print(f"Detailed comparison saved to: {comparison_path}")
    
    # Save reconstruction data
    reconstruction_data_path = output_dir / "reconstructed_data.pt"
    torch.save(reconstructed.cpu(), reconstruction_data_path)
    print(f"Reconstruction data saved to: {reconstruction_data_path}")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == "__main__":
    main()