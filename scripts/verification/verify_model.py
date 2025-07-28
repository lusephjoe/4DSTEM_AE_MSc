#!/usr/bin/env python3
"""
Model Verification Script
Loads a trained checkpoint and computes losses independently from training.
This helps verify if high training losses are due to computation issues or actual model performance.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from typing import Dict, List, Tuple

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "training"))

from models.autoencoder import Autoencoder
from models.losses import LossManager, create_loss_config_from_args
from models.summary import calculate_metrics, calculate_diffraction_metrics
from scripts.training.train import HDF5Dataset, LitAE
import pytorch_lightning as pl

def load_model_from_checkpoint(checkpoint_path: Path, device: str = "auto") -> Tuple[LitAE, Dict]:
    """Load model from Lightning checkpoint."""
    print(f"Loading checkpoint from: {checkpoint_path}")
    
    # Determine device
    if device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"  
        else:
            device = "cpu"
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract hyperparameters
    hparams = checkpoint.get('hyper_parameters', {})
    print(f"Model hyperparameters: {hparams}")
    
    # Create model with same config
    model = LitAE.load_from_checkpoint(checkpoint_path, map_location=device)
    model.eval()
    model.to(device)
    
    return model, hparams

def create_test_data(data_path: Path, num_samples: int = 100) -> torch.Tensor:
    """Load test data from HDF5 file."""
    print(f"Loading test data from: {data_path}")
    
    dataset = HDF5Dataset(data_path)
    
    # Take random samples
    indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
    
    data_list = []
    for idx in indices:
        sample = dataset[idx]
        if isinstance(sample, tuple):
            data_list.append(sample[0])
        else:
            data_list.append(sample)
    
    data = torch.stack(data_list)
    print(f"Loaded {data.shape[0]} samples of shape {data.shape[1:]}")
    print(f"Data range: {data.min():.4f} to {data.max():.4f}")
    
    return data, dataset

def compute_independent_losses(model: LitAE, data: torch.Tensor, dataset: HDF5Dataset, 
                             device: str, batch_size: int = 32) -> Dict:
    """Compute losses independently using the same logic as training."""
    model.eval()
    data = data.to(device)
    
    all_losses = []
    all_metrics = []
    all_diffraction_metrics = []
    
    print(f"Computing losses for {len(data)} samples...")
    
    # Process in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        
        with torch.no_grad():
            # Forward pass
            z = model.model.embed(batch)
            x_hat = model.model.decoder(z)
            
            # Apply same denormalization as training
            if hasattr(model, 'global_log_mean') and hasattr(model, 'global_log_std'):
                x_log = model.denormalize_to_log_space(batch)
                x_hat_log = model.denormalize_to_log_space(x_hat)
                print(f"Using scale-aligned loss (mean={model.global_log_mean:.4f}, std={model.global_log_std:.4f})")
            else:
                # Fallback to direct comparison
                x_log, x_hat_log = batch, x_hat
                print("Using direct loss computation (no scale alignment)")
            
            # Compute loss components
            loss_dict = model.model.compute_loss(x_log, x_hat_log, z)
            
            # Compute metrics
            metrics = calculate_metrics(x_log, x_hat_log)
            diffraction_metrics = calculate_diffraction_metrics(x_log, x_hat_log)
            
            all_losses.append({k: v.item() if torch.is_tensor(v) else v for k, v in loss_dict.items()})
            all_metrics.append(metrics)
            all_diffraction_metrics.append(diffraction_metrics)
    
    # Average results
    avg_losses = {}
    for key in all_losses[0].keys():
        avg_losses[key] = np.mean([l[key] for l in all_losses])
    
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    
    avg_diffraction = {}
    for key in all_diffraction_metrics[0].keys():
        avg_diffraction[key] = np.mean([d[key] for d in all_diffraction_metrics])
    
    return {
        'losses': avg_losses,
        'metrics': avg_metrics,
        'diffraction_metrics': avg_diffraction,
        'sample_count': len(data)
    }

def test_different_loss_scales(model: LitAE, data: torch.Tensor, device: str) -> Dict:
    """Test loss computation at different scales to identify scale issues."""
    model.eval()
    data = data.to(device)
    
    # Take small batch for testing
    batch = data[:4]
    
    with torch.no_grad():
        z = model.model.embed(batch)
        x_hat = model.model.decoder(z)
        
        results = {}
        
        # 1. Direct MSE (both in normalized space)
        mse_direct = torch.nn.functional.mse_loss(batch, x_hat).item()
        results['mse_normalized_space'] = mse_direct
        
        # 2. Scale-aligned MSE (both in log space)
        if hasattr(model, 'global_log_mean'):
            x_log = model.denormalize_to_log_space(batch)
            x_hat_log = model.denormalize_to_log_space(x_hat)
            mse_aligned = torch.nn.functional.mse_loss(x_log, x_hat_log).item()
            results['mse_log_space'] = mse_aligned
            results['scale_factor'] = mse_aligned / mse_direct
        
        # 3. Raw intensity space (convert from log)
        if hasattr(model, 'global_log_mean'):
            x_intensity = torch.exp(x_log) - 1e-6
            x_hat_intensity = torch.exp(x_hat_log) - 1e-6
            x_intensity = torch.clamp(x_intensity, min=0)
            x_hat_intensity = torch.clamp(x_hat_intensity, min=0)
            mse_intensity = torch.nn.functional.mse_loss(x_intensity, x_hat_intensity).item()
            results['mse_intensity_space'] = mse_intensity
        
        return results

def visualize_reconstructions(model: LitAE, data: torch.Tensor, device: str, 
                            output_dir: Path, num_examples: int = 6):
    """Create visualization of original vs reconstructed patterns."""
    model.eval()
    data = data.to(device)
    
    # Select examples
    indices = np.linspace(0, len(data)-1, num_examples, dtype=int)
    examples = data[indices]
    
    with torch.no_grad():
        z = model.model.embed(examples)
        reconstructed = model.model.decoder(z)
        
        # Denormalize for visualization if possible
        if hasattr(model, 'global_log_mean'):
            orig_log = model.denormalize_to_log_space(examples)
            recon_log = model.denormalize_to_log_space(reconstructed)
            
            # Convert to intensity space for better visualization
            orig_viz = torch.exp(orig_log) - 1e-6
            recon_viz = torch.exp(recon_log) - 1e-6
        else:
            orig_viz = examples
            recon_viz = reconstructed
    
    # Create visualization
    fig, axes = plt.subplots(3, num_examples, figsize=(2*num_examples, 6))
    
    for i in range(num_examples):
        # Original
        axes[0, i].imshow(orig_viz[i, 0].cpu().numpy(), cmap='hot')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon_viz[i, 0].cpu().numpy(), cmap='hot')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
        
        # Difference
        diff = torch.abs(orig_viz[i, 0] - recon_viz[i, 0]).cpu().numpy()
        axes[2, i].imshow(diff, cmap='viridis')
        axes[2, i].set_title(f'|Difference| {i+1}')
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    viz_path = output_dir / "reconstruction_verification.png"
    plt.savefig(viz_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Visualization saved to: {viz_path}")

def main():
    parser = argparse.ArgumentParser(description="Verify model checkpoint independently")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="Path to test data (HDF5)")
    parser.add_argument("--output_dir", type=Path, default=Path("./verification_results"), 
                       help="Output directory for results")
    parser.add_argument("--num_samples", type=int, default=200, help="Number of test samples")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="auto", help="Device: auto, cpu, cuda, mps")
    parser.add_argument("--visualize", action="store_true", help="Create reconstruction visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MODEL VERIFICATION SCRIPT")
    print("=" * 60)
    
    # Load model
    try:
        model, hparams = load_model_from_checkpoint(args.checkpoint, args.device)
        print(f"✅ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return
    
    # Load test data
    try:
        data, dataset = create_test_data(args.data, args.num_samples)
        print(f"✅ Test data loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Set normalization parameters if available
    if hasattr(dataset, 'global_log_mean') and hasattr(dataset, 'global_log_std'):
        model.global_log_mean = torch.tensor(dataset.global_log_mean)
        model.global_log_std = torch.tensor(dataset.global_log_std)
    
    device = next(model.parameters()).device
    
    print(f"\n🔍 RUNNING VERIFICATION ON {str(device).upper()}")
    print("-" * 40)
    
    # 1. Test different loss scales
    print("1. Testing loss computation at different scales...")
    scale_results = test_different_loss_scales(model, data, device)
    
    print("Scale Comparison:")
    for scale_name, value in scale_results.items():
        print(f"  {scale_name}: {value:.6f}")
    
    # 2. Compute comprehensive losses
    print(f"\n2. Computing comprehensive losses on {len(data)} samples...")
    loss_results = compute_independent_losses(model, data, dataset, device, args.batch_size)
    
    print("\n📊 VERIFICATION RESULTS:")
    print("=" * 40)
    
    print("Loss Components:")
    for loss_name, value in loss_results['losses'].items():
        print(f"  {loss_name}: {value:.6f}")
    
    print(f"\nStandard Metrics:")
    for metric_name, value in loss_results['metrics'].items():
        print(f"  {metric_name}: {value:.6f}")
    
    print(f"\nDiffraction-Specific Metrics:")
    for metric_name, value in loss_results['diffraction_metrics'].items():
        if not metric_name.endswith('_std'):
            print(f"  {metric_name}: {value:.6f}")
    
    # 3. Create visualizations if requested
    if args.visualize:
        print(f"\n3. Creating reconstruction visualizations...")
        visualize_reconstructions(model, data, device, args.output_dir)
    
    # 4. Save results
    results_file = args.output_dir / "verification_results.txt"
    with open(results_file, 'w') as f:
        f.write("MODEL VERIFICATION RESULTS\n")
        f.write("=" * 40 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Data: {args.data}\n")
        f.write(f"Samples: {loss_results['sample_count']}\n\n")
        
        f.write("Scale Comparison:\n")
        for k, v in scale_results.items():
            f.write(f"  {k}: {v:.6f}\n")
        
        f.write(f"\nLoss Components:\n")
        for k, v in loss_results['losses'].items():
            f.write(f"  {k}: {v:.6f}\n")
        
        f.write(f"\nMetrics:\n")
        for k, v in loss_results['metrics'].items():
            f.write(f"  {k}: {v:.6f}\n")
    
    print(f"\n✅ Verification complete! Results saved to: {results_file}")
    
    # Interpretation
    print(f"\n🔍 INTERPRETATION:")
    print("-" * 20)
    
    total_loss = loss_results['losses'].get('total_loss', 0)
    mse_loss = loss_results['losses'].get('mse_loss', 0)
    psnr = loss_results['metrics'].get('psnr', 0)
    
    if total_loss > 10:
        print("⚠️  HIGH LOSS: Model may not be learning effectively")
    elif total_loss > 1:
        print("🟡 MODERATE LOSS: Model is learning but could improve")
    else:
        print("✅ LOW LOSS: Model appears to be training well")
    
    if psnr > 20:
        print("✅ GOOD PSNR: Reconstructions have good quality")
    elif psnr > 15:
        print("🟡 MODERATE PSNR: Reconstructions are acceptable")
    else:
        print("⚠️  LOW PSNR: Reconstructions may have quality issues")

if __name__ == "__main__":
    main()