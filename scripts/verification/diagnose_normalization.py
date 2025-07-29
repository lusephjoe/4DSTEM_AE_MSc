#!/usr/bin/env python3
"""
Normalization Diagnostic Script
Identifies issues with data normalization that could cause high training losses.
"""

import argparse
import h5py
import numpy as np
import torch
from pathlib import Path
import sys
import json
import matplotlib.pyplot as plt

# Add project paths
sys.path.append(str(Path(__file__).parent.parent.parent))
sys.path.append(str(Path(__file__).parent.parent / "training"))

from scripts.training.dataset import HDF5Dataset

def analyze_raw_data(data_path: Path, num_samples: int = 1000):
    """Analyze raw HDF5 data before any processing."""
    print("🔍 ANALYZING RAW DATA")
    print("=" * 40)
    
    with h5py.File(data_path, 'r') as f:
        data = f['patterns']
        
        print(f"Dataset shape: {data.shape}")
        print(f"Dataset dtype: {data.dtype}")
        
        # Sample random patterns
        if len(data) > num_samples:
            indices = np.random.choice(len(data), num_samples, replace=False)
            sample = data[indices]
        else:
            sample = data[:]
        
        print(f"\nRaw data statistics (n={len(sample)}):")
        print(f"  Min: {sample.min():.8f}")
        print(f"  Max: {sample.max():.8f}")
        print(f"  Mean: {sample.mean():.8f}")
        print(f"  Std: {sample.std():.8f}")
        
        # Check for problematic values
        zero_fraction = np.mean(sample == 0.0)
        near_zero_fraction = np.mean(sample < 1e-6)
        
        print(f"\nData distribution:")
        print(f"  Exact zeros: {zero_fraction:.4f}")
        print(f"  Near-zero (< 1e-6): {near_zero_fraction:.4f}")
        
        # Check if data is quantized
        unique_count = len(np.unique(sample.flatten()))
        total_pixels = sample.size
        print(f"  Unique values: {unique_count} / {total_pixels} ({unique_count/total_pixels:.4f})")
        
        if unique_count < 1000:
            print("  ⚠️  Data appears highly quantized!")
        
        # Check attributes for metadata
        print(f"\nHDF5 attributes:")
        if hasattr(data, 'attrs') and len(data.attrs) > 0:
            for key, value in data.attrs.items():
                print(f"  {key}: {value}")
        else:
            print("  No attributes found")
        
        return sample

def test_normalization_computation(data_path: Path):
    """Test the normalization computation logic."""
    print(f"\n🧮 TESTING NORMALIZATION COMPUTATION")
    print("=" * 40)
    
    try:
        # Create dataset (this computes normalization)
        dataset = HDF5Dataset(data_path)
        
        print(f"Computed normalization:")
        print(f"  log_mean: {dataset.global_log_mean:.6f}")
        print(f"  log_std: {dataset.global_log_std:.6f}")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Failed to compute normalization: {e}")
        return None

def manual_normalization_check(sample_data, metadata=None):
    """Manually compute normalization to verify logic."""
    print(f"\n✋ MANUAL NORMALIZATION CHECK")
    print("=" * 40)
    
    # Apply same preprocessing as code
    print("Step 1: Dequantization")
    if metadata and metadata.get("dtype") == "uint16":
        processed = sample_data.astype(np.float32) / 65535.0 * metadata["data_range"] + metadata["data_min"]
        print(f"  uint16 -> float32: range {processed.min():.6f} to {processed.max():.6f}")
    else:
        processed = sample_data.astype(np.float32)
        print(f"  No dequantization: range {processed.min():.6f} to {processed.max():.6f}")
    
    print("Step 2: Log transform")
    log_data = np.log(processed + 1e-6)
    print(f"  After log(x + 1e-6): range {log_data.min():.6f} to {log_data.max():.6f}")
    print(f"  Log mean: {log_data.mean():.6f}")
    print(f"  Log std: {log_data.std():.6f}")
    
    # Check for issues
    if log_data.mean() < -5:
        print("  ⚠️  Very negative mean suggests most data near zero!")
    
    if processed.max() <= 1.0 and processed.min() >= 0.0:
        print("  ℹ️  Data appears to be in [0,1] range (normalized/probability)")
        
        if np.mean(processed < 0.01) > 0.5:
            print("  ⚠️  >50% of data is < 0.01 - this will create very negative log values!")
    
    return log_data

def test_sample_processing(dataset, num_samples: int = 10):
    """Test actual sample processing through dataset."""
    print(f"\n🔬 TESTING SAMPLE PROCESSING")
    print("=" * 40)
    
    samples = []
    for i in range(min(num_samples, len(dataset))):
        sample = dataset[i]
        if isinstance(sample, tuple):
            sample = sample[0]
        samples.append(sample)
    
    samples = torch.stack(samples)
    
    print(f"Processed samples (n={len(samples)}):")
    print(f"  Shape: {samples.shape}")
    print(f"  Range: {samples.min():.6f} to {samples.max():.6f}")
    print(f"  Mean: {samples.mean():.6f}")
    print(f"  Std: {samples.std():.6f}")
    
    # Check if normalization is working
    if abs(samples.mean()) < 0.1 and abs(samples.std() - 1.0) < 0.1:
        print("  ✅ Normalization appears correct (mean≈0, std≈1)")
    else:
        print("  ⚠️  Normalization may be incorrect")
    
    return samples

def diagnose_high_loss(log_mean: float, log_std: float):
    """Diagnose why loss might be high based on normalization parameters."""
    print(f"\n🩺 DIAGNOSING HIGH LOSS")
    print("=" * 40)
    
    print(f"Normalization parameters:")
    print(f"  log_mean: {log_mean:.6f}")
    print(f"  log_std: {log_std:.6f}")
    
    # Expected ranges for diffraction data
    expected_log_mean_range = (3.0, 8.0)  # log(20) to log(3000) - typical intensity ranges
    expected_log_std_range = (1.0, 3.0)   # reasonable variation in log space
    
    issues = []
    
    if log_mean < expected_log_mean_range[0]:
        issues.append(f"Very low log_mean ({log_mean:.2f}) suggests most data is near zero")
    elif log_mean > expected_log_mean_range[1]:
        issues.append(f"Very high log_mean ({log_mean:.2f}) suggests very high intensities")
    
    if log_std < expected_log_std_range[0]:
        issues.append(f"Low log_std ({log_std:.2f}) suggests little intensity variation")
    elif log_std > expected_log_std_range[1]:
        issues.append(f"High log_std ({log_std:.2f}) suggests extreme intensity variation")
    
    if issues:
        print("Potential issues:")
        for issue in issues:
            print(f"  ⚠️  {issue}")
        
        print("\nPossible causes:")
        if log_mean < -5:
            print("  • Data might be pre-normalized to [0,1] instead of raw intensities")
            print("  • Data might contain mostly background (low intensities)")
            print("  • Quantization/compression might have clipped high intensities")
        
        print("\nRecommended fixes:")
        print("  • Check if data should be multiplied by a scale factor")
        print("  • Verify data preprocessing pipeline")
        print("  • Consider using different loss function (L1, Huber)")
        print("  • Adjust learning rates for the actual data scale")
    else:
        print("✅ Normalization parameters appear reasonable")

def create_diagnostic_plots(raw_data, processed_samples, output_dir: Path):
    """Create diagnostic plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Raw data histogram
    axes[0, 0].hist(raw_data.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 0].set_title("Raw Data Distribution")
    axes[0, 0].set_xlabel("Intensity")
    axes[0, 0].set_ylabel("Density")
    axes[0, 0].set_yscale('log')
    
    # Log data histogram
    log_raw = np.log(raw_data + 1e-6)
    axes[0, 1].hist(log_raw.flatten(), bins=50, alpha=0.7, density=True)
    axes[0, 1].set_title("Log-transformed Data")
    axes[0, 1].set_xlabel("Log Intensity")
    axes[0, 1].set_ylabel("Density")
    
    # Processed (normalized) data histogram
    axes[0, 2].hist(processed_samples.flatten().numpy(), bins=50, alpha=0.7, density=True)
    axes[0, 2].set_title("Normalized Data")
    axes[0, 2].set_xlabel("Normalized Value")
    axes[0, 2].set_ylabel("Density")
    
    # Sample images
    for i in range(3):
        if i < len(raw_data):
            axes[1, i].imshow(raw_data[i], cmap='hot')
            axes[1, i].set_title(f"Sample {i+1}")
            axes[1, i].axis('off')
    
    plt.tight_layout()
    
    plot_path = output_dir / "normalization_diagnostics.png"
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    print(f"Diagnostic plots saved to: {plot_path}")

def main():
    parser = argparse.ArgumentParser(description="Diagnose normalization issues")
    parser.add_argument("--data", type=Path, required=True, help="Path to HDF5 data file")
    parser.add_argument("--output_dir", type=Path, default=Path("./normalization_diagnostics"),
                       help="Output directory for diagnostic plots")
    parser.add_argument("--samples", type=int, default=1000, help="Number of samples to analyze")
    
    args = parser.parse_args()
    
    print("🔧 NORMALIZATION DIAGNOSTIC TOOL")
    print("=" * 50)
    
    # Step 1: Analyze raw data
    raw_sample = analyze_raw_data(args.data, args.samples)
    
    # Step 2: Test dataset normalization
    dataset = test_normalization_computation(args.data)
    
    if dataset is None:
        print("❌ Cannot proceed - dataset loading failed")
        return
    
    # Step 3: Manual verification
    manual_log_data = manual_normalization_check(raw_sample, dataset.metadata)
    
    # Step 4: Test processed samples
    processed_samples = test_sample_processing(dataset, 100)
    
    # Step 5: Diagnose issues
    diagnose_high_loss(dataset.global_log_mean, dataset.global_log_std)
    
    # Step 6: Create diagnostic plots
    create_diagnostic_plots(raw_sample[:10], processed_samples[:10], args.output_dir)
    
    # Summary
    print(f"\n📋 SUMMARY")
    print("=" * 20)
    print(f"Raw data range: {raw_sample.min():.6f} to {raw_sample.max():.6f}")
    print(f"Computed log_mean: {dataset.global_log_mean:.6f}")
    print(f"Computed log_std: {dataset.global_log_std:.6f}")
    print(f"Processed data range: {processed_samples.min():.6f} to {processed_samples.max():.6f}")
    
    if dataset.global_log_mean < -5:
        print("\n⚠️  LIKELY ISSUE: Data appears to be pre-normalized to [0,1]")
        print("   This creates very negative log values and inflated losses.")
        print("   Consider scaling data back to physical intensity units.")

if __name__ == "__main__":
    main()