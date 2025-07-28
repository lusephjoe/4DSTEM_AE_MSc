#!/usr/bin/env python
"""Check for scale mismatches in the loss calculation pipeline."""

import torch
import h5py
import numpy as np
import json
from pathlib import Path
import sys
sys.path.append('.')

def check_scale_mismatch():
    print("=== CHECKING FOR SCALE MISMATCHES ===")
    
    # Load normalization stats
    stats_path = Path("ds4_test_input_data/inputs/train_tensor_ds4_compressed_normalization_stats.json")
    if stats_path.exists():
        with open(stats_path, 'r') as f:
            stats = json.load(f)
        global_log_mean = stats['log_mean']
        global_log_std = stats['log_std']
        print(f"1. Loaded normalization stats:")
        print(f"   log_mean: {global_log_mean:.6f}")
        print(f"   log_std: {global_log_std:.6f}")
    else:
        print("1. No normalization stats found - computing from data...")
        # Load data and compute stats
        with h5py.File("ds4_test_input_data/inputs/train_tensor_ds4_compressed.h5", 'r') as f:
            sample = f['patterns'][:1000].astype('float32')
        log_data = np.log(sample + 1)
        global_log_mean = float(log_data.mean())
        global_log_std = float(log_data.std())
        print(f"   Computed log_mean: {global_log_mean:.6f}")
        print(f"   Computed log_std: {global_log_std:.6f}")
    
    # Load test data
    with h5py.File("ds4_test_input_data/inputs/train_tensor_ds4_compressed.h5", 'r') as f:
        raw_data = f['patterns'][:5].astype('float32')  # Small sample
    
    print(f"\n2. Raw data analysis:")
    print(f"   Shape: {raw_data.shape}")
    print(f"   Range: [{raw_data.min():.6f}, {raw_data.max():.6f}]")
    print(f"   Mean: {raw_data.mean():.6f}")
    
    # Step 1: Apply log transform
    log_data = np.log(raw_data + 1)
    print(f"\n3. After log(x + 1):")
    print(f"   Range: [{log_data.min():.6f}, {log_data.max():.6f}]")
    print(f"   Mean: {log_data.mean():.6f}")
    print(f"   Std: {log_data.std():.6f}")
    
    # Step 2: Apply z-score normalization (what dataset does)
    normalized = (log_data - global_log_mean) / (global_log_std + 1e-8)
    print(f"\n4. After z-score normalization:")
    print(f"   Range: [{normalized.min():.6f}, {normalized.max():.6f}]")
    print(f"   Mean: {normalized.mean():.6f}")
    print(f"   Std: {normalized.std():.6f}")
    
    # Convert to tensors (what model sees)
    input_tensor = torch.from_numpy(normalized).unsqueeze(1)  # Add channel dim
    print(f"\n5. Input tensor (what model processes):")
    print(f"   Shape: {input_tensor.shape}")
    print(f"   Range: [{input_tensor.min():.6f}, {input_tensor.max():.6f}]")
    
    # Simulate model output (normalized space)
    # For testing, let's create a reconstruction that's slightly different
    torch.manual_seed(42)
    reconstruction = input_tensor + torch.randn_like(input_tensor) * 0.1  # Add small noise
    print(f"\n6. Simulated reconstruction (normalized space):")
    print(f"   Range: [{reconstruction.min():.6f}, {reconstruction.max():.6f}]")
    print(f"   Difference from input: {torch.mean((input_tensor - reconstruction)**2):.6f}")
    
    # Step 3: Denormalize BOTH back to log space (what scale-aligned loss does)
    def denormalize_to_log_space(x_normalized):
        return x_normalized * (global_log_std + 1e-8) + global_log_mean
    
    input_log_space = denormalize_to_log_space(input_tensor)
    recon_log_space = denormalize_to_log_space(reconstruction)
    
    print(f"\n7. After denormalization to log space:")
    print(f"   Input range: [{input_log_space.min():.6f}, {input_log_space.max():.6f}]")
    print(f"   Reconstruction range: [{recon_log_space.min():.6f}, {recon_log_space.max():.6f}]")
    
    # Check if denormalization is working correctly
    original_log_tensor = torch.from_numpy(log_data).unsqueeze(1)
    denorm_error = torch.mean((original_log_tensor - input_log_space)**2)
    print(f"   Denormalization error: {denorm_error:.10f}")
    
    if denorm_error < 1e-6:
        print("   ✓ Denormalization is working correctly")
    else:
        print("   ❌ Denormalization has errors!")
    
    # Step 4: Compute MSE in log space
    mse_log = torch.mean((input_log_space - recon_log_space)**2)
    print(f"\n8. MSE Analysis:")
    print(f"   MSE in normalized space: {torch.mean((input_tensor - reconstruction)**2):.6f}")
    print(f"   MSE in log space (scale-aligned): {mse_log:.6f}")
    print(f"   Root MSE in log space: {torch.sqrt(mse_log):.6f}")
    
    # Check scaling effect
    scaling_factor = (global_log_std + 1e-8) ** 2
    expected_scaled_mse = torch.mean((input_tensor - reconstruction)**2) * scaling_factor
    print(f"   Expected scaled MSE: {expected_scaled_mse:.6f}")
    print(f"   Scaling factor: {scaling_factor:.6f}")
    
    if abs(mse_log - expected_scaled_mse) < 1e-6:
        print("   ✓ MSE scaling is consistent")
    else:
        print("   ❌ MSE scaling mismatch detected!")
    
    # Step 5: Compare with different scenarios
    print(f"\n9. Comparison scenarios:")
    
    # Scenario A: Perfect reconstruction
    perfect_recon = input_tensor.clone()
    perfect_mse = torch.mean((denormalize_to_log_space(input_tensor) - denormalize_to_log_space(perfect_recon))**2)
    print(f"   Perfect reconstruction MSE: {perfect_mse:.10f}")
    
    # Scenario B: Random reconstruction
    random_recon = torch.randn_like(input_tensor)
    random_mse = torch.mean((denormalize_to_log_space(input_tensor) - denormalize_to_log_space(random_recon))**2)
    print(f"   Random reconstruction MSE: {random_mse:.6f}")
    
    # Scenario C: Your reported MSE
    your_mse = 0.002
    print(f"   Your reported MSE: {your_mse:.6f}")
    
    print(f"\n10. Scale Mismatch Assessment:")
    if your_mse < perfect_mse + 1e-6:
        print("   ❌ CRITICAL: Your MSE is lower than perfect reconstruction!")
        print("   This indicates a serious scale mismatch or bug.")
    elif your_mse < random_mse / 10:
        print("   ⚠️  WARNING: Your MSE is suspiciously low")
        print("   This could indicate:")
        print("   - Model is performing exceptionally well")
        print("   - Scale mismatch making loss appear smaller")
        print("   - Model learning identity mapping")
    else:
        print("   ✓ MSE appears reasonable")
    
    # Step 6: Test with different global stats
    print(f"\n11. Testing sensitivity to normalization stats:")
    wrong_mean = 0.0  # Wrong mean
    wrong_std = 1.0   # Wrong std
    
    wrong_denorm_input = input_tensor * wrong_std + wrong_mean
    wrong_denorm_recon = reconstruction * wrong_std + wrong_mean
    wrong_mse = torch.mean((wrong_denorm_input - wrong_denorm_recon)**2)
    
    print(f"   MSE with wrong stats (mean=0, std=1): {wrong_mse:.6f}")
    print(f"   MSE with correct stats: {mse_log:.6f}")
    print(f"   Ratio (wrong/correct): {wrong_mse/mse_log:.2f}")
    
    if abs(your_mse - wrong_mse) < abs(your_mse - mse_log):
        print("   ❌ Your MSE is closer to wrong stats - possible bug!")
    else:
        print("   ✓ Your MSE matches correct stats")

if __name__ == "__main__":
    check_scale_mismatch()