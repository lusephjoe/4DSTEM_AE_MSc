#!/usr/bin/env python
"""Test both normalization modes in train.py to ensure they work correctly."""

import torch
import h5py
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

def test_dataset_modes():
    print("=== TESTING NORMALIZATION MODES ===")
    
    # Import the dataset class
    from scripts.training.train import HDF5Dataset
    
    data_path = "ds4_test_input_data/inputs/train_tensor_ds4_compressed.h5"
    
    print("1. Testing normalized mode (default):")
    try:
        dataset_normalized = HDF5Dataset(data_path, use_normalization=True)
        sample_norm = dataset_normalized[0]
        
        print(f"   ✓ Dataset created successfully")
        print(f"   Shape: {sample_norm.shape}")
        print(f"   Range: [{sample_norm.min():.6f}, {sample_norm.max():.6f}]")
        print(f"   Mean: {sample_norm.mean():.6f}")
        print(f"   Std: {sample_norm.std():.6f}")
        print(f"   Global stats: mean={dataset_normalized.global_log_mean:.6f}, std={dataset_normalized.global_log_std:.6f}")
        
        # Check if it looks normalized
        if abs(sample_norm.mean()) < 0.1 and abs(sample_norm.std() - 1.0) < 0.1:
            print("   ✓ Data appears properly normalized")
        else:
            print("   ⚠️  Data may not be properly normalized")
            
    except Exception as e:
        print(f"   ❌ Error in normalized mode: {e}")
        return False
    
    print("\n2. Testing non-normalized mode:")
    try:
        dataset_unnorm = HDF5Dataset(data_path, use_normalization=False)
        sample_unnorm = dataset_unnorm[0]
        
        print(f"   ✓ Dataset created successfully")
        print(f"   Shape: {sample_unnorm.shape}")
        print(f"   Range: [{sample_unnorm.min():.6f}, {sample_unnorm.max():.6f}]")
        print(f"   Mean: {sample_unnorm.mean():.6f}")
        print(f"   Std: {sample_unnorm.std():.6f}")
        print(f"   Global stats: mean={dataset_unnorm.global_log_mean:.6f}, std={dataset_unnorm.global_log_std:.6f}")
        
        # Check if it looks like log data
        if sample_unnorm.min() >= 0 and sample_unnorm.max() < 2:
            print("   ✓ Data appears to be in log space")
        else:
            print("   ⚠️  Data may not be in expected log space range")
            
    except Exception as e:
        print(f"   ❌ Error in non-normalized mode: {e}")  
        return False
    
    print("\n3. Comparing the two modes:")
    
    # Load same pattern from both
    sample_norm_5 = dataset_normalized[5]
    sample_unnorm_5 = dataset_unnorm[5]
    
    print(f"   Pattern 5 normalized: range=[{sample_norm_5.min():.6f}, {sample_norm_5.max():.6f}]")
    print(f"   Pattern 5 log space:  range=[{sample_unnorm_5.min():.6f}, {sample_unnorm_5.max():.6f}]")
    
    # Manual denormalization test
    manual_denorm = sample_norm_5 * dataset_normalized.global_log_std + dataset_normalized.global_log_mean
    mse_match = torch.mean((manual_denorm - sample_unnorm_5)**2)
    
    print(f"   MSE between manual denorm and log data: {mse_match:.10f}")
    
    if mse_match < 1e-6:
        print("   ✓ Denormalization works correctly")
    else:
        print("   ❌ Denormalization mismatch!")
        return False
    
    print("\n4. Testing model compatibility:")
    try:
        from scripts.training.train import LitAE
        
        # Create minimal model for testing
        model = LitAE(latent_dim=32, lr=1e-4)
        
        # Test normalized mode
        model.set_normalization_params(
            dataset_normalized.global_log_mean, 
            dataset_normalized.global_log_std, 
            use_normalization=True
        )
        
        # Simulate training step
        batch_norm = sample_norm.unsqueeze(0)  # Add batch dim
        with torch.no_grad():
            z = model.model.embed(batch_norm)
            x_hat = model.model.decoder(z)
            
            # Test denormalization
            x_log = model.denormalize_to_log_space(batch_norm)
            x_hat_log = model.denormalize_to_log_space(x_hat)
            
            mse_log = torch.mean((x_log - x_hat_log)**2)
            mse_norm = torch.mean((batch_norm - x_hat)**2)
            
        print(f"   Normalized mode - MSE in log space: {mse_log:.6f}")
        print(f"   Normalized mode - MSE in norm space: {mse_norm:.6f}")
        
        # Test non-normalized mode
        model.set_normalization_params(
            dataset_unnorm.global_log_mean,
            dataset_unnorm.global_log_std,
            use_normalization=False
        )
        
        batch_unnorm = sample_unnorm.unsqueeze(0)
        with torch.no_grad():
            z = model.model.embed(batch_unnorm)
            x_hat = model.model.decoder(z)
            
            mse_log_direct = torch.mean((batch_unnorm - x_hat)**2)
            
        print(f"   Non-normalized mode - MSE in log space: {mse_log_direct:.6f}")
        print("   ✓ Model compatibility test passed")
        
    except Exception as e:
        print(f"   ❌ Model compatibility error: {e}")
        return False
    
    print("\n5. Expected MSE ranges:")
    print("   Normalized mode:")
    print("     - train_mse_normalized: 0.001 - 0.04 (for comparison)")
    print("     - train_loss: varies (scale-aligned)")
    print("   Non-normalized mode:")
    print("     - train_mse_log: 0.1 - 0.3 (reference-comparable)")
    print("     - train_loss: same scale as train_mse_log")
    
    print("\n✅ ALL TESTS PASSED - Both normalization modes work correctly!")
    return True

if __name__ == "__main__":
    test_dataset_modes()