#!/usr/bin/env python
"""Debug script to understand the suspiciously low MSE."""

import torch
import h5py
import numpy as np
from pathlib import Path
import sys
sys.path.append('.')

# Load a trained model checkpoint to test
def debug_mse_calculation():
    print("=== DEBUGGING MSE CALCULATION ===")
    
    # Load some real data
    data_path = Path("ds4_test_input_data/inputs/train_tensor_ds4_compressed.h5")
    with h5py.File(data_path, 'r') as f:
        # Get a small sample
        raw_data = f['patterns'][:10].astype('float32')
    
    print(f"1. Raw data shape: {raw_data.shape}")
    print(f"   Raw data range: [{raw_data.min():.6f}, {raw_data.max():.6f}]")
    
    # Apply exact same preprocessing as dataset
    log_data = np.log(raw_data + 1)
    log_mean = log_data.mean()
    log_std = log_data.std()
    
    normalized = (log_data - log_mean) / (log_std + 1e-8)
    
    print(f"\n2. After log + normalization:")
    print(f"   Range: [{normalized.min():.6f}, {normalized.max():.6f}]")
    print(f"   Mean: {normalized.mean():.6f}")
    print(f"   Std: {normalized.std():.6f}")
    
    # Convert to tensor
    input_tensor = torch.from_numpy(normalized).unsqueeze(1)  # Add channel dim
    
    print(f"\n3. Input tensor shape: {input_tensor.shape}")
    
    # Try loading a model if available
    checkpoint_path = Path("lightning_logs")
    ckpt_files = list(checkpoint_path.rglob("*.ckpt"))
    
    if ckpt_files:
        print(f"\n4. Found checkpoint: {ckpt_files[0]}")
        try:
            # Load checkpoint
            checkpoint = torch.load(ckpt_files[0], map_location='cpu')
            
            # Import the model class
            from scripts.training.lightning_model import LitAE
            from models.autoencoder import Autoencoder
            
            # Create model (you may need to adjust these parameters)
            model = LitAE(latent_dim=32, lr=1e-4)
            
            # Load state dict
            model.load_state_dict(checkpoint['state_dict'])
            model.eval()
            
            # Set normalization params
            model.global_log_mean.data = torch.tensor(log_mean)
            model.global_log_std.data = torch.tensor(log_std)
            
            print(f"   Model loaded successfully")
            print(f"   Normalization params: mean={model.global_log_mean:.6f}, std={model.global_log_std:.6f}")
            
            # Forward pass
            with torch.no_grad():
                z = model.model.embed(input_tensor)
                reconstruction = model.model.decoder(z)
                
                print(f"\n5. Model output:")
                print(f"   Embedding shape: {z.shape}")
                print(f"   Reconstruction shape: {reconstruction.shape}")
                print(f"   Reconstruction range: [{reconstruction.min():.6f}, {reconstruction.max():.6f}]")
                
                # Test denormalization
                input_denorm = model.denormalize_to_log_space(input_tensor)
                recon_denorm = model.denormalize_to_log_space(reconstruction)
                
                print(f"\n6. After denormalization to log space:")
                print(f"   Input range: [{input_denorm.min():.6f}, {input_denorm.max():.6f}]")
                print(f"   Reconstruction range: [{recon_denorm.min():.6f}, {recon_denorm.max():.6f}]")
                
                # Compute MSE in log space
                mse_log = torch.mean((input_denorm - recon_denorm)**2).item()
                
                print(f"\n7. MSE Analysis:")
                print(f"   MSE in log space: {mse_log:.6f}")
                print(f"   Root MSE: {np.sqrt(mse_log):.6f}")
                
                # Compare with normalized space MSE
                mse_normalized = torch.mean((input_tensor - reconstruction)**2).item()
                print(f"   MSE in normalized space: {mse_normalized:.6f}")
                
                # Check if reconstruction is too similar to input
                correlation = torch.corrcoef(torch.stack([
                    input_denorm.flatten(), 
                    recon_denorm.flatten()
                ]))[0, 1].item()
                
                print(f"   Correlation: {correlation:.6f}")
                
                if correlation > 0.99:
                    print("   ⚠️  Correlation > 0.99 suggests model might be learning identity!")
                elif mse_log < 0.005:
                    print("   ⚠️  Very low MSE suggests model is reconstructing very well")
                else:
                    print("   ✓ MSE seems reasonable for a trained model")
                    
        except Exception as e:
            print(f"   Error loading model: {e}")
    else:
        print(f"\n4. No checkpoint found in {checkpoint_path}")
        print("   Run training first to generate a checkpoint")

if __name__ == "__main__":
    debug_mse_calculation()