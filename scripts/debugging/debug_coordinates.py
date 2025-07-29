#!/usr/bin/env python3
"""
Debug coordinate mapping for garbled visualization issue.
"""
import numpy as np
import matplotlib.pyplot as plt
import h5py
import sys
from pathlib import Path

# Add visualization path
sys.path.append('scripts/visualization')
from scan_util import raster_coords, coords_to_sparse_image, validate_coords

def load_data():
    """Load the raw data and embeddings for analysis."""
    print("Loading data...")
    
    # Load raw data
    raw_file = "ds4_test_input_data/inputs/train_tensor_ds4_compressed.h5"
    with h5py.File(raw_file, 'r') as f:
        keys = list(f.keys())
        print(f"HDF5 keys: {keys}")
        raw_data = f[keys[0]][:]
        print(f"Raw data shape: {raw_data.shape}")
    
    # Load embeddings
    emb_file = "ds4_test_input_data/ds4_4epoch_embeddings.npz"
    emb_data = np.load(emb_file)
    print(f"Embeddings keys: {list(emb_data.keys())}")
    embeddings = emb_data['embeddings']
    coords_from_file = emb_data.get('spatial_coordinates', None)
    
    print(f"Embeddings shape: {embeddings.shape}")
    if coords_from_file is not None:
        print(f"File coordinates shape: {coords_from_file.shape}")
        print(f"File coordinates range: Y({coords_from_file[:,0].min()}-{coords_from_file[:,0].max()}), X({coords_from_file[:,1].min()}-{coords_from_file[:,1].max()})")
    
    return raw_data, embeddings, coords_from_file

def test_coordinate_systems(raw_data):
    """Test different coordinate systems."""
    N = raw_data.shape[0]
    print(f"\nTesting coordinate systems for {N} patterns...")
    
    # Test 1: Raster coordinates 194x209
    print("\n=== Test 1: Raster 194x209 ===")
    coords_raster = raster_coords(194, 209)
    print(f"Generated raster coords shape: {coords_raster.shape}")
    print(f"First 10 raster coords: {coords_raster[:10]}")
    print(f"Last 10 raster coords: {coords_raster[-10:]}")
    
    # Test mapping
    test_values = np.arange(N, dtype=float)  # Sequential values
    raster_image = coords_to_sparse_image(coords_raster, test_values, (194, 209))
    
    # Test 2: Different ordering - column-major
    print("\n=== Test 2: Column-major 194x209 ===")
    coords_colmajor = []
    for x in range(209):
        for y in range(194):
            coords_colmajor.append([y, x])
    coords_colmajor = np.array(coords_colmajor)
    print(f"First 10 colmajor coords: {coords_colmajor[:10]}")
    colmajor_image = coords_to_sparse_image(coords_colmajor, test_values, (194, 209))
    
    # Test 3: Transposed dimensions 209x194
    print("\n=== Test 3: Transposed 209x194 ===")
    coords_transposed = raster_coords(209, 194)
    transposed_image = coords_to_sparse_image(coords_transposed, test_values, (209, 194))
    
    return raster_image, colmajor_image, transposed_image

def analyze_real_data_structure(raw_data):
    """Analyze the actual structure of the raw data."""
    print(f"\n=== Analyzing real data structure ===")
    
    # Calculate raw mean for each pattern
    if raw_data.ndim == 4:
        raw_mean_values = raw_data.mean(axis=(2, 3))
        if raw_mean_values.ndim == 2:
            raw_mean_values = raw_mean_values.squeeze()
    else:
        raw_mean_values = raw_data.mean(axis=(1, 2))
    
    print(f"Raw mean values shape: {raw_mean_values.shape}")
    print(f"Raw mean range: {raw_mean_values.min():.3f} to {raw_mean_values.max():.3f}")
    
    # Look for patterns that might indicate scan ordering
    # Check if there are smooth transitions that would indicate proper spatial ordering
    N = len(raw_mean_values)
    
    # Test different reshape attempts
    print(f"\nTesting different arrangements...")
    
    # Test gradients to see if data has spatial structure
    for name, coords in [
        ("Raster 194x209", raster_coords(194, 209)),
        ("Column-major", np.array([[y, x] for x in range(209) for y in range(194)])),
    ]:
        try:
            img = coords_to_sparse_image(coords, raw_mean_values, coords[0].shape if name == "Transposed" else (194, 209))
            
            # Calculate spatial gradients as a measure of "smoothness"
            grad_y = np.gradient(img, axis=0)
            grad_x = np.gradient(img, axis=1)
            grad_magnitude = np.sqrt(grad_y**2 + grad_x**2)
            avg_gradient = np.nanmean(grad_magnitude)
            
            print(f"{name}: Average gradient = {avg_gradient:.6f}")
            
        except Exception as e:
            print(f"{name}: Error - {e}")
    
    return raw_mean_values

def create_diagnostic_plots(raw_data, raster_image, colmajor_image, transposed_image, raw_mean_values):
    """Create diagnostic plots to visualize different coordinate mappings."""
    print(f"\nCreating diagnostic plots...")
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Sequential test pattern
    N = len(raw_mean_values)
    test_pattern = np.arange(N).reshape(-1) % 100  # Pattern that repeats every 100
    
    # Plot 1: Raster mapping with test pattern
    coords_raster = raster_coords(194, 209)
    test_raster = coords_to_sparse_image(coords_raster, test_pattern, (194, 209))
    axes[0,0].imshow(test_raster, cmap='viridis')
    axes[0,0].set_title('Test Pattern: Raster 194x209')
    
    # Plot 2: Column-major mapping with test pattern  
    coords_colmajor = np.array([[y, x] for x in range(209) for y in range(194)])
    test_colmajor = coords_to_sparse_image(coords_colmajor, test_pattern, (194, 209))
    axes[0,1].imshow(test_colmajor, cmap='viridis')
    axes[0,1].set_title('Test Pattern: Column-major')
    
    # Plot 3: Raw data with raster mapping
    raw_raster = coords_to_sparse_image(coords_raster, raw_mean_values, (194, 209))
    axes[0,2].imshow(raw_raster, cmap='gray')
    axes[0,2].set_title('Raw Data: Raster mapping')
    
    # Plot 4: Raw data with column-major mapping
    raw_colmajor = coords_to_sparse_image(coords_colmajor, raw_mean_values, (194, 209))
    axes[1,0].imshow(raw_colmajor, cmap='gray')
    axes[1,0].set_title('Raw Data: Column-major')
    
    # Plot 5: Raw data simple reshape (if possible)
    try:
        if len(raw_mean_values) == 194 * 209:
            raw_simple = raw_mean_values.reshape(194, 209)
            axes[1,1].imshow(raw_simple, cmap='gray')
            axes[1,1].set_title('Raw Data: Simple reshape')
        else:
            axes[1,1].text(0.5, 0.5, 'Cannot reshape\n(size mismatch)', ha='center', va='center', transform=axes[1,1].transAxes)
            axes[1,1].set_title('Raw Data: Simple reshape (failed)')
    except:
        axes[1,1].text(0.5, 0.5, 'Reshape failed', ha='center', va='center', transform=axes[1,1].transAxes)
        axes[1,1].set_title('Raw Data: Simple reshape (failed)')
    
    # Plot 6: Show the difference between mappings
    diff = np.abs(raw_raster - raw_colmajor)
    axes[1,2].imshow(diff, cmap='hot')
    axes[1,2].set_title('Difference: Raster vs Column-major')
    
    for ax in axes.flat:
        ax.axis('off')
    
    plt.tight_layout()
    plt.savefig('coordinate_diagnostics.png', dpi=150, bbox_inches='tight')
    print("Saved diagnostic plots to: coordinate_diagnostics.png")
    
    return raw_raster, raw_colmajor

def main():
    """Main diagnostic function."""
    print("=== COORDINATE MAPPING DIAGNOSTICS ===")
    
    # Load data
    raw_data, embeddings, coords_from_file = load_data()
    
    # Test coordinate systems
    raster_image, colmajor_image, transposed_image = test_coordinate_systems(raw_data)
    
    # Analyze real data
    raw_mean_values = analyze_real_data_structure(raw_data)
    
    # Create plots
    raw_raster, raw_colmajor = create_diagnostic_plots(raw_data, raster_image, colmajor_image, transposed_image, raw_mean_values)
    
    # Summary
    print(f"\n=== SUMMARY ===")
    print(f"Data shape: {raw_data.shape}")
    print(f"Expected arrangement: 194 × 209 = {194*209}")
    print(f"Actual patterns: {raw_data.shape[0]}")
    print(f"Missing patterns: {194*209 - raw_data.shape[0]}")
    
    if coords_from_file is not None:
        print(f"File coordinates suggest: {int(coords_from_file[:,0].max())+1} × {int(coords_from_file[:,1].max())+1}")
    
    print(f"\nCheck coordinate_diagnostics.png to see which mapping looks correct!")
    print(f"Look for:")
    print(f"- Smooth spatial gradients (not random noise)")
    print(f"- Recognizable STEM-like features")
    print(f"- Coherent image structure")

if __name__ == "__main__":
    main()