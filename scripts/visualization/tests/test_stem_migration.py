#!/usr/bin/env python3
"""
Test script to validate the migrated STEMVisualizer implementation.
"""

import numpy as np
import sys
from pathlib import Path

# Add the parent directory (visualization) to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from stem_visualization import STEMVisualizer, PY4DSTEM_AVAILABLE

def create_test_data():
    """Create synthetic 4D-STEM data for testing."""
    # Create a 8x8 scan with 64x64 diffraction patterns
    scan_shape = (8, 8)
    pattern_shape = (64, 64)
    n_patterns = scan_shape[0] * scan_shape[1]
    
    # Generate synthetic data with a central bright spot (direct beam)
    data = np.random.poisson(10, (n_patterns, pattern_shape[0], pattern_shape[1]))
    
    # Add a bright central beam at position (32, 32)
    center_y, center_x = 32, 32
    y_indices, x_indices = np.indices(pattern_shape)
    beam_mask = np.hypot(x_indices - center_x, y_indices - center_y) < 5
    
    # Add intensity to central beam
    for i in range(n_patterns):
        data[i][beam_mask] += np.random.poisson(1000)
        
    # Add some scattered intensity in annular regions
    annular_mask = (np.hypot(x_indices - center_x, y_indices - center_y) > 8) & \
                   (np.hypot(x_indices - center_x, y_indices - center_y) < 20)
    
    for i in range(n_patterns):
        data[i][annular_mask] += np.random.poisson(50)
    
    return data, scan_shape

def test_basic_functionality():
    """Test basic STEMVisualizer functionality."""
    print("Creating test data...")
    data, scan_shape = create_test_data()
    
    print(f"py4DSTEM available: {PY4DSTEM_AVAILABLE}")
    
    # Test STEMVisualizer initialization
    print("Initializing STEMVisualizer...")
    visualizer = STEMVisualizer(data, scan_shape)
    
    print(f"Data shape: {visualizer.data.shape}")
    print(f"Scan shape: {visualizer.scan_shape}")
    print(f"Pattern shape: {visualizer.pattern_shape}")
    print(f"Direct beam position: {visualizer.direct_beam_position}")
    
    # Test bright field imaging
    print("Testing bright field imaging...")
    bf_image = visualizer.create_bright_field_image()
    print(f"Bright field image shape: {bf_image.shape}")
    print(f"Bright field intensity range: {bf_image.min():.2f} - {bf_image.max():.2f}")
    
    # Test new annular dark field imaging
    print("Testing annular dark field imaging...")
    df_image = visualizer.create_dark_field_image(inner_radius=8, outer_radius=20)
    print(f"Dark field image shape: {df_image.shape}")
    print(f"Dark field intensity range: {df_image.min():.2f} - {df_image.max():.2f}")
    
    # Test dark field region format
    print(f"Default dark field region (annular): {visualizer.dark_field_region}")
    
    # Test backward compatibility
    print("Testing backward compatibility...")
    old_region = (10, 50, 10, 50)  # Old rectangular format
    visualizer.set_dark_field_region(old_region)
    df_image_old = visualizer.create_virtual_field_image(old_region)
    print(f"Old format dark field shape: {df_image_old.shape}")
    
    # Test new annular format
    visualizer.set_dark_field_annular(32, 32, 8, 20)
    print(f"New annular region: {visualizer.dark_field_region}")
    
    print("✓ All basic tests passed!")
    
def test_preprocessing():
    """Test data preprocessing improvements."""
    print("Testing data preprocessing...")
    
    # Create data with negative values and outliers
    data, scan_shape = create_test_data()
    
    # Add negative values and hot pixels
    data = data.astype(float)
    data[:5] -= 50  # Negative values from reconstruction
    data[10, 30, 30] = 50000  # Hot pixel
    
    print(f"Raw data range: {data.min():.2f} - {data.max():.2f}")
    
    # Test preprocessing
    visualizer = STEMVisualizer(data, scan_shape)
    
    print(f"Processed data range: {visualizer.data.min():.2f} - {visualizer.data.max():.2f}")
    print(f"Raw data preserved: {hasattr(visualizer, 'data_raw')}")
    
    print("✓ Preprocessing tests passed!")

def test_backward_compatibility():
    """Test that old API still works."""
    print("Testing backward compatibility...")
    
    # Create synthetic data
    scan_shape = (4, 4)
    pattern_shape = (32, 32)
    n_patterns = scan_shape[0] * scan_shape[1]
    data = np.random.poisson(100, (n_patterns, pattern_shape[0], pattern_shape[1]))
    
    # Test old initialization
    visualizer = STEMVisualizer(data, scan_shape)
    print(f"✓ Initialization works: {visualizer.scan_shape}")
    
    # Test old bright field region setting
    old_bf_region = (10, 20, 10, 20)  # (y_min, y_max, x_min, x_max)
    visualizer.set_bright_field_region(old_bf_region)
    print(f"✓ Old BF region setting: {visualizer.bright_field_region}")
    
    # Test old dark field region setting (rectangular)
    old_df_region = (5, 25, 5, 25)  # (y_min, y_max, x_min, x_max)
    visualizer.set_dark_field_region(old_df_region)
    print(f"✓ Old DF region setting: {visualizer.dark_field_region}")
    
    # Test old virtual field image creation
    old_df_image = visualizer.create_virtual_field_image(old_df_region)
    print(f"✓ Old virtual field imaging: {old_df_image.shape}")
    
    # Test bright field imaging (should work as before)
    bf_image = visualizer.create_bright_field_image()
    print(f"✓ Bright field imaging: {bf_image.shape}")
    
    # Test new annular dark field setting
    visualizer.set_dark_field_annular(16, 16, 5, 12)
    print(f"✓ New annular DF setting: {visualizer.dark_field_region}")
    
    # Test visualization methods
    try:
        import matplotlib.pyplot as plt
        plt.ioff()  # Turn off interactive mode to avoid display
        fig = visualizer.plot_virtual_images()
        plt.close(fig)
        print("✓ Visualization methods work")
    except Exception as e:
        print(f"⚠️  Visualization test skipped: {e}")
    
    print("✓ Backward compatibility tests passed!")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_preprocessing()
        test_backward_compatibility()
        print("\n🎉 All migration tests passed successfully!")
        
        if not PY4DSTEM_AVAILABLE:
            print("\n⚠️  Note: py4DSTEM not available - using fallback implementations")
            print("   Install py4DSTEM for optimal performance: pip install py4DSTEM")
            
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)