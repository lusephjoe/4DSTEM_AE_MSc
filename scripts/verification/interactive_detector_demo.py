#!/usr/bin/env python3
"""
Interactive Detector Geometry Demo

This script demonstrates the new interactive detector geometry functionality
for 4D-STEM visualization.

Usage:
    python interactive_detector_demo.py
"""

import numpy as np
import sys
import os

# Add the scripts directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'scripts', 'visualization'))

from stem_visualization import STEMVisualizer


def create_synthetic_stem_data():
    """Create synthetic 4D-STEM data with realistic patterns."""
    print("Creating synthetic 4D-STEM data...")
    
    # Create scan grid
    scan_y, scan_x = 20, 20
    pattern_y, pattern_x = 128, 128
    
    # Initialize data array
    data = np.zeros((scan_y * scan_x, pattern_y, pattern_x), dtype=np.float32)
    
    # Create center coordinates
    center_y, center_x = pattern_y // 2, pattern_x // 2
    
    # Create coordinate grids
    y_coords, x_coords = np.meshgrid(np.arange(pattern_y), np.arange(pattern_x), indexing='ij')
    
    for i in range(scan_y * scan_x):
        # Create distance array from center
        distances = np.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
        
        # Central beam (Gaussian)
        central_beam = 1000 * np.exp(-(distances**2) / (2 * 8**2))
        
        # Add some Bragg spots at specific radii
        bragg_radius = 25
        bragg_spots = 200 * (
            np.exp(-((distances - bragg_radius)**2) / (2 * 3**2)) +
            np.exp(-((distances - bragg_radius * 1.5)**2) / (2 * 2**2)) +
            np.exp(-((distances - bragg_radius * 2)**2) / (2 * 1.5**2))
        )
        
        # Add some noise
        noise = np.random.exponential(5, (pattern_y, pattern_x))
        
        # Combine components
        pattern = central_beam + bragg_spots + noise
        
        # Add slight variations across the scan
        scan_row, scan_col = divmod(i, scan_x)
        variation = 1.0 + 0.2 * np.sin(scan_row * np.pi / scan_y) * np.cos(scan_col * np.pi / scan_x)
        pattern *= variation
        
        data[i] = pattern
    
    print(f"Created data shape: {data.shape}")
    print(f"Data range: {data.min():.1f} - {data.max():.1f}")
    
    return data, (scan_y, scan_x)


def main():
    """Main demo function."""
    print("="*60)
    print("Interactive Detector Geometry Demo")
    print("="*60)
    
    # Create synthetic data
    data, scan_shape = create_synthetic_stem_data()
    
    # Create visualizer
    print("\nInitializing STEM Visualizer...")
    visualizer = STEMVisualizer(data, scan_shape)
    
    print(f"Initial detector parameters:")
    print(f"  Beam center: {visualizer.direct_beam_position}")
    print(f"  Bragg radius: {visualizer.bragg_radius:.1f}")
    print(f"  BF region: {visualizer.bright_field_region}")
    print(f"  DF region: {visualizer.dark_field_region}")
    print(f"  HAADF region: {visualizer.haadf_region}")
    
    print("\n" + "="*60)
    print("INTERACTIVE DETECTOR SETUP")
    print("="*60)
    print("Instructions:")
    print("1. A window will open showing the mean diffraction pattern")
    print("2. Use the buttons on the left to select which detector to adjust:")
    print("   - Beam Center: Click center, then edge to set beam position and radius")
    print("   - BF Radius: Click to set bright field detector size")
    print("   - DF Inner/Outer: Click to set dark field annulus boundaries")
    print("   - HAADF Inner/Outer: Click to set high-angle detector boundaries")
    print("3. The preview window shows the resulting virtual image")
    print("4. Click 'Reset' to restore default values")
    print("5. Click 'Done' when satisfied with the detector setup")
    
    input("\nPress Enter to start interactive detector setup...")
    
    try:
        # Run interactive detector setup
        detector_params = visualizer.apply_interactive_detector_setup()
        
        print("\n" + "="*60)
        print("SETUP COMPLETE!")
        print("="*60)
        print("Final detector parameters:")
        for key, value in detector_params.items():
            if key == 'beam_center':
                print(f"  {key}: {value}")
            else:
                print(f"  {key}: {value:.1f}")
        
        # Create final visualization
        print("\nCreating final visualization...")
        fig = visualizer.create_comprehensive_visualization(figsize=(20, 10))
        
        # Save result
        output_path = "interactive_detector_demo_result.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Saved result to: {output_path}")
        
        # Show statistics
        bf_image = visualizer.create_bright_field_image()
        df_image = visualizer.create_dark_field_image(*visualizer.dark_field_region[2:4])
        haadf_image = visualizer.create_haadf_image()
        
        print("\nVirtual image statistics:")
        print(f"  BF range: [{bf_image.min():.1f}, {bf_image.max():.1f}]")
        print(f"  DF range: [{df_image.min():.1f}, {df_image.max():.1f}]")
        print(f"  HAADF range: [{haadf_image.min():.1f}, {haadf_image.max():.1f}]")
        
    except KeyboardInterrupt:
        print("\nDemo cancelled by user")
    except Exception as e:
        print(f"\nDemo failed with error: {e}")
        raise
    
    print("\nDemo completed!")


if __name__ == "__main__":
    main()