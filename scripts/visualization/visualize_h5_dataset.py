#!/usr/bin/env python3
"""
Standalone script for visualizing 4D-STEM datasets stored in compressed H5 format.

This script loads a compressed .h5 dataset and generates:
1. Mean diffraction pattern
2. Virtual bright field image (circular detector)
3. Virtual dark field image (annular detector)

Usage:
    python visualize_h5_dataset.py <h5_file> [options]

Example:
    python visualize_h5_dataset.py /path/to/train_tensor_ds4_compressed.h5 --output results.png
"""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
import json
from pathlib import Path
from typing import Tuple, Optional, Dict, Any

# Import the migrated STEMVisualizer
from stem_visualization import STEMVisualizer


def load_h5_dataset(h5_path: str) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load 4D-STEM data from compressed H5 file.
    
    Args:
        h5_path: Path to the .h5 file
        
    Returns:
        Tuple of (data_array, metadata)
    """
    print(f"Loading dataset from: {h5_path}")
    
    with h5py.File(h5_path, 'r') as f:
        # Load the main data array
        if 'data' in f:
            data = f['data'][:]
        elif 'diffraction_patterns' in f:
            data = f['diffraction_patterns'][:]
        else:
            # Try to find the main data array
            keys = list(f.keys())
            data_key = keys[0]  # Use first key as fallback
            print(f"Using dataset key: {data_key}")
            data = f[data_key][:]
        
        print(f"Loaded data shape: {data.shape}")
        print(f"Data type: {data.dtype}")
        print(f"Data range: {data.min():.2f} - {data.max():.2f}")
        
        # Try to load metadata from attributes
        metadata = {}
        for key, value in f.attrs.items():
            metadata[key] = value
            
    return data, metadata


# Interactive beam center detection is now handled by STEMVisualizer.interactive_beam_center_detection()


def load_metadata_file(h5_path: str) -> Optional[Dict[str, Any]]:
    """
    Load accompanying metadata JSON file if it exists.
    
    Args:
        h5_path: Path to the .h5 file
        
    Returns:
        Metadata dictionary or None if file doesn't exist
    """
    metadata_path = Path(h5_path).with_suffix('').with_suffix('.h5_metadata.json')
    if not metadata_path.exists():
        # Try alternative naming
        metadata_path = Path(h5_path).parent / f"{Path(h5_path).stem}_metadata.json"
    
    if metadata_path.exists():
        print(f"Loading metadata from: {metadata_path}")
        with open(metadata_path, 'r') as f:
            return json.load(f)
    else:
        print("No metadata file found")
        return None


def infer_scan_shape(data_shape: Tuple[int, ...]) -> Tuple[int, int]:
    """
    Infer scan shape from data dimensions.
    
    Args:
        data_shape: Shape of the 4D data array
        
    Returns:
        Tuple of (scan_y, scan_x)
    """
    if len(data_shape) == 4:
        # Already in (scan_y, scan_x, det_y, det_x) format
        return data_shape[:2]
    elif len(data_shape) == 3:
        # Flattened format (N, det_y, det_x)
        n_patterns = data_shape[0]
        # Find the most square-like factorization
        factors = []
        for i in range(1, int(np.sqrt(n_patterns)) + 1):
            if n_patterns % i == 0:
                factors.append((i, n_patterns // i))
        return min(factors, key=lambda x: abs(x[0] - x[1]))
    else:
        raise ValueError(f"Unexpected data shape: {data_shape}")


def create_visualization(visualizer: STEMVisualizer, output_path: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Create comprehensive visualization of the 4D-STEM dataset using STEMVisualizer methods.
    
    Args:
        visualizer: Pre-configured STEMVisualizer instance
        output_path: Path to save the visualization
        metadata: Optional metadata dictionary
    """
    print("Creating visualization with configured STEMVisualizer...")
    
    print(f"Direct beam position: {visualizer.direct_beam_position}")
    print(f"Pattern shape: {visualizer.pattern_shape}")
    print(f"Scan shape: {visualizer.scan_shape}")
    
    # Use the comprehensive visualization method from STEMVisualizer
    fig = visualizer.create_comprehensive_visualization(
        figsize=(20, 10), 
        include_metadata=True, 
        metadata=metadata
    )
    
    # Save figure
    print(f"Saving visualization to: {output_path}")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Return the virtual images for summary statistics
    bf_image = visualizer.create_bright_field_image()
    center_y, center_x, inner_radius, outer_radius = visualizer.dark_field_region
    df_image = visualizer.create_dark_field_image(inner_radius, outer_radius)
    haadf_image = visualizer.create_haadf_image()
    
    return bf_image, df_image, haadf_image


def print_dataset_info(data: np.ndarray, metadata: Optional[Dict[str, Any]] = None, 
                      external_metadata: Optional[Dict[str, Any]] = None):
    """Print comprehensive dataset information."""
    print("\n" + "="*60)
    print("DATASET INFORMATION")
    print("="*60)
    
    print(f"Data shape: {data.shape}")
    print(f"Data type: {data.dtype}")
    print(f"Data size: {data.nbytes / 1e6:.1f} MB")
    print(f"Data range: [{data.min():.2e}, {data.max():.2e}]")
    print(f"Data mean: {data.mean():.2e}")
    
    # Calculate std safely to avoid overflow with float16
    try:
        data_std = np.std(data.astype(np.float64))
        print(f"Data std: {data_std:.2e}")
    except:
        print("Data std: calculation overflow (large dataset)")
    
    if len(data.shape) == 3:
        scan_shape = infer_scan_shape(data.shape)
        pattern_shape = data.shape[-2:]
        print(f"Inferred scan shape: {scan_shape}")
        print(f"Pattern shape: {pattern_shape}")
        print(f"Total patterns: {data.shape[0]}")
    
    # Print metadata if available
    if metadata:
        print("\nH5 Metadata:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    
    if external_metadata:
        print("\nExternal Metadata:")
        for key, value in external_metadata.items():
            if isinstance(value, dict):
                print(f"  {key}:")
                for k, v in value.items():
                    print(f"    {k}: {v}")
            else:
                print(f"  {key}: {value}")


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description='Visualize compressed 4D-STEM datasets from H5 files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic visualization with automatic detection
  python visualize_h5_dataset.py data.h5
  
  # Interactive beam center detection (recommended for precise results)
  python visualize_h5_dataset.py data.h5 --interactive-center
  
  # Specify output file
  python visualize_h5_dataset.py data.h5 --output my_results.png
  
  # Custom detector radii
  python visualize_h5_dataset.py data.h5 --bf-radius 15 --df-inner 20 --df-outer 50
        """
    )
    
    parser.add_argument('h5_file', type=str, help='Path to the compressed H5 dataset')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output visualization file (default: auto-generated)')
    parser.add_argument('--bf-radius', type=int, default=None,
                       help='Bright field detector radius in pixels')
    parser.add_argument('--df-inner', type=int, default=None,
                       help='Dark field inner radius in pixels')
    parser.add_argument('--df-outer', type=int, default=None,
                       help='Dark field outer radius in pixels')
    parser.add_argument('--scan-shape', nargs=2, type=int, default=None,
                       help='Manual scan shape (scan_y scan_x)')
    parser.add_argument('--info-only', action='store_true',
                       help='Only print dataset information, skip visualization')
    parser.add_argument('--interactive-center', action='store_true',
                       help='Interactive beam center and radius detection (click center, then edge)')
    
    args = parser.parse_args()
    
    # Validate input file
    h5_path = Path(args.h5_file)
    if not h5_path.exists():
        print(f"Error: File not found: {h5_path}")
        return 1
    
    try:
        # Load data and metadata
        data, h5_metadata = load_h5_dataset(str(h5_path))
        external_metadata = load_metadata_file(str(h5_path))
        
        # Print dataset information
        print_dataset_info(data, h5_metadata, external_metadata)
        
        if args.info_only:
            return 0
        
        # Determine scan shape
        if args.scan_shape:
            scan_shape = tuple(args.scan_shape)
            print(f"Using manual scan shape: {scan_shape}")
        elif external_metadata and 'original_shape' in external_metadata:
            # Use original scan shape from metadata
            original_shape = external_metadata['original_shape']
            # Fix: Don't swap dimensions - use original order
            scan_shape = (original_shape[0], original_shape[1])  # (scan_y, scan_x)
            print(f"Using metadata scan shape: {scan_shape}")
        else:
            scan_shape = infer_scan_shape(data.shape)
            print(f"Inferred scan shape: {scan_shape}")
            
        # Validate scan shape matches data
        expected_patterns = scan_shape[0] * scan_shape[1]
        if expected_patterns != data.shape[0]:
            print(f"Warning: Scan shape mismatch! Expected {expected_patterns} patterns, got {data.shape[0]}")
            print("Falling back to automatic inference...")
            scan_shape = infer_scan_shape(data.shape)
            print(f"Corrected scan shape: {scan_shape}")
        
        # Generate output filename if not specified
        if args.output is None:
            output_path = h5_path.parent / f"{h5_path.stem}_visualization.png"
        else:
            output_path = Path(args.output)
        
        # Create visualizer
        visualizer = STEMVisualizer(data, scan_shape)
        
        # Interactive beam center detection if requested
        if args.interactive_center:
            print("Starting interactive beam center detection...")
            visualizer.apply_interactive_detection()
            print(f"Interactive detection complete: center={visualizer.direct_beam_position}, radius={visualizer.bragg_radius:.1f}")
        
        # Set custom detector parameters if provided
        if args.bf_radius is not None:
            print(f"Using custom bright field radius: {args.bf_radius}")
        
        if args.df_inner is not None and args.df_outer is not None:
            print(f"Using custom dark field radii: inner={args.df_inner}, outer={args.df_outer}")
            center_y, center_x = visualizer.direct_beam_position
            visualizer.set_dark_field_annular(center_y, center_x, args.df_inner, args.df_outer)
        
        # Create visualization using the configured visualizer
        bf_image, df_image, haadf_image = create_visualization(
            visualizer, str(output_path), external_metadata
        )
        
        # Print results summary
        print("\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        print(f"Bright field intensity range: [{bf_image.min():.1f}, {bf_image.max():.1f}]")
        print(f"Dark field intensity range: [{df_image.min():.1f}, {df_image.max():.1f}]")
        print(f"HAADF intensity range: [{haadf_image.min():.1f}, {haadf_image.max():.1f}]")
        print(f"Bragg spot radius: {visualizer.bragg_radius:.1f} pixels")
        print(f"BF detector: r={visualizer.bragg_radius*0.8:.1f}px")
        df_inner, df_outer = visualizer.dark_field_region[2:4]
        print(f"DF detector: r={df_inner:.1f}-{df_outer:.1f}px")
        haadf_inner, haadf_outer = visualizer.haadf_region[2:4]
        print(f"HAADF detector: r={haadf_inner:.1f}-{haadf_outer:.1f}px")
        print(f"Visualization saved: {output_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error processing dataset: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())