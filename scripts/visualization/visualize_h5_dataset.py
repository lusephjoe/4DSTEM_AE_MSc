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


def create_visualization(data: np.ndarray, scan_shape: Tuple[int, int], 
                        output_path: str, metadata: Optional[Dict[str, Any]] = None):
    """
    Create comprehensive visualization of the 4D-STEM dataset.
    
    Args:
        data: 4D-STEM data array
        scan_shape: Shape of the scan grid (scan_y, scan_x)
        output_path: Path to save the visualization
        metadata: Optional metadata dictionary
    """
    print("Creating STEMVisualizer...")
    
    # Initialize the visualizer
    visualizer = STEMVisualizer(data, scan_shape)
    
    print(f"Direct beam position: {visualizer.direct_beam_position}")
    print(f"Pattern shape: {visualizer.pattern_shape}")
    print(f"Scan shape: {visualizer.scan_shape}")
    
    # Create figure with subplots for all detector types
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Row 1: Mean patterns and analysis
    # 1. Mean diffraction pattern with all detector regions
    visualizer.plot_mean_diffraction_pattern(axes[0, 0], show_regions=True, use_log=True, add_scalebar=True)
    axes[0, 0].set_title('Mean Diffraction Pattern\n(All Detector Regions)')
    
    # 2. Radial profile analysis
    print("Creating radial profile analysis...")
    mean_pattern = np.mean(visualizer.data, axis=0)
    center_y, center_x = visualizer.direct_beam_position
    y_indices, x_indices = np.indices(mean_pattern.shape)
    distances = np.hypot(x_indices - center_x, y_indices - center_y)
    
    # Calculate radial profile
    max_radius = min(mean_pattern.shape) // 2
    radii = np.arange(1, max_radius)
    radial_profile = []
    for r in radii:
        ring_mask = (distances >= r - 0.5) & (distances < r + 0.5)
        if np.sum(ring_mask) > 0:
            radial_profile.append(np.mean(mean_pattern[ring_mask]))
        else:
            radial_profile.append(0)
    
    axes[0, 1].plot(radii, radial_profile, 'b-', linewidth=2)
    axes[0, 1].axvline(visualizer.bragg_radius, color='yellow', linestyle='--', label=f'Bragg: {visualizer.bragg_radius:.1f}')
    axes[0, 1].axvline(visualizer.bragg_radius * 0.8, color='red', linestyle=':', label='BF limit')
    axes[0, 1].axvline(visualizer.bragg_radius * 1.5, color='orange', linestyle=':', label='HAADF start')
    axes[0, 1].set_xlabel('Radius (pixels)')
    axes[0, 1].set_ylabel('Intensity')
    axes[0, 1].set_title('Radial Profile Analysis')
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].set_yscale('log')
    
    # 3. Mean pattern - log scale
    im2 = axes[0, 2].imshow(np.log(mean_pattern + 1), cmap='viridis')
    axes[0, 2].set_title('Mean Pattern (Log Scale)')
    axes[0, 2].axis('off')
    plt.colorbar(im2, ax=axes[0, 2], shrink=0.6)
    
    # 4. Detector geometry schematic
    axes[0, 3].set_xlim(0, 100)
    axes[0, 3].set_ylim(0, 100)
    axes[0, 3].set_aspect('equal')
    
    # Draw detector schematic
    center = 50
    bf_r = 15
    df_inner, df_outer = 18, 22
    haadf_inner, haadf_outer = 30, 45
    
    bf_circle = plt.Circle((center, center), bf_r, fill=True, alpha=0.3, color='red', label='BF')
    df_ring = plt.Circle((center, center), df_outer, fill=False, color='blue', linewidth=3, label='DF')
    df_inner_circle = plt.Circle((center, center), df_inner, fill=False, color='blue', linewidth=1, linestyle='--')
    haadf_ring = plt.Circle((center, center), haadf_outer, fill=False, color='orange', linewidth=3, label='HAADF')
    haadf_inner_circle = plt.Circle((center, center), haadf_inner, fill=False, color='orange', linewidth=1, linestyle='--')
    
    axes[0, 3].add_patch(bf_circle)
    axes[0, 3].add_patch(df_ring)
    axes[0, 3].add_patch(df_inner_circle)
    axes[0, 3].add_patch(haadf_ring)
    axes[0, 3].add_patch(haadf_inner_circle)
    axes[0, 3].plot(center, center, 'k+', markersize=10, markeredgewidth=2)
    
    axes[0, 3].set_title('Detector Geometry')
    axes[0, 3].legend(loc='upper right', fontsize=8)
    axes[0, 3].axis('off')
    
    # Row 2: Virtual images from all detector types
    # 1. Bright field image
    print("Creating bright field image...")
    bf_image = visualizer.create_bright_field_image()
    im3 = axes[1, 0].imshow(bf_image, cmap='gray')
    axes[1, 0].set_title(f'Virtual Bright Field\n(r={visualizer.bragg_radius*0.8:.1f}px)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.6)
    
    # 2. Conventional Dark field image
    print("Creating conventional dark field image...")
    center_y, center_x, inner_radius, outer_radius = visualizer.dark_field_region
    df_image = visualizer.create_dark_field_image(inner_radius, outer_radius)
    im4 = axes[1, 1].imshow(df_image, cmap='hot')
    axes[1, 1].set_title(f'Conventional Dark Field\n(r={inner_radius:.1f}-{outer_radius:.1f}px)')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.6)
    
    # 3. HAADF image
    print("Creating HAADF image...")
    haadf_image = visualizer.create_haadf_image()
    im5 = axes[1, 2].imshow(haadf_image, cmap='plasma')
    haadf_center_y, haadf_center_x, haadf_inner, haadf_outer = visualizer.haadf_region
    axes[1, 2].set_title(f'HAADF (Z-contrast)\n(r={haadf_inner:.1f}-{haadf_outer:.1f}px)')
    axes[1, 2].axis('off')
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.6)
    
    # 4. Comparison: BF vs HAADF
    print("Creating BF/HAADF comparison...")
    # Normalize both images for comparison
    bf_norm = (bf_image - bf_image.min()) / (bf_image.max() - bf_image.min())
    haadf_norm = (haadf_image - haadf_image.min()) / (haadf_image.max() - haadf_image.min())
    
    # Create RGB composite: BF=red, HAADF=green
    composite = np.zeros((*bf_image.shape, 3))
    composite[..., 0] = bf_norm  # Red channel = BF
    composite[..., 1] = haadf_norm  # Green channel = HAADF
    composite[..., 2] = 0.5 * (bf_norm + haadf_norm)  # Blue = average
    
    axes[1, 3].imshow(composite)
    axes[1, 3].set_title('BF/HAADF Composite\n(Red=BF, Green=HAADF)')
    axes[1, 3].axis('off')
    
    # Add metadata text if available
    if metadata:
        info_text = []
        if 'original_shape' in metadata:
            info_text.append(f"Original: {metadata['original_shape']}")
        if 'compression_ratio' in metadata:
            info_text.append(f"Compression: {metadata['compression_ratio']:.1f}x")
        if 'normalization_method' in metadata:
            info_text.append(f"Norm: {metadata['normalization_method']}")
            
        if info_text:
            fig.suptitle(' | '.join(info_text), y=0.02, fontsize=10)
    
    # Add statistics
    stats_text = f"Data: {data.shape} | Range: [{data.min():.1f}, {data.max():.1f}] | Direct beam: {visualizer.direct_beam_position}"
    fig.suptitle(stats_text, y=0.98, fontsize=10)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93, bottom=0.07)
    
    # Save figure
    print(f"Saving visualization to: {output_path}")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
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
  # Basic visualization
  python visualize_h5_dataset.py data.h5
  
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
        
        # Set custom detector parameters if provided
        if args.bf_radius is not None:
            print(f"Using custom bright field radius: {args.bf_radius}")
        
        if args.df_inner is not None and args.df_outer is not None:
            print(f"Using custom dark field radii: inner={args.df_inner}, outer={args.df_outer}")
            center_y, center_x = visualizer.direct_beam_position
            visualizer.set_dark_field_annular(center_y, center_x, args.df_inner, args.df_outer)
        
        # Create visualization
        bf_image, df_image, haadf_image = create_visualization(
            data, scan_shape, str(output_path), external_metadata
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