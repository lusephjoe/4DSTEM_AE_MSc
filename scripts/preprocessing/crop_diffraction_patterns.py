#!/usr/bin/env python3
"""
CLI script for automatic diffraction pattern cropping.

This script provides a command-line interface for the automatic diffraction pattern 
cropping functionality, allowing users to process 4D-STEM datasets with various 
configuration options.

Usage:
    python crop_diffraction_patterns.py --input data.pt --output cropped_data.pt --target-retention 0.98

Author: Claude Code Assistant
Date: 2025-01-18
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from scripts.preprocessing.diffraction_cropping import DiffractionCropper, CroppingConfig


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Automatic diffraction pattern cropping for 4D-STEM data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output
    parser.add_argument(
        "--input", "-i", 
        type=Path, 
        help="Input tensor file (.pt format)"
    )
    parser.add_argument(
        "--output", "-o", 
        type=Path, 
        required=True,
        help="Output cropped tensor file (.pt format)"
    )
    
    # Cropping parameters
    parser.add_argument(
        "--target-retention", 
        type=float, 
        default=0.98,
        help="Target intensity retention (0-1)"
    )
    parser.add_argument(
        "--min-retention", 
        type=float, 
        default=0.95,
        help="Minimum acceptable retention (0-1)"
    )
    parser.add_argument(
        "--margin-pixels", 
        type=int, 
        default=3,
        help="Additional pixels around computed radius"
    )
    
    # Processing parameters
    parser.add_argument(
        "--chunk-size", 
        type=int, 
        default=1000,
        help="Patterns per chunk for memory efficiency"
    )
    parser.add_argument(
        "--use-gpu", 
        action="store_true",
        help="Enable GPU acceleration with CuPy"
    )
    
    # Center detection
    parser.add_argument(
        "--center-method", 
        choices=["centroid", "center", "manual"], 
        default="centroid",
        help="Method for determining pattern center"
    )
    parser.add_argument(
        "--manual-center", 
        type=int, 
        nargs=2, 
        metavar=("Y", "X"),
        help="Manual center coordinates (y, x) for all patterns"
    )
    
    # Visualization and output
    parser.add_argument(
        "--no-visualization", 
        action="store_true",
        help="Disable visualization output"
    )
    parser.add_argument(
        "--quiet", "-q", 
        action="store_true",
        help="Suppress verbose output"
    )
    parser.add_argument(
        "--test-mode", 
        action="store_true",
        help="Use synthetic test data instead of input file"
    )
    parser.add_argument(
        "--test-patterns", 
        type=int, 
        default=1000,
        help="Number of test patterns to generate"
    )
    parser.add_argument(
        "--test-size", 
        type=int, 
        default=256,
        help="Size of test patterns"
    )
    
    return parser.parse_args()


def validate_arguments(args):
    """Validate command line arguments."""
    errors = []
    
    # Check retention values
    if not (0 < args.target_retention <= 1):
        errors.append("target-retention must be between 0 and 1")
    
    if not (0 < args.min_retention <= 1):
        errors.append("min-retention must be between 0 and 1")
    
    if args.min_retention > args.target_retention:
        errors.append("min-retention cannot be greater than target-retention")
    
    # Check file paths
    if not args.test_mode:
        if args.input is None:
            errors.append("Input file is required when not in test mode")
        elif not args.input.exists():
            errors.append(f"Input file does not exist: {args.input}")
        elif not args.input.suffix.lower() in ['.pt', '.pth']:
            errors.append("Input file must be a PyTorch tensor file (.pt or .pth)")
    
    # Check output directory
    if not args.output.parent.exists():
        try:
            args.output.parent.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            errors.append(f"Cannot create output directory: {e}")
    
    # Check manual center
    if args.center_method == "manual" and args.manual_center is None:
        errors.append("Manual center coordinates required when using manual center method")
    
    if errors:
        print("Error: Invalid arguments:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)


def load_data(input_path: Path, quiet: bool = False) -> torch.Tensor:
    """Load tensor data from file."""
    if not quiet:
        print(f"Loading data from {input_path}...")
    
    try:
        data = torch.load(input_path, map_location='cpu')
        if not quiet:
            print(f"Loaded tensor with shape: {data.shape}")
            print(f"Data type: {data.dtype}")
            print(f"Memory usage: {data.element_size() * data.numel() / 1024**3:.2f} GB")
        return data
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)


def create_test_data(n_patterns: int, image_size: int, quiet: bool = False) -> torch.Tensor:
    """Create synthetic test data."""
    if not quiet:
        print(f"Creating synthetic test data: {n_patterns} patterns of size {image_size}x{image_size}")
    
    from scripts.preprocessing.diffraction_cropping import create_test_data
    return create_test_data(n_patterns, image_size)


def main():
    """Main function."""
    args = parse_arguments()
    validate_arguments(args)
    
    # Create configuration
    config = CroppingConfig(
        target_retention=args.target_retention,
        min_retention=args.min_retention,
        margin_pixels=args.margin_pixels,
        chunk_size=args.chunk_size,
        use_gpu=args.use_gpu,
        center_method=args.center_method,
        manual_center=tuple(args.manual_center) if args.manual_center else None,
        visualization=not args.no_visualization,
        verbose=not args.quiet
    )
    
    # Load or create data
    if args.test_mode:
        data = create_test_data(args.test_patterns, args.test_size, args.quiet)
    else:
        data = load_data(args.input, args.quiet)
    
    # Process data
    if not args.quiet:
        print("\nStarting diffraction pattern cropping...")
        print(f"Configuration:")
        print(f"  Target retention: {config.target_retention:.3f}")
        print(f"  Min retention: {config.min_retention:.3f}")
        print(f"  Margin pixels: {config.margin_pixels}")
        print(f"  Chunk size: {config.chunk_size}")
        print(f"  GPU acceleration: {config.use_gpu}")
        print(f"  Center method: {config.center_method}")
        if config.manual_center:
            print(f"  Manual center: {config.manual_center}")
        print()
    
    # Create cropper and process
    cropper = DiffractionCropper(config)
    
    try:
        cropped_data, results = cropper.process_dataset(data, args.output)
        
        # Print summary
        if not args.quiet:
            print("\n" + "="*60)
            print("PROCESSING SUMMARY")
            print("="*60)
            
            analysis = results['analysis']
            validation = results['validation']
            
            print(f"Input data:")
            print(f"  Shape: {data.shape}")
            print(f"  Size: {data.element_size() * data.numel() / 1024**3:.2f} GB")
            
            print(f"\nOutput data:")
            print(f"  Shape: {cropped_data.shape}")
            print(f"  Size: {cropped_data.element_size() * cropped_data.numel() / 1024**3:.2f} GB")
            print(f"  Size reduction: {(1 - cropped_data.numel()/data.numel()):.1%}")
            
            print(f"\nCropping results:")
            print(f"  Global radius: {analysis['global_radius']} pixels")
            print(f"  Original size: {analysis['original_shape'][0]}×{analysis['original_shape'][1]}")
            print(f"  Cropped size: {analysis['cropped_shape'][0]}×{analysis['cropped_shape'][1]}")
            
            print(f"\nIntensity retention:")
            print(f"  Mean: {validation['mean_retention']:.3f}")
            print(f"  Min: {validation['min_retention']:.3f}")
            print(f"  Max: {validation['max_retention']:.3f}")
            print(f"  Std: {validation['std_retention']:.3f}")
            
            print(f"\nValidation:")
            print(f"  Patterns passing min threshold ({config.min_retention:.2f}): "
                  f"{validation['passes_min_threshold']}/{validation['n_patterns']} "
                  f"({validation['passes_min_threshold']/validation['n_patterns']:.1%})")
            print(f"  Patterns passing target threshold ({config.target_retention:.2f}): "
                  f"{validation['passes_target_threshold']}/{validation['n_patterns']} "
                  f"({validation['passes_target_threshold']/validation['n_patterns']:.1%})")
            
            print(f"\nProcessing time: {results['processing_time']:.2f} seconds")
            print(f"Output saved to: {args.output}")
            
            # Check if requirements are met
            if validation['min_retention'] >= config.min_retention:
                print("\n✓ SUCCESS: All quality requirements met!")
            else:
                print(f"\n⚠ WARNING: Minimum retention ({validation['min_retention']:.3f}) "
                      f"below threshold ({config.min_retention:.3f})")
                print("  Consider reducing target retention or increasing margin pixels")
        
    except Exception as e:
        print(f"Error during processing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()