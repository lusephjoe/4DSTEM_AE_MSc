#!/usr/bin/env python3
"""Convert dm4 files to hspy format for efficient lazy loading."""

import argparse
from pathlib import Path
import time
try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False

def convert_dm4_to_hspy(input_path: Path, output_path: Path, rechunk: bool = True):
    """Convert dm4 file to hspy format with optional rechunking for better performance."""
    
    if not HAS_HYPERSPY:
        raise ImportError("HyperSpy required. Install with: pip install hyperspy")
    
    print(f"Loading {input_path}...")
    start_time = time.time()
    
    # Load dm4 file lazily
    sig = hs.load(str(input_path), lazy=True)
    
    load_time = time.time() - start_time
    print(f"✓ Loaded in {load_time:.2f}s")
    print(f"  Original shape: {sig.data.shape}")
    print(f"  Original chunks: {sig.data.chunks if hasattr(sig.data, 'chunks') else 'N/A'}")
    
    # Rechunk for better performance if requested
    if rechunk and hasattr(sig.data, 'rechunk'):
        print("Rechunking for optimal performance...")
        # Rechunk to have reasonable chunk sizes (~64MB per chunk)
        # For 4D data (ny, nx, qy, qx), chunk along scan dimensions
        ny, nx, qy, qx = sig.data.shape
        
        # Calculate optimal chunk size (aim for ~64MB chunks)
        bytes_per_pixel = 4  # float32
        pixels_per_chunk = (64 * 1024 * 1024) // bytes_per_pixel  # 64MB
        patterns_per_chunk = pixels_per_chunk // (qy * qx)
        
        # Chunk size should be reasonable for scan dimensions
        chunk_size = min(64, max(8, patterns_per_chunk))
        
        chunks = (chunk_size, chunk_size, qy, qx)
        print(f"  Rechunking to: {chunks}")
        
        sig.data = sig.data.rechunk(chunks)
        print(f"  New chunks: {sig.data.chunks}")
    
    # Save as hspy
    print(f"Saving to {output_path}...")
    save_start = time.time()
    
    sig.save(str(output_path), overwrite=True)
    
    save_time = time.time() - save_start
    total_time = time.time() - start_time
    
    print(f"✓ Saved in {save_time:.2f}s")
    print(f"✓ Total time: {total_time:.2f}s")
    
    # Show file sizes
    input_size = input_path.stat().st_size / (1024**3)
    output_size = output_path.stat().st_size / (1024**3)
    
    print(f"✓ Input size: {input_size:.2f} GB")
    print(f"✓ Output size: {output_size:.2f} GB")
    print(f"✓ Compression ratio: {input_size/output_size:.2f}x")

def main():
    parser = argparse.ArgumentParser(description="Convert dm4 files to hspy format")
    parser.add_argument("--input", type=Path, required=True, help="Input dm4 file")
    parser.add_argument("--output", type=Path, required=True, help="Output hspy file")
    parser.add_argument("--no-rechunk", action="store_true", help="Skip rechunking optimization")
    
    args = parser.parse_args()
    
    if not args.input.exists():
        raise FileNotFoundError(f"Input file not found: {args.input}")
    
    if args.input.suffix.lower() not in ['.dm4', '.dm3']:
        raise ValueError(f"Input file must be dm4/dm3 format, got: {args.input.suffix}")
    
    # Set default output extension
    if args.output.suffix == '':
        args.output = args.output.with_suffix('.hspy')
    
    # Create output directory
    args.output.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("DM4 TO HSPY CONVERSION")
    print("="*60)
    
    convert_dm4_to_hspy(args.input, args.output, rechunk=not args.no_rechunk)
    
    print("="*60)
    print("CONVERSION COMPLETE!")
    print("="*60)

if __name__ == "__main__":
    main()