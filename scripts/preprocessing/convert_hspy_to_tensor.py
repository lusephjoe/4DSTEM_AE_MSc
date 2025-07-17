"""Convert .hspy files to tensor format for use with original training script."""

import argparse
import torch
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False


def convert_hspy_to_tensor(
    input_path: Path,
    output_path: Path,
    scan_step: int = 1,
    downsample: int = 1,
    downsample_mode: str = "bin",
    normalize: bool = True,
    sigma: float = 0.8
) -> None:
    """
    Convert .hspy file to tensor format compatible with original training script.
    
    Args:
        input_path: Path to input .hspy file
        output_path: Path to output .pt file
        scan_step: Subsample scan positions by this factor
        downsample: Downsample diffraction patterns by this factor
        downsample_mode: Downsampling method ('bin', 'stride', 'gauss')
        normalize: Whether to apply log normalization
        sigma: Gaussian sigma for 'gauss' mode
    """
    if not HAS_HYPERSPY:
        raise ImportError("HyperSpy required. Install with: pip install hyperspy")
    
    print(f"Loading {input_path}...")
    signal = hs.load(str(input_path))
    
    # Get original shape
    original_shape = signal.data.shape
    ny, nx, qy, qx = original_shape
    print(f"Original shape: {original_shape}")
    
    # Apply scan step subsampling
    if scan_step > 1:
        print(f"Subsampling scan positions by factor {scan_step}...")
        signal = signal.isig[::scan_step, ::scan_step]
        ny, nx = ny // scan_step, nx // scan_step
    
    # Load data into memory
    print("Loading data into memory...")
    data = signal.data.compute() if hasattr(signal.data, 'compute') else signal.data
    print(f"Data shape after scan subsampling: {data.shape}")
    
    # Apply downsampling to diffraction patterns
    if downsample > 1:
        print(f"Downsampling diffraction patterns by factor {downsample} using {downsample_mode}...")
        
        if downsample_mode == "bin":
            # Mean pooling
            k = downsample
            qy2, qx2 = (qy // k) * k, (qx // k) * k
            if qy2 < k or qx2 < k:
                print("Warning: Pattern too small for binning, using stride instead")
                data = data[:, :, ::downsample, ::downsample]
            else:
                # Trim to make divisible by k
                data = data[:, :, :qy2, :qx2]
                # Reshape and average
                data = data.reshape(ny, nx, qy2 // k, k, qx2 // k, k).mean(axis=(3, 5))
        
        elif downsample_mode == "stride":
            # Simple stride sampling
            data = data[:, :, ::downsample, ::downsample]
        
        elif downsample_mode == "gauss":
            # Gaussian filtering followed by stride
            try:
                from scipy.ndimage import gaussian_filter
                print(f"Applying Gaussian filter with sigma={sigma}...")
                # Apply gaussian filter to each pattern
                for i in range(ny):
                    for j in range(nx):
                        data[i, j] = gaussian_filter(data[i, j], sigma=sigma)
                # Then downsample
                data = data[:, :, ::downsample, ::downsample]
            except ImportError:
                print("Warning: scipy not available, falling back to stride sampling")
                data = data[:, :, ::downsample, ::downsample]
        
        qy, qx = data.shape[2], data.shape[3]
    
    print(f"Final data shape: {data.shape}")
    
    # Reshape to (N, H, W) format
    data = data.reshape(-1, qy, qx)
    print(f"Reshaped to: {data.shape}")
    
    # Apply normalization
    if normalize:
        print("Applying log normalization...")
        # Add small constant to avoid log(0)
        data = np.log(data + 1.0)
        
        # Calculate dataset-specific statistics
        mean = float(np.mean(data))
        std = float(np.std(data))
        print(f"Dataset statistics: mean={mean:.4f}, std={std:.4f}")
        
        # Apply normalization
        data = (data - mean) / (std + 1e-8)
        
        print(f"Normalized data range: {data.min():.4f} to {data.max():.4f}")
        
        # Store normalization parameters
        norm_stats = {
            'mean': mean,
            'std': std,
            'normalized': True
        }
    else:
        norm_stats = {
            'mean': 0.0,
            'std': 1.0,
            'normalized': False
        }
    
    # Convert to tensor and add channel dimension
    tensor = torch.from_numpy(data).float().unsqueeze(1)  # Shape: (N, 1, H, W)
    
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Memory usage: {tensor.element_size() * tensor.nelement() / 1024**3:.2f} GB")
    
    # Save tensor and metadata
    save_data = {
        'data': tensor,
        'original_shape': original_shape,
        'processed_shape': (ny, nx, qy, qx),
        'scan_step': scan_step,
        'downsample': downsample,
        'downsample_mode': downsample_mode,
        'normalization': norm_stats
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(save_data, output_path)
    print(f"Saved tensor to {output_path}")
    
    # Print summary
    print("\n" + "="*60)
    print("CONVERSION SUMMARY")
    print("="*60)
    print(f"Input file: {input_path}")
    print(f"Output file: {output_path}")
    print(f"Original shape: {original_shape}")
    print(f"Final tensor shape: {tensor.shape}")
    print(f"Scan step: {scan_step}")
    print(f"Downsample: {downsample} ({downsample_mode})")
    print(f"Normalized: {normalize}")
    if normalize:
        print(f"  Mean: {norm_stats['mean']:.4f}")
        print(f"  Std: {norm_stats['std']:.4f}")
    print(f"File size: {output_path.stat().st_size / 1024**3:.2f} GB")
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Convert .hspy files to tensor format")
    parser.add_argument("--input", type=Path, required=True, help="Input .hspy file")
    parser.add_argument("--output", type=Path, required=True, help="Output .pt file")
    parser.add_argument("--scan_step", type=int, default=1, help="Subsample scan positions")
    parser.add_argument("--downsample", type=int, default=1, help="Downsample diffraction patterns")
    parser.add_argument("--downsample_mode", choices=["bin", "stride", "gauss"], default="bin")
    parser.add_argument("--no_normalize", action="store_true", help="Skip normalization")
    parser.add_argument("--sigma", type=float, default=0.8, help="Gaussian sigma for gauss mode")
    
    args = parser.parse_args()
    
    try:
        convert_hspy_to_tensor(
            args.input,
            args.output,
            args.scan_step,
            args.downsample,
            args.downsample_mode,
            not args.no_normalize,
            args.sigma
        )
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())