"""Preprocess 4D-STEM datasets (HDF5, dm4, hspy) into training tensors with automatic file type detection."""
import argparse, h5py, numpy as np, torch
from pathlib import Path
try:
    import hyperspy.api as hs
    HAS_HYPERSPY = True
except ImportError:
    HAS_HYPERSPY = False
from scipy.ndimage import gaussian_filter

def normalise(x: np.ndarray) -> np.ndarray:
    """Z-score normalization with log scaling like m3_learning"""
    x = x.astype("float32")
    x = np.log(x + 1)  # Log scaling with +1 to avoid log(0) like m3_learning
    # Apply z-score normalization (mean=0, std=1)
    x_flat = x.reshape(-1)
    mean = np.mean(x_flat)
    std = np.std(x_flat)
    x = (x - mean) / (std + 1e-8)  # Small epsilon to avoid division by zero
    return x

def block_bin_mean(arr: np.ndarray, k: int) -> np.ndarray:
    """Reduce spatial resolution by k*k mean pooling."""
    *lead, qy, qx = arr.shape
    qy2, qx2 = (qy // k) * k, (qx // k) * k  # drop edges so divisible by k
    trimmed = arr[..., :qy2, :qx2]
    newshape = (*lead, qy2 // k, k, qx2 // k, k)
    return trimmed.reshape(newshape).mean(axis=(-1, -3))

def gaussian_downsample(arr: np.ndarray, k: int, sigma: float) -> np.ndarray:
    """Gaussian low-pass filter then stride-based down-sample."""
    blurred = gaussian_filter(arr, sigma=(0, 0, sigma, sigma), mode="reflect")
    return blurred[..., ::k, ::k]

def fft_crop(arr: np.ndarray, k: int) -> np.ndarray:
    """Down-sample via Fourier cropping (slow, high fidelity)."""
    *lead, qy, qx = arr.shape
    fq = np.fft.fftshift(np.fft.fft2(arr, axes=(-2, -1)), axes=(-2, -1))
    cy, cx = qy // 2, qx // 2
    ny, nx = qy // k, qx // k
    fq_crop = fq[..., cy - ny // 2: cy + (ny + 1) // 2,
                  cx - nx // 2: cx + (nx + 1) // 2]
    img = np.fft.ifft2(np.fft.ifftshift(fq_crop, axes=(-2, -1)),
                       s=(ny, nx), axes=(-2, -1)).real
    return img.astype(arr.dtype)

def downsample_patterns(data: np.ndarray, k: int, mode: str, sigma: float) -> np.ndarray:
    """Apply downsampling to diffraction patterns."""
    if k <= 1:
        return data
    if mode == "stride":
        return data[..., ::k, ::k]
    if mode == "bin":
        return block_bin_mean(data, k)
    if mode == "gauss":
        return gaussian_downsample(data, k, sigma)
    if mode == "fft":
        return fft_crop(data, k)
    raise ValueError(f"Unknown down-sampling mode: {mode}")

def detect_file_type(file_path: Path) -> str:
    """Detect file type based on extension."""
    suffix = file_path.suffix.lower()
    if suffix in [".dm4", ".dm3"]:
        return "dm4"
    elif suffix in [".hdf5", ".h5"]:
        return "hdf5"
    elif suffix == ".hspy":
        return "hspy"
    else:
        raise ValueError(f"Unsupported file type: {suffix}")

def load_data(file_path: Path) -> np.ndarray:
    """Load data based on file type."""
    file_type = detect_file_type(file_path)
    
    if file_type == "hdf5":
        with h5py.File(file_path, "r") as f:
            data = f["/data"][:]  # shape = (Ny, Nx, Qy, Qx)
    elif file_type in ["dm4", "hspy"]:
        if not HAS_HYPERSPY:
            raise ImportError("HyperSpy required for dm4/hspy files. Install with: pip install hyperspy")
        sig = hs.load(file_path.as_posix(), lazy=False)
        data = sig.data  # shape = (Ny, Nx, Qy, Qx)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    return data

def main():
    p = argparse.ArgumentParser(description="Preprocess 4D-STEM files → PyTorch tensor")
    p.add_argument("--input", type=Path, required=True, help="Input file (.dm4, .hdf5, .hspy)")
    p.add_argument("--output", type=Path, required=True, help="Output .pt tensor file")
    p.add_argument("--downsample", type=int, default=1, metavar="k",
                   help="Factor k; diffraction pattern size becomes Q/k * Q/k")
    p.add_argument("--mode", choices=["bin", "stride", "gauss", "fft"], default="bin",
                   help="Down-sampling strategy: bin (mean pooling), stride (pixel skip), gauss (Gaussian+stride), fft (Fourier crop)")
    p.add_argument("--sigma", type=float, default=0.8, metavar="sigma",
                   help="Gaussian sigma (pixels) if --mode gauss (default 0.8)")
    p.add_argument("--scan_step", type=int, default=1, metavar="n",
                   help="Take every n-th probe position along both scan axes")
    args = p.parse_args()

    # Load data with automatic file type detection
    data = load_data(args.input)
    print(f"Loaded data shape: {data.shape} from {detect_file_type(args.input)} file")

    # Optional scan grid thinning
    if args.scan_step > 1:
        data = data[::args.scan_step, ::args.scan_step, ...]
        print(f"Scan grid subsampled by {args.scan_step} → new grid {data.shape[:2]}")

    # Pattern down-sampling
    data = downsample_patterns(data, args.downsample, args.mode, args.sigma)
    print(f"Downsampled patterns by {args.downsample}x using {args.mode} method")

    # Intensity normalisation with log scaling
    data = normalise(data)

    # Reshape to samples × 1 × Qy × Qx
    ny, nx, qy, qx = data.shape
    data = data.reshape(ny*nx, 1, qy, qx)

    torch.save(torch.from_numpy(data), args.output)
    print(f"Saved {data.shape[0]} patterns of size {qy}*{qx} → {args.output}")

if __name__ == "__main__":
    main()