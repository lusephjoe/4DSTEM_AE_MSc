"""Convert Gatan .dm4 4D-STEM stacks to the PyTorch tensor format used by `train.py`.

Down-sampling strategies implemented (choose with `--mode`):

* **stride**  - plain pixel skipping (fastest, strong aliasing)
* **bin**     - k*k mean-pooling (good compromise, default)
* **gauss**   - Gaussian low-pass filter followed by stride (anti-alias)
* **fft**     - Fourier cropping (gold-standard frequency fidelity, slow)

Extra features
--------------
* Optional **scan grid sub-sampling** (take every *n*-th probe position) to thin very dense scans.
* All heavy lifting done with NumPy / SciPy - no GPU required, but script plays nicely inside
  a CUDA environment because the tensors are saved in CPU-compatible `.pt` format.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import torch
import hyperspy.api as hs  # pip install hyperspy
from scipy.ndimage import gaussian_filter

# ─────────────────────────── helpers ────────────────────────────

def normalise(x: np.ndarray) -> np.ndarray:
    """Scale array to [0,1] float32 with log scaling"""
    x = x.astype("float32", copy=False)
    x = np.log(x + 1e-6)  # Log scaling with small epsilon to avoid log(0)
    x -= x.min()
    x /= x.max() + 1e-6
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

# ───────────────────────────── CLI ─────────────────────────────

def main():
    p = argparse.ArgumentParser(description="Convert .dm4 4D-STEM file → PyTorch tensor")
    p.add_argument("--input", type=Path, required=True, help="Input .dm4 file")
    p.add_argument("--output", type=Path, required=True, help="Output .pt tensor file")
    p.add_argument("--downsample", type=int, default=1, metavar="k",
                   help="Factor k; diffraction pattern size becomes Q/k * Q/k")
    p.add_argument("--mode", choices=["bin", "stride", "gauss", "fft"], default="bin",
                   help="Down-sampling strategy - see script docstring for details")
    p.add_argument("--sigma", type=float, default=0.8, metavar="sigma",
                   help="Gaussian sigma (pixels) if --mode gauss (default 0.8)")
    p.add_argument("--scan_step", type=int, default=1, metavar="n",
                   help="Take every n-th probe position along both scan axes")
    args = p.parse_args()

    # load .dm4 (HyperSpy handles lazy loading, but here we force into memory for simplicity)
    sig = hs.load(args.input.as_posix(), lazy=False)
    data = sig.data  # shape = (Ny, Nx, Qy, Qx)

    # optional scan grid thinning
    if args.scan_step > 1:
        data = data[::args.scan_step, ::args.scan_step, ...]
        print(f"Scan grid subsampled by {args.scan_step} → new grid {data.shape[:2]}")

    # pattern down-sampling
    data = downsample_patterns(data, args.downsample, args.mode, args.sigma)

    # intensity normalisation
    data = normalise(data)

    # reshape to N * 1 * Qy * Qx tensor
    ny, nx, qy, qx = data.shape
    tensor = torch.from_numpy(data.reshape(ny * nx, 1, qy, qx))
    torch.save(tensor, args.output)
    print(f"Saved {tensor.shape[0]} patterns of size {qy}*{qx} → {args.output}")

if __name__ == "__main__":
    main()