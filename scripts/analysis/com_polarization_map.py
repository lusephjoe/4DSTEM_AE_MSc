#!/usr/bin/env python3
"""
Centre-of-Mass (DPC) analysis for 4D-STEM -> polarization direction map.

Requires: py4DSTEM, numpy; optional: tifffile, scipy (for gaussian prefilter).

Example:
  python com_polarization_map.py data.h5 --datapath /root/data --beamstop \
      --mask-radius 8 --bin-q 2 --rotate 0 --outdir out

Notes:
- If there's a beam stop or central saturation, use --beamstop and a small --mask-radius.
- By default, outputs are in pixel units (detector pixels). For direction mapping,
  only the angle matters.
"""
import argparse, os, sys
import numpy as np
import h5py

try:
    import tifffile
except Exception:
    tifffile = None

import py4DSTEM as p4
from py4DSTEM.process.calibration import origin as origin_mod

def save_npy(outdir, name, arr):
    np.save(os.path.join(outdir, name), arr)

def save_tif(outdir, name, arr):
    if tifffile is None:
        return
    a = np.array(arr, dtype=np.float32)
    a = np.nan_to_num(a, nan=0.0)
    # scale each image to usable range for visualization
    p = np.nanpercentile(np.abs(a), 99.0)
    if p <= 0:
        p = 1.0
    vis = (np.clip(a / p, -1, 1) * 32767 + 32768).astype(np.uint16)
    tifffile.imwrite(os.path.join(outdir, name), vis)

def hsv_angle_rgb(angle, magnitude):
    """Encode angle (rad) as hue and magnitude as value -> RGB uint8."""
    import matplotlib.colors as mcolors
    h = (angle + np.pi) / (2 * np.pi)           # 0..1 (wrap [-pi,pi] -> [0,1])
    s = np.ones_like(h, dtype=np.float32)
    v = magnitude / (np.nanpercentile(magnitude, 99.0) or 1.0)
    v = np.clip(v, 0, 1).astype(np.float32)
    hsv = np.stack([h.astype(np.float32), s, v], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

def main():
    ap = argparse.ArgumentParser(description="4D-STEM CoM/DPC -> polarization direction map")
    ap.add_argument("h5", help="Input .h5/.emd/.hspy file")
    ap.add_argument("--datapath", default=None,
                    help="Optional HDF5 path to the EMD data tree (see py4DSTEM.print_h5_tree)")
    ap.add_argument("--outdir", default="com_out", help="Output directory")
    ap.add_argument("--beamstop", action="store_true",
                    help="Use Friedel-symmetry origin finder (robust with a beam stop)")
    ap.add_argument("--mask-radius", type=float, default=0.0,
                    help="Mask radius (in q pixels) around origin to exclude central beam")
    ap.add_argument("--bin-q", type=int, default=1, help="Diffraction-space bin factor (speed/denoise)")
    ap.add_argument("--sigma", type=float, default=0.0,
                    help="Gaussian sigma (pixels) to denoise DPs before CoM (requires scipy)")
    ap.add_argument("--threshold", type=float, default=0.0,
                    help="Relative threshold (0..1) of max DP intensity; values below are zeroed")
    ap.add_argument("--invert", action="store_true",
                    help="Flip vector sign (depends on your E-field/polarization convention)")
    ap.add_argument("--rotate", type=float, default=0.0,
                    help="Rotate vector field by +CCW degrees to align to specimen coords")
    ap.add_argument("--scan-shape", type=int, nargs=2, metavar=("NY","NX"),
                    help="Scan dimensions (height width) - required for (N,Ky,Kx) data")
    ap.add_argument("--crop-rx", type=int, nargs=2, metavar=("X0","X1"),
                    help="Optional crop on scan x-index (inclusive, exclusive)")
    ap.add_argument("--crop-ry", type=int, nargs=2, metavar=("Y0","Y1"),
                    help="Optional crop on scan y-index (inclusive, exclusive)")
    args = ap.parse_args()

    # 1) Load the 4D dataset
    try:
        dc = p4.read(args.h5, datapath=args.datapath)  # EMD/py4DSTEM reader
    except Exception:
        # Fallback: load HDF5 directly and create DataCube
        try:
            print(f"py4DSTEM reader failed, loading HDF5 directly...")
            with h5py.File(args.h5, 'r') as f:
                if args.datapath and args.datapath in f:
                    data = f[args.datapath][:]
                else:
                    # Try to find the data automatically
                    keys = list(f.keys())
                    if args.datapath:
                        raise KeyError(f"Dataset '{args.datapath}' not found. Available: {keys}")
                    elif len(keys) == 1:
                        data = f[keys[0]][:]
                        print(f"Using dataset: {keys[0]}")
                    else:
                        raise KeyError(f"Multiple datasets found: {keys}. Specify --datapath")
            
            print(f"Loaded data shape: {data.shape}")
            
            # Handle reshaping if needed
            if len(data.shape) == 3 and args.scan_shape:
                # Reshape from (N, Ky, Kx) to (Ny, Nx, Ky, Kx)
                Ny, Nx = args.scan_shape
                N, Ky, Kx = data.shape
                expected_N = Ny * Nx
                if N != expected_N:
                    raise ValueError(f"Data frames ({N}) != scan size ({expected_N})")
                data = data.reshape(Ny, Nx, Ky, Kx)
                print(f"Reshaped to: {data.shape}")
            elif len(data.shape) == 3:
                raise ValueError("3D data requires --scan-shape parameter")
            
            # Create py4DSTEM DataCube
            dc = p4.DataCube(data)
        except Exception as e:
            print(f"Failed to load data: {e}")
            sys.exit(1)

    # Optional Q-binning to speed up and improve robustness
    if hasattr(dc, "bin_Q") and args.bin_q > 1:
        dc = dc.bin_Q(args.bin_q)

    # 2) Find the DP origin (per scan position)
    if args.beamstop:
        qx0, qy0 = origin_mod.get_origin_friedel(dc)  # robust with beam stop
    else:
        qx0, qy0 = origin_mod.get_origin(dc)          # assumes brightest central disk

    # Sanity & shapes (py4DSTEM uses (R_x, R_y, Q_x, Q_y))
    try:
        R_x, R_y, Q_x, Q_y = dc.data.shape
    except Exception as e:
        print(f"Unexpected data shape: {getattr(dc, 'data', None)}", file=sys.stderr)
        raise e

    # Optional crop on the scan area
    rx0, rx1 = (0, R_x) if not args.crop_rx else (max(0, args.crop_rx[0]), min(R_x, args.crop_rx[1]))
    ry0, ry1 = (0, R_y) if not args.crop_ry else (max(0, args.crop_ry[0]), min(R_y, args.crop_ry[1]))

    # Precompute pixel index arrays once
    yy, xx = np.indices((Q_x, Q_y), dtype=np.float32)

    comx = np.full((rx1 - rx0, ry1 - ry0), np.nan, dtype=np.float32)
    comy = np.full((rx1 - rx0, ry1 - ry0), np.nan, dtype=np.float32)

    # Optional filters
    gaussian_filter = None
    if args.sigma > 0:
        try:
            from scipy.ndimage import gaussian_filter as gf
            gaussian_filter = gf
        except Exception:
            print("scipy not installed: --sigma will be ignored.", file=sys.stderr)

    # 3) CoM per DP relative to origin (first moment / intensity)
    for ix in range(rx0, rx1):
        for iy in range(ry0, ry1):
            I = dc.data[ix, iy].astype(np.float32, copy=False)

            if gaussian_filter is not None:
                I = gaussian_filter(I, args.sigma, mode="nearest")

            if args.threshold > 0:
                t = args.threshold * float(I.max() if I.size else 0.0)
                if t > 0:
                    I = np.where(I >= t, I, 0.0)

            if args.mask_radius > 0:
                cx0, cy0 = qx0[ix, iy], qy0[ix, iy]
                r2 = (xx - cx0) ** 2 + (yy - cy0) ** 2
                I = np.where(r2 >= args.mask_radius ** 2, I, 0.0)

            S = I.sum(dtype=np.float64)
            if S <= 0:
                continue

            cx = float((I * xx).sum(dtype=np.float64) / S)
            cy = float((I * yy).sum(dtype=np.float64) / S)

            comx[ix - rx0, iy - ry0] = cx - qx0[ix, iy]
            comy[ix - rx0, iy - ry0] = cy - qy0[ix, iy]

    # 4) Vector post-processing and outputs
    if args.invert:
        comx *= -1.0
        comy *= -1.0

    if abs(args.rotate) > 0:
        th = np.deg2rad(args.rotate)
        rx =  np.cos(th) * comx - np.sin(th) * comy
        ry =  np.sin(th) * comx + np.cos(th) * comy
        comx, comy = rx, ry

    angle = np.arctan2(comy, comx)              # radians, [-pi, pi]
    magnitude = np.hypot(comx, comy).astype(np.float32)

    os.makedirs(args.outdir, exist_ok=True)
    save_npy(args.outdir, "com_x.npy", comx)
    save_npy(args.outdir, "com_y.npy", comy)
    save_npy(args.outdir, "angle_rad.npy", angle)
    save_npy(args.outdir, "magnitude.npy", magnitude)

    save_tif(args.outdir, "com_x.tif", comx)
    save_tif(args.outdir, "com_y.tif", comy)
    save_tif(args.outdir, "magnitude.tif", magnitude)

    # Save a quick-look polarization direction RGB (HSV wheel)
    try:
        rgb = hsv_angle_rgb(angle, magnitude)
        if tifffile is not None:
            tifffile.imwrite(os.path.join(args.outdir, "polarization_rgb.png"), rgb)
        else:
            # fall back to numpy .npy if tifffile is missing
            np.save(os.path.join(args.outdir, "polarization_rgb.npy"), rgb)
    except Exception:
        pass

    print(f"Done. Outputs in: {args.outdir}")

if __name__ == "__main__":
    main()