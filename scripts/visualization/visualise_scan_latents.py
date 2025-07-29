#!/usr/bin/env python
"""
Make a mosaic:

    • panel 0  : raw-average image (grayscale)
    • panel 1  : virtual detector image (BF or DF)
    • panel 2+ : each latent channel overlaid on that background

Usage - default bright-field virtual detector
--------------------------------------------
python scripts/visualise_scan_latents.py \
       --raw      data/patterns.h5 \         # .pt, .h5, .npy, or .npz supported
       --latents  outputs/embeddings.npz \   # .pt, .npy, or .npz supported
       --scan     512 512 \
       --virtual  bf             # or df
       --lat_max_cols 6          # grid width for latent maps
       --outfig   outputs/latent_mosaic.png

If your converter stored coordinates:
-------------------------------------
python … --coords data/coords.npy  # shape (N,2) float32 (y,x)

Tune radial masks (fractions of max radius):
-------------------------------------------
--bf_radius 0.1        # <10% central disk
--df_inner  0.2 --df_outer 1.0
"""
import argparse, numpy as np, torch
from pathlib import Path
import matplotlib.pyplot as plt

try:
    import tifffile
except ImportError:
    tifffile = None


# ─────────────────────────── helpers ────────────────────────────
def load_tensor(path: Path) -> np.ndarray:
    """Load .pt, .npy, .npz, or .h5 → numpy array"""
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu").numpy()
    elif path.suffix == ".npy":
        return np.load(path)
    elif path.suffix == ".npz":
        # For .npz files, assume 'data' key for raw data, 'embeddings' key for latents
        data = np.load(path)
        if 'embeddings' in data:
            return data['embeddings']
        elif 'data' in data:
            return data['data']
        else:
            # If no standard keys, use the first array
            keys = list(data.keys())
            if keys:
                print(f"Warning: Using first array '{keys[0]}' from .npz file")
                return data[keys[0]]
            else:
                raise ValueError(f"No arrays found in .npz file: {path}")
    elif path.suffix in {".h5", ".hdf5"}:
        # For HDF5 files, look for common dataset names
        import h5py
        with h5py.File(path, 'r') as f:
            # Try common dataset names for 4D-STEM data
            if 'data' in f:
                data = f['data'][:]
            elif 'patterns' in f:
                data = f['patterns'][:]
            elif 'array' in f:
                data = f['array'][:]
            else:
                # Use first dataset found
                keys = list(f.keys())
                if keys:
                    dataset_name = keys[0]
                    print(f"Warning: Using first dataset '{dataset_name}' from .h5 file")
                    data = f[dataset_name][:]
                else:
                    raise ValueError(f"No datasets found in .h5 file: {path}")
        
        # Convert to float32 and add channel dimension if needed
        data = data.astype(np.float32)
        if len(data.shape) == 3:  # (N, Qy, Qx) -> (N, 1, Qy, Qx)
            data = data[:, np.newaxis, :, :]
        
        return data
    else:
        raise ValueError(f"Unknown file type {path.suffix}. Supported: .pt, .pth, .npy, .npz, .h5, .hdf5")


def radial_mask(shape: tuple[int, int], r_min: float, r_max: float) -> np.ndarray:
    """Build a boolean ring mask in normalised radius units (0…1)."""
    qy, qx = shape
    y, x = np.ogrid[:qy, :qx]
    cy, cx = (qy - 1) / 2, (qx - 1) / 2
    r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
    return (r >= r_min) & (r < r_max)


# Virtual detector functionality is now handled directly by STEMVisualizer in main()
# This eliminates the need for wrapper functions and import complexity


# ───────────────────────────── CLI ─────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, type=Path,
                   help="4D-STEM data: .pt, .h5/.hdf5, .npy, or .npz format")
    p.add_argument("--latents", required=True, type=Path,
                   help="Latent embeddings: .pt, .npy, or .npz format")
    p.add_argument("--scan", nargs=2, type=int, metavar=("Ny", "Nx"),
                   help="Scan grid dimensions (optional - will auto-detect if not provided)")
    p.add_argument("--coords", type=Path,
                   help="Optional .npy (N,2) (y,x) coordinates for irregular scan")
    p.add_argument("--virtual", choices=["bf", "df"], default="bf",
                   help="Virtual bright-field or dark-field")
    p.add_argument("--bf_radius", type=float, default=0.1,
                   help="BF: radius (0-1) of central disk")
    p.add_argument("--df_inner", type=float, default=0.2)
    p.add_argument("--df_outer", type=float, default=1.0)
    p.add_argument("--lat_max_cols", type=int, default=6)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--outfig", type=Path, required=True)
    return p.parse_args()


# ──────────────────────────── main ─────────────────────────────
def main():
    args = parse()

    # 1. Load data ------------------------------------------------------------
    raw = load_tensor(args.raw)          # (N,1,Qy,Qx)
    Z = load_tensor(args.latents)        # (N, latent_dim)
    N = raw.shape[0]

    # Import scan utilities and STEM visualization
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent))
    from scan_util import factorise_scan, coords_to_sparse_image, validate_coords, print_coord_summary
    from stem_visualization import STEMVisualizer
    
    # Coordinate handling following FR-3.1, FR-3.2, FR-3.3
    coords = None
    Ny = Nx = None
    
    # Priority 1: Use explicit --scan dimensions if provided (FR-3.3)
    if args.scan is not None:
        Ny, Nx = args.scan
        # Generate raster coordinates for explicit scan
        from scan_util import raster_coords
        coords = raster_coords(Ny, Nx)
        print(f"Using explicit scan dimensions: {Ny} x {Nx}")
        
    elif args.coords is not None:
        # Priority 2: Load coordinates from separate file
        if args.coords.suffix == ".npz":
            coord_data = np.load(args.coords)
            if 'coords' in coord_data:
                coords = coord_data['coords']
            elif 'spatial_coordinates' in coord_data:  # Legacy support
                coords = coord_data['spatial_coordinates']
            else:
                # Use first available array
                keys = list(coord_data.keys())
                coords = coord_data[keys[0]]
                print(f"Warning: Using '{keys[0]}' as coordinates from .npz file")
        else:
            coords = np.load(args.coords)    # (N,2) int32 (row,col)
        
        coords = np.array(coords, dtype=np.int32)
        print_coord_summary(coords, "External coordinates")
        
    elif args.latents.suffix == ".npz":
        # Priority 3: Check if latents file contains coordinates
        latent_data = np.load(args.latents)
        if 'coords' in latent_data:
            coords = latent_data['coords']
            print("✓ Using coordinates from latents file")
        elif 'spatial_coordinates' in latent_data:  # Legacy support
            coords = latent_data['spatial_coordinates']
            print("⚠️ Using legacy spatial_coordinates from latents file")
        
        if coords is not None:
            coords = np.array(coords, dtype=np.int32)
            print_coord_summary(coords, "Latents file coordinates")
    
    # If no coordinates found, fall back to factorization
    if coords is None:
        print(f"No coordinates found, auto-detecting scan shape for {N} patterns...")
        try:
            Ny, Nx = factorise_scan(N)
            print(f"Auto-detected scan shape: {Ny} x {Nx}")
            from scan_util import raster_coords
            coords = raster_coords(Ny, Nx)
        except Exception as e:
            print(f"ERROR: Could not determine scan shape: {e}")
            return
    
    # Validate coordinates
    if len(coords) != N:
        print(f"ERROR: Coordinate count ({len(coords)}) != pattern count ({N})")
        return
    
    # Get scan dimensions from coordinates
    coord_info = validate_coords(coords)
    Ny, Nx = coord_info['inferred_shape']
    
    print(f"\nFinal configuration:")
    print(f"  Scan dimensions: {Ny} x {Nx}")
    print(f"  Patterns: {N}")
    print(f"  Coordinate type: {coord_info['scan_type']}")
    print(f"  Grid completeness: {coord_info['completeness']:.1%}")
    
    if not coord_info['is_complete_grid']:
        print(f"  ⚠️ Using sparse mapping (some positions may be empty)")

    # 2. Build background images ---------------------------------------------
    # raw-average (mean over diffraction pattern)
    # Use sparse mapping instead of reshape (FR-3.1)
    raw_mean_values = raw.mean(axis=(2, 3)).squeeze() # Remove channel dimension
    raw_mean = coords_to_sparse_image(coords, raw_mean_values, (Ny, Nx))

    # virtual detector using STEMVisualizer (proper implementation) --------
    # Create STEMVisualizer instance
    stem_viz = STEMVisualizer(raw[:, 0], scan_shape=(Ny, Nx))
    print(f"Direct beam detected at: {stem_viz.direct_beam_position} (y, x)")
    
    # Generate virtual detector image
    if args.virtual == "bf":
        # Calculate appropriate radius in pixels from fractional radius
        pattern_size = min(raw.shape[-2:])
        radius_pixels = int(args.bf_radius * pattern_size // 2)
        virt_image_full = stem_viz.create_bright_field_image(radius=radius_pixels)
        print(f"Bright field radius: {radius_pixels} pixels ({args.bf_radius:.2f} fraction)")
    elif args.virtual == "df":
        # Create custom dark field region based on parameters
        center_y, center_x = stem_viz.direct_beam_position
        pattern_size = min(raw.shape[-2:])
        
        # Convert fractional radii to pixel radii
        inner_radius = int(args.df_inner * pattern_size // 2)
        outer_radius = int(args.df_outer * pattern_size // 2)
        
        # Create annular region
        df_region = (
            max(0, center_y - outer_radius),
            min(raw.shape[-2], center_y + outer_radius),
            max(0, center_x - outer_radius), 
            min(raw.shape[-1], center_x + outer_radius)
        )
        
        virt_image_full = stem_viz.create_virtual_field_image(df_region)
        print(f"Dark field region: {df_region} (inner: {inner_radius}px, outer: {outer_radius}px)")
    
    # Convert full grid image to sparse mapping if needed
    if coord_info['is_complete_grid']:
        virt_map = virt_image_full
    else:
        # For irregular coordinates, we need to extract values and map sparsely
        virt_values = virt_image_full.ravel()
        virt_map = coords_to_sparse_image(coords, virt_values, (Ny, Nx))

    # 3. Plot mosaic ----------------------------------------------------------
    latent_dim = Z.shape[1]
    n_cols = args.lat_max_cols
    n_rows_lat = int(np.ceil(latent_dim / n_cols))
    n_rows_total = n_rows_lat + 1                   # +1 row for the two backgrounds
    fig_w = 3 * n_cols
    fig_h = 3 * n_rows_total
    fig, axes = plt.subplots(n_rows_total, n_cols,
                             figsize=(fig_w, fig_h),
                             sharex=True, sharey=True)
    axes = axes.ravel()

    # Panel 0: raw-average (FR-3.2: handle NaN for sparse scans)
    axes[0].imshow(raw_mean, cmap="gray")
    axes[0].set_title("Raw average")
    axes[0].axis("off")

    # Panel 1: virtual detector (FR-3.2: handle NaN for sparse scans)
    axes[1].imshow(virt_map, cmap="gray")
    axes[1].set_title(f"Virtual {args.virtual.upper()}")
    axes[1].axis("off")

    # empty out other slots in first row if needed
    for idx in range(2, n_cols):
        axes[idx].axis("off")

    # latent overlays starting at row 2
    for k in range(latent_dim):
        ax = axes[n_cols + k]            # offset by first row
        colour = Z[:, k]
        colour = (colour - colour.min()) / (colour.ptp() + 1e-9)
        ax.imshow(raw_mean, cmap="gray")
        sc = ax.scatter(coords[:, 1], coords[:, 0], c=colour,
                        cmap="viridis", s=6, alpha=0.9)
        ax.set_title(f"Latent[{k}]")
        ax.axis("off")

    # hide any unused axes
    for ax in axes[n_cols + latent_dim:]:
        ax.axis("off")

    # single colourbar for latent panels
    cbar = fig.colorbar(sc, ax=axes[n_cols:], fraction=0.02, pad=0.02)
    cbar.set_label("normalised value", rotation=270, labelpad=15)

    plt.tight_layout()
    plt.savefig(args.outfig, dpi=args.dpi)
    print("Saved", args.outfig)


if __name__ == "__main__":
    main()
