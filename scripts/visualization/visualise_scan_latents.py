#!/usr/bin/env python
"""
Make a mosaic showing both BF and DF virtual detectors:

    • panel 0  : virtual bright field image (grayscale)
    • panel 1  : virtual dark field image (grayscale) 
    • panel 2+ : each latent channel overlaid on the bright field background

Usage - dual detector visualization with interactive center detection
---------------------------------------------------------------------
python scripts/visualise_scan_latents.py \
       --raw      data/patterns.h5 \         # .pt, .h5, .npy, or .npz supported
       --latents  outputs/embeddings.npz \   # .pt, .npy, or .npz supported
       --scan     512 512 \
       --lat_max_cols 6          # grid width for latent maps
       --outfig   outputs/latent_mosaic.png

For batch processing (no interactive GUI):
-----------------------------------------
python … --no-interactive

If your converter stored coordinates:
-------------------------------------
python … --coords data/coords.npy  # shape (N,2) float32 (y,x)

Advanced: Custom detector parameters (overrides physics-informed defaults):
--------------------------------------------------------------------------
--bf_radius 0.1        # <10% central disk (fraction of pattern size)
--df_inner  0.2 --df_outer 1.0  # annular ring fractions

Note: By default, physics-informed detector positioning is used based on 
automatic Bragg spot detection and interactive beam center refinement.
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


# Virtual detector functionality is handled by STEMVisualizer


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
    p.add_argument("--bf_radius", type=float, default=0.1,
                   help="BF: radius (0-1) of central disk (overrides physics-informed default)")
    p.add_argument("--df_inner", type=float, default=0.2,
                   help="DF: inner radius (0-1) fraction (overrides physics-informed default)")
    p.add_argument("--df_outer", type=float, default=1.0,
                   help="DF: outer radius (0-1) fraction (overrides physics-informed default)")
    p.add_argument("--interactive-center", action="store_true", default=True,
                   help="Use interactive beam center detection (default: True)")
    p.add_argument("--no-interactive", action="store_true", 
                   help="Disable interactive detection for batch processing")
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
    # Create both BF and DF virtual detector backgrounds

    # Create virtual detector image using STEMVisualizer
    stem_viz = STEMVisualizer(raw[:, 0], scan_shape=(Ny, Nx))
    
    # Interactive beam center detection by default
    use_interactive = args.interactive_center and not args.no_interactive
    if use_interactive:
        print("Starting interactive beam center detection...")
        print("Click center, then edge of beam. Window will close automatically.")
        try:
            stem_viz.apply_interactive_detection()
            print(f"✓ Interactive detection complete: center={stem_viz.direct_beam_position}")
        except Exception as e:
            print(f"⚠️ Interactive detection failed ({e}), using automatic detection")
            use_interactive = False
    
    if not use_interactive:
        print(f"Using automatic beam center detection: {stem_viz.direct_beam_position} (y, x)")
    
    # Create both BF and DF images using STEMVisualizer
    print("\nCreating virtual detector backgrounds...")
    
    # Create bright field image (for left panel and latent overlays)
    bf_image_full = stem_viz.create_bright_field_image()
    
    # Create dark field image (for right panel)  
    df_image_full = stem_viz.create_dark_field_image()
    
    # Convert full grid images to sparse mapping if needed
    if coord_info['is_complete_grid']:
        bf_map = bf_image_full
        df_map = df_image_full
    else:
        # For irregular coordinates, extract values and map sparsely
        bf_values = bf_image_full.ravel()
        bf_map = coords_to_sparse_image(coords, bf_values, (Ny, Nx))
        df_values = df_image_full.ravel()
        df_map = coords_to_sparse_image(coords, df_values, (Ny, Nx))

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

    # Panel 0: Bright field image (left background)
    axes[0].imshow(bf_map, cmap="gray")
    axes[0].set_title("Virtual Bright Field")
    axes[0].axis("off")

    # Panel 1: Dark field image (right background)
    axes[1].imshow(df_map, cmap="gray")
    axes[1].set_title("Virtual Dark Field")
    axes[1].axis("off")

    # empty out other slots in first row if needed
    for idx in range(2, n_cols):
        axes[idx].axis("off")

    # latent overlays starting at row 2
    for k in range(latent_dim):
        ax = axes[n_cols + k]            # offset by first row
        colour = Z[:, k]
        colour = (colour - colour.min()) / (colour.ptp() + 1e-9)
        sc = ax.scatter(coords[:, 1], coords[:, 0], c=colour,
                        cmap="viridis", s=6, alpha=0.9)
        ax.set_title(f"Latent[{k}]")
        ax.axis("off")

    # hide any unused axes
    for ax in axes[n_cols + latent_dim:]:
        ax.axis("off")

    # Apply tight_layout with padding for colorbar
    plt.tight_layout(rect=[0, 0, 0.9, 1])  # Leave space on right for colorbar
    
    # Add single colourbar for latent panels with proper positioning
    # Use fig.add_axes to manually position colorbar to avoid tight_layout conflicts
    cbar_ax = fig.add_axes([0.91, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(sc, cax=cbar_ax)
    cbar.set_label("Normalized value", rotation=270, labelpad=20)
    
    plt.savefig(args.outfig, dpi=args.dpi, bbox_inches='tight')
    
    # Final summary
    print(f"\n{'='*60}")
    print("LATENT VISUALIZATION COMPLETE")
    print(f"{'='*60}")
    print(f"Virtual detectors: BF (left) + DF (right) ({'Interactive' if use_interactive else 'Automatic'} center)")
    print(f"Beam center: {stem_viz.direct_beam_position}")
    print(f"Latent channels: {latent_dim}")
    print(f"Output saved: {args.outfig}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
