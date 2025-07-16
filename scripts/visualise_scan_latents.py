#!/usr/bin/env python
"""
Make a mosaic:

    • panel 0  : raw-average image (grayscale)
    • panel 1  : virtual detector image (BF or DF)
    • panel 2+ : each latent channel overlaid on that background

Usage - default bright-field virtual detector
--------------------------------------------
python scripts/visualise_scan_latents.py \
       --raw      data/ae_train.pt \
       --latents  outputs/embeddings.pt \
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
    """Load .pt or .npy → numpy array"""
    if path.suffix in {".pt", ".pth"}:
        return torch.load(path, map_location="cpu").numpy()
    if path.suffix == ".npy":
        return np.load(path)
    raise ValueError(f"Unknown file type {path}")


def radial_mask(shape: tuple[int, int], r_min: float, r_max: float) -> np.ndarray:
    """Build a boolean ring mask in normalised radius units (0…1)."""
    qy, qx = shape
    y, x = np.ogrid[:qy, :qx]
    cy, cx = (qy - 1) / 2, (qx - 1) / 2
    r = np.sqrt(((y - cy) / cy) ** 2 + ((x - cx) / cx) ** 2)
    return (r >= r_min) & (r < r_max)


# Use STEMVisualizer for better compatibility and features
try:
    from stem_visualization import STEMVisualizer
    
    def create_virtual_detector_image(data: np.ndarray, scan_shape: tuple, mode: str = "bf", **kwargs) -> np.ndarray:
        """Create virtual detector image using STEMVisualizer."""
        visualizer = STEMVisualizer(data, scan_shape=scan_shape)
        
        if mode == "bf":
            radius = kwargs.get('bf_radius', 0.1)
            pattern_size = min(data.shape[-2:])
            actual_radius = int(radius * pattern_size // 2)
            return visualizer.create_bright_field_image(radius=actual_radius)
        elif mode == "df":
            # Use default dark field region for now
            return visualizer.create_virtual_field_image(visualizer.dark_field_region)
        else:
            raise ValueError(f"Unknown mode: {mode}")
            
except ImportError:
    # Fallback to simple radial mask implementation
    def create_virtual_detector_image(data: np.ndarray, scan_shape: tuple, mode: str = "bf", **kwargs) -> np.ndarray:
        """Fallback virtual detector using simple radial masks."""
        print("Warning: Using fallback virtual detector implementation")
        
        if mode == "bf":
            radius = kwargs.get('bf_radius', 0.1)
            mask = radial_mask(data.shape[-2:], 0.0, radius)
        elif mode == "df":
            r_inner = kwargs.get('df_inner', 0.2)
            r_outer = kwargs.get('df_outer', 1.0)
            mask = radial_mask(data.shape[-2:], r_inner, r_outer)
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        # Apply mask and reshape
        virtual_image = np.sum(data * mask, axis=(-2, -1))
        return virtual_image.reshape(scan_shape)


# ───────────────────────────── CLI ─────────────────────────────
def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--raw", required=True, type=Path,
                   help="4D-STEM tensor N*1*Qy*Qx (output of convert_dm4.py)")
    p.add_argument("--latents", required=True, type=Path)
    p.add_argument("--scan", nargs=2, type=int, metavar=("Ny", "Nx"),
                   help="Scan grid if coords not given")
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

    # coordinates & mapping ---------------------------------------------------
    if args.coords is not None:
        coords = np.load(args.coords)    # (N,2) float32 (y,x)
        Ny = coords[:, 0].max()+1
        Nx = coords[:, 1].max()+1
    else:
        Ny, Nx = args.scan
        coords = np.stack(np.unravel_index(np.arange(N), (Ny, Nx)), axis=-1)

    # 2. Build background images ---------------------------------------------
    # raw-average (mean over diffraction pattern)
    raw_mean = raw.mean(axis=(2, 3)).reshape(Ny, Nx)

    # virtual detector --------------------------------------------------------
    # Use unified virtual detector function
    virt_map = create_virtual_detector_image(
        raw[:, 0], (Ny, Nx), mode=args.virtual,
        bf_radius=args.bf_radius, df_inner=args.df_inner, df_outer=args.df_outer
    )

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

    # Panel 0: raw-average
    axes[0].imshow(raw_mean, cmap="gray")
    axes[0].set_title("Raw average")
    axes[0].axis("off")

    # Panel 1: virtual detector
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
