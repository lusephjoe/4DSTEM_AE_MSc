#!/usr/bin/env python3
"""
STEM Visualization Module

Enhanced visualization tools for 4D-STEM data including:
- Virtual bright/dark field imaging
- Diffraction pattern analysis
- Support for .dm4 files
- Interactive visualization capabilities
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from typing import Tuple, Optional, Union
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
import hyperspy.api as hs
from matplotlib.widgets import RectangleSelector


class STEMVisualizer:
    """Enhanced STEM data visualization with interactive capabilities."""
    
    def __init__(self, data: Union[np.ndarray, torch.Tensor], scan_shape: Optional[Tuple[int, int]] = None, 
                 scalebar_info: Optional[dict] = None):
        """
        Initialize STEM visualizer.
        
        Args:
            data: Diffraction patterns (N, H, W) or (N, C, H, W)
            scan_shape: Shape of the scan grid (scan_y, scan_x)
            scalebar_info: Dict with keys 'width', 'scale_length', 'units'
        """
        self.data = self._prepare_data(data)
        self.scan_shape = scan_shape or self._infer_scan_shape()
        self.pattern_shape = self.data.shape[-2:]
        self.center = (self.pattern_shape[0] // 2, self.pattern_shape[1] // 2)
        
        # Default field regions
        self.bright_field_region = self._default_bright_field_region()
        self.dark_field_region = self._default_dark_field_region()
        
        # Scalebar information
        self.scalebar_info = scalebar_info or {
            'width': max(self.scan_shape) * 10,  # Default: 10nm per pixel
            'scale_length': 100,  # Default: 100nm scalebar
            'units': 'nm'
        }
        
    def _prepare_data(self, data: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Convert input data to numpy array with proper shape."""
        if isinstance(data, torch.Tensor):
            data = data.detach().cpu().numpy()
        
        # Handle batch and channel dimensions
        if data.ndim == 4:
            data = data.squeeze(1)  # Remove channel dimension
        
        # Apply log transformation like m3_learning (add 1 to avoid log(0))
        self.log_data = np.log(data + 1)
        
        return data
    
    def _infer_scan_shape(self) -> Tuple[int, int]:
        """Infer scan shape from data dimensions."""
        n_patterns = self.data.shape[0]
        # Try to find square or rectangular scan
        factors = []
        for i in range(1, int(np.sqrt(n_patterns)) + 1):
            if n_patterns % i == 0:
                factors.append((i, n_patterns // i))
        # Choose the most square-like factor pair
        return min(factors, key=lambda x: abs(x[0] - x[1]))
    
    def _default_bright_field_region(self) -> Tuple[int, int, int, int]:
        """Default bright field region (central) - more conservative size."""
        center_h, center_w = self.center
        region_size = min(self.pattern_shape) // 16  # Smaller region like m3_learning examples
        return (center_h - region_size, center_h + region_size,
                center_w - region_size, center_w + region_size)
    
    def _default_dark_field_region(self) -> Tuple[int, int, int, int]:
        """Default dark field region (annular) - more conservative size."""
        center_h, center_w = self.center
        outer_radius = min(self.pattern_shape) // 6  # Smaller region like m3_learning examples
        return (center_h - outer_radius, center_h + outer_radius,
                center_w - outer_radius, center_w + outer_radius)
    
    def set_bright_field_region(self, region: Tuple[int, int, int, int]):
        """Set bright field region (y_min, y_max, x_min, x_max)."""
        self.bright_field_region = region
    
    def set_dark_field_region(self, region: Tuple[int, int, int, int]):
        """Set dark field region (y_min, y_max, x_min, x_max)."""
        self.dark_field_region = region
    
    def create_virtual_field_image(self, field_region: Tuple[int, int, int, int], 
                                  field_type: str = 'bright') -> np.ndarray:
        """
        Create virtual bright/dark field image following m3_learning implementation.
        
        Args:
            field_region: Region for integration (y_min, y_max, x_min, x_max)
            field_type: 'bright' or 'dark' field
        
        Returns:
            Virtual field image (scan_y, scan_x)
        """
        y_min, y_max, x_min, x_max = field_region
        
        # Extract the field region from raw data (not log data)
        # Following m3_learning: data.data.reshape(-1, shape_[2], shape_[3])[:, y_min:y_max, x_min:x_max]
        field_data = self.data[:, y_min:y_max, x_min:x_max]
        
        # Calculate mean intensity in the region for each scan position
        # Following m3_learning: np.mean(field_data.reshape(shape_[0]*shape_[1], -1), axis=1)
        virtual_image = np.mean(field_data.reshape(self.data.shape[0], -1), axis=1)
        
        # Reshape to scan dimensions
        virtual_image = virtual_image.reshape(self.scan_shape)
        
        return virtual_image
    
    def _add_scalebar(self, ax, pattern_scale=False):
        """Add scalebar to the plot following m3_learning style."""
        if pattern_scale:
            # For diffraction patterns - typically in reciprocal space
            scale_size = 20  # Example: 20 pixels
            scale_length = 1.0  # Example: 1.0 nm⁻¹
            units = "nm⁻¹"
            image_size = max(self.pattern_shape)
        else:
            # For real space images
            scale_size = self.scalebar_info['scale_length']
            scale_length = scale_size
            units = self.scalebar_info['units']
            image_size = self.scalebar_info['width']
        
        # Get axis limits
        x_lim, y_lim = ax.get_xlim(), ax.get_ylim()
        x_size = int(np.abs(x_lim[1] - x_lim[0]))
        y_size = int(np.abs(y_lim[1] - y_lim[0]))
        
        # Calculate scalebar position (bottom right)
        fract = scale_size / image_size
        x_start = x_lim[0] + 0.9 * x_size
        x_end = x_lim[0] + (0.9 - fract) * x_size
        y_start = y_lim[0] + 0.1 * y_size
        y_end = y_lim[0] + 0.125 * y_size
        y_label = y_lim[0] + 0.175 * y_size
        
        # Draw scalebar
        ax.plot([x_end, x_start], [y_start, y_start], 'w-', linewidth=3)
        ax.plot([x_end, x_start], [y_start, y_start], 'k-', linewidth=1)
        ax.plot([x_end, x_end], [y_start, y_end], 'w-', linewidth=3)
        ax.plot([x_end, x_end], [y_start, y_end], 'k-', linewidth=1)
        ax.plot([x_start, x_start], [y_start, y_end], 'w-', linewidth=3)
        ax.plot([x_start, x_start], [y_start, y_end], 'k-', linewidth=1)
        
        # Add text label
        ax.text((x_start + x_end) / 2, y_label, f"{scale_length} {units}",
                ha='center', va='bottom', color='white', fontsize=10, weight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.7))

    def plot_mean_diffraction_pattern(self, ax=None, show_regions=True, use_log=True, add_scalebar=False):
        """Plot mean diffraction pattern with field regions."""
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 8))
        
        # Use log data for visualization following m3_learning
        # Following m3_learning: np.mean(data.log_data.reshape(-1, shape_[2], shape_[3]), axis=0)
        if use_log and hasattr(self, 'log_data'):
            mean_pattern = np.mean(self.log_data, axis=0)
        else:
            mean_pattern = np.mean(self.data, axis=0)
            
        im = ax.imshow(mean_pattern, cmap='viridis')
        ax.set_title('Mean Diffraction Pattern')
        ax.axis('off')
        
        if show_regions:
            # Bright field region
            y_min, y_max, x_min, x_max = self.bright_field_region
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                            fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min-5, 'BF', color='red', fontsize=12, weight='bold')
            
            # Dark field region
            y_min, y_max, x_min, x_max = self.dark_field_region
            rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                            fill=False, edgecolor='blue', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min-5, 'DF', color='blue', fontsize=12, weight='bold')
        
        if add_scalebar:
            self._add_scalebar(ax, pattern_scale=True)
        
        return im
    
    def plot_virtual_images(self, figsize=(12, 4)):
        """Plot virtual bright and dark field images."""
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Mean diffraction pattern (log scale)
        self.plot_mean_diffraction_pattern(axes[0], use_log=True)
        axes[0].set_title('Mean Diffraction Pattern (Log)')
        
        # Bright field image
        bf_image = self.create_virtual_field_image(self.bright_field_region, 'bright')
        im1 = axes[1].imshow(bf_image, cmap='gray')
        axes[1].set_title('Virtual Bright Field')
        axes[1].axis('off')
        divider = make_axes_locatable(axes[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im1, cax=cax)
        
        # Dark field image
        df_image = self.create_virtual_field_image(self.dark_field_region, 'dark')
        im2 = axes[2].imshow(df_image, cmap='hot')
        axes[2].set_title('Virtual Dark Field')
        axes[2].axis('off')
        divider = make_axes_locatable(axes[2])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im2, cax=cax)
        
        plt.tight_layout()
        return fig
    
    def interactive_region_selector(self):
        """Interactive region selector for virtual field imaging."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        mean_pattern = np.mean(self.data, axis=0)
        im = ax.imshow(mean_pattern, cmap='viridis')
        ax.set_title('Select Virtual Field Regions (Click and drag)')
        
        # Store selected regions
        self.selected_regions = []
        
        def onselect(eclick, erelease):
            x1, y1 = int(eclick.xdata), int(eclick.ydata)
            x2, y2 = int(erelease.xdata), int(erelease.ydata)
            
            # Ensure proper order
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            
            region = (y_min, y_max, x_min, x_max)
            self.selected_regions.append(region)
            
            print(f"Selected region: {region}")
            print(f"To use: visualizer.set_bright_field_region({region})")
        
        selector = RectangleSelector(ax, onselect, useblit=True,
                                   button=[1], minspanx=5, minspany=5,
                                   spancoords='pixels', interactive=True)
        
        plt.colorbar(im)
        plt.show()
        
        return selector
    
    def save_complete_visualization(self, output_path: str, comparison_data: Optional[np.ndarray] = None):
        """Save complete STEM visualization with all components."""
        # Determine subplot configuration
        n_plots = 3  # Mean pattern + BF + DF
        if comparison_data is not None:
            n_plots += 3  # + reconstructed versions
        
        cols = 3
        rows = (n_plots + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
        
        # Ensure axes is 2D
        if rows == 1:
            axes = axes.reshape(1, -1)
        if cols == 1:
            axes = axes.reshape(-1, 1)
        
        ax_idx = 0
        
        # Original data visualizations
        self.plot_mean_diffraction_pattern(axes[ax_idx // cols, ax_idx % cols], use_log=True)
        axes[ax_idx // cols, ax_idx % cols].set_title('Mean Diffraction Pattern (Log)')
        ax_idx += 1
        
        # Bright field
        bf_image = self.create_virtual_field_image(self.bright_field_region, 'bright')
        im = axes[ax_idx // cols, ax_idx % cols].imshow(bf_image, cmap='gray')
        axes[ax_idx // cols, ax_idx % cols].set_title('Virtual Bright Field')
        axes[ax_idx // cols, ax_idx % cols].axis('off')
        divider = make_axes_locatable(axes[ax_idx // cols, ax_idx % cols])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        # Dark field
        df_image = self.create_virtual_field_image(self.dark_field_region, 'dark')
        im = axes[ax_idx // cols, ax_idx % cols].imshow(df_image, cmap='hot')
        axes[ax_idx // cols, ax_idx % cols].set_title('Virtual Dark Field')
        axes[ax_idx // cols, ax_idx % cols].axis('off')
        divider = make_axes_locatable(axes[ax_idx // cols, ax_idx % cols])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        # Comparison data if provided
        if comparison_data is not None:
            comp_viz = STEMVisualizer(comparison_data, self.scan_shape)
            comp_viz.bright_field_region = self.bright_field_region
            comp_viz.dark_field_region = self.dark_field_region
            
            # Reconstructed mean pattern
            comp_viz.plot_mean_diffraction_pattern(axes[ax_idx // cols, ax_idx % cols], show_regions=False)
            axes[ax_idx // cols, ax_idx % cols].set_title('Reconstructed Mean Pattern')
            ax_idx += 1
            
            # Reconstructed bright field
            comp_bf = comp_viz.create_virtual_field_image(self.bright_field_region, 'bright')
            axes[ax_idx // cols, ax_idx % cols].imshow(comp_bf, cmap='gray')
            axes[ax_idx // cols, ax_idx % cols].set_title('Reconstructed Bright Field')
            axes[ax_idx // cols, ax_idx % cols].axis('off')
            ax_idx += 1
            
            # Reconstructed dark field
            comp_df = comp_viz.create_virtual_field_image(self.dark_field_region, 'dark')
            axes[ax_idx // cols, ax_idx % cols].imshow(comp_df, cmap='hot')
            axes[ax_idx // cols, ax_idx % cols].set_title('Reconstructed Dark Field')
            axes[ax_idx // cols, ax_idx % cols].axis('off')
            ax_idx += 1
        
        # Hide unused subplots
        for i in range(ax_idx, rows * cols):
            axes[i // cols, i % cols].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"STEM visualization saved to: {output_path}")


def load_dm4_data(filepath: str) -> Tuple[np.ndarray, dict, Tuple[int, int]]:
    """
    Load 4D-STEM data from .dm4 file following m3_learning approach.
    
    Args:
        filepath: Path to .dm4 file
    
    Returns:
        Tuple of (data_array, metadata, scan_shape)
    """
    # Load using hyperspy
    signal = hs.load(filepath)
    
    # Extract data and metadata
    data = signal.data
    metadata = signal.metadata.as_dictionary()
    
    # Reshape if needed - 4D STEM data should be (scan_y, scan_x, det_y, det_x)
    if data.ndim == 4:
        scan_y, scan_x, det_y, det_x = data.shape
        # Reshape to (N, det_y, det_x) format for processing
        data = data.reshape(scan_y * scan_x, det_y, det_x)
        scan_shape = (scan_y, scan_x)
    else:
        scan_shape = None
    
    print(f"Loaded data shape: {data.shape}")
    print(f"Scan shape: {scan_shape}")
    print(f"Pattern shape: {data.shape[-2:]}")
    
    return data, metadata, scan_shape


def main():
    """Main function for standalone execution."""
    parser = argparse.ArgumentParser(description='STEM Data Visualization Tool')
    parser.add_argument('input_file', type=str, help='Input file (.dm4, .pt, .npy)')
    parser.add_argument('--output', '-o', type=str, default='stem_visualization.png',
                       help='Output visualization file')
    parser.add_argument('--comparison', '-c', type=str, default=None,
                       help='Comparison data file (reconstructed data)')
    parser.add_argument('--scan_shape', nargs=2, type=int, default=None,
                       help='Scan shape (scan_y scan_x)')
    parser.add_argument('--bf_region', nargs=4, type=int, default=None,
                       help='Bright field region (y_min y_max x_min x_max)')
    parser.add_argument('--df_region', nargs=4, type=int, default=None,
                       help='Dark field region (y_min y_max x_min x_max)')
    parser.add_argument('--interactive', '-i', action='store_true',
                       help='Interactive region selection')
    
    args = parser.parse_args()
    
    # Load data
    input_path = Path(args.input_file)
    if input_path.suffix == '.dm4':
        data, metadata, scan_shape = load_dm4_data(args.input_file)
        print(f"Loaded .dm4 file with shape: {data.shape}")
        print(f"Inferred scan shape: {scan_shape}")
    elif input_path.suffix == '.pt':
        data = torch.load(args.input_file)
        scan_shape = tuple(args.scan_shape) if args.scan_shape else None
        print(f"Loaded .pt file with shape: {data.shape}")
    elif input_path.suffix == '.npy':
        data = np.load(args.input_file)
        scan_shape = tuple(args.scan_shape) if args.scan_shape else None
        print(f"Loaded .npy file with shape: {data.shape}")
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}")
    
    # Create visualizer
    visualizer = STEMVisualizer(data, scan_shape)
    
    # Set custom regions if provided
    if args.bf_region:
        visualizer.set_bright_field_region(tuple(args.bf_region))
    if args.df_region:
        visualizer.set_dark_field_region(tuple(args.df_region))
    
    # Interactive mode
    if args.interactive:
        print("Starting interactive region selector...")
        print("Click and drag to select regions on the diffraction pattern")
        selector = visualizer.interactive_region_selector()
        return
    
    # Load comparison data if provided
    comparison_data = None
    if args.comparison:
        comp_path = Path(args.comparison)
        if comp_path.suffix == '.pt':
            comparison_data = torch.load(args.comparison)
        elif comp_path.suffix == '.npy':
            comparison_data = np.load(args.comparison)
        else:
            raise ValueError(f"Unsupported comparison file format: {comp_path.suffix}")
        print(f"Loaded comparison data with shape: {comparison_data.shape}")
    
    # Generate and save complete visualization (all panels)
    visualizer.save_complete_visualization(args.output, comparison_data)
    
    # Create 3-panel layout with scalebars
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    # Mean diffraction pattern (log scale) with scalebar
    visualizer.plot_mean_diffraction_pattern(axes[0], use_log=True, add_scalebar=True)
    axes[0].set_title('Mean Diffraction Pattern (Log)')
    
    # Bright field image with scalebar
    bf_image = visualizer.create_virtual_field_image(visualizer.bright_field_region, 'bright')
    axes[1].imshow(bf_image, cmap='gray')
    axes[1].set_title('Virtual Bright Field')
    axes[1].axis('off')
    visualizer._add_scalebar(axes[1], pattern_scale=False)
    
    # Dark field image with scalebar
    df_image = visualizer.create_virtual_field_image(visualizer.dark_field_region, 'dark')
    axes[2].imshow(df_image, cmap='hot')
    axes[2].set_title('Virtual Dark Field')
    axes[2].axis('off')
    visualizer._add_scalebar(axes[2], pattern_scale=False)
    
    plt.tight_layout()
    panel_output = args.output.replace('.png', '_3panel.png')
    fig.savefig(panel_output, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"3-panel visualization saved to: {panel_output}")
    
    # Create separate individual images for each component
    base_name = args.output.replace('.png', '')
    
    # Mean diffraction pattern
    fig, ax = plt.subplots(figsize=(8, 8))
    visualizer.plot_mean_diffraction_pattern(ax, use_log=True, add_scalebar=True)
    fig.savefig(f"{base_name}_mean_diffraction.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Bright field
    fig, ax = plt.subplots(figsize=(8, 8))
    bf_image = visualizer.create_virtual_field_image(visualizer.bright_field_region, 'bright')
    ax.imshow(bf_image, cmap='gray')
    ax.set_title('Virtual Bright Field')
    ax.axis('off')
    visualizer._add_scalebar(ax, pattern_scale=False)
    fig.savefig(f"{base_name}_bright_field.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # Dark field
    fig, ax = plt.subplots(figsize=(8, 8))
    df_image = visualizer.create_virtual_field_image(visualizer.dark_field_region, 'dark')
    ax.imshow(df_image, cmap='hot')
    ax.set_title('Virtual Dark Field')
    ax.axis('off')
    visualizer._add_scalebar(ax, pattern_scale=False)
    fig.savefig(f"{base_name}_dark_field.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Individual images saved: {base_name}_mean_diffraction.png, {base_name}_bright_field.png, {base_name}_dark_field.png")


if __name__ == "__main__":
    main()