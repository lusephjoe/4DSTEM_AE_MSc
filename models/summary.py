# models/summary.py
"""
Pretty-print a PyTorch model *and* the tensor shapes that flow through it.

Example
-------
from models.summary import show
show(model, example_input=torch.randn(1, 1, 64, 64, device="cuda"))
"""
from collections import OrderedDict
from typing import Tuple
import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import matplotlib.pyplot as plt
import os
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable

def _add_hooks(model: nn.Module, example: torch.Tensor):
    summary = OrderedDict()
    hooks = []
    processed_modules = set()

    def hook(module, inp, out):
        # Skip if this module has already been processed
        if id(module) in processed_modules:
            return
            
        class_name = module.__class__.__name__
        module_name = _get_module_name(model, module)
        key = f"{len(summary):03d}_{module_name}_{class_name}"
        
        # Only process leaf modules (modules without child modules that have parameters)
        if _is_leaf_module(module):
            summary[key] = {
                "in": tuple(inp[0].shape) if inp and len(inp) > 0 else "N/A",
                "out": tuple(out.shape) if hasattr(out, 'shape') else "N/A",
                "params": sum(p.numel() for p in module.parameters() if p.requires_grad),
                "trainable": any(p.requires_grad for p in module.parameters()),
            }
            processed_modules.add(id(module))

    for name, module in model.named_modules():
        # Skip the top-level module and container modules
        if module == model:
            continue
        if isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)):
            continue
        hooks.append(module.register_forward_hook(hook))

    with torch.no_grad():
        model(example)

    for h in hooks:
        h.remove()

    return summary

def _get_module_name(model: nn.Module, target_module: nn.Module) -> str:
    """Get the name of a module within the model."""
    for name, module in model.named_modules():
        if module is target_module:
            return name.replace('.', '_') if name else 'root'
    return 'unknown'

def _is_leaf_module(module: nn.Module) -> bool:
    """Check if a module is a leaf module (no child modules with parameters)."""
    # Check if module has parameters
    has_params = any(p.requires_grad for p in module.parameters())
    
    # Check if module has child modules with parameters
    has_child_with_params = False
    for child in module.children():
        if any(p.requires_grad for p in child.parameters()):
            has_child_with_params = True
            break
    
    # A module is a leaf if it has parameters but no child modules with parameters
    return has_params and not has_child_with_params

def calculate_metrics(original: torch.Tensor, reconstructed: torch.Tensor) -> dict:
    """Calculate reconstruction metrics between original and reconstructed images."""
    # Convert to numpy and handle batch dimension
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch dimension - calculate metrics for each sample then average
    if orig_np.ndim == 4:  # (batch, channels, height, width)
        orig_np = orig_np.squeeze(1)  # Remove channel dimension
        recon_np = recon_np.squeeze(1)
    
    mse_values = []
    psnr_values = []
    ssim_values = []
    
    for i in range(orig_np.shape[0]):
        # MSE
        mse = np.mean((orig_np[i] - recon_np[i]) ** 2)
        mse_values.append(mse)
        
        # PSNR
        if mse > 0:
            psnr = peak_signal_noise_ratio(orig_np[i], recon_np[i], data_range=1.0)
            psnr_values.append(psnr)
        else:
            psnr_values.append(float('inf'))
        
        # SSIM
        ssim = structural_similarity(orig_np[i], recon_np[i], data_range=1.0)
        ssim_values.append(ssim)
    
    return {
        'mse': np.mean(mse_values),
        'psnr': np.mean(psnr_values),
        'ssim': np.mean(ssim_values),
        'mse_std': np.std(mse_values),
        'psnr_std': np.std(psnr_values),
        'ssim_std': np.std(ssim_values)
    }

def save_comparison_images(original: torch.Tensor, reconstructed: torch.Tensor, 
                          output_path: str, num_samples: int = 4):
    """Save comparison images showing original vs reconstructed patterns."""
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if orig_np.ndim == 4:
        orig_np = orig_np.squeeze(1)
        recon_np = recon_np.squeeze(1)
    
    num_samples = min(num_samples, orig_np.shape[0])
    
    fig, axes = plt.subplots(2, num_samples, figsize=(num_samples * 3, 6))
    if num_samples == 1:
        axes = axes.reshape(2, 1)
    
    for i in range(num_samples):
        # Original
        axes[0, i].imshow(orig_np[i], cmap='viridis')
        axes[0, i].set_title(f'Original {i+1}')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(recon_np[i], cmap='viridis')
        axes[1, i].set_title(f'Reconstructed {i+1}')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def create_virtual_field_image(data: np.ndarray, scan_shape: tuple, field_region: tuple = None, 
                              field_type: str = 'bright'):
    """Create virtual bright/dark field image from diffraction data.
    
    Args:
        data: Diffraction patterns (N, H, W) or (N, C, H, W)
        scan_shape: Shape of the scan grid (scan_y, scan_x)
        field_region: Region for integration (y_min, y_max, x_min, x_max)
        field_type: 'bright' or 'dark' field
    
    Returns:
        Virtual field image (scan_y, scan_x)
    """
    # Handle tensor input
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if data.ndim == 4:
        data = data.squeeze(1)  # Remove channel dimension
    
    # Default field regions
    if field_region is None:
        h, w = data.shape[-2:]
        if field_type == 'bright':
            # Central region for bright field
            center_h, center_w = h // 2, w // 2
            region_size = min(h, w) // 8
            field_region = (center_h - region_size, center_h + region_size,
                           center_w - region_size, center_w + region_size)
        else:
            # Annular region for dark field
            center_h, center_w = h // 2, w // 2
            inner_radius = min(h, w) // 6
            outer_radius = min(h, w) // 3
            field_region = (center_h - outer_radius, center_h + outer_radius,
                           center_w - outer_radius, center_w + outer_radius)
    
    y_min, y_max, x_min, x_max = field_region
    
    # Extract the field region and sum
    field_data = data[:, y_min:y_max, x_min:x_max]
    
    if field_type == 'dark':
        # For dark field, exclude the central bright region
        center_h, center_w = field_data.shape[-2:][0] // 2, field_data.shape[-2:][1] // 2
        exclude_size = min(field_data.shape[-2:]) // 4
        field_data[:, center_h-exclude_size:center_h+exclude_size, 
                   center_w-exclude_size:center_w+exclude_size] = 0
    
    virtual_image = np.sum(field_data, axis=(1, 2))
    
    # Reshape to scan dimensions
    virtual_image = virtual_image.reshape(scan_shape)
    
    return virtual_image, field_region

def save_stem_visualization(original: torch.Tensor, reconstructed: torch.Tensor, 
                           output_path: str, scan_shape: tuple = None,
                           bright_field_region: tuple = None, dark_field_region: tuple = None):
    """Save STEM visualization with raw and virtual bright/dark field images.
    
    Args:
        original: Original diffraction patterns
        reconstructed: Reconstructed diffraction patterns  
        output_path: Path to save the visualization
        scan_shape: Shape of the scan grid (scan_y, scan_x)
        bright_field_region: Region for bright field (y_min, y_max, x_min, x_max)
        dark_field_region: Region for dark field (y_min, y_max, x_min, x_max)
    """
    orig_np = original.detach().cpu().numpy()
    recon_np = reconstructed.detach().cpu().numpy()
    
    # Handle batch and channel dimensions
    if orig_np.ndim == 4:
        orig_np = orig_np.squeeze(1)
        recon_np = recon_np.squeeze(1)
    
    # Auto-detect scan shape if not provided
    if scan_shape is None:
        n_patterns = orig_np.shape[0]
        # Try to find square or rectangular scan
        factors = []
        for i in range(1, int(np.sqrt(n_patterns)) + 1):
            if n_patterns % i == 0:
                factors.append((i, n_patterns // i))
        # Choose the most square-like factor pair
        scan_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
    
    # Calculate number of subplots
    fig_count = 2  # Raw patterns (original + reconstructed)
    
    # Add virtual field images if regions are specified
    if bright_field_region is not None:
        fig_count += 2  # Original + reconstructed bright field
    if dark_field_region is not None:
        fig_count += 2  # Original + reconstructed dark field
    
    # Create figure
    cols = min(fig_count, 3)
    rows = (fig_count + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    
    # Ensure axes is always 2D
    if rows == 1:
        axes = axes.reshape(1, -1)
    if cols == 1:
        axes = axes.reshape(-1, 1)
    
    ax_idx = 0
    
    # Plot raw diffraction patterns (mean)
    ax = axes[ax_idx // cols, ax_idx % cols]
    mean_orig = np.mean(orig_np, axis=0)
    im = ax.imshow(mean_orig, cmap='viridis')
    ax.set_title('Mean Original Diffraction')
    ax.axis('off')
    
    # Add field region indicators
    if bright_field_region is not None:
        y_min, y_max, x_min, x_max = bright_field_region
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                        fill=False, edgecolor='red', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min, y_min-5, 'BF', color='red', fontsize=12, weight='bold')
    
    if dark_field_region is not None:
        y_min, y_max, x_min, x_max = dark_field_region
        rect = Rectangle((x_min, y_min), x_max-x_min, y_max-y_min, 
                        fill=False, edgecolor='blue', linewidth=2)
        ax.add_patch(rect)
        ax.text(x_min, y_min-5, 'DF', color='blue', fontsize=12, weight='bold')
    
    ax_idx += 1
    
    # Plot reconstructed diffraction patterns (mean)
    ax = axes[ax_idx // cols, ax_idx % cols]
    mean_recon = np.mean(recon_np, axis=0)
    im = ax.imshow(mean_recon, cmap='viridis')
    ax.set_title('Mean Reconstructed Diffraction')
    ax.axis('off')
    ax_idx += 1
    
    # Create and plot virtual bright field images
    if bright_field_region is not None:
        # Original bright field
        ax = axes[ax_idx // cols, ax_idx % cols]
        bf_orig, _ = create_virtual_field_image(orig_np, scan_shape, bright_field_region, 'bright')
        im = ax.imshow(bf_orig, cmap='gray')
        ax.set_title('Original Bright Field')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        # Reconstructed bright field
        ax = axes[ax_idx // cols, ax_idx % cols]
        bf_recon, _ = create_virtual_field_image(recon_np, scan_shape, bright_field_region, 'bright')
        im = ax.imshow(bf_recon, cmap='gray')
        ax.set_title('Reconstructed Bright Field')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
    
    # Create and plot virtual dark field images
    if dark_field_region is not None:
        # Original dark field
        ax = axes[ax_idx // cols, ax_idx % cols]
        df_orig, _ = create_virtual_field_image(orig_np, scan_shape, dark_field_region, 'dark')
        im = ax.imshow(df_orig, cmap='hot')
        ax.set_title('Original Dark Field')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
        
        # Reconstructed dark field
        ax = axes[ax_idx // cols, ax_idx % cols]
        df_recon, _ = create_virtual_field_image(recon_np, scan_shape, dark_field_region, 'dark')
        im = ax.imshow(df_recon, cmap='hot')
        ax.set_title('Reconstructed Dark Field')
        ax.axis('off')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)
        ax_idx += 1
    
    # Hide unused subplots
    for i in range(ax_idx, rows * cols):
        axes[i // cols, i % cols].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def show(model: nn.Module, example_input: torch.Tensor, output_dir: str = None, *_, **__):
    device = example_input.device
    model = model.to(device).eval()

    summary = _add_hooks(model, example_input)
    print("─" * 80)
    print(f"{'Layer':<35}{'Input → Output':<30}{'Params':>10}")
    print("─" * 80)
    total, trainable = 0, 0
    for k, v in summary.items():
        io = f"{v['in']} → {v['out']}"
        print(f"{k:<35}{io:<30}{v['params']:>10,}")
        total += v["params"]
        if v["trainable"]:
            trainable += v["params"]
    print("─" * 80)
    print(f"{'Total params':<65}{total:>10,}")
    print(f"{'Trainable':<65}{trainable:>10,}")
    print("─" * 80)
    
    # Add performance evaluation if output_dir is provided
    if output_dir and hasattr(model, 'forward'):
        print("\n" + "─" * 80)
        print("PERFORMANCE EVALUATION")
        print("─" * 80)
        
        with torch.no_grad():
            reconstructed = model(example_input)
            metrics = calculate_metrics(example_input, reconstructed)
            
            print(f"{'MSE':<20}{metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
            print(f"{'PSNR (dB)':<20}{metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"{'SSIM':<20}{metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
            
            # Save comparison images
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                comparison_path = os.path.join(output_dir, "reconstruction_comparison.png")
                save_comparison_images(example_input, reconstructed, comparison_path)
                print(f"{'Comparison saved to':<20}{comparison_path}")
                
                # Save STEM visualization with virtual field images
                stem_path = os.path.join(output_dir, "stem_visualization.png")
                
                # Auto-detect scan shape and define default field regions
                n_patterns = example_input.shape[0]
                factors = []
                for i in range(1, int(np.sqrt(n_patterns)) + 1):
                    if n_patterns % i == 0:
                        factors.append((i, n_patterns // i))
                scan_shape = min(factors, key=lambda x: abs(x[0] - x[1]))
                
                # Define default bright and dark field regions
                h, w = example_input.shape[-2:]
                center_h, center_w = h // 2, w // 2
                
                # Bright field: central region
                bf_size = min(h, w) // 8
                bright_field_region = (center_h - bf_size, center_h + bf_size,
                                     center_w - bf_size, center_w + bf_size)
                
                # Dark field: annular region
                df_inner = min(h, w) // 6
                df_outer = min(h, w) // 3
                dark_field_region = (center_h - df_outer, center_h + df_outer,
                                   center_w - df_outer, center_w + df_outer)
                
                save_stem_visualization(example_input, reconstructed, stem_path, 
                                      scan_shape, bright_field_region, dark_field_region)
                print(f"{'STEM visualization':<20}{stem_path}")
        
        print("─" * 80)
