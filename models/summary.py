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

# STEM visualization functions moved to scripts/stem_visualization.py
# Import them here for backwards compatibility
try:
    from scripts.visualization.stem_visualization import STEMVisualizer
except ImportError:
    print("Warning: stem_visualization module not found. STEM visualization features unavailable.")
    STEMVisualizer = None


# Compatibility functions for tests
def create_virtual_field_image(data: np.ndarray, region: tuple, scan_shape: tuple = None) -> np.ndarray:
    """Create virtual field image (compatibility wrapper)."""
    if STEMVisualizer is None:
        raise ImportError("STEMVisualizer not available")
    
    visualizer = STEMVisualizer(data, scan_shape=scan_shape)
    return visualizer.create_virtual_field_image(region)


def save_stem_visualization(data: np.ndarray, output_path: str, reconstructed: np.ndarray = None, scan_shape: tuple = None):
    """Save STEM visualization (compatibility wrapper)."""
    if STEMVisualizer is None:
        raise ImportError("STEMVisualizer not available")
    
    visualizer = STEMVisualizer(data, scan_shape=scan_shape)
    if reconstructed is not None:
        visualizer.save_complete_visualization(output_path, reconstructed)
    else:
        visualizer.save_stem_visualization(output_path)


def show(model: nn.Module, example_input: torch.Tensor, output_dir: str = None, include_evaluation: bool = True):
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
    
    # Add performance evaluation if output_dir is provided and evaluation is requested
    if output_dir and hasattr(model, 'forward') and include_evaluation:
        print("\n" + "─" * 80)
        print("PERFORMANCE EVALUATION")
        print("─" * 80)
        
        # Ensure model is in evaluation mode
        model.eval()
        
        with torch.no_grad():
            # Generate reconstructions
            reconstructed = model(example_input)
            
            # Ensure tensors are on CPU for metrics calculation
            example_cpu = example_input.cpu()
            reconstructed_cpu = reconstructed.cpu()
            
            # Calculate metrics
            metrics = calculate_metrics(example_cpu, reconstructed_cpu)
            
            print(f"{'MSE':<20}{metrics['mse']:.6f} ± {metrics['mse_std']:.6f}")
            print(f"{'PSNR (dB)':<20}{metrics['psnr']:.2f} ± {metrics['psnr_std']:.2f}")
            print(f"{'SSIM':<20}{metrics['ssim']:.4f} ± {metrics['ssim_std']:.4f}")
            
            # Check reconstruction quality
            if metrics['mse'] > 0.1:
                print(f"{'WARNING':<20}High MSE - check model training")
            if metrics['psnr'] < 20:
                print(f"{'WARNING':<20}Low PSNR - poor reconstruction quality")
            
            # Save comparison images
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)
                comparison_path = os.path.join(output_dir, "reconstruction_comparison.png")
                save_comparison_images(example_cpu, reconstructed_cpu, comparison_path)
                print(f"{'Comparison saved to':<20}{comparison_path}")
                
                # Save STEM visualization using new module
                if STEMVisualizer is not None:
                    try:
                        stem_path = os.path.join(output_dir, "stem_visualization.png")
                        visualizer = STEMVisualizer(example_cpu.numpy())
                        visualizer.save_complete_visualization(stem_path, reconstructed_cpu.numpy())
                        print(f"{'STEM visualization':<20}{stem_path}")
                    except Exception as e:
                        print(f"{'STEM viz warning':<20}Could not create: {e}")
                else:
                    print("STEM visualization skipped (module not available)")
        
        print("─" * 80)
