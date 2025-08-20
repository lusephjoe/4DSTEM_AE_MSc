#!/usr/bin/env python3
"""
Two-level TorchLens visualizations for your 4D-STEM Autoencoder:

A) Block-level (full model) at shallow nesting depth
B) Inside-the-block diagrams: visualize one or more submodules (e.g., encoder.resnet1)
   at deeper nesting depth, as separate images.

Examples (run from repo root):

  # Default: blocks + first matching resnet down & up blocks
  python -m scripts.visualization.diagrams.visualize_with_torchlens \
    --model models.autoencoder:Autoencoder \
    --scan-y 209 --scan-x 194 --det 256 --latent-dim 32 --out torchlens_viz

  # Control nesting depths and which blocks to show:
  python -m scripts.visualization.diagrams.visualize_with_torchlens \
    --model models.autoencoder:Autoencoder \
    --blocks-depth 1 --inside-depth 4 \
    --focus-modules encoder.resnet1,decoder.resnet_up1 \
    --out torchlens_viz

Requires: torch, torchlens (pip install torchlens)
"""

from __future__ import annotations
import argparse
import contextlib
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Tuple

import torch


# ---------- robust importer (dotted path or file path) ----------
def import_class(spec: str):
    """
    Import 'pkg.mod:Class' or 'path/to/mod.py:Class'.
    Auto-adds repo root (that contains 'models/') to sys.path if needed.
    """
    if ":" not in spec:
        raise ValueError("--model must be like 'module:Class' or 'file.py:Class'")
    module_str, class_name = spec.split(":", 1)

    # file path import
    if module_str.endswith(".py") and Path(module_str).exists():
        mod_path = Path(module_str).resolve()
        spec_obj = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
        if not spec_obj or not spec_obj.loader:
            raise ImportError(f"Could not load module from {mod_path}")
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)  # type: ignore[attr-defined]
        return getattr(mod, class_name)

    # dotted import (try; if missing, add likely repo root)
    try:
        mod = importlib.import_module(module_str)
        return getattr(mod, class_name)
    except ModuleNotFoundError:
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / "models").exists():
                sys.path.insert(0, str(parent))
                break
        mod = importlib.import_module(module_str)
        return getattr(mod, class_name)


# ---------- model instantiation helpers ----------
def parse_model_kwargs(s: Optional[str]) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--model-kwargs must be valid JSON: {e}") from e


def maybe_add_kwarg(kwargs: Dict[str, Any], name: str, value: Any, fn) -> None:
    """Add kwarg if __init__ accepts it and the user didn't provide it."""
    try:
        sig = inspect.signature(fn)
        if name in sig.parameters and name not in kwargs:
            kwargs[name] = value
    except (TypeError, ValueError):
        pass


def instantiate_model(cls, latent_dim: int, det: int, device: str, kwargs_json: Optional[str]):
    kwargs = parse_model_kwargs(kwargs_json)
    maybe_add_kwarg(kwargs, "latent_dim", latent_dim, cls.__init__)
    maybe_add_kwarg(kwargs, "out_shape", (det, det), cls.__init__)
    maybe_add_kwarg(kwargs, "in_channels", 1, cls.__init__)
    model = cls(**kwargs).to(device).eval()
    return model


# ---------- Output suppression context manager ----------
@contextlib.contextmanager
def suppress_output():
    """Suppress stdout to prevent TorchLens from printing memory info."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout


# ---------- TorchLens wrappers (version-robust) ----------
def tl_render_full_or_history(obj, example, outpath: str, fileformat: str, nesting_depth: int, title: str = ""):
    """
    Try the newer 'show_model_graph(model, x, ...)' signature first.
    Fallback to 'log_forward_pass -> show_model_graph(history, ...)'.
    Suppresses console output to prevent memory info display.
    """
    from torchlens import show_model_graph, log_forward_pass

    # Newer API path - try minimal clean approach first
    try:
        with suppress_output():
            show_model_graph(
                obj, example,
                vis_outpath=outpath,
                vis_fileformat=fileformat,
                vis_nesting_depth=nesting_depth,
                vis_buffer_layers=False,  # Hide buffer layers
                save_only=True
            )
        return
    except TypeError:
        pass  # fall through

    # Older API path via model history
    with torch.no_grad(), suppress_output():
        res = log_forward_pass(obj, example, vis_nesting_depth=nesting_depth)
    history = res[-1] if isinstance(res, tuple) else res  # handle variant returns
    
    # Quick sanity check without printing the full history
    if hasattr(history, 'layer_labels'):
        layer_count = len(history.layer_labels)
    else:
        layer_count = "unknown"

    # Try different fallback approaches for older API versions
    try:
        with suppress_output():
            show_model_graph(history, vis_outpath=outpath, vis_fileformat=fileformat, 
                            vis_buffer_layers=False, save_only=True)
    except TypeError:
        try:
            with suppress_output():
                show_model_graph(history, vis_outpath=outpath, vis_buffer_layers=False, save_only=True)
        except TypeError:
            try:
                with suppress_output():
                    show_model_graph(history, vis_outpath=outpath, save_only=True)
            except TypeError:
                # Last resort - minimal parameters (might still show memory info)
                print(f"  Warning: Using minimal TorchLens API with {layer_count} layers")
                with suppress_output():
                    show_model_graph(history)


# ---------- utilities to pick blocks & their input shapes ----------
def find_first_matching_modules(model: torch.nn.Module, patterns: List[str]) -> List[str]:
    """
    Return first match for each pattern from model.named_modules().
    """
    names = [n for n, _ in model.named_modules()]
    hits: List[str] = []
    for pat in patterns:
        hit = next((n for n in names if pat in n), None)
        if hit:
            hits.append(hit)
    return hits


def capture_input_shape(model: torch.nn.Module, module_name: str, example: torch.Tensor) -> Optional[Tuple[int, int, int, int]]:
    """
    Run a forward pass with a pre-hook on the target submodule to capture the first tensor argument shape.
    Returns (B, C, H, W) or None if not seen.
    """
    target = dict(model.named_modules()).get(module_name, None)
    if target is None:
        return None

    captured: Dict[str, Tuple[int, ...]] = {}

    def pre_hook(_, args):
        if args and isinstance(args[0], torch.Tensor):
            captured["shape"] = tuple(args[0].shape)

    handle = target.register_forward_pre_hook(pre_hook)
    try:
        with torch.no_grad():
            _ = model(example)
    except Exception:
        handle.remove()
        return None
    handle.remove()

    shp = captured.get("shape", None)
    if shp and len(shp) == 4:
        return shp  # type: ignore[return-value]
    return None


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="pkg.mod:Class OR path/to/mod.py:Class")
    ap.add_argument("--model-kwargs", default=None, help="JSON dict of kwargs for the model ctor")
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--scan-y", type=int, default=209)
    ap.add_argument("--scan-x", type=int, default=194)
    ap.add_argument("--det", type=int, default=256)
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--channels", type=int, default=1)
    ap.add_argument("--micro-batch", type=int, default=1)
    ap.add_argument("--out", default="torchlens_viz", help="Output *stem*")
    ap.add_argument("--format", default="svg", choices=["svg", "pdf", "html"])
    ap.add_argument("--blocks-depth", type=int, default=1, help="Nesting depth for block-level figure (whole model)")
    ap.add_argument("--inside-depth", type=int, default=4, help="Nesting depth for inside-the-block figures")
    ap.add_argument("--focus-modules", default="", help="Comma-separated submodule names to visualize (e.g., encoder.resnet1,decoder.resnet_up1). If empty, auto-pick.")
    args = ap.parse_args()

    # 1) Model
    ModelCls = import_class(args.model)
    model = instantiate_model(ModelCls, args.latent_dim, args.det, args.device, args.model_kwargs)
    if args.ckpt:
        sd = torch.load(args.ckpt, map_location=args.device)
        if isinstance(sd, dict) and "state_dict" in sd:
            sd = sd["state_dict"]
            sd = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in sd.items()}
        model.load_state_dict(sd, strict=False)

    # 2) Example input (small batch; spatial size matters for graph)
    B = max(1, args.micro_batch)
    x = torch.randn(B, args.channels, args.det, args.det, device=args.device)

    title_base = (
        f"4D-STEM Autoencoder — logical input {args.scan_y}×{args.scan_x}×{args.det}×{args.det} "
        f"(batched as B×{args.channels}×{args.det}×{args.det}); latent_dim={args.latent_dim}"
    )

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)

    # ---------------- A) BLOCK-LEVEL (whole model) ----------------
    print(f"\n[A] Block-level figure (nesting depth={args.blocks_depth})")
    out_blocks = str(out_stem.parent / f"{out_stem.stem}_blocks.{args.format}")
    tl_render_full_or_history(
        model, x,
        outpath=out_blocks,
        fileformat=args.format,
        nesting_depth=args.blocks_depth,
        title=f"{title_base} — Block Level",
    )
    print(f"  -> {out_blocks}")

    # ---------------- B) INSIDE-THE-BLOCK (submodules) ------------
    # Decide which submodules to show
    if args.focus_modules.strip():
        focus_names = [s.strip() for s in args.focus_modules.split(",") if s.strip()]
    else:
        # Try to auto-pick a down block and an up block by common substrings.
        focus_names = find_first_matching_modules(model, ["encoder.resnet", "decoder.decoder.resnet_up", "decoder.resnet_up"])
        # fallback: any 'resnet' present
        if not focus_names:
            focus_names = find_first_matching_modules(model, ["resnet"])

    # For each focus module, capture the required tensor shape and visualize the submodule only
    for name in focus_names:
        print(f"\n[B] Inside-the-block figure for '{name}' (nesting depth={args.inside_depth})")

        in_shape = capture_input_shape(model, name, x)
        if in_shape is None:
            print(f"  ! Could not capture input shape for '{name}'. Skipping.")
            continue

        submod = dict(model.named_modules()).get(name, None)
        if submod is None:
            print(f"  ! Submodule '{name}' not found. Skipping.")
            continue

        # Build a dummy tensor that matches what the block expects
        dummy = torch.randn(in_shape, device=args.device)

        nice_name = name.replace(".", "_")
        out_inside = str(out_stem.parent / f"{out_stem.stem}_inside_{nice_name}.{args.format}")

        tl_render_full_or_history(
            submod, dummy,
            outpath=out_inside,
            fileformat=args.format,
            nesting_depth=args.inside_depth,
            title=f"{title_base} — Inside '{name}'",
        )
        print(f"  -> {out_inside}")

    print("\nDone.")


if __name__ == "__main__":
    main()
