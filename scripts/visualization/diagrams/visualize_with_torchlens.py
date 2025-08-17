#!/usr/bin/env python3
"""
Visualize your autoencoder with TorchLens.

- Accepts either a dotted import path (e.g. models.autoencoder:Autoencoder)
  or a direct file path (e.g. models/autoencoder.py:Autoencoder).
- Treats a 4D-STEM array of shape (scan_y, scan_x, det, det) as a batch of
  frames for the model (B x C x H x W). TorchLens only needs a tiny batch.
- Saves an interactive HTML graph (and a static SVG when TorchLens supports it).

Usage example (run from repo root):
  python -m scripts.visualization.diagrams.visualize_with_torchlens \
    --model models.autoencoder:Autoencoder \
    --scan-y 209 --scan-x 194 --det 256 --latent-dim 32 --out torchlens_viz

Optional:
  --model-kwargs '{"latent_dim":32,"out_shape":[256,256]}'
  --ckpt /path/to/checkpoint.pt
"""

from __future__ import annotations
import argparse
import importlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Tuple

import torch


# ----------------------------- util: robust importer ----------------------------- #
def import_class(spec: str):
    """
    Import a class from either:
      - dotted path "package.module:Class", or
      - file path "path/to/module.py:Class".
    Also auto-adds likely repo roots (that contain 'models/') to sys.path
    so you can run this script from anywhere.
    """
    if ":" not in spec:
        raise ValueError("--model must be like 'module:Class' or 'file.py:Class'")
    module_str, class_name = spec.split(":", 1)

    # 1) Direct file import
    if module_str.endswith(".py") and Path(module_str).exists():
        mod_path = Path(module_str).resolve()
        spec_obj = importlib.util.spec_from_file_location(mod_path.stem, mod_path)
        if spec_obj is None or spec_obj.loader is None:
            raise ImportError(f"Could not load module from {mod_path}")
        mod = importlib.util.module_from_spec(spec_obj)
        spec_obj.loader.exec_module(mod)  # type: ignore[attr-defined]
        return getattr(mod, class_name)

    # 2) Try normal import
    try:
        mod = importlib.import_module(module_str)
        return getattr(mod, class_name)
    except ModuleNotFoundError:
        # 3) Search upwards for a repo root that contains 'models' and add to sys.path
        here = Path(__file__).resolve()
        for parent in [here.parent] + list(here.parents):
            if (parent / "models").exists():
                sys.path.insert(0, str(parent))
                break
        mod = importlib.import_module(module_str)
        return getattr(mod, class_name)


# ----------------------------- util: model instantiation ----------------------------- #
def parse_model_kwargs(s: str | None) -> Dict[str, Any]:
    if not s:
        return {}
    try:
        return json.loads(s)
    except json.JSONDecodeError as e:
        raise SystemExit(f"--model-kwargs must be valid JSON: {e}") from e


def maybe_add_kwarg(kwargs: Dict[str, Any], name: str, value: Any, fn) -> None:
    """Add kwarg if the model __init__ accepts it and user didn't supply it."""
    try:
        sig = inspect.signature(fn)
        if name in sig.parameters and name not in kwargs:
            kwargs[name] = value
    except (TypeError, ValueError):
        # if we can't inspect, just don't add
        pass


def instantiate_model(cls, latent_dim: int, det: int, device: str, kwargs_json: str | None):
    kwargs = parse_model_kwargs(kwargs_json)

    # populate common kwargs if the constructor supports them
    maybe_add_kwarg(kwargs, "latent_dim", latent_dim, cls.__init__)
    maybe_add_kwarg(kwargs, "out_shape", (det, det), cls.__init__)
    maybe_add_kwarg(kwargs, "in_channels", 1, cls.__init__)

    model = cls(**kwargs).to(device).eval()
    return model


def maybe_load_checkpoint(model: torch.nn.Module, ckpt_path: str | None, device: str) -> None:
    if not ckpt_path:
        return
    map_location = torch.device(device)
    sd = torch.load(ckpt_path, map_location=map_location)
    # tolerate common ckpt formats
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
        # strip lightning prefixes if present
        sd = {k.split("model.", 1)[-1] if k.startswith("model.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=False)


# ----------------------------- main ----------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="Import path 'pkg.mod:Class' OR file 'path/to/mod.py:Class'")
    ap.add_argument("--model-kwargs", default=None,
                    help="JSON dict of kwargs to pass to the model constructor")
    ap.add_argument("--ckpt", default=None, help="Optional checkpoint to load")
    ap.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    ap.add_argument("--scan-y", type=int, default=209)
    ap.add_argument("--scan-x", type=int, default=194)
    ap.add_argument("--det", type=int, default=256, help="Detector H=W")
    ap.add_argument("--latent-dim", type=int, default=32)
    ap.add_argument("--channels", type=int, default=1, help="Input channels (default 1)")
    ap.add_argument("--micro-batch", type=int, default=1,
                    help="Safe batch size for the tracing forward pass")
    ap.add_argument("--out", default="torchlens_viz", help="Output file stem (no extension)")
    args = ap.parse_args()

    # 1) import & instantiate model
    ModelCls = import_class(args.model)
    model = instantiate_model(ModelCls, args.latent_dim, args.det, args.device, args.model_kwargs)
    maybe_load_checkpoint(model, args.ckpt, args.device)

    # 2) prepare safe example input (TorchLens only needs spatial size)
    B = max(1, args.micro_batch)
    x = torch.randn(B, args.channels, args.det, args.det, device=args.device)

    # 3) run TorchLens
    try:
        from torchlens import log_forward_pass, show_model_graph
    except Exception as e:
        raise SystemExit(
            "TorchLens is not installed in this environment. "
            "Install with: pip install torchlens"
        ) from e

    torch.set_grad_enabled(False)
    with torch.no_grad():
        res = log_forward_pass(model, x)

    # 4) compatibility shim: TorchLens versions return different shapes
    if isinstance(res, tuple):
        model_history = res[-1]  # last item is the history/graph object
    else:
        model_history = res

    title = (
        f"4D-STEM Autoencoder — logical input {args.scan_y}×{args.scan_x}×{args.det}×{args.det} "
        f"(batched to B×{args.channels}×{args.det}×{args.det}); latent_dim={args.latent_dim}"
    )

    out_base = Path(args.out)
    out_base.parent.mkdir(parents=True, exist_ok=True)
    html_path = str(out_base.with_suffix(".html"))
    svg_path = str(out_base.with_suffix(".svg"))

    # 5) save HTML (and try SVG if supported)
    saved_html = False
    try:
        show_model_graph(model, x, vis_outpath=html_path, vis_fileformat='pdf', save_only=True)
        saved_html = True
    except Exception as e:
        print(f"Could not save visualization: {e}")

    # try SVG
    try:
        show_model_graph(model, x, vis_outpath=svg_path, vis_fileformat='svg', save_only=True)
    except Exception:
        pass

    print("\n=== TorchLens Visualization ===")
    if saved_html:
        print(f"Interactive HTML : {html_path}")
    if Path(svg_path).exists():
        print(f"Static SVG       : {svg_path}")
    print(f"Tracing input    : shape={tuple(x.shape)} on device={args.device}")
    print(f"Logical 4D input : ({args.scan_y}, {args.scan_x}, {args.det}, {args.det}) "
          f"=> batched as (B, C, H, W)")


if __name__ == "__main__":
    main()
