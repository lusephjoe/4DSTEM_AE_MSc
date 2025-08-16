#!/usr/bin/env python3
"""
npy_pair_to_json.py

Combine two .npy files into one JSON object:
{
  "spatial_coordinates": [[x, y], ...],
  "cluster_labels": [l0, l1, ...]
}

Designed for large files: uses mmap + chunked streaming writes.

Usage:
  python npy_pair_to_json.py coords.npy labels.npy out.json
  python npy_pair_to_json.py coords.npy labels.npy out.json.gz   # gzip by extension

Options:
  --chunk-rows 100000     Number of rows per write chunk (default: 131072)
  --indent 0              Pretty-print indent (default: none/compact)
  --nan-as-null           Convert NaN/Inf to JSON null instead of erroring
  --overwrite             Overwrite output file if it exists
"""

import argparse
import gzip
import io
import json
import os
import sys
from typing import Iterable

import numpy as np


def open_text_auto(path: str):
    """
    Open a text file for writing. If path ends with .gz (case-insensitive),
    write gzip-compressed text; otherwise write plain UTF-8 text.
    """
    if path.lower().endswith(".gz"):
        # Use mtime=0 for deterministic gzip header, helpful for reproducibility.
        return gzip.open(path, mode="wt", encoding="utf-8", newline="", mtime=0)
    return open(path, mode="w", encoding="utf-8", newline="")


def iter_row_chunks(arr: np.ndarray, chunk_rows: int) -> Iterable[np.ndarray]:
    """Yield consecutive row chunks of size chunk_rows from a 2D array."""
    n = arr.shape[0]
    for start in range(0, n, chunk_rows):
        yield arr[start : min(start + chunk_rows, n)]


def ndarray_to_list(chunk: np.ndarray, nan_as_null: bool):
    """
    Convert a 1D or 2D ndarray to nested Python lists.
    If nan_as_null is True, replace NaN/Inf with None (-> JSON null).
    """
    if nan_as_null and np.issubdtype(chunk.dtype, np.floating):
        obj = chunk.astype(object, copy=False)
        # We need a separate finite mask from the original float view
        # to avoid comparisons on object dtype.
        float_view = chunk.astype(float, copy=False)
        mask = np.isfinite(float_view)
        # Replace non-finite with None
        obj[~mask] = None
        return obj.tolist()
    return chunk.tolist()


def write_json_array_of_rows(
    f: io.TextIOBase,
    arr: np.memmap | np.ndarray,
    chunk_rows: int,
    indent: int | None,
    nan_as_null: bool,
):
    """
    Stream-write a 2D array as a JSON array of row arrays: [[...],[...],...]
    """
    first = True
    for chunk in iter_row_chunks(arr, chunk_rows):
        rows = ndarray_to_list(chunk, nan_as_null)
        for row in rows:
            if not first:
                f.write(",")
                if indent:
                    f.write("\n" + " " * (indent * 2))  # align items nicely
            json.dump(row, f, ensure_ascii=False, allow_nan=not nan_as_null, separators=(",", ":"), indent=None)
            first = False


def write_json_array_1d(
    f: io.TextIOBase,
    arr: np.memmap | np.ndarray,
    chunk_rows: int,
    indent: int | None,
    nan_as_null: bool,
):
    """
    Stream-write a 1D array as a JSON array: [a,b,c,...]
    """
    n = arr.shape[0]
    step = max(1, chunk_rows)
    first = True
    for start in range(0, n, step):
        chunk = arr[start : min(start + step, n)]
        vals = ndarray_to_list(chunk, nan_as_null)
        for v in vals:
            if not first:
                f.write(",")
                if indent:
                    f.write("\n" + " " * (indent * 2))
            json.dump(v, f, ensure_ascii=False, allow_nan=not nan_as_null, separators=(",", ":"), indent=None)
            first = False


def main():
    parser = argparse.ArgumentParser(description="Zip two .npy files into a single JSON object.")
    parser.add_argument("spatial_npy", help="Path to spatial coordinates .npy (shape: N x D)")
    parser.add_argument("labels_npy", help="Path to cluster labels .npy (shape: N or N x 1)")
    parser.add_argument("output_json", help="Output path (.json or .json.gz)")
    parser.add_argument("--chunk-rows", type=int, default=131072, help="Rows per streamed write chunk")
    parser.add_argument("--indent", type=int, default=0, help="Pretty-print JSON with this indent (0 = compact)")
    parser.add_argument("--nan-as-null", action="store_true", help="Convert NaN/Inf to JSON null (default: error)")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists")
    args = parser.parse_args()

    if os.path.exists(args.output_json) and not args.overwrite:
        print(f"ERROR: Output file exists: {args.output_json}. Use --overwrite to replace.", file=sys.stderr)
        sys.exit(2)

    # Load with mmap so we don't bring entire arrays into RAM.
    try:
        spatial = np.load(args.spatial_npy, mmap_mode="r", allow_pickle=False)
        labels = np.load(args.labels_npy, mmap_mode="r", allow_pickle=False)
    except Exception as e:
        print(f"ERROR: Failed to load .npy files: {e}", file=sys.stderr)
        sys.exit(1)

    # Validate shapes
    if spatial.ndim != 2:
        print(f"ERROR: spatial_coordinates must be 2D (got shape {spatial.shape}).", file=sys.stderr)
        sys.exit(1)

    labels_1d = labels.ravel()  # accept (N,) or (N,1) etc.
    if spatial.shape[0] != labels_1d.shape[0]:
        print(
            f"ERROR: Row count mismatch: spatial rows={spatial.shape[0]} vs labels={labels_1d.shape[0]}.",
            file=sys.stderr,
        )
        sys.exit(1)

    # Optional: provide a heads-up if coords look degenerate
    if spatial.shape[1] < 2:
        print(
            f"WARNING: spatial_coordinates has {spatial.shape[1]} columns (expected 2+). Continuing...",
            file=sys.stderr,
        )

    # Write JSON streaming
    try:
        with open_text_auto(args.output_json) as f:
            indent = None if args.indent <= 0 else args.indent
            nl = "\n" if indent else ""
            sp = " " * (indent if indent else 0)

            # Opening brace
            f.write("{" + nl)

            # spatial_coordinates
            f.write((sp if indent else "") + '"spatial_coordinates": [' + (nl if indent else ""))
            if indent:
                f.write(sp * 2)
            write_json_array_of_rows(
                f,
                spatial,
                chunk_rows=max(1, args.chunk_rows),
                indent=indent,
                nan_as_null=args.nan_as_null,
            )
            f.write(nl + (sp if indent else "") + "],"+ nl)

            # cluster_labels
            f.write((sp if indent else "") + '"cluster_labels": [' + (nl if indent else ""))
            if indent:
                f.write(sp * 2)
            write_json_array_1d(
                f,
                labels_1d,
                chunk_rows=max(1, args.chunk_rows),
                indent=indent,
                nan_as_null=args.nan_as_null,
            )
            f.write(nl + (sp if indent else "") + "]" + nl + "}")
    except ValueError as ve:
        # Commonly triggered by NaN/Inf with allow_nan=False; suggest flag
        msg = str(ve)
        if "Out of range float values are not JSON compliant" in msg:
            print(
                "ERROR: Found NaN/Inf in data and strict JSON forbids them. "
                "Re-run with --nan-as-null to convert these to JSON null.",
                file=sys.stderr,
            )
        else:
            print(f"ERROR: {msg}", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"ERROR: Failed while writing JSON: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
