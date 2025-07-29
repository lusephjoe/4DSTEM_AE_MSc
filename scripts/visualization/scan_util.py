#!/usr/bin/env python3
"""
Scan coordinate utilities for 4D-STEM data processing.

Provides robust coordinate generation and handling for various scan patterns
including raster, serpentine, and sparse arrangements.
"""
import numpy as np
from typing import Tuple, Optional


def raster_coords(Ny: int, Nx: int) -> np.ndarray:
    """Generate raster scan coordinates (row-major order).
    
    Creates coordinates for a standard raster scan where the probe moves
    left-to-right on each row, then jumps to the next row.
    
    Args:
        Ny: Number of rows in scan
        Nx: Number of columns in scan
        
    Returns:
        Array of shape (Ny*Nx, 2) with (row, col) coordinates
    """
    coords = []
    for y in range(Ny):
        for x in range(Nx):
            coords.append([y, x])
    return np.array(coords, dtype=np.int32)


def snake_coords(Ny: int, Nx: int) -> np.ndarray:
    """Generate serpentine scan coordinates.
    
    Creates coordinates for a serpentine scan where odd rows are scanned
    right-to-left to minimize probe travel time.
    
    Args:
        Ny: Number of rows in scan
        Nx: Number of columns in scan
        
    Returns:
        Array of shape (Ny*Nx, 2) with (row, col) coordinates
    """
    coords = []
    for y in range(Ny):
        if y % 2 == 0:  # Even rows: left to right
            for x in range(Nx):
                coords.append([y, x])
        else:  # Odd rows: right to left
            for x in range(Nx-1, -1, -1):
                coords.append([y, x])
    return np.array(coords, dtype=np.int32)


def factorise_scan(N: int) -> Tuple[int, int]:
    """Find the best rectangular factorization of N patterns.
    
    Returns factor pair closest to square for optimal visualization.
    
    Args:
        N: Total number of patterns
        
    Returns:
        Tuple (Ny, Nx) where Ny * Nx = N and the aspect ratio is close to 1
    """
    if N <= 0:
        raise ValueError("N must be positive")
    
    # Find all factor pairs
    factors = []
    for i in range(1, int(np.sqrt(N)) + 1):
        if N % i == 0:
            factors.append((i, N // i))
    
    if not factors:
        raise ValueError(f"Could not factorize {N}")
    
    # Choose factors closest to square (minimize aspect ratio)
    best_factors = min(factors, key=lambda x: abs(x[0] - x[1]))
    return best_factors


def coords_to_sparse_image(coords: np.ndarray, values: np.ndarray, 
                          scan_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """Convert coordinate-value pairs to a sparse 2D image.
    
    Handles irregular/sparse scans by creating a 2D array and filling only
    the positions specified by coordinates. Missing positions remain as NaN.
    
    Args:
        coords: Array of shape (N, 2) with (row, col) coordinates
        values: Array of shape (N,) with values for each coordinate
        scan_shape: Optional (Ny, Nx) tuple. If None, inferred from coords.
        
    Returns:
        2D array of shape (Ny, Nx) with values at coordinate positions
    """
    if len(coords) != len(values):
        raise ValueError("Coordinates and values must have same length")
    
    if scan_shape is None:
        Ny = int(coords[:, 0].max()) + 1
        Nx = int(coords[:, 1].max()) + 1
    else:
        Ny, Nx = scan_shape
    
    # Initialize with NaN for missing positions
    image = np.full((Ny, Nx), np.nan, dtype=np.float32)
    
    # Fill in the available data
    valid_mask = (coords[:, 0] >= 0) & (coords[:, 0] < Ny) & \
                 (coords[:, 1] >= 0) & (coords[:, 1] < Nx)
    
    valid_coords = coords[valid_mask]
    valid_values = values[valid_mask]
    
    # Ensure values are 1D (squeeze any extra dimensions)
    valid_values = np.squeeze(valid_values)
    
    image[valid_coords[:, 0], valid_coords[:, 1]] = valid_values
    
    return image


def validate_coords(coords: np.ndarray, expected_shape: Optional[Tuple[int, int]] = None) -> dict:
    """Validate coordinate array and return diagnostic information.
    
    Args:
        coords: Array of shape (N, 2) with (row, col) coordinates
        expected_shape: Optional (Ny, Nx) expected scan dimensions
        
    Returns:
        Dictionary with validation results and statistics
    """
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("Coordinates must be (N, 2) array")
    
    N = len(coords)
    min_row, max_row = coords[:, 0].min(), coords[:, 0].max()
    min_col, max_col = coords[:, 1].min(), coords[:, 1].max()
    
    inferred_shape = (int(max_row) + 1, int(max_col) + 1)
    inferred_N = inferred_shape[0] * inferred_shape[1]
    
    # Check for duplicates
    unique_coords = np.unique(coords, axis=0)
    has_duplicates = len(unique_coords) != N
    
    # Check if coordinates are within expected bounds
    bounds_ok = True
    if expected_shape is not None:
        bounds_ok = (max_row < expected_shape[0]) and (max_col < expected_shape[1])
    
    # Detect scan pattern
    if N >= 2:
        # Check if it looks like raster vs serpentine
        expected_raster = raster_coords(inferred_shape[0], inferred_shape[1])
        expected_snake = snake_coords(inferred_shape[0], inferred_shape[1])
        
        if N == len(expected_raster):
            raster_match = np.array_equal(coords, expected_raster)
            snake_match = np.array_equal(coords, expected_snake)
        else:
            raster_match = snake_match = False
        
        if raster_match:
            scan_type = "raster"
        elif snake_match:
            scan_type = "serpentine"
        else:
            scan_type = "irregular"
    else:
        scan_type = "unknown"
    
    return {
        'N_patterns': N,
        'coordinate_range': {
            'row': (int(min_row), int(max_row)),
            'col': (int(min_col), int(max_col))
        },
        'inferred_shape': inferred_shape,
        'inferred_N': inferred_N,
        'expected_shape': expected_shape,
        'completeness': N / inferred_N if inferred_N > 0 else 0,
        'has_duplicates': has_duplicates,
        'bounds_ok': bounds_ok,
        'scan_type': scan_type,
        'is_complete_grid': N == inferred_N and not has_duplicates
    }


def print_coord_summary(coords: np.ndarray, name: str = "Coordinates") -> None:
    """Print a human-readable summary of coordinate array."""
    info = validate_coords(coords)
    
    print(f"\n{name} Summary:")
    print(f"  Patterns: {info['N_patterns']}")
    print(f"  Range: rows {info['coordinate_range']['row'][0]}-{info['coordinate_range']['row'][1]}, "
          f"cols {info['coordinate_range']['col'][0]}-{info['coordinate_range']['col'][1]}")
    print(f"  Inferred scan shape: {info['inferred_shape'][0]} × {info['inferred_shape'][1]} "
          f"({info['inferred_N']} positions)")
    print(f"  Completeness: {info['completeness']:.1%}")
    print(f"  Scan type: {info['scan_type']}")
    
    if info['has_duplicates']:
        print("  ⚠️  Contains duplicate coordinates")
    if not info['bounds_ok']:
        print("  ⚠️  Some coordinates exceed expected bounds")
    if not info['is_complete_grid']:
        print("  ⚠️  Incomplete or irregular grid - will use sparse mapping")