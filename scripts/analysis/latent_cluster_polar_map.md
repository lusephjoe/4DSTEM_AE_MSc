"""
Data-driven Virtual Detectors for 4D-STEM Polarisation Mapping

This script implements a novel approach to 4D-STEM analysis that combines autoencoder-based 
clustering with physics-informed virtual detector design for polarization mapping in 
ferroelectric and other functional materials.

METHODOLOGY OVERVIEW
====================

The approach consists of four main stages:

1. CLUSTER-GUIDED TEMPLATE GENERATION
   - Uses pre-computed cluster labels from autoencoder latent space analysis
   - Builds mean diffraction patterns μ_c(k) for each cluster c
   - Creates differential templates: T_c(k) = (μ_c(k) - μ(k)) / (σ(k) + ε)
   - L2-normalizes templates to ensure fair comparison across clusters
   - Templates highlight k-space features that distinguish each cluster

2. VIRTUAL DETECTOR DESIGN
   - Converts templates to detector gates G_c(k) via percentile thresholding
   - Keeps only top X% of template values (default: 10%) to focus on most distinctive features
   - Applies symmetrization: G_c = 0.5 * (G_c + G_c[::-1, ::-1]) to preserve CoM accuracy
   - Optional radial band-limiting to exclude unwanted reflections (HOLZ, etc.)
   - Gates define which k-space regions contribute to analysis for each cluster

3. MATCHED-FILTER CONTRAST MAPPING
   - For each probe position i: S_c(i) = Σ_k T_c(k) * [(I_i(k) - μ(k)) / (σ(k) + ε)]
   - This gives cluster-specific contrast maps S_c(x,y) in real space
   - Softmax with temperature creates confidence maps and argmax cluster assignment
   - Reveals spatial distribution of different diffraction behaviors

4. CLUSTER-GATED DPC/COM ANALYSIS
   - Computes center-of-mass (CoM) for each diffraction pattern using cluster-specific gates
   - Two modes: 'union' (use max of all gates) or 'adaptive' (use gate of assigned cluster)
   - CoM shifts C_x, C_y are converted to projected electric fields E_x, E_y
   - Baseline subtraction removes systematic offsets from beam center uncertainties
   - Generates polarization maps: |E|, angle θ, and HSV visualization

PHYSICS INTERPRETATION
======================

The cluster-guided approach assumes that different local structural environments 
(ferroelectric domains, defects, interfaces) produce characteristic diffraction signatures.
By learning these signatures via unsupervised clustering, we can:

- Design optimal virtual detectors that are sensitive to specific structural features
- Reduce noise by focusing analysis on informative k-space regions
- Map polarization textures with enhanced contrast and spatial resolution
- Distinguish between different types of domains or structural phases

The DPC analysis measures the deflection of the electron beam due to local electric fields,
which are related to ferroelectric polarization, space charge, and built-in fields.
Cluster gating enhances sensitivity by weighting the analysis toward k-space regions
that correlate with the structural features of interest.

MATHEMATICAL FORMULATION
========================

Template Generation:
    μ_c(k) = mean_{i∈cluster_c} I_i(k)                    # cluster mean
    μ(k) = mean_i I_i(k)                                   # global mean
    σ(k) = std_i I_i(k)                                    # global std
    T_c(k) = (μ_c(k) - μ(k)) / (σ(k) + ε)                # z-scored template
    T_c(k) = T_c(k) / ||T_c(k)||_2                        # L2 normalize

Gate Generation:
    threshold = percentile(T_c, 100 - top_percent)
    G_c(k) = T_c(k) >= threshold                           # binary mask
    G_c(k) = 0.5 * (G_c(k) + G_c(k)[::-1, ::-1])         # symmetrize
    G_c(k) = G_c(k) * radial_mask(k)                      # optional ROI

Matched Filtering:
    Z_i(k) = (I_i(k) - μ(k)) / (σ(k) + ε)                # z-score normalize
    # Optional diagonal whitening: Z_i(k) = (I_i(k) - μ(k)) / (σ(k) + ε)
    S_c(i) = Σ_k T_c(k) * Z_i(k)                         # template correlation
    
    # Softmax with temperature for confidence
    confidence_c(i) = exp(S_c(i)/τ) / Σ_j exp(S_j(i)/τ)

DPC/CoM with Gating:
    # Center of mass calculation
    C_x(i) = Σ_k k_x * G(k) * I_i(k) / Σ_k G(k) * I_i(k)
    C_y(i) = Σ_k k_y * G(k) * I_i(k) / Σ_k G(k) * I_i(k)
    
    # Baseline correction options
    C_x(i) = C_x(i) - median(C_x)                    # global median
    C_x(i) = C_x(i) - median(C_x, axis=1)[:, None]  # row-wise median
    C_x(i) = C_x(i) - mean(C_x[ROI])                # ROI-based
    
    # Convert to electric field (simplified)
    E_x(i) = α * C_x(i)
    E_y(i) = α * C_y(i)
    
    where α is a calibration factor depending on camera length and pixel size.

USAGE EXAMPLES
==============

Basic usage with auto-detection:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --dset patterns \
      --labels cluster_results.npz --scan-shape 128 128 \
      --out polarization_analysis

With custom configuration:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 256 256 --config config.yaml \
      --device cuda:0 --chunks 2048 --out results

With interactive beam center picking:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 128 128 --interactive-center \
      --out results

With interactive k-space calibration:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 128 128 --interactive-kspace \
      --known-spacing 3.14 --out results

With both interactive modes:
    python scripts/analysis/latent_cluster_polar_map.py \
      --data patterns.h5 --labels clusters.npy \
      --scan-shape 128 128 --interactive-center --interactive-kspace \
      --known-spacing 2.45 --out results

Configuration options (config.yaml):
    templates:
      top_percent: 10        # Keep top 10% of template values
      smooth_sigma: 1.5      # Gaussian smoothing of gates
      roi_r: [8, 80]        # Radial ROI limits (inner, outer radius)
    
    preprocess:
      log_transform: true    # Apply log(I+1) transform
      background_subtract: true  # Subtract mean background
      central_mask_px: 8     # Mask central beam (radius in pixels)
      whiten: diag           # Diagonal whitening: 'none', 'diag'
    
    matched_filter:
      softmax_temp: 2.0      # Temperature for softmax confidence
    
    dpc:
      gate_mode: adaptive    # 'union' or 'adaptive' gating
      baseline: rowcol       # 'median', 'rowcol', or 'roi' baseline correction
      roi: [8, 80]          # Radial ROI for DPC analysis
      field_scale: 1.0       # Conversion factor to physical field units
    
    calibration:
      center_y: null         # Auto-detect beam center if null
      center_x: null
      pixel_size: null       # Direct pixel size in nm/pixel (takes precedence)
      pixel_size_units: "nm" # Units for pixel_size: "nm" or "angstrom" 
      pixel_size_k: 1.0      # k-space pixel size (1/Å or relative units)
      pixel_size_ky: null    # Anisotropic k-space calibration (y-axis, same units as pixel_size_k)
      pixel_size_kx: null    # Anisotropic k-space calibration (x-axis, same units as pixel_size_k)
      convergence_angle: null # Convergence angle in mrad (for advanced calculations)
    
    validation:
      enable_split_half: false  # Enable template stability validation

OUTPUTS
=======

The script generates a structured output directory:

    outdir/
      kspace/                    # K-space artifacts
        template_T_{c}.npy       # Differential templates for each cluster
        gate_G_{c}.npy           # Virtual detector gates
        mu_cluster_{c}.npy       # True cluster mean patterns (.npy)
        mu_cluster_{c}.png       # Mean patterns (visualization)
        mu_global.npy            # Global mean pattern
        sigma_global.npy         # Global standard deviation
      
      realspace/                 # Real-space maps
        S_cluster_{c}.tif        # Contrast maps for each cluster
        S_argmax.tif             # Cluster assignment map
        S_confidence.tif         # Assignment confidence
        Ex.tif, Ey.tif           # Electric field components
        E_mag.tif, E_angle.tif   # Field magnitude and angle
        com_x.tif, com_y.tif     # Raw center-of-mass maps
      
      figures/                   # Visualizations
        montage_templates.png    # Template and gate overview
        hsv_polarisation.png     # HSV polarization visualization
        DPC_quiver.png           # Vector field quiver plot
      
      logs/                      # Provenance and metadata
        run.json                 # Complete run parameters and timing
        config.yaml              # Configuration used
        timings.txt              # Performance breakdown

INTERACTIVE CALIBRATION FEATURES
=================================

The script provides advanced interactive calibration capabilities:

1. **Interactive Beam Center Picking** (--interactive-center):
   - Opens a GUI window displaying the mean diffraction pattern
   - Shows auto-detected center as red dashed crosshairs for reference
   - Click once to select the desired beam center
   - Window closes automatically after selection
   - Updates config and uses selected center for all calculations

2. **Interactive K-space Calibration** (--interactive-kspace):
   - Enables calibration of k-space pixel sizes by clicking Bragg peaks
   - Displays mean pattern with log scaling for better peak visibility
   - Shows guiding circles and beam center marker
   - Click two opposite Bragg peaks (180° apart) or ring boundaries
   - Automatically calculates pixel_size_ky and pixel_size_kx
   - If pixel_size is specified in config, it takes precedence over calculations
   - Supports both absolute calibration (with --known-spacing) and relative
   - Updates config without reloading data for efficient workflow

3. **Combined Interactive Mode**:
   - Both modes can be used together for complete interactive setup
   - Center picking happens first, then k-space calibration
   - All calibration results are saved to config and used immediately

Interactive Calibration Workflow:
- Use --interactive-center to precisely set beam center by eye
- Use --interactive-kspace to calibrate against known Bragg reflections
- Provide --known-spacing X.XX (in Angstroms) for absolute k-space units
- Without known spacing, relative calibration normalizes selected radius to 1.0
- Results update the config and are used immediately without data reload

Example with known silicon (220) reflection (d = 1.92 Å):
    python script.py --data si.h5 --labels clusters.npy \\
                     --interactive-kspace --known-spacing 1.92 \\
                     --scan-shape 256 256 --out results

Example polar_config.yaml with direct pixel size specification:
    calibration:
      pixel_size: 0.1         # Direct specification takes precedence
      pixel_size_units: "nm"  # Units: "nm" or "angstrom"
      convergence_angle: 20.0 # mrad (logged for calculations)

PERFORMANCE AND OPTIMIZATION
=============================

The implementation includes several performance optimizations:

1. Vectorized Operations: CoM computation uses vectorized numpy/cupy operations
   - Processes entire chunks at once rather than frame-by-frame
   - Precomputed gate lookup tables for adaptive mode efficiency
   - Significant speedup especially for large datasets

2. Memory Efficiency: Streaming data processing with configurable chunk sizes
   - No need to load entire dataset into memory
   - Cached HDF5 file handles reduce I/O overhead
   - Support for both CPU and GPU acceleration via CuPy

3. Early Center Detection: Beam center detected once and used consistently
   - Eliminates disagreement between preprocessor and DPC centering
   - Uses small data sample for fast auto-detection

VALIDATION AND QUALITY ASSURANCE
================================

The script includes several built-in validation mechanisms:

1. Gate Coverage Analysis: Reports the fraction of detector covered by each gate
   - Healthy gates typically cover 10-40% of the detector
   - 100% coverage indicates poor template discrimination

2. Auto-centering Validation: Uses smoothed mean pattern to find beam center
   - Reports detected center coordinates
   - Applied consistently across preprocessing and analysis

3. Confidence Analysis: Non-saturated confidence maps indicate good cluster separation
   - Mean confidence ~1.0 suggests over-fitting or poor templates
   - Well-separated clusters should show confidence ~0.3-0.8
   - Optional diagonal whitening reduces central beam dominance

4. Baseline Correction: Ensures DPC fields are properly centered
   - Median baseline correction removes global systematic offsets
   - Row/column baseline correction suppresses scan drift and residual tilt
   - ROI-based correction uses reference vacuum region

5. Template Stability: Split-half validation checks template reproducibility
   - Randomly splits data and compares template similarity using SSIM
   - Warns if SSIM < 0.5, indicating potential overfitting or noise
   - Enabled via validation.enable_split_half config option

6. Physics Consistency: DPC fields should show expected symmetries
   - Field magnitude should be low in uniform regions
   - Domain walls should show enhanced field gradients

REFERENCES
==========

This implementation draws on several key concepts:

1. 4D-STEM and DPC theory:
   - Müller-Caspary et al., Ultramicroscopy 178, 62 (2017)
   - Lazić et al., Ultramicroscopy 160, 265 (2016)

2. Virtual detector design:
   - Ophus, Microsc. Microanal. 25, 563 (2019)
   - Close et al., Ultramicroscopy 159, 124 (2015)

3. Machine learning for diffraction analysis:
   - Kalinin et al., npj Comput. Mater. 7, 78 (2021)
   - Spurgeon et al., npj Comput. Mater. 7, 200 (2021)

AUTHORS AND ACKNOWLEDGEMENTS
============================

Implementation: Claude AI (Anthropic)
Scientific guidance: User requirements and domain expertise
Framework: Built on numpy, scipy, matplotlib, scikit-image

For questions or issues, please refer to the project documentation.
"""