# Visualization Scripts

This directory contains scripts for visualizing, evaluating, and analyzing trained 4D-STEM autoencoders.

## Scripts

### `evaluate_autoencoder.py`
**Comprehensive model evaluation and analysis**
- Loads trained model checkpoints
- Computes detailed reconstruction metrics
- Generates comparison visualizations
- Statistical analysis of performance

**Usage:**
```bash
python evaluate_autoencoder.py --model_path outputs/ae.ckpt --data_path data.pt
```

**Key Features:**
- PSNR, SSIM, MSE metrics with confidence intervals
- Reconstruction quality histograms
- Side-by-side comparison plots
- Statistical significance testing
- Performance summary reports

### `generate_embeddings.py`
**Extract latent embeddings from trained models**
- Generate embeddings for entire datasets
- Save embeddings for further analysis
- Dimensionality reduction visualization
- Clustering analysis

**Usage:**
```bash
python generate_embeddings.py --model_path outputs/ae.ckpt --data_path data.pt --output embeddings.npy
```

**Key Features:**
- Batch processing for large datasets
- Memory-efficient embedding generation
- PCA and t-SNE visualization
- Embedding quality metrics

### `reconstruct.py`
**Generate reconstructions from trained models**
- Reconstruct diffraction patterns
- Compare original vs reconstructed patterns
- Generate reconstruction galleries
- Export high-quality figures

**Usage:**
```bash
python reconstruct.py --model_path outputs/ae.ckpt --data_path data.pt --output_dir reconstructions/
```

**Key Features:**
- Batch reconstruction processing
- High-quality figure generation
- Customizable visualization parameters
- Export options (PNG, PDF, SVG)

### `stem_visualization.py`
**STEM-specific visualization utilities**
- Specialized plotting functions for 4D-STEM data
- Colormap optimization for diffraction patterns
- Scale bar and annotation tools
- Publication-ready figures

**Key Features:**
- STEMVisualizer class for consistent plotting
- Optimized colormaps for diffraction data
- Automatic scaling and normalization
- Professional figure formatting

### `visualise_scan_latents.py`
**Visualize latent space across scan positions**
- Map latent representations to scan positions
- Spatial analysis of latent features
- Clustering visualization
- Feature importance maps

**Usage:**
```bash
python visualise_scan_latents.py --model_path outputs/ae.ckpt --data_path data.pt
```

**Key Features:**
- Spatial latent maps
- Feature clustering analysis
- Interactive visualizations
- Latent space navigation

## Workflow Examples

### Model Evaluation Pipeline
```bash
# 1. Evaluate model performance
python evaluate_autoencoder.py --model_path outputs/ae.ckpt --data_path data.pt

# 2. Generate embeddings
python generate_embeddings.py --model_path outputs/ae.ckpt --data_path data.pt --output embeddings.npy

# 3. Create reconstructions
python reconstruct.py --model_path outputs/ae.ckpt --data_path data.pt --output_dir reconstructions/

# 4. Visualize latent space
python visualise_scan_latents.py --model_path outputs/ae.ckpt --data_path data.pt
```

### Publication Figure Generation
```bash
# High-quality reconstruction comparisons
python reconstruct.py --model_path outputs/ae.ckpt --data_path data.pt \
    --output_dir figures/ --dpi 300 --format pdf

# Evaluation metrics with error bars
python evaluate_autoencoder.py --model_path outputs/ae.ckpt --data_path data.pt \
    --save_figures --output_dir figures/
```

## Output Types

### Evaluation Outputs
- **Metrics summary**: CSV files with PSNR, SSIM, MSE statistics
- **Comparison plots**: Side-by-side original vs reconstructed
- **Histograms**: Distribution of reconstruction quality
- **Statistical reports**: Confidence intervals and significance tests

### Embedding Outputs
- **Embedding arrays**: NumPy arrays with latent representations
- **Visualization plots**: PCA, t-SNE, UMAP projections
- **Clustering results**: Cluster assignments and centroids
- **Quality metrics**: Embedding quality scores

### Reconstruction Outputs
- **Image galleries**: Grid layouts of reconstructions
- **Individual comparisons**: Paired original-reconstruction images
- **Difference maps**: Reconstruction error visualization
- **Quality metrics**: Per-pattern reconstruction scores

### Latent Space Outputs
- **Spatial maps**: Latent features mapped to scan positions
- **Feature importance**: Which latent dimensions are most informative
- **Clustering maps**: Spatial distribution of clusters
- **Navigation tools**: Interactive latent space exploration

## Customization Options

### Figure Styling
```python
# Custom colormap and styling
python visualize_script.py --colormap viridis --figsize 10,8 --dpi 300
```

### Data Subsampling
```python
# Process subset of data for faster visualization
python visualize_script.py --subsample 1000 --random_seed 42
```

### Export Formats
```python
# Multiple export formats
python visualize_script.py --formats png,pdf,svg --dpi 300
```

## Performance Tips

### Memory Management
- Use `--batch_size` to control memory usage
- Process data in chunks for large datasets
- Use `--subsample` for quick visualization tests

### Speed Optimization
- Use GPU acceleration when available
- Reduce `--dpi` for faster preview generation
- Cache embeddings for repeated analysis

### Quality Settings
- Use `--dpi 300` for publication figures
- Use `--format pdf` for vector graphics
- Use `--colormap` optimized for your data

## Integration with Training

### Using with train.py outputs
```bash
python evaluate_autoencoder.py --model_path outputs/ae.ckpt --data_path processed_data.pt
```

### Using with train_hyperspy.py outputs
```bash
# Convert hspy to tensor for visualization (if needed)
python ../preprocessing/preprocess.py --input data.hspy --output vis_data.pt
python evaluate_autoencoder.py --model_path outputs/ae.ckpt --data_path vis_data.pt
```

## Common Analysis Workflows

### Quality Assessment
1. **evaluate_autoencoder.py** - Overall performance metrics
2. **reconstruct.py** - Visual quality inspection
3. **stem_visualization.py** - Specialized STEM analysis

### Latent Space Analysis
1. **generate_embeddings.py** - Extract embeddings
2. **visualise_scan_latents.py** - Spatial analysis
3. **stem_visualization.py** - Publication figures

### Comparative Analysis
1. Train multiple models with different parameters
2. Use evaluation scripts to compare performance
3. Generate comparative visualizations

## Dependencies

- PyTorch
- PyTorch Lightning
- Matplotlib
- Seaborn
- scikit-learn
- NumPy
- SciPy
- HyperSpy (optional, for some visualizations)

## Troubleshooting

### "Model not found" errors
- Check that the model path points to a valid `.ckpt` file
- Ensure the model architecture matches the saved checkpoint

### Memory issues during visualization
- Use `--batch_size` to reduce memory usage
- Use `--subsample` to process smaller datasets
- Close figure windows to free memory

### Poor visualization quality
- Increase `--dpi` for higher resolution
- Use appropriate colormaps for your data type
- Adjust contrast and brightness settings