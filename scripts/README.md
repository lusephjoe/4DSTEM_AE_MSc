# 4D-STEM Autoencoder Scripts

Complete pipeline for training, analysis, and visualization of 4D-STEM autoencoders.

## Directory Structure

```
scripts/
├── training/           # Model training and configuration
│   ├── train.py                 # Main training script
│   ├── dataset.py               # HDF5 dataset implementation
│   ├── lightning_model.py       # PyTorch Lightning model wrapper
│   └── README.md               # Training documentation
├── preprocessing/      # Data conversion and preprocessing
│   ├── convert_dm4_to_hspy.py  # dm4 to hspy conversion
│   ├── preprocess.py           # Traditional preprocessing
│   └── README.md               # Preprocessing documentation
├── visualization/      # Model evaluation and embedding generation
│   ├── evaluate_autoencoder.py # Model evaluation
│   ├── generate_embeddings.py  # Extract latent embeddings
│   ├── reconstruct.py          # Generate reconstructions
│   └── README.md               # Visualization documentation
├── analysis/          # Advanced analysis and clustering
│   ├── dimension_reduction.py   # UMAP/PCA dimension reduction
│   ├── clustering_analysis.py   # Comprehensive clustering analysis
│   ├── optimize_clustering.py   # Parameter optimization
│   └── umap_latent_visualization.py # Legacy UMAP visualization
└── README.md          # This file
```

## Quick Start

### Complete Analysis Pipeline

```bash
# 1. Data preprocessing
cd preprocessing/
python convert_dm4.py --input ../data/sample.dm4 --output ../data/sample.h5

# 2. Model training
cd ../training/
python train.py --data ../data/sample.h5 --output_dir ../outputs --epochs 50

# 3. Generate embeddings
cd ../visualization/
python generate_embeddings.py --checkpoint ../outputs/ae_final.ckpt \
    --data ../data/sample.h5 --output ../embeddings/latent_embeddings.npz

# 4. Dimension reduction analysis
cd ../analysis/
python dimension_reduction.py --embeddings ../embeddings/latent_embeddings.npz \
    --method umap --optimize_parameters --output_dir ../results/umap_analysis

# 5. Clustering analysis
python clustering_analysis.py --latent_embeddings ../embeddings/latent_embeddings.npz \
    --method all --compare_methods --standardize --output_dir ../results/clustering
```

## Key Features

### Training
- Professional PyTorch Lightning architecture with comprehensive regularization
- HDF5 lazy loading for memory-efficient processing
- Mixed precision training and model compilation support
- Extensive configuration options and parameter validation

### Analysis
- Dimension reduction with UMAP/PCA and parameter optimization
- Multiple clustering algorithms (HDBSCAN, K-means, DBSCAN, Gaussian Mixture)
- High-dimensional and reduced embedding support
- Comprehensive evaluation metrics and statistical analysis

### Visualization
- Publication-ready figures with spatial mapping
- Interactive reconstructions and embedding visualizations  
- Automated parameter optimization plots
- Clean, non-overlapping legends and annotations

## Usage Examples

### Training with Regularization
```bash
python training/train.py --data data.h5 --output_dir outputs --epochs 100 \
    --batch 64 --latent 256 --lambda_act 1e-4 --lambda_l2 1e-6 --precision 16
```

### Clustering Parameter Optimization
```bash
python analysis/optimize_clustering.py --latent_embeddings embeddings.npz \
    --standardize --max_clusters 20 --subsample 10000 --output_dir results
```

### Method Comparison
```bash
python analysis/clustering_analysis.py --latent_embeddings embeddings.npz \
    --method all --compare_methods --standardize --output_dir comparison
```

## Dependencies

### Required
- PyTorch >= 1.13 
- PyTorch Lightning >= 1.8
- NumPy, h5py, scikit-learn
- matplotlib, seaborn
- UMAP-learn, HDBSCAN

### Installation
```bash
pip install -r "requirements.txt"
# install your relevant GPU-accelerated Pytorch version

## Documentation

Each directory contains detailed documentation:
- **training/README.md**: Training configuration and regularization
- **preprocessing/README.md**: Data conversion workflows  
- **visualization/README.md**: Model evaluation tools
- **analysis/README.md**: Clustering and dimension reduction analysis

For script-specific help: `python script_name.py --help`