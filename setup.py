from setuptools import setup, find_packages

setup(
    name="4dstem-ae",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        # "pytorch", your version will vary by device
        "h5py~=3.12",
        "numpy~=1.26",
        "tqdm~=4.66",
        "matplotlib~=3.9",
        "pytorch-lightning~=2.2",
        "hyperspy~=2.3.0",
        "numba~=0.61.2",
        "scikit-learn~=1.7.0",
        "tifffile",
        "umap-learn~=0.5.9",
        "plotly~=6.2.0",
        "scikit-image",
    ],
    extras_require={
        "dev": ["pytest"],
    },
    python_requires=">=3.8",
    author="Your Name",
    description="4D-STEM Autoencoder for MSc project",
    entry_points={
        "console_scripts": [
            "4dstem-train=scripts.train:main",
            "4dstem-preprocess=scripts.preprocess:main",
            "4dstem-convert=scripts.convert_dm4:main",
            "4dstem-embeddings=scripts.generate_embeddings:main",
            "4dstem-visualize=scripts.visualise_scan_latents:main",
            "4dstem-stem-viz=scripts.stem_visualization:main",
            "4dstem-reconstruct=scripts.reconstruct:main",
        ],
    },
)