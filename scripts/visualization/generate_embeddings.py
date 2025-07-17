#!/usr/bin/env python
# scripts/generate_embeddings.py
"""
Batch-extract latent vectors from a trained autoencoder encoder.

Usage
-----
python scripts/generate_embeddings.py \
    --input data/ae_train.pt \
    --checkpoint checkpoints/ae_best.pt \
    --output embeddings/ae_train_latents.pt \
    --batch_size 1024 \
    --device cuda \
    --pca 50
"""
from pathlib import Path
import argparse, torch
from sklearn.decomposition import PCA  # CPU PCA is fine for ≤10⁴ samples
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent.parent))
from models.summary import show

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, help=".pt tensor from convert_dm4")
    p.add_argument("--checkpoint", required=True, help="trained AE or encoder .pt")
    p.add_argument("--output", required=True, help="file to write latents to (.pt or .npy)")
    p.add_argument("--batch_size", type=int, default=2048)
    p.add_argument("--device", default="cpu")
    p.add_argument("--pca", type=int, default=0,
                   help="If >0: reduce dimensionality with PCA to this many comps")
    return p.parse_args()

@torch.no_grad()
def main():
    args = parse_args()
    device = torch.device(args.device)

    # ---------- load model ---------- #
    ckpt = torch.load(args.checkpoint, map_location=device)
    if "model_state_dict" in ckpt:
        state = ckpt["model_state_dict"]
    else:
        state = ckpt
    from models.autoencoder import Encoder  # adjust if path differs
    encoder = Encoder()
    encoder.load_state_dict(state, strict=False)
    encoder.eval().to(device)

    # ---------- load data ---------- #
    print("Loading data...")
    data = torch.load(args.input, map_location="cpu")  # shape [N, signal_dim]
    print(f"Loaded {data.shape[0]} samples of shape {data.shape[1:]}")
    
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True)

    example = data[:1].to(device)          # <‑‑ one real sample, preserves dims
    show(encoder, example_input=example)  
    
    print("Generating embeddings...")
    latents = []
    for batch in tqdm(loader, desc="Processing batches", unit="batch"):
        batch = batch.to(device, non_blocking=True)
        z = encoder(batch).cpu()
        latents.append(z)

    print("Concatenating results...")
    latents = torch.cat(latents, dim=0)  # [N, latent_dim]
    print(f"Generated embeddings shape: {latents.shape}")

    # ---------- optional PCA ---------- #
    if args.pca > 0 and args.pca < latents.size(1):
        print(f"Running PCA to reduce from {latents.size(1)} to {args.pca} dimensions...")
        pca = PCA(args.pca, svd_solver="randomized")
        latents_pca = pca.fit_transform(latents.numpy())
        latents = torch.tensor(latents_pca)
        print(f"PCA completed. Explained variance ratio: {pca.explained_variance_ratio_[:5]}")  # Show first 5 components

    # ---------- save ---------- #
    print("Saving embeddings...")
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents, args.output)
    print(f"✓ Saved embeddings {latents.shape} → {args.output}")
    
    # Show some statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {latents.mean():.4f}")
    print(f"  Std:  {latents.std():.4f}")
    print(f"  Min:  {latents.min():.4f}")
    print(f"  Max:  {latents.max():.4f}")

if __name__ == "__main__":
    main()
