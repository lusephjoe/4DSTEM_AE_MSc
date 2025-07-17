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
    data = torch.load(args.input, map_location="cpu")  # shape [N, signal_dim]
    loader = torch.utils.data.DataLoader(data,
                                         batch_size=args.batch_size,
                                         shuffle=False,
                                         pin_memory=True)

    example = data[:1].to(device)          # <‑‑ one real sample, preserves dims
    show(encoder, example_input=example)  
    
    latents = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        z = encoder(batch).cpu()
        latents.append(z)

    latents = torch.cat(latents, dim=0)  # [N, latent_dim]

    # ---------- optional PCA ---------- #
    if args.pca > 0 and args.pca < latents.size(1):
        print(f"Running PCA → {args.pca} D …")
        pca = PCA(args.pca, svd_solver="randomized")
        latents = torch.tensor(pca.fit_transform(latents.numpy()))

    # ---------- save ---------- #
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(latents, args.output)
    print(f"Wrote {latents.shape}  →  {args.output}")

if __name__ == "__main__":
    main()
