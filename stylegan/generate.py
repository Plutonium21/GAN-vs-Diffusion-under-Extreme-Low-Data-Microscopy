#!/usr/bin/env python3
"""
Generate grayscale microscopy patches from a StyleGAN2-ADA snapshot (network-snapshot-*.pkl).

Truncation psi = 1 (no truncation), as used for FID/KID in the ADA paper, so the full
learned distribution (and its diversity) is evaluated.

    python stylegan/generate.py --network runs/.../network-snapshot-001000.pkl \
        --n 2000 --outdir samples/stylegan_scratch_n25_seed0
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "third_party" / "stylegan3"))
import dnnlib  # noqa: E402
import legacy  # noqa: E402


def load_generator(pkl, device):
    with dnnlib.util.open_url(str(pkl)) as f:
        return legacy.load_network_pkl(f)["G_ema"].to(device).eval().requires_grad_(False)


@torch.no_grad()
def generate(G, n, outdir, batch=32, seed=123, psi=1.0, device="cuda", start=0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    gen = torch.Generator(device=device).manual_seed(seed + start)
    c = torch.zeros([batch, G.c_dim], device=device)
    done = start
    while done < n:
        b = min(batch, n - done)
        z = torch.randn([b, G.z_dim], generator=gen, device=device)
        img = G(z, c[:b], truncation_psi=psi, noise_mode="const")
        img = (img.float().mean(dim=1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()  # RGB -> gray
        for i in range(b):
            Image.fromarray(img[i]).save(outdir / f"gen_{done + i:06d}.png")
        done += b
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--psi", type=float, default=1.0)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    G = load_generator(args.network, device)
    generate(G, args.n, args.outdir, args.batch, args.seed, args.psi, device)
    print(f"Saved {args.n} images to {args.outdir}")


if __name__ == "__main__":
    main()
