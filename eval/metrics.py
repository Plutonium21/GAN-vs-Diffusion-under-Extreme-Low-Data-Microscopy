#!/usr/bin/env python3

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


class GrayFolder(torch.utils.data.Dataset):
    """Grayscale PNGs -> uint8 3xHxW tensors (gray replicated), as torch-fidelity expects."""

    def __init__(self, folder, limit=None):
        self.files = sorted(Path(folder).expanduser().glob("*.png"))[:limit]
        if not self.files:
            raise FileNotFoundError(f"No PNG files in {folder}")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        x = torch.from_numpy(np.array(Image.open(self.files[i]).convert("L")))
        return x[None].expand(3, -1, -1).contiguous()


def compute(gen_dir, real_dir, prc=False, kid_subsets=100, max_gen=None):
    import torch_fidelity
    gen, real = GrayFolder(gen_dir, max_gen), GrayFolder(real_dir)
    m = torch_fidelity.calculate_metrics(
        input1=gen, input2=real, fid=True, kid=True, prc=prc,
        kid_subset_size=min(1000, len(gen), len(real)), kid_subsets=kid_subsets,
        cuda=torch.cuda.is_available(), batch_size=64, verbose=False, cache=False)
    out = {"fid": float(m["frechet_inception_distance"]),
           "kid": float(m["kernel_inception_distance_mean"]),
           "kid_std": float(m["kernel_inception_distance_std"]),
           "n_gen": len(gen), "n_real": len(real)}
    if prc:
        out.update(precision=float(m["precision"]), recall=float(m["recall"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--prc", action="store_true", help="Also precision/recall.")
    ap.add_argument("--out", default=None, help="Optional JSON file to write.")
    args = ap.parse_args()
    res = compute(args.gen, args.real, args.prc)
    print(json.dumps(res, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
