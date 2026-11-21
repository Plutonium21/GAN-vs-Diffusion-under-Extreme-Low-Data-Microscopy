#!/usr/bin/env python3
"""
Build an in-domain test set from BBBC038 fluorescence images that NO model saw in training.

"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_bbbc import is_fluorescence, normalise, pick_crops  # noqa: E402
from prepare_data import save_png  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bbbc038", required=True)
    ap.add_argument("--data-root", required=True, help="prepared data (contains subsets/seed0/n100/frames.txt)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--stride", type=int, default=64)
    ap.add_argument("--max-patches", type=int, default=8, help="cap per image (keeps the set balanced)")
    ap.add_argument("--max-total", type=int, default=2000)
    args = ap.parse_args()

    used = set()
    for f in Path(args.data_root).expanduser().glob("subsets/seed*/n*/frames.txt"):
        used |= {l.strip() for l in f.read_text().splitlines() if l.strip()}
    if not used:
        sys.exit("No frames.txt found - is --data-root the prepared bbbc_128 folder?")

    root = Path(args.bbbc038).expanduser()
    out = Path(args.out).expanduser()
    patches, n_img = [], 0
    for img_dir in sorted(root.rglob("images")):
        pngs = sorted(img_dir.glob("*.png")) if img_dir.is_dir() else []
        if not pngs or img_dir.parent.name in used:
            continue
        rgb = np.array(Image.open(pngs[0]))
        if min(rgb.shape[:2]) < args.crop or not is_fluorescence(rgb):
            continue
        g = normalise(rgb if rgb.ndim == 2 else rgb[..., :3].mean(-1))
        n_img += 1
        for y, x, c in pick_crops(g, args.crop, args.stride, args.max_patches):
            patches.append((f"{img_dir.parent.name[:16]}_y{y:04d}_x{x:04d}.png", c, n_img))
    if not patches:
        sys.exit("No unseen fluorescence images found.")
    rng = np.random.default_rng(0)
    if len(patches) > args.max_total:
        patches = [patches[i] for i in sorted(rng.choice(len(patches), args.max_total, replace=False))]
    # split by IMAGE (not patch) so the two halves share no source image
    ids = sorted({p[2] for p in patches})
    half_a = set(rng.permutation(ids)[: len(ids) // 2].tolist())
    for name, c, i in patches:
        save_png(c, out / "images" / name)
        save_png(c, out / ("half_a" if i in half_a else "half_b") / name)
    print(f"Unseen BBBC038 fluorescence images: {n_img} | test patches: {len(patches)} "
          f"(half A {sum(p[2] in half_a for p in patches)}, half B {sum(p[2] not in half_a for p in patches)})")


if __name__ == "__main__":
    main()
