#!/usr/bin/env python3
"""
Prepare BBBC038 (training) + BBBC039 (validation / testing) for the low-data benchmark.

Downloads (both CC0):
  BBBC038  https://data.broadinstitute.org/bbbc/BBBC038/stage1_train.zip
  BBBC039  https://data.broadinstitute.org/bbbc/BBBC039/images.zip
           https://data.broadinstitute.org/bbbc/BBBC039/masks.zip
           https://data.broadinstitute.org/bbbc/BBBC039/metadata.zip 

"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_data import iter_crops, save_png  # noqa: E402


# ----------------------------------------------------------------------------- helpers
def normalise(img, pclip=(1.0, 99.8)):
    img = img.astype(np.float32)
    lo, hi = np.percentile(img, pclip)
    return (np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1) * 255 + 0.5).astype(np.uint8)


def label_image(arr):
    """Colour-coded mask -> integer label image (touching nuclei have different colours)."""
    if arr.ndim == 3:
        arr = arr[..., :3].astype(np.int64)
        arr = arr[..., 0] * 65536 + arr[..., 1] * 256 + arr[..., 2]
    try:
        from skimage.measure import label
        return label(arr, background=0, connectivity=1).astype(np.uint16)
    except ImportError:                     # fallback: connected components per colour
        from scipy import ndimage
        out, nxt = np.zeros(arr.shape, np.int32), 0
        for v in np.unique(arr):
            if v == 0:
                continue
            lab, k = ndimage.label(arr == v)
            out[lab > 0] = lab[lab > 0] + nxt
            nxt += k
        return out.astype(np.uint16)


def is_fluorescence(rgb, gray_tol=2.0, max_median=0.35):
    """Grayscale (R=G=B) and dark background -> fluorescence. H&E is coloured,
    brightfield is grayscale with a bright background."""
    if rgb.ndim == 2:
        g = rgb.astype(np.float32)
    else:
        c = rgb[..., :3].astype(np.float32)
        if np.abs(c[..., 0] - c[..., 1]).mean() > gray_tol or np.abs(c[..., 1] - c[..., 2]).mean() > gray_tol:
            return False
        g = c.mean(-1)
    g = g / max(g.max(), 1.0)
    return float(np.median(g)) < max_median


def pick_crops(arr, crop, stride, max_patches):
    crops = list(iter_crops(arr, crop, stride))
    if max_patches and len(crops) > max_patches:          # evenly spread, deterministic
        idx = np.linspace(0, len(crops) - 1, max_patches).round().astype(int)
        crops = [crops[i] for i in idx]
    return crops


def find_bbbc039(root):
    root = Path(root).expanduser()
    imgs = {p.stem: p for p in root.rglob("*.tif")}
    masks = {p.stem: p for p in root.rglob("*.png") if "mask" in str(p.parent).lower()}
    splits = {}
    for name in ("training", "validation", "test"):
        f = next(iter(root.rglob(f"{name}.txt")), None)
        if f:
            splits[name] = [Path(l.strip()).stem for l in f.read_text().splitlines() if l.strip()]
    if not imgs:
        sys.exit(f"No .tif images found under {root} (unzip images.zip there).")
    if not splits:
        print("  ! BBBC039 metadata (training/validation/test.txt) not found - using a 50/25/25 split.")
        keys = sorted(imgs)
        a, b = len(keys) // 2, len(keys) * 3 // 4
        splits = {"training": keys[:a], "validation": keys[a:b], "test": keys[b:]}
    return imgs, masks, splits


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bbbc038", required=True, help="Extracted stage1_train folder.")
    ap.add_argument("--bbbc039", default=None, help="Folder with extracted images/, masks/, metadata/.")
    ap.add_argument("--out", required=True)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 25, 100])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    ap.add_argument("--crop", type=int, default=128)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--max-patches", type=int, default=64, help="Cap on training patches per image.")
    ap.add_argument("--heldout-frac", type=float, default=0.15,
                    help="BBBC038 images kept aside as an extra real reference (0 = use all for training).")
    ap.add_argument("--all-types", action="store_true", help="Keep brightfield/H&E images too (not recommended).")
    args = ap.parse_args()
    out = Path(args.out).expanduser()
    rng = np.random.default_rng(2018)

    # ---------------------------------------------------------------- BBBC038
    root = Path(args.bbbc038).expanduser()
    entries = []
    for img_dir in sorted(root.rglob("images")):
        if not img_dir.is_dir():
            continue
        pngs = sorted(img_dir.glob("*.png"))
        if not pngs:
            continue
        rgb = np.array(Image.open(pngs[0]))
        if min(rgb.shape[:2]) < args.crop:
            continue
        if not args.all_types and not is_fluorescence(rgb):
            continue
        entries.append((img_dir.parent.name, pngs[0], img_dir.parent / "masks"))
    if not entries:
        sys.exit(f"No usable images under {root} (expected stage1_train/<id>/images/<id>.png).")
    print(f"BBBC038: {len(entries)} fluorescence images kept")

    order = rng.permutation(len(entries))
    n_held = int(round(len(entries) * args.heldout_frac))
    held = [entries[i] for i in order[:n_held]]
    pool = [entries[i] for i in order[n_held:]]
    print(f"  held-out (FID/KID reference): {len(held)} images | training pool: {len(pool)} images")

    def gray_of(png):
        a = np.array(Image.open(png))
        return normalise(a if a.ndim == 2 else a[..., :3].mean(-1))

    def mask_of(mdir, shape):
        lab = np.zeros(shape, np.uint16)
        for k, mp in enumerate(sorted(Path(mdir).glob("*.png")), 1):
            lab[np.array(Image.open(mp)) > 0] = k
        return lab

    n_held_p = 0
    for iid, png, mdir in held:
        g = gray_of(png)
        m = mask_of(mdir, g.shape) if mdir.is_dir() else None
        for y, x, c in pick_crops(g, args.crop, args.crop // 2, args.max_patches):
            name = f"{iid[:16]}_y{y:04d}_x{x:04d}.png"
            save_png(c, out / "heldout" / "images" / name)
            if m is not None:
                save_png(m[y:y + args.crop, x:x + args.crop], out / "heldout" / "masks" / name)
            n_held_p += 1
    print(f"  held-out patches: {n_held_p}")

    stats = {}
    for seed in args.seeds:
        perm = np.random.default_rng(seed).permutation(len(pool))
        for n in sorted(args.sizes):
            if n > len(pool):
                print(f"  ! n={n} > pool {len(pool)} - skipped")
                continue
            d = out / "subsets" / f"seed{seed}" / f"n{n}"
            chosen = [pool[i] for i in perm[:n]]
            count = 0
            for iid, png, _ in chosen:
                for y, x, c in pick_crops(gray_of(png), args.crop, args.stride, args.max_patches):
                    save_png(c, d / "images" / f"{iid[:16]}_y{y:04d}_x{x:04d}.png")
                    count += 1
            (d / "frames.txt").write_text("\n".join(i for i, _, _ in chosen) + "\n")
            stats[f"seed{seed}/n{n}"] = {"images": n, "patches": count}
            print(f"  seed {seed} n={n:4d}: {count:6d} patches")

    # ---------------------------------------------------------------- BBBC039
    b39 = {}
    if args.bbbc039:
        import tifffile
        imgs, masks, splits = find_bbbc039(args.bbbc039)
        for split, keys in splits.items():
            cnt = 0
            for k in keys:
                if k not in imgs:
                    continue
                g = normalise(tifffile.imread(str(imgs[k])))
                save_png(g, out / "bbbc039" / split / "full_images" / f"{k}.png")
                m = label_image(np.array(Image.open(masks[k]))) if k in masks else None
                if m is not None:
                    save_png(m, out / "bbbc039" / split / "full_masks" / f"{k}.png")
                for y, x, c in pick_crops(g, args.crop, args.crop, None):
                    name = f"{k[:40]}_y{y:04d}_x{x:04d}.png"
                    save_png(c, out / "bbbc039" / split / "images" / name)
                    if m is not None:
                        save_png(m[y:y + args.crop, x:x + args.crop], out / "bbbc039" / split / "masks" / name)
                    cnt += 1
            b39[split] = {"images": len([k for k in keys if k in imgs]), "patches": cnt}
            print(f"BBBC039 {split:10s}: {b39[split]['images']} images, {cnt} patches")

    manifest = dict(vars(args), out=str(out), bbbc038_fluorescence=len(entries), heldout_images=len(held),
                    heldout_patches=n_held_p, pool_images=len(pool), subsets=stats, bbbc039=b39)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
