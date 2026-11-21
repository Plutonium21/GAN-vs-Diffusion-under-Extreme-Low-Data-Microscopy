#!/usr/bin/env python3
"""
Prepare Cell Tracking Challenge (CTC) microscopy data for the low-data benchmark.

"""
import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

try:
    import tifffile
except ImportError:
    sys.exit("Please `pip install tifffile` first.")


SEQ_RE = re.compile(r"^\d\d$")


# ----------------------------------------------------------------------------- helpers
def find_sequences(src_dirs):
    """Return {label: (src_dir, seq_name, [tif paths])} for every CTC sequence found."""
    seqs = {}
    for src in src_dirs:
        src = Path(src).expanduser()
        if not src.is_dir():
            sys.exit(f"--src folder not found: {src}")
        # The zip may extract into src/<DatasetName>/01 or directly src/01. Search 2 levels.
        candidates = [p for p in src.rglob("*") if p.is_dir() and SEQ_RE.match(p.name)]
        for seq_dir in sorted(candidates):
            tifs = sorted(seq_dir.glob("t*.tif"))
            if not tifs:
                continue
            label = f"{src.name}/{seq_dir.name}"
            i = 2
            while label in seqs:                     # two --src folders with the same name
                label = f"{src.name}#{i}/{seq_dir.name}"; i += 1
            seqs[label] = (seq_dir.parent, seq_dir.name, tifs)
    if not seqs:
        sys.exit("No CTC sequences found (expected folders like 01/ with t000.tif ...).")
    return seqs


def read_tif(path):
    img = tifffile.imread(str(path)).astype(np.float32)
    if img.ndim == 3:            # safety: take first channel / plane
        img = img[0] if img.shape[0] < img.shape[-1] else img[..., 0]
    return img


def resize(arr, scale, nearest=False):
    if scale == 1.0:
        return arr
    h, w = arr.shape
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    if nearest:
        return np.array(Image.fromarray(arr.astype(np.int32)).resize(size, Image.NEAREST)).astype(arr.dtype)
    resample = Image.BOX if scale < 1 else Image.BICUBIC        # BOX = area averaging when shrinking
    return np.array(Image.fromarray(arr.astype(np.float32)).resize(size, resample))


def crop_positions(length, crop, stride):
    """Start positions so that crops cover the whole axis (last crop flush with the border)."""
    if length < crop:
        raise ValueError(f"Frame side {length}px is smaller than crop {crop}px; lower --crop or raise --scale.")
    pos = list(range(0, length - crop + 1, stride))
    if pos[-1] != length - crop:
        pos.append(length - crop)
    return pos


def iter_crops(arr, crop, stride):
    h, w = arr.shape[:2]
    for y in crop_positions(h, crop, stride):
        for x in crop_positions(w, crop, stride):
            yield y, x, arr[y:y + crop, x:x + crop]


def save_png(arr, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path, compress_level=1)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+", required=True, help="Extracted CTC dataset folder(s).")
    ap.add_argument("--out", required=True, help="Output folder.")
    ap.add_argument("--heldout", default=None,
                    help='Held-out sequence label "<srcfolder>/<seq>" (default: sequence 02 of the first --src).')
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 10, 25, 50, 100, 250],
                    help="Numbers of training FRAMES per low-data regime.")
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2], help="Random draws per regime.")
    ap.add_argument("--scale", type=float, default=0.5,
                    help="Rescale factor applied to frames before cropping (0.5: 1024px GOWT1 frame -> 512px).")
    ap.add_argument("--crop", type=int, default=256, help="Patch size (training resolution).")
    ap.add_argument("--stride", type=int, default=64, help="Stride between overlapping training patches.")
    ap.add_argument("--heldout-stride", type=int, default=128, help="Stride for held-out patches.")
    ap.add_argument("--pclip", nargs=2, type=float, default=[0.5, 99.8],
                    help="Global low/high percentiles mapped to 0/255.")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    seqs = find_sequences(args.src)
    print("Found sequences:")
    for k, (_, _, tifs) in seqs.items():
        print(f"  {k:40s} {len(tifs):4d} frames")

    held_label = args.heldout or f"{Path(args.src[0]).expanduser().name}/02"
    if held_label not in seqs:
        sys.exit(f"Held-out sequence '{held_label}' not found. Choose one of: {list(seqs)}")
    pool_labels = [k for k in seqs if k != held_label]
    print(f"Held-out: {held_label} | Training pool: {pool_labels}")

    # ---- global percentile window, estimated on a pixel sample of pool frames
    rng = np.random.default_rng(1234)
    samples = []
    for k in pool_labels:
        for p in seqs[k][2][:: max(1, len(seqs[k][2]) // 20)]:
            img = read_tif(p).ravel()
            samples.append(rng.choice(img, size=min(200_000, img.size), replace=False))
    samples = np.concatenate(samples)
    lo, hi = np.percentile(samples, args.pclip)
    print(f"Intensity window: [{lo:.1f}, {hi:.1f}] (percentiles {args.pclip})")

    def to_uint8(img):
        img = resize(img, args.scale)
        return (np.clip((img - lo) / (hi - lo), 0, 1) * 255 + 0.5).astype(np.uint8)

    # ---- normalise & store all frames once
    frame_index = {}                       # label -> list of png paths
    for k, (_, seq_name, tifs) in seqs.items():
        frame_index[k] = []
        for p in tifs:
            dst = out / "frames" / k.replace("/", "__") / (p.stem + ".png")
            if not dst.exists():
                save_png(to_uint8(read_tif(p)), dst)
            frame_index[k].append(dst)
    pool = [p for k in pool_labels for p in frame_index[k]]
    print(f"Training pool: {len(pool)} frames")

    # ---- training subsets
    stats = {}
    for seed in args.seeds:
        perm = np.random.default_rng(seed).permutation(len(pool))
        for n in sorted(args.sizes):
            if n > len(pool):
                print(f"  ! seed {seed}: n={n} > pool size {len(pool)} - skipped "
                      f"(add the CTC challenge set to --src to get more frames).")
                continue
            chosen = [pool[i] for i in perm[:n]]            # nested: prefix of the same permutation
            d = out / "subsets" / f"seed{seed}" / f"n{n}"
            (d / "images").mkdir(parents=True, exist_ok=True)
            count = 0
            for fp in chosen:
                arr = np.array(Image.open(fp))
                tag = f"{fp.parent.name}__{fp.stem}"
                for y, x, c in iter_crops(arr, args.crop, args.stride):
                    save_png(c, d / "images" / f"{tag}_y{y:04d}_x{x:04d}.png")
                    count += 1
            (d / "frames.txt").write_text("\n".join(str(p.relative_to(out)) for p in chosen) + "\n")
            stats[f"seed{seed}/n{n}"] = {"frames": n, "patches": count}
            print(f"  seed {seed} n={n:4d}: {count:6d} patches")

    # ---- held-out real set (+ masks if present)
    src_dir, seq_name, tifs = seqs[held_label]
    mask_dirs = {t: src_dir / f"{seq_name}_{t}" / "SEG" for t in ("ST", "GT")}
    n_held = 0
    n_masks = {"ST": 0, "GT": 0}
    for fp, tif in zip(frame_index[held_label], tifs):
        arr = np.array(Image.open(fp))
        frame_id = tif.stem[1:]                             # t012 -> 012
        masks = {}
        for t, md in mask_dirs.items():
            mp = md / f"man_seg{frame_id}.tif"
            if mp.exists():
                masks[t] = resize(tifffile.imread(str(mp)).astype(np.int32), args.scale, nearest=True)
        for y, x, c in iter_crops(arr, args.crop, args.heldout_stride):
            name = f"{fp.stem}_y{y:04d}_x{x:04d}.png"
            save_png(c, out / "heldout" / "images" / name)
            n_held += 1
            for t, m in masks.items():
                save_png(m[y:y + args.crop, x:x + args.crop].astype(np.uint16), out / "heldout" / f"masks_{t}" / name)
                n_masks[t] += 1
    print(f"Held-out: {n_held} real patches, mask patches: {n_masks}")

    manifest = dict(vars(args), out=str(out), intensity_window=[float(lo), float(hi)],
                    sequences={k: len(v[2]) for k, v in seqs.items()}, heldout_label=held_label,
                    pool_labels=pool_labels, pool_frames=len(pool), subsets=stats,
                    heldout_patches=n_held, heldout_mask_patches=n_masks)
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Done. Manifest: {out / 'manifest.json'}")


if __name__ == "__main__":
    main()
