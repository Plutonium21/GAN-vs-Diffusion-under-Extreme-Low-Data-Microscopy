#!/usr/bin/env python3
"""
Paper A - StyleGAN2-ADA (Karras et al., NeurIPS 2020) on the low-data microscopy subsets.

"""
import argparse
import json
import subprocess
import sys
import zipfile
from io import BytesIO
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SG3 = ROOT / "third_party" / "stylegan3"

# Same-config FFHQ 256x256 source network published with stylegan2-ada-pytorch
# (train.py: resume_specs['ffhq256'], config "paper256").
FFHQ256 = ("https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/"
           "transfer-learning-source-nets/ffhq-res256-mirror-paper256-noaug.pkl")


def build_zip(img_dir: Path, zip_path: Path) -> int:
    """Pack grayscale patches as RGB PNGs in an uncompressed zip (StyleGAN dataset format).

    Gray is replicated to 3 channels so the same data works for scratch and for transfer
    learning from an RGB network. Generated images are converted back to gray at sampling.
    """
    files = sorted(p for p in img_dir.iterdir() if p.suffix.lower() == ".png")
    if not files:
        sys.exit(f"No PNGs in {img_dir}")
    if zip_path.exists():
        return len(files)
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = zip_path.with_suffix(".tmp")
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_STORED) as zf:
        for i, p in enumerate(files):
            buf = BytesIO()
            Image.open(p).convert("L").convert("RGB").save(buf, format="png", compress_level=0)
            zf.writestr(f"{i // 1000:05d}/img{i:08d}.png", buf.getvalue())
        zf.writestr("dataset.json", json.dumps({"labels": None}))
    tmp.rename(zip_path)
    return len(files)


def snapshot_kimg(p: Path) -> int:
    return int(p.stem.split("-")[-1])


def loads_ok(pkl: Path) -> bool:
    """A snapshot written while Colab disconnected can be cut off; check it loads."""
    sys.path.insert(0, str(SG3))
    try:
        import dnnlib, legacy  # noqa: E401
        with dnnlib.util.open_url(str(pkl)) as f:
            legacy.load_network_pkl(f)
        return True
    except Exception as e:  # noqa: BLE001
        print(f"  ! snapshot {pkl.name} is damaged ({e.__class__.__name__}) - ignoring it")
        return False


def progress(run_dir: Path):
    """(kimg already trained, snapshot to continue from) summed over earlier sessions.

    Every Colab session that continues a run writes a new numbered sub-folder whose
    snapshots count from 0 again, so the total is the sum of each sub-folder's last snapshot."""
    done, latest = 0, None
    for sub in sorted(d for d in run_dir.glob("*") if d.is_dir()):
        snaps = sorted(sub.glob("network-snapshot-*.pkl"), key=snapshot_kimg)
        while snaps and not loads_ok(snaps[-1]):
            snaps[-1].rename(snaps[-1].with_suffix(".pkl.damaged"))
            snaps.pop()
        if snaps:
            done += snapshot_kimg(snaps[-1])
            latest = snaps[-1]
    return done, latest


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-root", required=True, help="Output folder of data/prepare_data.py")
    ap.add_argument("--outdir", default=str(ROOT / "runs" / "stylegan2ada"))
    ap.add_argument("--mode", choices=["scratch", "transfer"], default="scratch")
    ap.add_argument("--sizes", nargs="+", type=int, default=None, help="Regimes to run (default: all prepared).")
    ap.add_argument("--seeds", nargs="+", type=int, default=None, help="Seeds to run (default: all prepared).")
    ap.add_argument("--kimg", type=int, default=2000,
                    help="Training length in thousands of real images shown to D. "
                         "Paper: 25000; transfer learning converges much faster (~500-1000).")
    ap.add_argument("--snap-kimg", type=int, default=100, help="Snapshot interval (kimg).")
    ap.add_argument("--gpus", type=int, default=1)
    ap.add_argument("--batch", type=int, default=64, help="Total minibatch (paper: 64).")
    ap.add_argument("--batch-gpu", type=int, default=32,
                    help="Per-GPU batch; lower to 16 or 8 on <=16 GB GPUs (gradient accumulation, same results). "
                         "Must be a multiple of 8 (minibatch-stddev group).")
    ap.add_argument("--gamma", type=float, default=1.0, help="R1 weight (paper, 256x256: 1).")
    ap.add_argument("--freezed", type=int, default=0, help="Freeze-D layers for transfer (paper: small gain with ADA).")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--resume-pkl", default=None,
                    help="Continue an interrupted run from its latest network-snapshot-*.pkl "
                         "(use with a single --sizes/--seeds). Snapshots then go to a new numbered sub-folder.")
    ap.add_argument("--dry-run", action="store_true", help="Print the commands/options without training.")
    args = ap.parse_args()

    if not (SG3 / "train.py").exists():
        sys.exit("Run `python stylegan/setup_stylegan3.py` first.")
    if args.batch_gpu % 8:
        sys.exit("--batch-gpu must be a multiple of 8.")

    subsets_root = Path(args.data_root).expanduser().resolve() / "subsets"
    runs = []
    for sd in sorted(subsets_root.glob("seed*")):
        seed = int(sd.name[4:])
        if args.seeds is not None and seed not in args.seeds:
            continue
        for nd in sorted(sd.glob("n*"), key=lambda p: int(p.name[1:])):
            n = int(nd.name[1:])
            if args.sizes is None or n in args.sizes:
                runs.append((seed, n, nd / "images"))
    if not runs:
        sys.exit(f"No matching subsets under {subsets_root}")
    if args.resume_pkl and len(runs) != 1:
        sys.exit("--resume-pkl needs exactly one run: pass a single value to --sizes and --seeds.")

    for seed, n, img_dir in runs:
        zip_path = Path(args.data_root).expanduser().resolve() / "stylegan_zips" / f"seed{seed}_n{n}.zip"
        n_imgs = build_zip(img_dir, zip_path)
        desc = f"{args.mode}-n{n}-seed{seed}"
        outdir = Path(args.outdir).expanduser().resolve() / desc
        done, latest = progress(outdir) if outdir.exists() and not args.resume_pkl else (0, None)
        if done >= args.kimg:
            print(f"\n=== {desc}: already finished ({done} kimg) - skipping ===")
            continue
        kimg_left = args.kimg - done
        cmd = [
            sys.executable, str(SG3 / "train.py"),
            f"--outdir={outdir}", f"--data={zip_path}", "--cfg=stylegan2",
            f"--gpus={args.gpus}", f"--batch={args.batch}", f"--batch-gpu={args.batch_gpu}",
            f"--gamma={args.gamma}", "--cbase=16384", "--glr=0.0025", "--dlr=0.0025",
            "--mbstd-group=8", "--map-depth=8", "--mirror=1", "--aug=ada", "--target=0.6",
            f"--kimg={kimg_left}", "--tick=4", f"--snap={max(1, args.snap_kimg // 4)}",
            "--metrics=none", f"--seed={seed}", f"--workers={args.workers}", f"--desc={desc}",
        ]
        if args.resume_pkl:
            cmd += [f"--resume={Path(args.resume_pkl).expanduser()}", f"--freezed={args.freezed}"]
        elif latest is not None:
            print(f"\n=== {desc}: continuing from {latest.parent.name}/{latest.name} "
                  f"({done} of {args.kimg} kimg done) ===")
            cmd += [f"--resume={latest}", f"--freezed={args.freezed}"]
        elif args.mode == "transfer":
            cmd += [f"--resume={FFHQ256}", f"--freezed={args.freezed}"]
        if args.dry_run:
            cmd.append("--dry-run")
        print(f"\n=== StyleGAN2-ADA | {desc} | {n_imgs} patches from {n} frames ===")
        print(" ".join(cmd))
        subprocess.run(cmd, check=True, cwd=str(SG3))


if __name__ == "__main__":
    main()
