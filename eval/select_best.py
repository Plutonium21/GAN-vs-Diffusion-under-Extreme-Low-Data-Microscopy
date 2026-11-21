#!/usr/bin/env python3

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import compute  # noqa: E402


def snapshots(kind, run):
    pat = "network-snapshot-*.pkl" if kind == "stylegan" else "ema_*.pt"
    files = sorted(Path(run).expanduser().rglob(pat))
    if kind == "stylegan":
        files = [f for f in files if not f.name.endswith("000000.pkl")]   # skip the untrained init
    return files


def make_sampler(kind, device, args):
    if kind == "stylegan":
        sys.path.insert(0, str(ROOT / "stylegan"))
        from generate import generate, load_generator

        def sample(snap, n, out, final, start=0):
            generate(load_generator(snap, device), n, out, batch=args.batch, seed=args.seed, device=device, start=start)
    else:
        sys.path.insert(0, str(ROOT / "ddpm"))
        from sample import generate, load_model

        def sample(snap, n, out, final, start=0):
            steps = 1000 if final else args.screen_steps
            generate(load_model(snap, device), n, out, batch=args.batch, steps=steps, seed=args.seed, device=device,
                     start=start)
    return sample


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kind", choices=["stylegan", "ddpm"], required=True)
    ap.add_argument("--run", required=True, help="Run folder (searched recursively for snapshots).")
    ap.add_argument("--real", required=True,
                    help="Real patches used to PICK the best snapshot (validation set, e.g. bbbc039/validation/images).")
    ap.add_argument("--real-final", default=None,
                    help="Real patches for the FINAL scores (test set, e.g. bbbc039/test/images). Default: --real.")
    ap.add_argument("--n-screen", type=int, default=1000, help="Images per snapshot while screening.")
    ap.add_argument("--n-final", type=int, default=5000, help="Images for the final evaluation.")
    ap.add_argument("--screen-steps", type=int, default=250, help="DDPM sampler steps while screening.")
    ap.add_argument("--every", type=int, default=1, help="Screen only every k-th snapshot (saves time).")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--keep-screen-images", action="store_true")
    ap.add_argument("--force", action="store_true", help="Re-evaluate even if final_metrics.json exists.")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    run = Path(args.run).expanduser()
    if (run / "final_metrics.json").exists() and not args.force:
        print(f"{run.name}: already evaluated (final_metrics.json exists) - skipping.")
        return
    snaps = snapshots(args.kind, run)
    if not snaps:
        sys.exit(f"No snapshots found in {run}")
    snaps = snaps[::-1][::args.every][::-1]            # always keep the latest
    sample = make_sampler(args.kind, device, args)

    res_file = run / "snapshot_kid.json"
    results = json.loads(res_file.read_text()) if res_file.exists() else {}
    for s in snaps:
        if str(s) in results:
            continue
        tmp = run / "_screen" / s.stem
        sample(s, args.n_screen, tmp, final=False)
        results[str(s)] = compute(tmp, args.real)
        print(f"{s.name}: KID={results[str(s)]['kid']:.5f}  FID={results[str(s)]['fid']:.2f}", flush=True)
        res_file.write_text(json.dumps(results, indent=2))
        if not args.keep_screen_images:
            shutil.rmtree(tmp)

    best = min(results, key=lambda k: results[k]["kid"])
    print(f"\nBest snapshot by KID: {best}")
    final_dir = run / "final_samples"
    marker = final_dir / "snapshot.txt"
    if final_dir.exists() and not (marker.exists() and marker.read_text() == f"{best}|{args.n_final}"):
        shutil.rmtree(final_dir)                  # samples from another snapshot/size: start over
    final_dir.mkdir(parents=True, exist_ok=True)
    marker.write_text(f"{best}|{args.n_final}")
    have = len(list(final_dir.glob("gen_*.png")))
    if have < args.n_final:
        if have:
            print(f"Continuing final generation: {have}/{args.n_final} images already done")
        sample(Path(best), args.n_final, final_dir, final=True, start=have)
    try:
        final = compute(final_dir, args.real_final or args.real, prc=True)
    except Exception as e:                      # precision/recall needs extra VGG weights
        print(f"precision/recall skipped ({e.__class__.__name__}); computing FID/KID only")
        final = compute(final_dir, args.real_final or args.real, prc=False)
    final["snapshot"] = best
    final["selected_on"] = args.real
    final["tested_on"] = args.real_final or args.real
    (run / "final_metrics.json").write_text(json.dumps(final, indent=2))
    print(json.dumps(final, indent=2))


if __name__ == "__main__":
    main()
