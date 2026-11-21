#!/usr/bin/env python3
"""
Final comparison table: StyleGAN2-ADA vs DDPM, every training size, on two test sets,
plus two real-vs-real reference rows.

  * BBBC039 test           - the report's cross-dataset test (different microscope / cell line)
  * BBBC038 unseen         - same kind of images as training, never seen by any model
  * Real BBBC038 A vs B    - two halves of the unseen set: the best score achievable in-domain
  * Real BBBC038 vs BBBC039 - real images only: how much of the BBBC039 score is the dataset gap

"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "eval"))
from metrics import compute  # noqa: E402


def first_session_snapshot(run):
    """The last snapshot of the first training session = the original 300-kimg model."""
    for sub in sorted(d for d in run.glob("*") if d.is_dir() and not d.name.startswith("_")):
        snaps = sorted(sub.glob("network-snapshot-*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
        if snaps and int(snaps[-1].stem.split("-")[-1]) > 0:
            return snaps[-1]
    return None


def last_snapshot(run):
    """The last snapshot of the last training session = the model at the end of all training (600 kimg)."""
    last = None
    for sub in sorted(d for d in run.glob("*") if d.is_dir() and not d.name.startswith("_")):
        snaps = sorted(sub.glob("network-snapshot-*.pkl"), key=lambda p: int(p.stem.split("-")[-1]))
        if snaps and int(snaps[-1].stem.split("-")[-1]) > 0:
            last = snaps[-1]
    return last


def generate_stylegan(pkl, out, n):
    import torch
    sys.path.insert(0, str(ROOT / "stylegan"))
    from generate import generate, load_generator
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"generating {n} images from {pkl.parent.name}/{pkl.name}", flush=True)
    generate(load_generator(pkl, dev), n, out, batch=32, seed=123, device=dev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True)
    ap.add_argument("--test039", required=True)
    ap.add_argument("--test038", required=True, help="output folder of data/make_bbbc038_test.py")
    ap.add_argument("--n-gen", type=int, default=300)
    ap.add_argument("--sizes", nargs="+", type=int, default=[5, 25, 100])
    args = ap.parse_args()
    work = Path(args.work).expanduser()
    t38 = Path(args.test038).expanduser()
    out_dir = work / "results"; out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / "final_table_cache.json"          # resume-safe: finished rows are kept
    rows = json.loads(cache.read_text()) if cache.exists() else {}

    jobs = []
    for n in args.sizes:                                   # StyleGAN at 300 kimg and after continuing
        run = work / "runs" / f"stylegan2ada/scratch-n{n}-seed0"
        first = first_session_snapshot(run)
        if first is not None:
            d = Path(f"/tmp/gvd_sg300_n{n}") if Path("/content").exists() else run / "_sg300_samples"
            if len(list(d.glob("gen_*.png"))) < args.n_gen:
                generate_stylegan(first, d, args.n_gen)
            jobs.append((f"StyleGAN2-ADA, 300 kimg | n={n}", d, args.n_gen))
        last = last_snapshot(run)
        fm = run / "final_metrics.json"
        if fm.exists():
            print(f"StyleGAN n={n}: snapshot chosen by KID on BBBC039 validation = {json.loads(fm.read_text())['snapshot']}")
        if last is not None and last != first:              # training was continued
            d = Path(f"/tmp/gvd_sg600_n{n}") if Path("/content").exists() else run / "_sg600_samples"
            if len(list(d.glob("gen_*.png"))) < args.n_gen:
                generate_stylegan(last, d, args.n_gen)
            jobs.append((f"StyleGAN2-ADA, 600 kimg | n={n}", d, args.n_gen))
        elif first is None:
            print(f"skip StyleGAN n={n}: no snapshots found")
        else:
            print(f"StyleGAN n={n}: no continued training found (only one session with snapshots)")
    for n in args.sizes:
        d = work / "runs" / f"ddpm/n{n}-seed0" / "final_samples"
        if d.exists() and len(list(d.glob("gen_*.png"))) >= args.n_gen:
            jobs.append((f"DDPM | n={n}", d, args.n_gen))
        else:
            print(f"skip DDPM n={n}: final_samples missing (run its evaluation first)")
    jobs.append(("Real BBBC038 half A (reference)", t38 / "half_a", None))
    for name, gen, ng in jobs:
        for test_name, test_dir in [("BBBC039 test", args.test039),
                                    ("BBBC038 unseen", t38 / ("half_b" if "half A" in name else "images"))]:
            key = f"{name} || {test_name}"
            if key in rows:
                continue
            try:                                           # precision/recall (VGG) can fail to download
                m = compute(gen, test_dir, prc=True, max_gen=ng)
            except Exception as e:  # noqa: BLE001
                print(f"precision/recall failed ({e.__class__.__name__}); FID/KID only", flush=True)
                m = compute(gen, test_dir, max_gen=ng)
            rows[key] = {k: m[k] for k in ("fid", "kid", "kid_std", "n_gen", "precision", "recall") if k in m}
            cache.write_text(json.dumps(rows, indent=2))
            print(f"{key:60s} KID={m['kid']:.4f}  FID={m['fid']:.1f}  "
                  f"P={m.get('precision', float('nan')):.3f}  R={m.get('recall', float('nan')):.3f}", flush=True)

    names = [j[0] for j in jobs]
    lines = ["| Model | Training images | KID ↓ 039 test | FID ↓ 039 test | Prec ↑ 039 | Rec ↑ 039 "
             "| KID ↓ 038 unseen | FID ↓ 038 unseen | Prec ↑ 038 | Rec ↑ 038 |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for name in names:
        a, b = rows.get(f"{name} || BBBC039 test"), rows.get(f"{name} || BBBC038 unseen")
        model, size = (name.split(" | ") + [""])[:2] if " | " in name else (name, "–")
        fmt = lambda r, k, p: f"{r[k]:.{p}f}" if r else "–"  # noqa: E731
        kid = lambda r: f"{r['kid']:.3f} ± {r['kid_std']:.3f}" if r else "–"  # noqa: E731
        pr = lambda r, k: f"{r[k]:.3f}" if r and k in r else "–"  # noqa: E731
        lines.append(f"| {model} | {size.replace('n=', '')} | {kid(a)} | {fmt(a, 'fid', 1)} | {pr(a, 'precision')} | "
                     f"{pr(a, 'recall')} | {kid(b)} | {fmt(b, 'fid', 1)} | {pr(b, 'precision')} | {pr(b, 'recall')} |")
    lines += ["", f"Generated images per model: {args.n_gen}. The 'Real BBBC038 half A' row compares two disjoint "
              "halves of real unseen BBBC038 patches (in-domain floor) and real BBBC038 vs real BBBC039 "
              "(dataset-gap floor)."]
    table = "\n".join(lines)
    (out_dir / "final_table.md").write_text(table + "\n")
    print("\n" + table + f"\n\nSaved: {out_dir / 'final_table.md'}")


if __name__ == "__main__":
    main()
