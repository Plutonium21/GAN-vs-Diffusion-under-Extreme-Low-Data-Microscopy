#!/usr/bin/env python3
"""
Train a DDPM (Ho et al., 2020) on one low-data microscopy subset.

"""
import argparse
import copy
import json
import os
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddpm import GaussianDiffusion, UNet, default_config  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
PAPER_LR = {256: 2e-5, 128: 5e-5, 64: 1e-4, 32: 2e-4}   # 256 and 32 from the paper; others interpolated


def load_images(img_dir):
    files = sorted(Path(img_dir).glob("*.png"))
    if not files:
        sys.exit(f"No PNG images in {img_dir}")
    arr = np.stack([np.array(Image.open(f).convert("L")) for f in files])
    return torch.from_numpy(arr)[:, None]          # uint8 [N,1,H,W]


def augment(x, mode):
    """Random flips (paper: horizontal). 'dihedral' adds vertical flips + 90 deg rotations:
    microscopy has no canonical orientation, so rotated cells are still realistic samples."""
    B = x.shape[0]
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(3), x)
    if mode == "dihedral":
        k = torch.randint(0, 4, (B,), device=x.device)
        out = x.clone()
        for r in (1, 2, 3):
            m = k == r
            if m.any():
                out[m] = torch.rot90(x[m], r, dims=(2, 3))
        x = out
    return x


def amp_dtype(choice, device):
    if device.type != "cuda" or choice == "none":
        return None
    if choice == "bf16" or (choice == "auto" and torch.cuda.is_bf16_supported()):
        return torch.bfloat16
    return torch.float16


def save_grid(x, path, nrow=4):
    x = ((x.float().clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()[:, 0]
    n, h, w = x.shape
    rows = math.ceil(n / nrow)
    grid = np.zeros((rows * h, nrow * w), np.uint8)
    for i in range(n):
        grid[(i // nrow) * h:(i // nrow + 1) * h, (i % nrow) * w:(i % nrow + 1) * w] = x[i]
    Image.fromarray(grid).save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_argument_group("data")
    g.add_argument("--data-root", help="Output folder of data/prepare_data.py")
    g.add_argument("--n", type=int, help="Low-data regime (number of frames).")
    g.add_argument("--seed", type=int, default=0, help="Which random subset (and training seed).")
    g.add_argument("--images", help="Alternatively: a folder of PNG patches.")
    g.add_argument("--outdir", default=None)
    g = ap.add_argument_group("model / diffusion (paper defaults)")
    g.add_argument("--ch", type=int, default=128)
    g.add_argument("--ch-mult", type=int, nargs="+", default=None)
    g.add_argument("--dropout", type=float, default=0.1)
    g.add_argument("--T", type=int, default=1000)
    g = ap.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=60000, help="Training iterations.")
    g.add_argument("--batch", type=int, default=64, help="Effective batch (paper: 64 for large images).")
    g.add_argument("--batch-gpu", type=int, default=16,
                   help="Micro-batch per forward pass; gradients are accumulated up to --batch (same result). "
                        "Raise to 32/64 on 40/80 GB GPUs for speed; use 8 + --grad-ckpt on 16 GB GPUs.")
    g.add_argument("--lr", type=float, default=None, help="Default: paper value for the resolution.")
    g.add_argument("--warmup", type=int, default=5000)
    g.add_argument("--grad-clip", type=float, default=1.0)
    g.add_argument("--ema", type=float, default=0.9999)
    g.add_argument("--aug", choices=["hflip", "dihedral"], default="dihedral")
    g = ap.add_argument_group("speed")
    g.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    g.add_argument("--compile", action="store_true", help="torch.compile the U-Net (faster after warm-up).")
    g.add_argument("--grad-ckpt", action="store_true", help="Gradient checkpointing (less memory, ~25%% slower).")
    g = ap.add_argument_group("logging")
    g.add_argument("--ckpt-every", type=int, default=5000, help="Save an EMA snapshot (for evaluation) every N steps.")
    g.add_argument("--save-minutes", type=float, default=10.0,
                   help="Also save resumable progress at least this often, so a disconnect loses little work.")
    g.add_argument("--sample-every", type=int, default=5000, help="Preview grid every N steps (100-step sampler).")
    g.add_argument("--log-every", type=int, default=100)
    args = ap.parse_args()

    # ---------------------------------------------------------------- setup
    if args.images:
        img_dir = Path(args.images).expanduser()
        name = img_dir.parent.name
    else:
        if not (args.data_root and args.n):
            sys.exit("Give --data-root and --n (or --images).")
        img_dir = Path(args.data_root).expanduser() / "subsets" / f"seed{args.seed}" / f"n{args.n}" / "images"
        name = f"n{args.n}-seed{args.seed}"
    outdir = Path(args.outdir or ROOT / "runs" / "ddpm" / name).expanduser()
    (outdir / "checkpoints").mkdir(parents=True, exist_ok=True)
    prog = outdir / "checkpoints" / "progress.json"
    if prog.exists() and json.loads(prog.read_text()).get("step", 0) >= args.steps:
        print(f"{name}: already finished ({args.steps} steps) - skipping.")
        return
    (outdir / "previews").mkdir(exist_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
    dtype = amp_dtype(args.amp, device)

    data = load_images(img_dir)
    if device.type == "cuda":
        data = data.pin_memory()
    res = data.shape[-1]
    cfg = default_config(res)
    cfg.update(ch=args.ch, dropout=args.dropout, in_ch=1)
    if args.ch_mult:
        cfg["ch_mult"] = tuple(args.ch_mult)
    lr = args.lr or PAPER_LR[res]
    bgpu = min(args.batch_gpu or args.batch, args.batch)
    if args.batch % bgpu:
        sys.exit("--batch must be divisible by --batch-gpu")
    accum = args.batch // bgpu

    model = UNet(**cfg, grad_ckpt=args.grad_ckpt).to(device).to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    diffusion = GaussianDiffusion(T=args.T).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, fused=device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=dtype == torch.float16)
    fwd = torch.compile(model) if args.compile else model

    step, resumed_at = 0, None
    ck = outdir / "checkpoints"
    slots = [ck / "progress_a.pt", ck / "progress_b.pt"]
    if prog.exists():
        order = [ck / json.loads(prog.read_text()).get("file", "progress_a.pt")]
        order += [f for f in slots if f not in order]
        for f in order:                          # newest first; fall back to the other slot if damaged
            try:
                st = torch.load(f, map_location=device, weights_only=False)
                model.load_state_dict(st["model"]); ema.load_state_dict(st["ema"]); step = st["step"]
                resumed_at = step
                print(f"Resumed from step {step} ({f.name})")
                break
            except Exception as e:  # noqa: BLE001
                print(f"  ! could not read {f.name} ({e.__class__.__name__})")

    meta = dict(args=vars(args), model=cfg, lr=lr, n_images=len(data), resolution=res,
                amp=str(dtype), device=str(device), params_M=sum(p.numel() for p in model.parameters()) / 1e6)
    (outdir / "config.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))

    def half(sd):
        return {k: (v.half() if v.is_floating_point() else v) for k, v in sd.items()}

    slot = [0]

    def save_progress():
        """Resumable progress, written alternately into two fixed files (no new files, so Drive's
        trash does not fill up). progress.json is updated only after a write completes, so a
        disconnect mid-save falls back to the other, complete file. Saved in half precision to
        keep Drive usage low; the optimizer restarts with a short warm-up after resuming."""
        f = slots[slot[0]]
        try:
            torch.save({"model": half(model.state_dict()), "ema": half(ema.state_dict()),
                        "step": step, "config": cfg}, f)
            prog.write_text(json.dumps({"step": step, "file": f.name}))
            slot[0] ^= 1
            return True
        except OSError as e:
            print(f"  ! could not save progress ({e}) - training continues; check Google Drive space", flush=True)
            return False

    def save_snapshot():
        try:
            torch.save({"config": cfg, "ema": half(ema.state_dict()), "step": step},
                       ck / f"ema_{step:07d}.pt")
        except OSError as e:
            print(f"  ! could not save snapshot ({e}) - check Google Drive space", flush=True)

    # ---------------------------------------------------------------- loop
    log = open(outdir / "log.jsonl", "a")
    last_save = time.time()
    t0, run_loss, seen = time.time(), 0.0, 0
    gen = torch.Generator(device=device).manual_seed(1234)
    while step < args.steps:
        model.train()
        ramp = min(1.0, (step + 1) / args.warmup)
        if resumed_at is not None:               # fresh optimizer after a resume: short re-warm-up
            ramp = min(ramp, (step - resumed_at + 1) / 200)
        for g_ in opt.param_groups:
            g_["lr"] = lr * ramp
        opt.zero_grad(set_to_none=True)
        loss_acc = 0.0
        for _ in range(accum):
            idx = torch.randint(0, len(data), (bgpu,))
            x0 = data[idx].to(device, non_blocking=True).float().div_(127.5).sub_(1)
            x0 = augment(x0, args.aug).contiguous(memory_format=torch.channels_last)
            with torch.autocast(device_type=device.type, dtype=dtype or torch.float32, enabled=dtype is not None):
                loss = diffusion.loss(fwd, x0) / accum
            scaler.scale(loss).backward()
            loss_acc += loss.item()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        step += 1

        decay = min(args.ema, (1 + step) / (10 + step))
        with torch.no_grad():
            for pe, pm in zip(ema.parameters(), model.parameters()):
                pe.lerp_(pm, 1 - decay)

        run_loss += loss_acc
        seen += 1
        if step % args.log_every == 0 or step == args.steps:
            el = time.time() - t0
            sps = seen / el
            rec = dict(step=step, loss=run_loss / seen, grad_norm=float(gnorm), lr=opt.param_groups[0]["lr"],
                       steps_per_s=round(sps, 3))
            log.write(json.dumps(rec) + "\n"); log.flush()
            eta_h = (args.steps - step) / max(sps, 1e-9) / 3600
            print(f"step {step:7d} | loss {rec['loss']:.4f} | {sps:.2f} it/s | ETA {eta_h:.1f} h", flush=True)
            t0, run_loss, seen = time.time(), 0.0, 0

        if step % args.sample_every == 0 or step == args.steps:
            x = diffusion.sample(ema, (16, 1, res, res), steps=100, generator=gen, device=device,
                                 dtype=dtype or torch.float32)
            save_grid(x, outdir / "previews" / f"step{step:07d}.png")
        if step % args.ckpt_every == 0 or step == args.steps:
            save_snapshot()
            save_progress(); last_save = time.time()
        elif time.time() - last_save > args.save_minutes * 60:
            if save_progress():
                print(f"  progress saved at step {step}", flush=True)
            last_save = time.time()
    for f in slots:                              # finished: progress files no longer needed
        f.unlink(missing_ok=True)
    print(f"Finished. Checkpoints in {outdir / 'checkpoints'}")


if __name__ == "__main__":
    main()
