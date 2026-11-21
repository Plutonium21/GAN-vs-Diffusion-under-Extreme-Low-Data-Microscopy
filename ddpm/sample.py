#!/usr/bin/env python3
"""
Generate grayscale patches from a trained DDPM checkpoint (ema_XXXXXXX.pt).

Use --steps 1000 (default, Algorithm 2 of the paper) for the numbers you report.
A smaller --steps (e.g. 100) uses a re-spaced chain and is only meant for quick screening.

    python ddpm/sample.py --ckpt runs/ddpm/n25-seed0/checkpoints/ema_0060000.pt \
        --n 2000 --outdir samples/ddpm_n25_seed0
"""
import argparse
import sys
from pathlib import Path

import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ddpm import GaussianDiffusion, UNet  # noqa: E402


def load_model(ckpt, device):
    st = torch.load(ckpt, map_location="cpu", weights_only=False)
    model = UNet(**st["config"])
    model.load_state_dict(st["ema"])
    return model.to(device).eval().to(memory_format=torch.channels_last)


@torch.no_grad()
def generate(model, n, outdir, batch=64, steps=1000, var="large", seed=123, device="cuda", amp="auto", start=0):
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    res, in_ch = model.config["img_res"], model.config["in_ch"]
    diffusion = GaussianDiffusion().to(device)
    if device == "cuda" and amp != "none":
        dtype = torch.bfloat16 if (amp == "bf16" or (amp == "auto" and torch.cuda.is_bf16_supported())) else torch.float16
    else:
        dtype = torch.float32
    gen = torch.Generator(device=device).manual_seed(seed + start)
    done = start
    while done < n:
        b = min(batch, n - done)
        x = diffusion.sample(model, (b, in_ch, res, res), steps=steps, var=var, generator=gen,
                             device=device, dtype=dtype, progress=True)
        x = ((x[:, 0].clamp(-1, 1) + 1) * 127.5).round().to(torch.uint8).cpu().numpy()
        for i in range(b):
            Image.fromarray(x[i]).save(outdir / f"gen_{done + i:06d}.png")
        done += b
        print(f"{done}/{n}", flush=True)
    return outdir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--n", type=int, default=2000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1000)
    ap.add_argument("--var", choices=["large", "small"], default="large", help="sigma_t^2 = beta_t or beta~_t")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--amp", choices=["auto", "bf16", "fp16", "none"], default="auto")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(args.ckpt, device)
    generate(model, args.n, args.outdir, args.batch, args.steps, args.var, args.seed, device, args.amp)


if __name__ == "__main__":
    main()
