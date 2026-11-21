"""

Paper B - Denoising Diffusion Probabilistic Models (Ho, Jain, Abbeel, NeurIPS 2020).

"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


# ============================================================================ U-Net
def timestep_embedding(t, dim):
    """Transformer sinusoidal embedding (same formula as the official DDPM code)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1))
    args = t.float()[:, None] * freqs[None]
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def norm(c):
    return nn.GroupNorm(32, c, eps=1e-6)


def zero_init(m):
    nn.init.zeros_(m.weight)
    nn.init.zeros_(m.bias)
    return m


class ResBlock(nn.Module):
    def __init__(self, cin, cout, temb_ch, dropout):
        super().__init__()
        self.norm1, self.conv1 = norm(cin), nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(temb_ch, cout)
        self.norm2, self.drop = norm(cout), nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.norm, self.qkv = norm(c), nn.Conv2d(c, 3 * c, 1)
        self.proj = zero_init(nn.Conv2d(c, c, 1))

    def forward(self, x, temb=None):
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, 1, C, H * W).transpose(-1, -2).unbind(1)
        h = F.scaled_dot_product_attention(q, k, v)            # softmax(q k^T / sqrt(C)) v
        return x + self.proj(h.transpose(-1, -2).reshape(B, C, H, W))


class Down(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, stride=2, padding=1)

    def forward(self, x, temb=None):
        return self.conv(x)


class Up(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Conv2d(c, c, 3, padding=1)

    def forward(self, x, temb=None):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class UNet(nn.Module):
    def __init__(self, img_res=256, in_ch=1, ch=128, ch_mult=(1, 1, 2, 2, 4, 4), num_res_blocks=2,
                 attn_res=(16,), dropout=0.0, grad_ckpt=False):
        super().__init__()
        self.config = dict(img_res=img_res, in_ch=in_ch, ch=ch, ch_mult=tuple(ch_mult),
                           num_res_blocks=num_res_blocks, attn_res=tuple(attn_res), dropout=dropout)
        self.ch, self.grad_ckpt = ch, grad_ckpt
        temb_ch = 4 * ch
        self.temb = nn.Sequential(nn.Linear(ch, temb_ch), nn.SiLU(), nn.Linear(temb_ch, temb_ch))
        self.conv_in = nn.Conv2d(in_ch, ch, 3, padding=1)

        self.down, skip_ch, cur, res = nn.ModuleList(), [ch], ch, img_res
        for i, mult in enumerate(ch_mult):
            for _ in range(num_res_blocks):
                blocks = [ResBlock(cur, ch * mult, temb_ch, dropout)]
                cur = ch * mult
                if res in attn_res:
                    blocks.append(AttnBlock(cur))
                self.down.append(nn.ModuleList(blocks))
                skip_ch.append(cur)
            if i != len(ch_mult) - 1:
                self.down.append(nn.ModuleList([Down(cur)]))
                skip_ch.append(cur)
                res //= 2

        self.mid = nn.ModuleList([ResBlock(cur, cur, temb_ch, dropout), AttnBlock(cur),
                                  ResBlock(cur, cur, temb_ch, dropout)])

        self.up = nn.ModuleList()
        for i, mult in reversed(list(enumerate(ch_mult))):
            for _ in range(num_res_blocks + 1):
                blocks = [ResBlock(cur + skip_ch.pop(), ch * mult, temb_ch, dropout)]
                cur = ch * mult
                if res in attn_res:
                    blocks.append(AttnBlock(cur))
                self.up.append(nn.ModuleList(blocks))
            if i != 0:
                self.up.append(nn.ModuleList([Up(cur)]))
                res *= 2
        assert not skip_ch
        self.out = nn.Sequential(norm(cur), nn.SiLU(), zero_init(nn.Conv2d(cur, in_ch, 3, padding=1)))
        self._init()

    def _init(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)) and m.weight.abs().sum() > 0:
                nn.init.xavier_uniform_(m.weight)     # variance scaling, fan_avg (official init)
                nn.init.zeros_(m.bias)

    def _run(self, blocks, h, temb):
        for b in blocks:
            if self.grad_ckpt and self.training and isinstance(b, (ResBlock, AttnBlock)):
                h = checkpoint(b, h, temb, use_reentrant=False)
            else:
                h = b(h, temb)
        return h

    def forward(self, x, t):
        temb = self.temb(timestep_embedding(t, self.ch))
        h = self.conv_in(x)
        hs = [h]
        for blocks in self.down:
            h = self._run(blocks, h, temb)
            hs.append(h)
        h = self._run(self.mid, h, temb)
        for blocks in self.up:
            if not isinstance(blocks[0], Up):
                h = torch.cat([h, hs.pop()], dim=1)
            h = self._run(blocks, h, temb)
        return self.out(h)


def default_config(res):
    """Architectures per resolution. 256 and 32 are the paper's; 128/64 follow the same pattern."""
    ch_mult = {256: (1, 1, 2, 2, 4, 4), 128: (1, 1, 2, 3, 4), 64: (1, 2, 3, 4), 32: (1, 2, 2, 2)}
    if res not in ch_mult:
        raise ValueError(f"Unsupported resolution {res}; use 32, 64, 128 or 256.")
    return dict(img_res=res, ch=128, ch_mult=ch_mult[res], num_res_blocks=2, attn_res=(16,))


# ============================================================================ Diffusion
class GaussianDiffusion(nn.Module):
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.T = T
        betas = torch.linspace(beta_start, beta_end, T, dtype=torch.float64)
        ab = torch.cumprod(1.0 - betas, 0)
        self.register_buffer("alphas_cumprod", ab, persistent=False)
        self.register_buffer("sqrt_ab", ab.sqrt().float(), persistent=False)
        self.register_buffer("sqrt_1mab", (1 - ab).sqrt().float(), persistent=False)

    def q_sample(self, x0, t, noise):
        return self.sqrt_ab[t][:, None, None, None] * x0 + self.sqrt_1mab[t][:, None, None, None] * noise

    def loss(self, model, x0):
        """L_simple (Algorithm 1)."""
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        noise = torch.randn_like(x0)
        pred = model(self.q_sample(x0, t, noise), t)
        return F.mse_loss(pred.float(), noise)

    def _schedule(self, steps):
        """Timesteps to visit and the matching (re-spaced) betas. steps == T -> original chain."""
        if steps >= self.T:
            ts = torch.arange(self.T, device=self.alphas_cumprod.device)
        else:
            ts = torch.linspace(0, self.T - 1, steps, device=self.alphas_cumprod.device).round().long().unique()
        ab = self.alphas_cumprod[ts]
        ab_prev = torch.cat([ab.new_ones(1), ab[:-1]])
        betas = 1 - ab / ab_prev
        return ts, ab, ab_prev, betas

    @torch.no_grad()
    def sample(self, model, shape, steps=None, var="large", generator=None, device="cuda",
               dtype=torch.float32, progress=False):
        """Ancestral sampling (Algorithm 2), with x0 clipped to [-1, 1] as in the official code."""
        ts, ab, ab_prev, betas = self._schedule(steps or self.T)
        post_var = betas * (1 - ab_prev) / (1 - ab)                      # beta~_t
        if var == "large":                                               # sigma_t^2 = beta_t
            sig2 = torch.cat([post_var[1:2], betas[1:]])
        else:                                                            # sigma_t^2 = beta~_t
            sig2 = post_var
        coef_x0 = betas * ab_prev.sqrt() / (1 - ab)
        coef_xt = (1 - ab_prev) * (1 - betas).sqrt() / (1 - ab)

        x = torch.randn(shape, generator=generator, device=device)
        it = range(len(ts) - 1, -1, -1)
        if progress:
            from tqdm import tqdm
            it = tqdm(it, desc="sampling", leave=False)
        for i in it:
            t = ts[i].expand(shape[0])
            with torch.autocast(device_type=x.device.type, dtype=dtype, enabled=dtype != torch.float32):
                eps = model(x, t).float()
            x0 = ((x - (1 - ab[i]).sqrt().float() * eps) / ab[i].sqrt().float()).clamp(-1, 1)
            mean = coef_x0[i].float() * x0 + coef_xt[i].float() * x
            if i > 0:
                x = mean + sig2[i].sqrt().float() * torch.randn(shape, generator=generator, device=device)
            else:
                x = mean
        return x
