"""Text -> voxel latents by flow matching (rectified flow) on the tokenizer's FSQ latent.

The tokenizer turns a 32^3 build into 8x8x8 positions with 5 numbers each (its FSQ latent:
values on a small grid in [-1, 1]). This model learns to turn Gaussian noise into such a
latent, guided by CLIP text features:

    x_t = (1 - t) * x0 + t * noise          t in [0, 1], x0 = data, t = 1 is pure noise
    the network predicts the velocity  v = noise - x0

Sampling starts from noise at t = 1 and steps to t = 0 with classifier-free guidance, then
rounds to the FSQ grid and decodes with the tokenizer. Unlike the MaskGIT model (t2v.py),
which commits to hundreds of discrete codes per step independently, all 512 positions are
refined together at every step, so neighbours stay consistent.

Architecture: a DiT-style transformer. Each block has self-attention, cross-attention to the
CLIP token features and an MLP; the timestep and the pooled prompt steer every block through
adaptive LayerNorm (adaLN-Zero). QK-normalization keeps attention logits bounded (the MaskGIT
runs diverged twice without it).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int = 256) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half)
    args = t.float()[:, None] * 1000 * freqs[None]
    return torch.cat([args.cos(), args.sin()], -1)


def modulate(x, shift, scale):
    return x * (1 + scale) + shift


class RMSNorm(nn.Module):
    """RMSNorm computed in the input's dtype (nn.RMSNorm under bf16 autocast falls back to a slow
    unfused path because its weight stays float32)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight.to(x.dtype)


class Attention(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.h = heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))
        self.qn = RMSNorm(d // heads)
        self.kn = RMSNorm(d // heads)

    def forward(self, x, ctx=None, mask=None):
        B, L, D = x.shape
        ctx = x if ctx is None else ctx
        q = self.qn(self.q(x).view(B, L, self.h, -1)).transpose(1, 2)
        k = self.kn(self.k(ctx).view(B, ctx.shape[1], self.h, -1)).transpose(1, 2)
        v = self.v(ctx).view(B, ctx.shape[1], self.h, -1).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(out.transpose(1, 2).reshape(B, L, D))


class Block(nn.Module):
    def __init__(self, d: int, heads: int):
        super().__init__()
        self.n1 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(d, heads)
        self.n2 = nn.LayerNorm(d, eps=1e-6)
        self.cross = Attention(d, heads)
        self.n3 = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.ada = nn.Linear(d, 6 * d)
        nn.init.zeros_(self.ada.weight)
        nn.init.zeros_(self.ada.bias)

    def forward(self, x, c, ctx, mask):
        s1, k1, g1, s2, k2, g2 = self.ada(F.silu(c))[:, None].chunk(6, -1)
        x = x + g1 * self.attn(modulate(self.n1(x), s1, k1))
        x = x + self.cross(self.n2(x), ctx, mask)
        return x + g2 * self.mlp(modulate(self.n3(x), s2, k2))


class FlowT2V(nn.Module):
    def __init__(self, latent_dim: int = 5, seq: int = 512, d: int = 512, layers: int = 12,
                 heads: int = 8, text_dim: int = 512):
        super().__init__()
        self.latent_dim, self.seq = latent_dim, seq
        self.inp = nn.Linear(latent_dim, d)
        self.pos = nn.Parameter(torch.randn(1, seq, d) * 0.02)
        self.t_mlp = nn.Sequential(nn.Linear(256, d), nn.SiLU(), nn.Linear(d, d))
        self.pool_proj = nn.Linear(text_dim, d)          # pooled prompt (EOS token) -> adaLN
        self.text_proj = nn.Linear(text_dim, d)          # all prompt tokens -> cross-attention
        self.blocks = nn.ModuleList(Block(d, heads) for _ in range(layers))
        self.norm = nn.LayerNorm(d, elementwise_affine=False, eps=1e-6)
        self.ada_out = nn.Linear(d, 2 * d)
        self.out = nn.Linear(d, latent_dim)
        for m in (self.ada_out, self.out):
            nn.init.zeros_(m.weight)
            nn.init.zeros_(m.bias)

    def forward(self, x, t, text_h, text_pad):
        """x (B, seq, latent_dim) noisy latent, t (B,), text_h (B, 77, text_dim), text_pad (B, 77)."""
        eos = (~text_pad).sum(1) - 1                     # CLIP's last real token summarizes the prompt
        # gather instead of text_h[arange(B), eos]: the same values, and it exports to ONNX for any batch
        pooled = text_h.gather(1, eos[:, None, None].expand(-1, 1, text_h.shape[-1])).squeeze(1)
        c = self.t_mlp(timestep_embedding(t)) + self.pool_proj(pooled)
        ctx = self.text_proj(text_h)
        mask = ~text_pad[:, None, None, :]               # True = attend
        h = self.inp(x) + self.pos
        for b in self.blocks:
            h = b(h, c, ctx, mask)
        shift, scale = self.ada_out(F.silu(c))[:, None].chunk(2, -1)
        return self.out(modulate(self.norm(h), shift, scale))


# ---------------------------------------------------------------- FSQ codes <-> latent

def codes_to_latent(idx: torch.Tensor, fsq) -> torch.Tensor:
    """(..., ) integer codes -> (..., dim) latent values in [-1, 1] (FSQ.indices_to_codes)."""
    digits = (idx[..., None] // fsq.basis) % fsq.levels
    half = fsq.levels // 2
    return (digits - half).float() / half


def latent_to_codes(z: torch.Tensor, fsq) -> torch.Tensor:
    """Round a latent to the nearest FSQ grid point and return its integer code."""
    half = fsq.levels // 2
    digits = torch.round(z * half + half)
    digits = torch.minimum(digits.clamp_min(0), (fsq.levels - 1).to(digits.dtype)).long()
    return (digits * fsq.basis).sum(-1)


@torch.no_grad()
def sample(model: FlowT2V, text, prompts: list[str], steps: int = 40, cfg: float = 5.0,
           seed: int | None = None, shift: float = 1.0, init: torch.Tensor | None = None,
           strength: float = 1.0) -> torch.Tensor:
    """Euler integration from noise (t=1) to data (t=0). Returns (B, seq, latent_dim).
    shift > 1 spends more of the steps at high noise, where the overall shape is decided
    (the timestep shift of Stable Diffusion 3); 1 = evenly spaced.
    init (B, seq, latent_dim) + strength < 1: start from an existing build's latent pushed to
    noise level `strength` instead of pure noise (SDEdit): low strength keeps its shape and
    colours, high strength keeps only the rough layout."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)
    B = len(prompts)
    h, pad = text(prompts)
    h0, pad0 = text([""] * B)
    hh, pp = torch.cat([h, h0]), torch.cat([pad, pad0])
    z = torch.randn(B, model.seq, model.latent_dim, device=dev, generator=g)
    ts = torch.linspace(1, 0, steps + 1, device=dev)
    ts = shift * ts / (1 + (shift - 1) * ts)
    if init is not None:
        ts = ts * strength                                # all `steps` steps go from strength to 0
        z = (1 - strength) * init.to(dev) + strength * z
    for i in range(steps):
        t = ts[i].expand(2 * B)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(torch.cat([z, z]), t, hh, pp).float()
        vc, vu = v.chunk(2)
        z = z + (ts[i + 1] - ts[i]) * (vu + cfg * (vc - vu))
    return z
