"""3D VQ-VAE ("voxel tokenizer"): 64^3 block grid <-> 16^3 grid of discrete codes.

Quantization uses FSQ (finite scalar quantization, Mentzer et al. 2023): each
latent channel is rounded to a few fixed levels, so there is no codebook to
collapse and no commitment loss. levels=(8,5,5,5) gives 1000 codes.
"""
from __future__ import annotations

import math
from pathlib import Path, PureWindowsPath

import torch
import torch.nn as nn
import torch.nn.functional as F


class FSQ(nn.Module):
    def __init__(self, levels: tuple[int, ...] = (8, 5, 5, 5)):
        super().__init__()
        levels_t = torch.tensor(levels, dtype=torch.int64)
        self.register_buffer("levels", levels_t, persistent=False)
        basis = torch.cumprod(torch.tensor((1,) + levels[:-1], dtype=torch.int64), 0)
        self.register_buffer("basis", basis, persistent=False)
        self.dim = len(levels)
        self.codebook_size = int(math.prod(levels))

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        half_l = (self.levels - 1) * (1 + 1e-3) / 2
        offset = torch.where(self.levels % 2 == 0, 0.5, 0.0)
        shift = torch.atanh(offset / half_l)
        return torch.tanh(z + shift) * half_l - offset

    def forward(self, z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """z: (B, dim, D, H, W) -> (quantized in [-1,1], code indices (B, D, H, W))."""
        z = z.movedim(1, -1).float()
        bounded = self._bound(z)
        q = bounded + (bounded.round() - bounded).detach()        # straight-through
        half_w = (self.levels // 2).to(q.dtype)
        idx = ((q.round() + half_w).long() * self.basis).sum(-1)
        return (q / half_w).movedim(-1, 1), idx

    def indices_to_codes(self, idx: torch.Tensor) -> torch.Tensor:
        """(B, D, H, W) -> (B, dim, D, H, W) normalized codes."""
        digits = (idx.unsqueeze(-1) // self.basis) % self.levels
        half_w = self.levels // 2
        return ((digits - half_w).float() / half_w).movedim(-1, 1)


class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch), nn.SiLU(), nn.Conv3d(ch, ch, 3, padding=1),
        )

    def forward(self, x):
        return x + self.net(x)


class VoxelVQVAE(nn.Module):
    def __init__(self, vocab: int, embed: int = 32, channels: tuple[int, ...] = (32, 128, 256),
                 levels: tuple[int, ...] = (8, 5, 5, 5)):
        """len(channels)-1 stride-2 stages: (32,128,256) downsamples 4x -> 16^3 codes.

        The full-resolution (64^3) stage is a single conv on each side; residual
        blocks only run at 32^3 and below, which keeps memory within 8 GB GPUs.
        """
        super().__init__()
        self.vocab = vocab
        self.fsq = FSQ(levels)
        self.embed = nn.Embedding(vocab, embed)

        enc = [nn.Conv3d(embed, channels[0], 3, padding=1)]
        for cin, cout in zip(channels, channels[1:]):
            enc += [nn.Conv3d(cin, cout, 4, stride=2, padding=1), ResBlock(cout)]
        enc += [ResBlock(channels[-1]), nn.GroupNorm(8, channels[-1]), nn.SiLU(),
                nn.Conv3d(channels[-1], self.fsq.dim, 1)]
        self.encoder = nn.Sequential(*enc)

        dec = [nn.Conv3d(self.fsq.dim, channels[-1], 3, padding=1), ResBlock(channels[-1]),
               ResBlock(channels[-1])]
        rev = channels[::-1]
        for i, (cin, cout) in enumerate(zip(rev, rev[1:])):
            dec += [nn.ConvTranspose3d(cin, cout, 4, stride=2, padding=1)]
            if i < len(rev) - 2:                                  # skip at full resolution
                dec += [ResBlock(cout)]
        dec += [nn.GroupNorm(8, channels[0]), nn.SiLU(), nn.Conv3d(channels[0], vocab, 3, padding=1)]
        self.decoder = nn.Sequential(*dec)

    def encode(self, blocks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """blocks: (B, D, H, W) long -> (quantized latents, code indices)."""
        x = self.embed(blocks).movedim(-1, 1)
        return self.fsq(self.encoder(x))

    def decode(self, q: torch.Tensor) -> torch.Tensor:
        return self.decoder(q)                                    # (B, vocab, D, H, W) logits

    def decode_indices(self, idx: torch.Tensor) -> torch.Tensor:
        return self.decode(self.fsq.indices_to_codes(idx))

    def forward(self, blocks: torch.Tensor):
        q, idx = self.encode(blocks)
        return self.decode(q), idx


ROOT = Path(__file__).resolve().parents[1]


def project_path(path) -> Path:
    """Resolve a stored path on any machine: as given if it exists, else relative to the
    project root (absolute paths are re-rooted at their 'data' folder)."""
    p = Path(path)
    if p.exists():
        return p
    parts = PureWindowsPath(str(path)).parts if "\\" in str(path) else p.parts
    if "data" in parts:
        parts = parts[parts.index("data"):]
    return ROOT.joinpath(*parts)


def load_tokenizer(path, device: str = "cuda") -> "VoxelVQVAE":
    """Rebuild a VoxelVQVAE from a checkpoint's saved config, in eval mode."""
    ck = torch.load(project_path(path), map_location=device)
    cfg = ck["config"]
    model = VoxelVQVAE(cfg["vocab"], channels=tuple(cfg.get("channels", (32, 128, 256))),
                       levels=tuple(cfg.get("levels", (8, 5, 5, 5))))
    model.load_state_dict(ck["model"])
    model.res = cfg.get("res", 64)
    return model.to(device).eval()


def recon_loss(logits: torch.Tensor, target: torch.Tensor, air_weight: float = 0.1) -> torch.Tensor:
    """Cross-entropy with air down-weighted; most voxels are air."""
    ce = F.cross_entropy(logits.float(), target, reduction="none")
    w = torch.where(target == 0, air_weight, 1.0)
    return (ce * w).sum() / w.sum()


@torch.no_grad()
def recon_metrics(logits: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred = logits.argmax(1)
    p_occ, t_occ = pred > 0, target > 0
    inter = (p_occ & t_occ).sum().item()
    union = (p_occ | t_occ).sum().item()
    both = p_occ & t_occ
    return {
        "occ_iou": inter / max(union, 1),
        "block_acc": (pred[both] == target[both]).float().mean().item() if both.any() else 0.0,
    }
