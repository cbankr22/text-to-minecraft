"""Text -> voxel-code transformer (MaskGIT-style) conditioned on frozen CLIP text features.

The tokenizer turns a 32^3 build into an 8x8x8 grid of codes (512 tokens). This model
learns to fill in masked codes given the caption; generation starts fully masked and
unmasks the most confident codes over a few parallel steps.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

CLIP_NAME = "openai/clip-vit-base-patch32"


class TextEncoder(nn.Module):
    """Frozen CLIP text model -> (hidden states, padding mask)."""

    def __init__(self, device: str = "cuda"):
        super().__init__()
        from transformers import CLIPTextModel, CLIPTokenizer
        self.tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
        self.model = CLIPTextModel.from_pretrained(CLIP_NAME).to(device).eval()
        self.model.requires_grad_(False)
        self.device = device
        self.dim = self.model.config.hidden_size

    @torch.no_grad()
    def forward(self, texts: list[str]) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.tok(texts, padding="max_length", max_length=77, truncation=True,
                     return_tensors="pt").to(self.device)
        h = self.model(**x).last_hidden_state
        return h, x["attention_mask"] == 0          # True = padding


class T2V(nn.Module):
    def __init__(self, codebook: int, seq: int = 512, d: int = 512, layers: int = 12,
                 heads: int = 8, text_dim: int = 512, dropout: float = 0.1):
        super().__init__()
        self.codebook, self.seq = codebook, seq
        self.mask_id = codebook
        self.tok_emb = nn.Embedding(codebook + 1, d)
        self.pos = nn.Parameter(torch.randn(1, seq, d) * 0.02)
        self.text_proj = nn.Linear(text_dim, d)
        layer = nn.TransformerDecoderLayer(d, heads, 4 * d, dropout, activation="gelu",
                                           batch_first=True, norm_first=True)
        self.blocks = nn.TransformerDecoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, codebook)

    def forward(self, tokens, text_h, text_pad, only: torch.Tensor | None = None):
        """only: optional bool mask; if given, return logits just for those positions
        (flattened), which saves memory with large codebooks."""
        x = self.tok_emb(tokens) + self.pos
        mem = self.text_proj(text_h)
        x = self.norm(self.blocks(x, mem, memory_key_padding_mask=text_pad))
        return self.head(x[only] if only is not None else x)


def mask_tokens(codes: torch.Tensor, mask_id: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Cosine-schedule masking (MaskGIT). Returns (inputs, bool mask of positions to predict)."""
    B, L = codes.shape
    r = torch.cos(torch.rand(B, device=codes.device) * math.pi / 2)
    n = (r * L).ceil().clamp(1, L).long()
    ranks = torch.rand(B, L, device=codes.device).argsort(1).argsort(1)
    mask = ranks < n[:, None]
    return torch.where(mask, mask_id, codes), mask


@torch.no_grad()
def generate(model: T2V, text: TextEncoder, prompts: list[str], steps: int = 12,
             cfg: float = 3.0, temperature: float = 1.0, choice_temp: float = 4.5,
             seed: int | None = None) -> torch.Tensor:
    """Returns (B, seq) code indices."""
    dev = next(model.parameters()).device
    g = torch.Generator(device=dev)
    if seed is not None:
        g.manual_seed(seed)
    B, L = len(prompts), model.seq
    h, pad = text(prompts)
    h0, pad0 = text([""] * B)
    tokens = torch.full((B, L), model.mask_id, device=dev)
    for t in range(steps):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(torch.cat([tokens, tokens]), torch.cat([h, h0]), torch.cat([pad, pad0]))
        cond, uncond = logits.float().chunk(2)
        logits = uncond + cfg * (cond - uncond)
        probs = F.softmax(logits / temperature, -1)
        sampled = torch.multinomial(probs.view(-1, model.codebook), 1, generator=g).view(B, L)
        conf = probs.gather(-1, sampled[..., None])[..., 0]
        unknown = tokens == model.mask_id
        sampled = torch.where(unknown, sampled, tokens)
        conf = torch.where(unknown, conf, torch.full_like(conf, float("inf")))
        n_mask = int(math.floor(math.cos(math.pi / 2 * (t + 1) / steps) * L))
        if n_mask == 0:
            tokens = sampled
            break
        # randomness in which tokens stay masked, annealed to zero
        gumbel = -torch.log(-torch.log(torch.rand(B, L, device=dev, generator=g).clamp_min(1e-20)))
        score = torch.log(conf.clamp_min(1e-20)) + choice_temp * (1 - (t + 1) / steps) * gumbel
        score = torch.where(unknown, score, torch.full_like(score, float("inf")))
        remask = score.argsort(1)[:, :n_mask]
        tokens = sampled.scatter(1, remask, model.mask_id)
    return tokens
