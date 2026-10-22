"""Prompt -> blocks with only ONNX Runtime and numpy: the reference for the Java mod.

Every step here has a line-for-line twin in mod/src/main/java/.../ (ClipTokenizer, Generator),
so when the mod misbehaves, compare against this file. Uses the files from
scripts/export_onnx.py (data/onnx).

    .venv/Scripts/python -m ttmc.onnx_infer "a wooden chair" --seed 0
"""
from __future__ import annotations

import html
import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import regex

ONNX_DIR = Path(__file__).resolve().parents[1] / "data" / "onnx"


# ---------------------------------------------------------------- CLIP tokenizer (BPE)

@lru_cache(maxsize=1)
def bytes_to_unicode() -> dict[int, str]:
    """GPT-2/CLIP byte <-> printable character table."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) \
        + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, map(chr, cs)))


class ClipTokenizer:
    PATTERN = regex.compile(r"""<\|startoftext\|>|<\|endoftext\|>|'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+""",
                            regex.IGNORECASE)

    def __init__(self, folder: Path = ONNX_DIR, context: int = 77):
        self.encoder = json.loads((folder / "vocab.json").read_text(encoding="utf-8"))
        merges = (folder / "merges.txt").read_text(encoding="utf-8").split("\n")[1:]
        self.ranks = {tuple(m.split()): i for i, m in enumerate(merges) if m}
        self.bos, self.eos = self.encoder["<|startoftext|>"], self.encoder["<|endoftext|>"]
        self.context = context
        self.cache: dict[str, list[str]] = {}

    def bpe(self, token: str) -> list[str]:
        if token in self.cache:
            return self.cache[token]
        word = list(token[:-1]) + [token[-1] + "</w>"]
        while len(word) > 1:
            pairs = [(self.ranks.get((a, b), math.inf), i) for i, (a, b) in enumerate(zip(word, word[1:]))]
            rank, _ = min(pairs)
            if rank == math.inf:
                break
            a, b = next((word[i], word[i + 1]) for r, i in pairs if r == rank)
            merged, i = [], 0
            while i < len(word):                 # merge every occurrence of (a, b), left to right
                if i < len(word) - 1 and word[i] == a and word[i + 1] == b:
                    merged.append(a + b)
                    i += 2
                else:
                    merged.append(word[i])
                    i += 1
            word = merged
        self.cache[token] = word
        return word

    def encode(self, text: str) -> tuple[np.ndarray, np.ndarray]:
        """(input_ids, attention_mask), both (context,) int64, padded with eos like HF."""
        text = html.unescape(html.unescape(text))
        text = regex.sub(r"\s+", " ", text).strip().lower()
        b2u = bytes_to_unicode()
        ids = []
        for tok in self.PATTERN.findall(text):
            tok = "".join(b2u[b] for b in tok.encode("utf-8"))
            ids += [self.encoder[t] for t in self.bpe(tok)]
        ids = [self.bos] + ids[: self.context - 2] + [self.eos]
        mask = [1] * len(ids) + [0] * (self.context - len(ids))
        ids += [self.eos] * (self.context - len(ids))
        return np.array(ids, np.int64), np.array(mask, np.int64)


# ---------------------------------------------------------------- generation

class OnnxBuilder:
    def __init__(self, folder: Path = ONNX_DIR, threads: int = 0):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        if threads:
            opts.intra_op_num_threads = threads
        load = lambda n: ort.InferenceSession(str(folder / n), opts, providers=["CPUExecutionProvider"])  # noqa: E731
        self.clip, self.t2v, self.decoder = load("clip_text.onnx"), load("t2v.onnx"), load("decoder.onnx")
        self.meta = json.loads((folder / "ttmc.json").read_text())
        self.tok = ClipTokenizer(folder, self.meta["context"])

    def text(self, prompt: str) -> tuple[np.ndarray, np.ndarray]:
        ids, mask = self.tok.encode(prompt)
        h = self.clip.run(None, {"input_ids": ids[None], "attention_mask": mask[None]})[0]
        return h, mask[None] == 0

    def generate(self, prompt: str, steps: int = 12, cfg: float = 3.0, temperature: float = 1.0,
                 choice_temp: float = 4.5, seed: int = 0) -> np.ndarray:
        """Returns (32, 32, 32) uint8 block tokens. Same algorithm as ttmc.t2v.generate."""
        rng = np.random.default_rng(seed)
        L, K, mask_id = self.meta["seq"], self.meta["codebook"], self.meta["mask_id"]
        h, pad = self.text(prompt)
        h0, pad0 = self.text("")
        text_h, text_pad = np.concatenate([h, h0]), np.concatenate([pad, pad0])
        tokens = np.full(L, mask_id, np.int64)
        for t in range(steps):
            logits = self.t2v.run(None, {"tokens": np.stack([tokens, tokens]), "text_h": text_h,
                                         "text_pad": text_pad})[0]
            logits = logits[1] + cfg * (logits[0] - logits[1])            # (L, K) guided
            logits = logits / temperature
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)
            # sample one code per position by inverse CDF
            u = rng.random(L)[:, None]
            sampled = np.minimum((probs.cumsum(1) < u).sum(1), K - 1)
            conf = probs[np.arange(L), sampled]
            unknown = tokens == mask_id
            sampled = np.where(unknown, sampled, tokens)
            n_mask = int(math.floor(math.cos(math.pi / 2 * (t + 1) / steps) * L))
            if n_mask == 0:
                tokens = sampled
                break
            gumbel = -np.log(-np.log(np.clip(rng.random(L), 1e-20, None)))
            score = np.log(np.clip(conf, 1e-20, None)) + choice_temp * (1 - (t + 1) / steps) * gumbel
            score = np.where(unknown, score, np.inf)
            remask = np.argsort(score, kind="stable")[:n_mask]
            tokens = sampled.copy()
            tokens[remask] = mask_id
        g = self.meta["grid"]
        return self.decoder.run(None, {"codes": tokens.reshape(1, g, g, g)})[0][0]


def main() -> None:
    import argparse
    import time
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", nargs="+")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    b = OnnxBuilder()
    tiles = []
    for p in args.prompts:
        t = time.time()
        blocks = b.generate(p, steps=args.steps, seed=args.seed)
        print(f"{p}: {int((blocks > 0).sum())} blocks in {time.time() - t:.1f} s", flush=True)
        tiles.append((blocks, p))
    if args.out:
        from ttmc.render import contact_sheet, render_iso
        from ttmc.voxelize import block_vocab
        _, vrgb = block_vocab()
        contact_sheet([([render_iso(vrgb[x].astype(np.uint8), x > 0, 200)], p) for x, p in tiles],
                      cols=4).save(args.out)


if __name__ == "__main__":
    main()
