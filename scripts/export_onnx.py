"""Export the three networks to ONNX for the Minecraft mod (Java + ONNX Runtime, no Python).

    .venv/Scripts/python scripts/export_onnx.py                     # data/runs/t2v/best.pt
    .venv/Scripts/python scripts/export_onnx.py --ckpt <t2v.pt> --out data/onnx

Writes to --out (default data/onnx):
    clip_text.onnx   input_ids, attention_mask (B,77) int64  -> hidden (B,77,512) float
    t2v.onnx         tokens (B,512) int64, text_h (B,77,512), text_pad (B,77) bool -> logits (B,512,K)
    decoder.onnx     codes (B,8,8,8) int64 -> blocks (B,32,32,32) uint8 (argmax over block ids)
  or, for a flow-matching checkpoint (config kind "flow"):
    flow.onnx        latent (B,512,5), t (B,), text_h, text_pad -> velocity (B,512,5)
    decoder.onnx     latent (B,512,5) float -> blocks (B,32,32,32) uint8 (rounds to the FSQ grid first)
    vocab.json, merges.txt   CLIP BPE tokenizer files (MIT, from openai/clip-vit-base-patch32)
    ttmc.json        codebook size, mask id, sequence length, block names (index = token id)
Then checks every file against PyTorch with ONNX Runtime. Runs on CPU.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.flow import FlowT2V, latent_to_codes  # noqa: E402
from ttmc.t2v import CLIP_NAME, T2V  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

OPSET = 18


class ClipText(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state


class Decoder(nn.Module):
    """codes -> most likely block id per voxel (uint8), so Java never sees the logits."""

    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, codes):
        # FSQ.indices_to_codes with an explicit permute: movedim(-1, 1) exports a negative axis
        fsq = self.tok.fsq
        digits = (codes.unsqueeze(-1) // fsq.basis) % fsq.levels
        half = fsq.levels // 2
        q = ((digits - half).float() / half).permute(0, 4, 1, 2, 3)
        return self.tok.decode(q).argmax(1).to(torch.uint8)


class LatentDecoder(nn.Module):
    """Flow model output (B, 512, dim) -> blocks: rounds to the FSQ grid (latent_to_codes) and
    decodes, so Java passes the sampled latent straight in."""

    def __init__(self, tok):
        super().__init__()
        self.tok = tok

    def forward(self, latent):
        fsq = self.tok.fsq
        half = (fsq.levels // 2).float()
        digits = torch.minimum(torch.round(latent * half + half).clamp_min(0), (fsq.levels - 1).float())
        q = ((digits - half) / half).reshape(-1, 8, 8, 8, latent.shape[-1]).permute(0, 4, 1, 2, 3)
        return self.tok.decode(q).argmax(1).to(torch.uint8)


def export(model: nn.Module, args: tuple, path: Path, inputs: list[str], outputs: list[str]) -> None:
    dyn = {n: {0: "batch"} for n in inputs + outputs}
    torch.onnx.export(model, args, str(path), input_names=inputs, output_names=outputs,
                      dynamic_axes=dyn, opset_version=OPSET, dynamo=False)
    print(f"{path.name}: {path.stat().st_size / 1e6:.0f} MB", flush=True)


def check(path: Path, feeds: dict, expected: np.ndarray, name: str, tol: float = 1e-3) -> None:
    import onnxruntime as ort
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    got = sess.run(None, feeds)[0]
    if expected.dtype == np.uint8:
        err = float((got != expected).mean())
        print(f"  check {name}: {err:.4%} voxels differ")
        assert err < 0.01, name
    else:
        err = float(np.abs(got - expected).max())
        print(f"  check {name}: max abs diff {err:.2e}")
        assert err < tol, name


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(ROOT / "data" / "runs" / "t2v" / "best.pt"))
    ap.add_argument("--out", default=str(ROOT / "data" / "onnx"))
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)
    # the fused attention kernel PyTorch uses in eval mode has no ONNX equivalent
    torch.backends.mha.set_fastpath_enabled(False)

    from transformers import CLIPTextModel, CLIPTokenizer
    clip_tok = CLIPTokenizer.from_pretrained(CLIP_NAME)
    clip = ClipText(CLIPTextModel.from_pretrained(CLIP_NAME).eval())
    ck = torch.load(args.ckpt, map_location="cpu")
    cfg = ck["config"]
    flow = cfg.get("kind") == "flow"
    tok = load_tokenizer(cfg["tokenizer"], device="cpu").eval()
    if flow:
        net = FlowT2V(latent_dim=cfg["latent_dim"], d=cfg["d"], layers=cfg["layers"], heads=cfg["heads"]).eval()
        net.load_state_dict(ck["ema"])
        dec = LatentDecoder(tok).eval()
    else:
        net = T2V(cfg["codebook"], d=cfg["d"], layers=cfg["layers"]).eval()
        net.load_state_dict(ck["model"])
        dec = Decoder(tok).eval()

    x = clip_tok(["a wooden chair", ""], padding="max_length", max_length=77, truncation=True,
                 return_tensors="pt")
    ids, am = x["input_ids"], x["attention_mask"]
    with torch.no_grad():
        h = clip(ids, am)
        pad = am == 0
        export(clip, (ids, am), out / "clip_text.onnx", ["input_ids", "attention_mask"], ["hidden"])
        if flow:
            latent = torch.randn(2, net.seq, net.latent_dim)
            t = torch.tensor([0.7, 0.2])
            gen_out = net(latent, t, h, pad)
            gen_feeds = {"latent": latent.numpy(), "t": t.numpy(), "text_h": h.numpy(), "text_pad": pad.numpy()}
            export(net, (latent, t, h, pad), out / "flow.onnx", ["latent", "t", "text_h", "text_pad"], ["velocity"])
            dec_in = (torch.randn(2, net.seq, net.latent_dim) * 0.6).clamp(-1, 1)
            blocks = dec(dec_in)
            # same blocks as the PyTorch path: latent_to_codes, then the tokenizer's own decode
            ref = tok.decode_indices(latent_to_codes(dec_in, tok.fsq).view(-1, 8, 8, 8)).argmax(1)
            assert torch.equal(blocks.long(), ref), "LatentDecoder differs from latent_to_codes + decode_indices"
            export(dec, (dec_in,), out / "decoder.onnx", ["latent"], ["blocks"])
            dec_feeds = {"latent": dec_in.numpy()}
        else:
            tokens = torch.randint(0, cfg["codebook"] + 1, (2, net.seq))
            gen_out = net(tokens, h, pad)
            gen_feeds = {"tokens": tokens.numpy(), "text_h": h.numpy(), "text_pad": pad.numpy()}
            export(net, (tokens, h, pad), out / "t2v.onnx", ["tokens", "text_h", "text_pad"], ["logits"])
            codes = torch.randint(0, cfg["codebook"], (2, 8, 8, 8))
            blocks = dec(codes)
            export(dec, (codes,), out / "decoder.onnx", ["codes"], ["blocks"])
            dec_feeds = {"codes": codes.numpy()}

    from huggingface_hub import hf_hub_download
    for f in ("vocab.json", "merges.txt"):
        shutil.copy(hf_hub_download(CLIP_NAME, f), out / f)
    names, _ = block_vocab()
    meta = {"kind": "flow" if flow else "maskgit", "seq": net.seq, "grid": 8, "res": 32,
            "context": 77, "bos": clip_tok.bos_token_id, "eos": clip_tok.eos_token_id,
            "pad": clip_tok.pad_token_id, "blocks": list(names),          # index = token, 0 = air
            "step": ck["step"]}
    if flow:   # sampling defaults from scripts/tune_flow.py (2026-10-21)
        meta.update(latent_dim=net.latent_dim, steps=10, cfg=7.5, shift=3.0)
    else:
        meta.update(codebook=cfg["codebook"], mask_id=net.mask_id)
    (out / "ttmc.json").write_text(json.dumps(meta, indent=1))

    check(out / "clip_text.onnx", {"input_ids": ids.numpy(), "attention_mask": am.numpy()},
          h.numpy(), "clip_text")
    check(out / ("flow.onnx" if flow else "t2v.onnx"), gen_feeds, gen_out.numpy(), "generator", tol=5e-3)
    check(out / "decoder.onnx", dec_feeds, blocks.numpy(), "decoder")
    print(f"exported to {out}")


if __name__ == "__main__":
    main()
