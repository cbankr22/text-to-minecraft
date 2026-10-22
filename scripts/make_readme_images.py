"""Images for README.md -> docs/images/.

    .venv/Scripts/python scripts/make_readme_images.py [progression gallery compare sketches]

progression.gif  six preview prompts (seed 0) from data/runs/flow_big/samples_*.png across training
gallery.png      the final model on a mix of prompts, seed 0, uncurated (mod settings: 10 steps,
                 cfg 7.5, shift 3)
compare.png      the same prompts and seed through the old 55M MaskGIT, the 175M MaskGIT and the
                 final flow model
sketch.png       gray-wool sketches (scripts/sketch_build.py) and the build made from each, strength 0.8
Needs the GPU for a minute or two.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from ttmc.flow import FlowT2V, latent_to_codes, sample  # noqa: E402
from ttmc.render import render_iso  # noqa: E402
from ttmc.t2v import T2V, TextEncoder, generate  # noqa: E402
from ttmc.voxelize import block_vocab  # noqa: E402
from ttmc.vqvae import load_tokenizer  # noqa: E402

OUT = ROOT / "docs" / "images"
RUN = ROOT / "data" / "runs" / "flow_big"
BG, FG, DIM = (22, 24, 29), (235, 235, 235), (150, 155, 165)
STEPS, CFG, SHIFT = 10, 7.5, 3.0
_, VRGB = block_vocab()


def font(size: int):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def render(x: np.ndarray, size: int = 300) -> Image.Image:
    return render_iso(VRGB[x].astype(np.uint8), x > 0, size)


def grid(images: list[Image.Image], labels: list[str], cols: int, size: int, title: str = "",
         label_h: int = 34) -> Image.Image:
    rows = -(-len(images) // cols)
    top = 56 if title else 0
    sheet = Image.new("RGB", (cols * size, top + rows * (size + label_h)), BG)
    d = ImageDraw.Draw(sheet)
    if title:
        d.text((16, 16), title, fill=FG, font=font(24))
    for i, (im, lab) in enumerate(zip(images, labels)):
        x, y = (i % cols) * size, top + (i // cols) * (size + label_h)
        sheet.paste(im.resize((size, size)), (x, y))
        d.text((x + size // 2, y + size + 6), lab, fill=DIM, font=font(17), anchor="ma")
    return sheet


def progression() -> None:
    picks = {2: "small house, red roof", 3: "stone castle", 4: "red sports car",
             5: "sailing ship", 6: "turtle", 7: "dragon"}
    sheets = sorted(RUN.glob("samples_*.png"), key=lambda p: int(re.findall(r"\d+", p.stem)[0]))
    want = [10_000, 20_000, 30_000, 50_000, 70_000, 100_000, 150_000, 200_000, 300_000, 400_000,
            500_000, 600_000, 700_000, 800_000, 910_001]
    by_step = {int(re.findall(r"\d+", p.stem)[0]): p for p in sheets}
    frames = []
    for s in want:
        if s not in by_step:
            continue
        src = Image.open(by_step[s]).convert("RGB")
        # contact_sheet layout: 3 columns of (2 views x 256 px) + 48 px caption, 6 px gaps
        tiles = [src.crop(((i % 3) * 518, (i // 3) * 310, (i % 3) * 518 + 256, (i // 3) * 310 + 256))
                 for i in picks]
        frame = grid(tiles, list(picks.values()), cols=3, size=256,
                     title=f"training step {s:,}  ({s / 910_001:.0%})")
        frames.append(frame.convert("P", palette=Image.ADAPTIVE, colors=255))
    durations = [600] * (len(frames) - 1) + [3000]
    frames[0].save(OUT / "progression.gif", save_all=True, append_images=frames[1:], duration=durations, loop=0)
    print("progression.gif", len(frames), "frames")


def load_flow():
    ck = torch.load(RUN / "best.pt", map_location="cuda")
    c = ck["config"]
    m = FlowT2V(latent_dim=c["latent_dim"], d=c["d"], layers=c["layers"], heads=c["heads"]).cuda().eval()
    m.load_state_dict(ck["ema"])
    return m, load_tokenizer(c["tokenizer"])


def flow_builds(model, tok, text, prompts, seed=0) -> np.ndarray:
    out = []
    for i in range(0, len(prompts), 8):
        z = sample(model, text, prompts[i:i + 8], steps=STEPS, cfg=CFG, seed=seed, shift=SHIFT)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            out.append(tok.decode_indices(latent_to_codes(z, tok.fsq).view(-1, 8, 8, 8)).argmax(1).cpu().numpy())
    return np.concatenate(out)


def gallery(model, tok, text) -> None:
    prompts = ["a medieval castle", "a mushroom house", "a red sports car", "a pirate ship",
               "a lighthouse", "a wooden chair", "a turtle", "a dragon", "a palm tree",
               "a treasure chest", "a windmill", "a pumpkin"]
    b = flow_builds(model, tok, text, prompts)
    grid([render(x) for x in b], [p[2:] if p.startswith("a ") else p for p in prompts], cols=4, size=300,
         title="TTMC 1.0: one try per prompt (seed 0, not cherry-picked)").save(OUT / "gallery.png")
    print("gallery.png")


def compare(model, tok, text) -> None:
    prompts = ["a small house with a red roof", "a coffee mug", "a turtle", "a sailing ship"]
    rows = []
    old = []
    for ck_path in (ROOT / "data/runs/t2v/best.pt", ROOT / "data/runs/t2v_big/best.pt"):
        ck = torch.load(ck_path, map_location="cuda")
        c = ck["config"]
        m = T2V(c["codebook"], d=c["d"], layers=c["layers"]).cuda().eval()
        m.load_state_dict(ck["model"])
        t = load_tokenizer(c["tokenizer"])
        codes = generate(m, text, prompts, steps=32, cfg=7.0, temperature=0.8, choice_temp=2.0, seed=0)
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            old.append(t.decode_indices(codes.view(-1, 8, 8, 8)).argmax(1).cpu().numpy())
        del m, t, ck
        torch.cuda.empty_cache()
    new = flow_builds(model, tok, text, prompts)
    names = ["v0 (MaskGIT 55M)", "v0.5 (MaskGIT 175M)", "1.0 (flow 211M)"]
    images, labels = [], []
    for i, p in enumerate(prompts):
        for name, b in zip(names, old + [new]):
            images.append(render(b[i]))
            labels.append(f"{'red-roof house' if 'house' in p else p[2:]}: {name}")
    sheet = grid(images, labels, cols=3, size=300, title="Same prompt, same seed, three versions of the model")
    sheet.save(OUT / "compare.png")
    print("compare.png")


def sketches(model, tok, text) -> None:
    import sketch_build as sb
    picks = [(sb.blob_animal, "a cow"), (sb.blob_animal, "a dragon"), (sb.mushroom, "a mushroom house"),
             (sb.church, "a japanese temple"), (sb.arch_bridge, "a stone bridge"), (sb.ship, "a pirate ship")]
    sk = np.stack([f() for f, _ in picks])
    prompts = [p for _, p in picks]
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        q, _ = tok.encode(torch.from_numpy(sk).long().cuda())
    init = q.float().permute(0, 2, 3, 4, 1).reshape(len(q), -1, q.shape[1])
    z = sample(model, text, prompts, steps=20, cfg=CFG, seed=0, shift=SHIFT, init=init, strength=0.8)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        b = tok.decode_indices(latent_to_codes(z, tok.fsq).view(-1, 8, 8, 8)).argmax(1).cpu().numpy()
    light = VRGB.copy()
    light[sb.WOOL] = (170, 172, 178)                    # show the sketch lighter than gray wool's real color
    images, labels = [], []
    for i, p in enumerate(prompts):
        images += [render_iso(light[sk[i]].astype(np.uint8), sk[i] > 0, 260), render(b[i], 260)]
        labels += ["sketch", f'"{p[2:] if p.startswith("a ") else p}"']
    grid(images, labels, cols=4, size=260, title="Sketch to build: a rough shape + a prompt (experimental)").save(OUT / "sketch.png")
    print("sketch.png")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    only = sys.argv[1:] or ["progression", "gallery", "compare", "sketches"]
    if "progression" in only:
        progression()
    model, tok = load_flow()
    text = TextEncoder()
    for name in ("gallery", "compare", "sketches"):
        if name in only:
            globals()[name](model, tok, text)


if __name__ == "__main__":
    main()
