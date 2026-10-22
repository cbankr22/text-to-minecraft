---
license: mit
tags:
  - text-to-3d
  - voxel
  - minecraft
  - flow-matching
  - diffusion
library_name: pytorch
pipeline_tag: text-to-3d
---

# TTMC 1.0: Text To Minecraft

A text-to-voxel model that turns a short prompt into a 32×32×32 build of Minecraft blocks.
It powers the TTMC Minecraft mod (Fabric, Minecraft 26.2), where it runs offline on the
player's CPU through ONNX Runtime. Code, mod and training pipeline: see the GitHub repository
`text-to-minecraft`.

## Files

| File | What |
|---|---|
| `onnx/clip_text.onnx` | CLIP ViT-B/32 text encoder (OpenAI, MIT), exported |
| `onnx/flow.onnx` | the flow transformer (211M parameters, fp32) |
| `onnx/decoder.onnx` | the tokenizer's decoder: latent → block ids |
| `onnx/ttmc.json` | settings and the block list (index = block id, 0 = air) |
| `onnx/vocab.json`, `onnx/merges.txt` | CLIP tokenizer files |
| `pytorch/flow_best.pt` | flow transformer, EMA weights (`ttmc/flow.py FlowT2V`) |
| `pytorch/tokenizer.pt` | 3D FSQ tokenizer (`ttmc/vqvae.py VoxelVQVAE`) |
| `attribution.csv` | all 527,084 training models with authors and licenses |

## Model

- **Tokenizer:** 3D convolutional autoencoder with FSQ quantization (levels 8,8,8,6,5,
  15,360 codes). 32³ blocks (one of 149 block types or air) → 8³ × 5 latent. Held-out
  reconstruction: IoU 0.771, block accuracy 75.8%.
- **Generator:** rectified-flow transformer on the tokenizer's continuous latent. DiT-style
  blocks (self-attention, cross-attention to CLIP token features, MLP), adaLN-Zero
  conditioning on the timestep and the pooled prompt, QK RMSNorm. d 768, 16 layers, 12 heads,
  211M parameters.
- **Sampling (mod defaults):** 10 Euler steps with an SD3-style timestep shift of 3,
  classifier-free guidance 7.5.

## Training

- **Data:** 527,084 Objaverse 1.0 models (CC BY 4.0: 525,015; CC0: 2,069), voxelized to 64³
  with colors mapped to the nearest Minecraft block in OKLab, downsampled to 32³, 8
  orientations each. Captions from Cap3D (ODC-By), 30% of the time replaced by a short casual
  version.
- **Compute:** one RTX 4090 for 40.8 hours, 910,001 steps, AdamW (β 0.9/0.95, LR 1.5e-4,
  warmup 3k, cosine decay over the time budget to 5%), batch 32 then 16, bf16 autocast,
  EMA 0.9999, 10% prompt dropout for guidance. Final validation loss 0.2526.

## Intended use and limitations

Made for generating builds in Minecraft from short prompts. Builds are 32³ and use full blocks
only (no stairs, slabs or block states); fine detail, rare objects and exact colors are
unreliable; interiors are not designed to be usable. The model reflects its training data:
general 3D models from Sketchfab, not Minecraft builds.

## License and attribution

MIT. The training data must be credited: see `attribution.csv` and ATTRIBUTION.md in the
GitHub repository. Objaverse (Deitke et al., 2023), Cap3D (Luo et al., 2023), CLIP
(Radford et al., 2021). Not affiliated with Mojang or Microsoft.
