"""Mesh (glb/obj/...) -> colored voxel shell -> Minecraft block grid.

Grids are (Y, Z, X) with Y up, matching ttmc.formats.Build. Token 0 is air;
token i>0 is blocks.json["blocks"][i-1].
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import trimesh

from .block_colors import linear_to_srgb, srgb_to_linear

BLOCKS_JSON = Path(__file__).with_name("blocks.json")
DEFAULT_RGB = np.array([180, 180, 180], dtype=np.float64)   # untextured surfaces


@lru_cache(maxsize=1)
def block_vocab() -> tuple[list[str], np.ndarray]:
    """(names incl. air at 0, (N,3) sRGB colors incl. a dummy for air)."""
    data = json.loads(BLOCKS_JSON.read_text())["blocks"]
    names = ["minecraft:air"] + [b["name"] for b in data]
    rgb = np.array([[0, 0, 0]] + [b["rgb"] for b in data], dtype=np.float64)
    return names, rgb


# --------------------------------------------------------------------------- #
# Color science: nearest block in OKLab (CIELAB shifts saturated blues to purple)
# --------------------------------------------------------------------------- #

def srgb_to_oklab(rgb: np.ndarray) -> np.ndarray:
    lin = srgb_to_linear(rgb)
    m1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                   [0.2119034982, 0.6806995451, 0.1073969566],
                   [0.0883024619, 0.2817188376, 0.6299787005]])
    m2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                   [1.9779984951, -2.4285922050, 0.4505937099],
                   [0.0259040371, 0.7827717662, -0.8086757660]])
    return np.cbrt(lin @ m1.T) @ m2.T


def nearest_blocks(rgb: np.ndarray) -> np.ndarray:
    """(N,3) sRGB -> (N,) block tokens (>=1)."""
    _, vocab_rgb = block_vocab()
    lab = srgb_to_oklab(rgb)
    vocab_lab = srgb_to_oklab(vocab_rgb[1:])
    d = ((lab[:, None, :] - vocab_lab[None, :, :]) ** 2).sum(-1)
    return d.argmin(1).astype(np.uint8) + 1


# --------------------------------------------------------------------------- #
# Surface color sampling
# --------------------------------------------------------------------------- #

def _sample_colors(mesh: trimesh.Trimesh, face_index: np.ndarray, bary: np.ndarray) -> np.ndarray:
    """RGBA (0-255 float) at the sampled surface points."""
    n = len(face_index)
    out = np.empty((n, 4))
    out[:, :3], out[:, 3] = DEFAULT_RGB, 255
    vis = mesh.visual
    try:
        if vis.kind == "texture":
            mat = vis.material
            factor = getattr(mat, "baseColorFactor", None)
            if factor is None:
                factor = getattr(mat, "diffuse", None)
            img = getattr(mat, "baseColorTexture", None) or getattr(mat, "image", None)
            if img is not None and vis.uv is not None and len(vis.uv) == len(mesh.vertices):
                tex = np.asarray(img.convert("RGBA"), dtype=np.float64)
                h, w = tex.shape[:2]
                uv = (vis.uv[mesh.faces[face_index]] * bary[:, :, None]).sum(1) % 1.0
                px = np.clip((uv[:, 0] * w).astype(int), 0, w - 1)
                py = np.clip(((1 - uv[:, 1]) * h).astype(int), 0, h - 1)
                out = tex[py, px]
            if factor is not None:
                f = np.asarray(factor, dtype=np.float64).ravel()
                f = np.pad(f, (0, 4 - len(f)), constant_values=255)[:4]
                out = out * (f / 255.0) if img is not None else np.broadcast_to(f, (n, 4)).copy()
        elif vis.kind in ("vertex", "face"):
            out = np.asarray(vis.face_colors[face_index], dtype=np.float64)
    except Exception:
        pass
    return out


# --------------------------------------------------------------------------- #
# Voxelization
# --------------------------------------------------------------------------- #

@dataclass
class VoxelResult:
    blocks: np.ndarray        # (R,R,R) uint8 tokens
    rgb: np.ndarray           # (R,R,R,3) uint8 mean surface color (0 where empty)
    n_faces: int
    extent: tuple[float, float, float]


def load_meshes(path: Path) -> list[trimesh.Trimesh]:
    loaded = trimesh.load(str(path), force="scene", process=False)
    return [g for g in loaded.dump(concatenate=False)
            if isinstance(g, trimesh.Trimesh) and len(g.faces)]


def voxelize(path: Path, res: int = 64, samples_per_voxel_area: float = 6.0,
             max_samples: int = 4_000_000, min_alpha: float = 128) -> VoxelResult:
    meshes = load_meshes(path)
    if not meshes:
        raise ValueError("no triangle geometry")
    verts = np.concatenate([m.vertices for m in meshes])
    lo, hi = verts.min(0), verts.max(0)
    extent = hi - lo
    span = float(extent.max())
    if span <= 0:
        raise ValueError("degenerate bounds")
    scale = res / span
    # Center X/Z, sit on the floor in Y.
    offset = (res - extent * scale) / 2
    offset[1] = 0

    areas = np.array([m.area for m in meshes]) * scale ** 2
    budget = np.minimum(max_samples, areas.sum() * samples_per_voxel_area)
    counts = np.maximum(64, (areas / max(areas.sum(), 1e-9) * budget).astype(int))

    flat_idx, colors = [], []
    for mesh, count in zip(meshes, counts):
        pts, face_idx, bary = _sample_with_bary(mesh, int(count))
        rgba = _sample_colors(mesh, face_idx, bary)
        keep = _alpha_keep(mesh, rgba[:, 3], min_alpha)
        g = np.floor((pts[keep] - lo) * scale + offset).astype(int)
        g = np.clip(g, 0, res - 1)
        # glTF is Y-up: grid index order (y, z, x)
        flat_idx.append((g[:, 1] * res + g[:, 2]) * res + g[:, 0])
        colors.append(srgb_to_linear(rgba[keep, :3]))
    flat_idx = np.concatenate(flat_idx)
    lin = np.concatenate(colors)

    n_cells = res ** 3
    counts = np.bincount(flat_idx, minlength=n_cells)
    occupied = counts > 0
    mean = np.zeros((n_cells, 3))
    for c in range(3):
        mean[:, c] = np.bincount(flat_idx, weights=lin[:, c], minlength=n_cells)
    mean[occupied] /= counts[occupied, None]
    srgb = linear_to_srgb(mean)

    blocks = np.zeros(n_cells, dtype=np.uint8)
    blocks[occupied] = nearest_blocks(srgb[occupied])
    rgb = np.where(occupied[:, None], srgb, 0).astype(np.uint8)
    return VoxelResult(blocks.reshape(res, res, res), rgb.reshape(res, res, res, 3),
                       int(sum(len(m.faces) for m in meshes)), tuple(extent.tolist()))


def _alpha_keep(mesh: trimesh.Trimesh, alpha: np.ndarray, min_alpha: float) -> np.ndarray:
    """glTF semantics: alpha only matters for MASK/BLEND materials; OPAQUE ignores it."""
    mat = getattr(mesh.visual, "material", None)
    mode = str(getattr(mat, "alphaMode", None) or "OPAQUE").upper()
    if mode == "MASK":
        cutoff = getattr(mat, "alphaCutoff", None)
        return alpha >= (255 * cutoff if cutoff is not None else 127.5)
    if mode == "BLEND":
        return alpha >= min_alpha
    return np.ones(len(alpha), dtype=bool)


def _sample_with_bary(mesh: trimesh.Trimesh, count: int):
    pts, face_idx = trimesh.sample.sample_surface(mesh, count)
    tri = mesh.triangles[face_idx]
    bary = trimesh.triangles.points_to_barycentric(tri, pts)
    return pts, face_idx, np.nan_to_num(bary, nan=1 / 3)
