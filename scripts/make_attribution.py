"""Attribution list for the objects the released weights were trained on (CC-BY requires it).

    .venv/Scripts/python scripts/make_attribution.py                 # objects in data/codes32
    .venv/Scripts/python scripts/make_attribution.py --codes data/codes32_release

Writes data/release/attribution.csv (one row per object: title, author, source link,
license) and data/release/ATTRIBUTION.md (what was used, how it was changed, the licenses,
Cap3D and Objaverse credits). Author names come from data/objaverse/annotations_slim.jsonl
(scripts/license_census.py).
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSES = {
    "by": ("CC BY 4.0", "https://creativecommons.org/licenses/by/4.0/"),
    "cc0": ("CC0 1.0", "https://creativecommons.org/publicdomain/zero/1.0/"),
    "by-sa": ("CC BY-SA 4.0", "https://creativecommons.org/licenses/by-sa/4.0/"),
    "by-nc": ("CC BY-NC 4.0", "https://creativecommons.org/licenses/by-nc/4.0/"),
    "by-nc-sa": ("CC BY-NC-SA 4.0", "https://creativecommons.org/licenses/by-nc-sa/4.0/"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--codes", default=str(ROOT / "data" / "codes32"))
    ap.add_argument("--out", default=str(ROOT / "data" / "release"))
    args = ap.parse_args()
    uids = json.loads((Path(args.codes) / "meta.json").read_text())["uids"]
    used = set(uids)
    ann = {}
    with open(ROOT / "data" / "objaverse" / "annotations_slim.jsonl", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["uid"] in used:
                ann[r["uid"]] = r
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    counts = collections.Counter()
    missing = 0
    with open(out / "attribution.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["uid", "title", "author", "author_profile", "source", "license", "license_url"])
        for uid in uids:
            a = ann.get(uid)
            if a is None:
                missing += 1
                continue
            name, url = LICENSES.get(a["license"], (a["license"], ""))
            counts[name] += 1
            profile = f"https://sketchfab.com/{a['username']}" if a.get("username") else ""
            w.writerow([uid, a.get("name") or "", a.get("author") or "", profile, a.get("url") or "",
                        name, url])
    lines = "\n".join(f"| {k} | {v:,} |" for k, v in counts.most_common())
    (out / "ATTRIBUTION.md").write_text(f"""# Attribution

The TTMC model weights were trained on {sum(counts.values()):,} 3D models from
[Objaverse 1.0](https://objaverse.allenai.org/) (Deitke et al., 2023), originally published on
[Sketchfab](https://sketchfab.com/) by their authors, with text captions from
[Cap3D](https://huggingface.co/datasets/tiange/Cap3D) (Luo et al., 2023; ODC-By 1.0).

**Changes made:** every model was converted to a grid of Minecraft blocks (voxelized at 64³,
colors mapped to the nearest block, downsampled to 32³) and used only to train the model.
No original model files are distributed.

| License | Models |
|---|---|
{lines}

Every model, with its title, author, source link and license, is listed in
[attribution.csv](attribution.csv).
""", encoding="utf-8")
    print(f"{sum(counts.values())} objects -> {out}  (no metadata for {missing})  {dict(counts)}")


if __name__ == "__main__":
    main()
