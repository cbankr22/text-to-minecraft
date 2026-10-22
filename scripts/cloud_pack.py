"""Bundle what a rented GPU needs to train the text -> voxel transformer.

    .venv/Scripts/python scripts/cloud_pack.py                       # -> data/cloud_bundle.tar
    .venv/Scripts/python scripts/cloud_pack.py --with-t2v            # also continue from t2v/last.pt
    .venv/Scripts/python scripts/cloud_pack.py --hf-repo NAME/ttmc-data   # upload to a private HF repo

The bundle holds the code, data/codes32 (codes.npy + meta.json) and the tokenizer checkpoint
named in meta.json; no raw voxels. On the rented machine run scripts/cloud_setup.sh.
"""
from __future__ import annotations

import argparse
import json
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ttmc.vqvae import project_path  # noqa: E402


def bundle_files(with_t2v: bool) -> list[Path]:
    meta = json.loads((ROOT / "data" / "codes32" / "meta.json").read_text())
    files = [ROOT / "requirements.txt", ROOT / "CLOUD_PLAN.md"]
    files += sorted((ROOT / "ttmc").glob("*.py")) + [ROOT / "ttmc" / "blocks.json"]
    files += sorted((ROOT / "scripts").glob("*.py")) + sorted((ROOT / "scripts").glob("*.sh"))
    files += [ROOT / "data" / "codes32" / "codes.npy", ROOT / "data" / "codes32" / "meta.json",
              project_path(meta["tokenizer"])]
    if with_t2v:
        files.append(ROOT / "data" / "runs" / "t2v" / "last.pt")
    missing = [f for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    return files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-t2v", action="store_true", help="include data/runs/t2v/last.pt")
    ap.add_argument("--out", default=str(ROOT / "data" / "cloud_bundle.tar"))
    ap.add_argument("--hf-repo", default="", help="upload files to this private HF dataset repo")
    args = ap.parse_args()

    files = bundle_files(args.with_t2v)
    gb = sum(f.stat().st_size for f in files) / 1e9
    print(f"{len(files)} files, {gb:.2f} GB")
    rel = [f.relative_to(ROOT).as_posix() for f in files]

    if args.hf_repo:
        from huggingface_hub import HfApi
        api = HfApi()
        api.create_repo(args.hf_repo, repo_type="dataset", private=True, exist_ok=True)
        for f, r in zip(files, rel):
            print("upload", r, flush=True)
            api.upload_file(path_or_fileobj=str(f), path_in_repo=r, repo_id=args.hf_repo,
                            repo_type="dataset")
        print(f"done: on the GPU box run  bash scripts/cloud_setup.sh hf:{args.hf_repo}")
        return

    with tarfile.open(args.out, "w") as tar:          # codes are uint16 and barely compress
        for f, r in zip(files, rel):
            tar.add(f, arcname=r)
    print(f"wrote {args.out}\ncopy it to the GPU box (scp / runpodctl send), then:\n"
          f"  tar xf cloud_bundle.tar && bash scripts/cloud_setup.sh")


if __name__ == "__main__":
    main()
