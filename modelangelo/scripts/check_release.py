"""CryoAtlas addition: release integrity and input-manifest checks.

Checks source syntax, portable paths, required inference files, file hashes,
and optional train/validation path and entry-ID separation.
"""
import argparse
import ast
import hashlib
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def read_manifest(path):
    paths = [Path(line.strip()).resolve() for line in path.read_text().splitlines() if line.strip()]
    if not paths:
        raise ValueError(f"empty manifest: {path}")
    if len(paths) != len(set(paths)):
        raise ValueError(f"duplicate paths in {path}")
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise ValueError(f"missing inputs: {missing[:3]}")
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-list", type=Path)
    parser.add_argument("--val-list", type=Path)
    args = parser.parse_args()
    sources = sorted((ROOT / "model_angelo").rglob("*.py"))
    for source in sources:
        text = source.read_text(encoding="utf-8")
        ast.parse(text, filename=str(source))
        if any(token in text for token in ("/2023533011/", "/defaultShare/", "wandb_v1_")):
            raise ValueError(f"private configuration in {source}")
    required = ["apps/build.py", "apps/build_no_seq.py", "apps/refine.py", "gnn/inference.py",
                "gnn/inference_no_seq.py", "gnn/flood_fill.py", "c_alpha/diag_infer.py"]
    for name in required:
        assert (ROOT / "model_angelo" / name).is_file(), name
    assert not (ROOT / "model_angelo/gnn/train.py").exists()
    if bool(args.train_list) != bool(args.val_list):
        parser.error("provide both manifest arguments")
    if args.train_list:
        train = read_manifest(args.train_list)
        val = read_manifest(args.val_list)
        overlap = {p.stem for p in train} & {p.stem for p in val}
        if overlap:
            raise ValueError(f"train/validation entry overlap: {len(overlap)}")
        print(f"Independent paths and entry IDs: train={len(train)}, validation={len(val)}")
    checksums = ROOT / "SHA256SUMS"
    if checksums.is_file():
        for line in checksums.read_text().splitlines():
            digest, name = line.split("  ", 1)
            assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    print(f"Source checks passed: {len(sources)} Python modules")


if __name__ == "__main__":
    main()
