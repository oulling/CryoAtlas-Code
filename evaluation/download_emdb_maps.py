"""Download CryoAtlas density maps from the official EMDB archive.

The manifest is the `metadata/source_accessions.csv` file distributed with
CryoAtlas. Re-running the command is safe: complete existing files are skipped.

Examples
--------
Download one map into data/EMD-14075/:

    python scripts/download_emdb_maps.py \
        --manifest /path/to/CryoAtlas/metadata/source_accessions.csv \
        --out data --emd-id EMD-14075

Download all maps into one directory:

    python scripts/download_emdb_maps.py \
        --manifest /path/to/CryoAtlas/metadata/source_accessions.csv \
        --out emdb_maps --layout flat
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import requests
from tqdm import tqdm


def normalize_emd_id(value: str) -> str:
    value = value.strip().upper()
    if not value.startswith("EMD-"):
        value = "EMD-" + value
    return value


def download(url: str, output_path: Path, timeout: int) -> bool:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")

    try:
        with requests.get(url, stream=True, timeout=timeout) as response:
            response.raise_for_status()
            with temporary_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        os.replace(temporary_path, output_path)
        return True
    except Exception as error:
        if temporary_path.exists():
            temporary_path.unlink()
        print(f"FAIL {output_path.name}: {error}", file=sys.stderr)
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--manifest",
        required=True,
        help="path to the released metadata/source_accessions.csv",
    )
    parser.add_argument(
        "--out",
        default="data",
        help="output root directory (default: data)",
    )
    parser.add_argument(
        "--emd-id",
        action="append",
        default=[],
        help="download only this EMD accession; repeat for multiple entries",
    )
    parser.add_argument(
        "--split",
        choices=("all", "train", "test"),
        default="all",
        help="download all entries or only one released split (default: all)",
    )
    parser.add_argument(
        "--layout",
        choices=("emd", "flat", "split"),
        default="emd",
        help=(
            "emd: OUT/EMD-XXXX/file; flat: OUT/file; "
            "split: OUT/{train,test}/EMD-XXXX/file (default: emd)"
        ),
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=300,
        help="per-request timeout in seconds (default: 300)",
    )
    return parser.parse_args()


def output_path(root: Path, row: dict, layout: str) -> Path:
    filename = row["map_filename"].strip()
    emd_id = normalize_emd_id(row["emd_id"])
    if layout == "flat":
        return root / filename
    if layout == "split":
        return root / row["split"].strip().lower() / emd_id / filename
    return root / emd_id / filename


def main() -> int:
    args = parse_args()
    requested_ids = {normalize_emd_id(value) for value in args.emd_id}

    with open(args.manifest, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    required_columns = {
        "emd_id",
        "split",
        "emdb_map_url",
        "map_filename",
    }
    if not rows or not required_columns.issubset(rows[0]):
        missing = required_columns - (set(rows[0]) if rows else set())
        raise ValueError(f"manifest is empty or missing columns: {sorted(missing)}")

    selected = []
    for row in rows:
        emd_id = normalize_emd_id(row["emd_id"])
        if requested_ids and emd_id not in requested_ids:
            continue
        if args.split != "all" and row["split"].strip().lower() != args.split:
            continue
        selected.append(row)

    if requested_ids:
        found = {normalize_emd_id(row["emd_id"]) for row in selected}
        missing_ids = requested_ids - found
        if missing_ids:
            raise ValueError(
                "EMD accession(s) not found in manifest: "
                + ", ".join(sorted(missing_ids))
            )

    root = Path(args.out)
    fetched = skipped = failed = 0
    for row in tqdm(selected, desc="EMDB maps"):
        destination = output_path(root, row, args.layout)
        if destination.exists() and destination.stat().st_size > 0:
            skipped += 1
            continue
        if download(row["emdb_map_url"].strip(), destination, args.timeout):
            fetched += 1
        else:
            failed += 1

    print(
        f"fetched={fetched} skipped={skipped} "
        f"failed={failed} selected={len(selected)}"
    )
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
