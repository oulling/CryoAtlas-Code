"""CryoAtlas addition: prepare chain-aware CNN training pickles.

Reads metadata and map/mmCIF pairs, corrects MRC axes/origins, resamples and
normalizes maps, and writes polymer anchors, manifests, and failure records.
"""

from __future__ import annotations

import argparse
import csv
import json
import multiprocessing as mp
import os
import pickle
import random
import traceback
from pathlib import Path

import numpy as np
import mrcfile
from Bio.PDB import MMCIFParser
from Bio.PDB.Atom import DisorderedAtom
from Bio.PDB.Polypeptide import is_aa, is_nucleic

from model_angelo.data.dataset_preprocess import compress_data
from model_angelo.utils.grid import (
    get_spectral_indices,
    make_model_angelo_grid,
    spectral_amplitude,
)


def atom_coord(atom):
    if isinstance(atom, DisorderedAtom):
        atom = atom.disordered_get_list()[0]
    occupancy = atom.get_occupancy()
    if occupancy is not None and occupancy <= 0:
        return None
    return atom.get_vector().get_array().astype(np.float32)


def polymer_anchors(cif_path: Path):
    """Return chain-safe CA/P anchors; exclude ions and zero-occupancy atoms."""
    model = MMCIFParser(QUIET=True).get_structure("structure", str(cif_path))[0]
    chains = []
    for chain in model:
        residues = list(chain)
        protein = [r for r in residues if is_aa(r, standard=False)]
        nucleic = [r for r in residues if is_nucleic(r, standard=False)]
        if protein:
            coords = [atom_coord(r["CA"]) for r in protein if "CA" in r]
            coords = [x for x in coords if x is not None]
            if coords:
                chains.append({"chain_id": chain.id, "kind": "protein",
                               "coords": np.asarray(coords, dtype=np.float32)})
        if nucleic:
            coords = [atom_coord(r["P"]) for r in nucleic if "P" in r]
            coords = [x for x in coords if x is not None]
            if coords:
                names = {r.get_resname().strip().upper() for r in nucleic}
                kind = "dna" if any(n.startswith("D") or n == "T" for n in names) else "rna"
                chains.append({"chain_id": chain.id, "kind": kind,
                               "coords": np.asarray(coords, dtype=np.float32)})
    return chains


def normalize(grid):
    mean, std = float(grid.mean()), float(grid.std())
    return ((grid - mean) / max(std, 1e-6)).astype(np.float32)


def load_mrc_axis_corrected(path):
    """Load an MRC grid in physical ZYX order with an XYZ origin in Angstrom."""
    with mrcfile.open(path, permissive=True) as mrc:
        labels = [int(mrc.header.maps), int(mrc.header.mapr), int(mrc.header.mapc)]
        if sorted(labels) != [1, 2, 3]:
            raise ValueError(f"invalid MAPC/MAPR/MAPS: {labels[::-1]}")
        order = [labels.index(axis) for axis in (3, 2, 1)]
        grid = np.transpose(np.asarray(mrc.data), order).copy()
        voxel = np.array(
            [mrc.voxel_size.x, mrc.voxel_size.y, mrc.voxel_size.z], dtype=np.float32
        )
        explicit = np.array(
            [mrc.header.origin.x, mrc.header.origin.y, mrc.header.origin.z],
            dtype=np.float32,
        )
        if np.any(np.abs(explicit) > 1e-6):
            origin = explicit
        else:
            starts = [
                int(mrc.header.nzstart),
                int(mrc.header.nystart),
                int(mrc.header.nxstart),
            ]
            origin = np.array(
                [starts[labels.index(axis)] * voxel[axis - 1] for axis in (1, 2, 3)],
                dtype=np.float32,
            )
    return grid, float(voxel[0]), origin


def process_one(task):
    row, dataset_root, output_dir, target_voxel = task
    emdb_id = row["emdb_id"]
    try:
        entry = Path(dataset_root) / emdb_id
        map_path = entry / row["map_filename"]
        cif_path = entry / row["primary_cif_filename"]
        split_dir = Path(output_dir) / row["split"]
        split_dir.mkdir(parents=True, exist_ok=True)
        output = split_dir / f"{emdb_id}.pkl"
        if output.is_file():
            with output.open("rb") as handle:
                existing = pickle.load(handle)
            return {"emdb_id": emdb_id, "split": row["split"], "path": str(output),
                    "ca_count": len(existing["cas"]["CA"]),
                    "p_count": len(existing["cas"]["P"]), "error": None}
        chains = polymer_anchors(cif_path)
        ca = [c["coords"] for c in chains if c["kind"] == "protein"]
        p = [c["coords"] for c in chains if c["kind"] in {"rna", "dna"}]
        ca = np.concatenate(ca) if ca else np.zeros((0, 3), dtype=np.float32)
        p = np.concatenate(p) if p else np.zeros((0, 3), dtype=np.float32)
        if len(ca) + len(p) < 5:
            raise ValueError("fewer than five positive-occupancy polymer anchors")

        grid, voxel_size, origin = load_mrc_axis_corrected(str(map_path))
        grid, voxel_size, origin = make_model_angelo_grid(
            np.asarray(grid, dtype=np.float32), voxel_size, origin,
            target_voxel_size=target_voxel,
        )
        grid = normalize(grid)
        grid_ft = np.fft.rfftn(grid)
        radius = get_spectral_indices(grid_ft)
        payload = {
            "grid_io": compress_data(grid),
            "grid_spectral_amplitude": spectral_amplitude(grid_ft, radius).astype(np.float32),
            "voxel_size": float(voxel_size),
            "global_origin": np.asarray(origin, dtype=np.float32),
            "cella": (np.asarray(grid.shape) * voxel_size).astype(np.float32),
            "shape": np.asarray(grid.shape, dtype=np.int64),
            "cas": {"CA": ca, "P": p},
            "cas_chains": chains,
            "source_map": str(map_path),
            "source_struct": str(cif_path),
            "struct_is_aligned": "_aligned.cif" in cif_path.name,
            "grid_is_global_normalized": True,
            "sample_type": row["sample_type"],
            "resolution_angstrom": float(row["resolution_angstrom"]),
        }
        temporary = output.with_suffix(".pkl.tmp")
        with temporary.open("wb") as handle:
            pickle.dump(payload, handle, protocol=4)
        os.replace(temporary, output)
        return {"emdb_id": emdb_id, "split": row["split"], "path": str(output),
                "ca_count": len(ca), "p_count": len(p), "error": None}
    except Exception:
        return {"emdb_id": emdb_id, "split": row["split"], "path": None,
                "ca_count": 0, "p_count": 0, "error": traceback.format_exc()}


def select_rows(args):
    with open(args.metadata_csv, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    eligible = []
    for row in rows:
        entry = Path(args.dataset_root) / row["emdb_id"]
        try:
            size = int(row["uncompressed_map_bytes"])
        except ValueError:
            continue
        if (row["final_status"] == "kept" and size <= args.max_map_mb * (1 << 20)
                and (entry / row["map_filename"]).is_file()
                and (entry / row["primary_cif_filename"]).is_file()):
            eligible.append(row)

    rng = random.Random(args.seed)
    if args.all_entries:
        eligible.sort(key=lambda r: int(r["emdb_id"].split("-")[1]))
        for row in eligible:
            row["split"] = "train"
        return eligible, {"eligible": len(eligible),
                          "eligible_nucleic": sum("rna" in r["sample_type"] or "dna" in r["sample_type"] for r in eligible),
                          "selected": len(eligible), "selected_nucleic": sum("rna" in r["sample_type"] or "dna" in r["sample_type"] for r in eligible)}

    nucleic = [r for r in eligible if "rna" in r["sample_type"] or "dna" in r["sample_type"]]
    protein = [r for r in eligible if "rna" not in r["sample_type"] and "dna" not in r["sample_type"]]
    rng.shuffle(nucleic)
    rng.shuffle(protein)
    train_nuc = round(args.train_entries * args.nucleic_fraction)
    val_nuc = round(args.val_entries * args.nucleic_fraction)
    train = nucleic[:train_nuc] + protein[:args.train_entries - train_nuc]
    val = (nucleic[train_nuc:train_nuc + val_nuc]
           + protein[args.train_entries - train_nuc:
                     args.train_entries - train_nuc + args.val_entries - val_nuc])
    rng.shuffle(train)
    rng.shuffle(val)
    chosen = train + val
    total = args.train_entries + args.val_entries
    if len(chosen) != total:
        raise RuntimeError(f"requested {total} entries but selected {len(chosen)}")
    for i, row in enumerate(chosen):
        row["split"] = "train" if i < args.train_entries else "val"
    return chosen, {"eligible": len(eligible), "eligible_nucleic": len(nucleic),
                    "selected": len(chosen), "selected_nucleic": train_nuc + val_nuc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--metadata-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--all-entries", action="store_true")
    parser.add_argument("--train-entries", type=int, default=512)
    parser.add_argument("--val-entries", type=int, default=128)
    parser.add_argument("--nucleic-fraction", type=float, default=0.25)
    parser.add_argument("--max-map-mb", type=int, default=3072)
    parser.add_argument("--target-voxel", type=float, default=1.5)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260826)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows, selection = select_rows(args)
    (output / "selection.json").write_text(json.dumps(
        {"summary": selection, "entries": [
            {k: r[k] for k in ("emdb_id", "pdb_id", "sample_type",
                               "resolution_angstrom", "split")}
            for r in rows]}, indent=2), encoding="utf-8")

    tasks = [(r, args.dataset_root, args.output_dir, args.target_voxel) for r in rows]
    results = []
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for index, result in enumerate(pool.imap_unordered(process_one, tasks, chunksize=1), 1):
            results.append(result)
            if index % 100 == 0 or index == len(tasks):
                failures = sum(bool(x["error"]) for x in results)
                print(f"processed={index}/{len(tasks)} failed={failures}", flush=True)
    kept = [r for r in results if not r["error"]]
    failed = [r for r in results if r["error"]]
    for split in ("train", "val"):
        paths = sorted(r["path"] for r in kept if r["split"] == split)
        (output / f"{split}.txt").write_text("\n".join(paths) + "\n", encoding="utf-8")
    if args.all_entries:
        train_paths = sorted(r["path"] for r in kept)
        val_paths = train_paths[:min(args.val_entries, len(train_paths))]
        (output / "train.txt").write_text("\n".join(train_paths) + "\n", encoding="utf-8")
        (output / "val.txt").write_text("\n".join(val_paths) + "\n", encoding="utf-8")
    summary = {
        **selection,
        "produced": len(kept),
        "failed": len(failed),
        "train_produced": sum(r["split"] == "train" for r in kept),
        "val_produced": sum(r["split"] == "val" for r in kept),
        "entries_with_p": sum(r["p_count"] > 0 for r in kept),
        "ca_count": sum(r["ca_count"] for r in kept),
        "p_count": sum(r["p_count"] for r in kept),
    }
    (output / "preprocess_summary.json").write_text(
        json.dumps({"summary": summary, "failures": failed}, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
