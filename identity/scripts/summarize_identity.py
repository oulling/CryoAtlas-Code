#!/usr/bin/env python3
"""Summarize MMseqs2 test-to-training alignments without treating no-hit as 0%."""

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-fasta", type=Path, required=True)
    parser.add_argument("--map-manifest", type=Path, required=True)
    parser.add_argument("--alignments", type=Path, required=True)
    parser.add_argument("--outdir", type=Path, required=True)
    return parser.parse_args()


def read_queries(test_fasta):
    queries = []
    query_to_pdb = {}
    with test_fasta.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.startswith(">"):
                continue
            query = line[1:].strip().split()[0].split("|", 1)[0]
            if "_" not in query:
                raise ValueError(
                    f"FASTA identifier must use PDB_CHAIN format: {query}"
                )
            pdb_id = query.split("_", 1)[0].upper()
            if query in query_to_pdb:
                raise ValueError(f"Duplicate FASTA identifier: {query}")
            queries.append(query)
            query_to_pdb[query] = pdb_id
    if not queries:
        raise ValueError("No FASTA records were found")
    return queries, query_to_pdb


def read_best_hits(alignments, known_queries):
    best = {query: None for query in known_queries}
    with alignments.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.rstrip("\n").split("\t")
            if len(fields) != 8:
                raise ValueError(
                    f"{alignments}:{line_number}: expected 8 tab-separated fields"
                )
            query, target = fields[:2]
            if query not in best:
                continue
            identity = float(fields[2])
            if identity <= 1.0:
                identity *= 100.0
            record = {
                "query": query,
                "target": target,
                "max_pident": identity,
                "alnlen": int(fields[3]),
                "qcov": float(fields[4]),
                "tcov": float(fields[5]),
                "evalue": float(fields[6]),
                "bits": float(fields[7]),
            }
            if best[query] is None or identity > best[query]["max_pident"]:
                best[query] = record
    return best


def read_map_manifest(path):
    rows = []
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"emd_id", "pdb_id"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(
                f"{path} must contain columns: {', '.join(sorted(required))}"
            )
        seen = set()
        for row in reader:
            emd_id = row["emd_id"].strip()
            pdb_id = row["pdb_id"].strip().upper()
            if emd_id in seen:
                raise ValueError(f"Duplicate map identifier: {emd_id}")
            seen.add(emd_id)
            rows.append((emd_id, pdb_id))
    return rows


def identity_stratum(best_hit):
    if best_hit is None:
        return "no_qualifying_hit"
    identity = best_hit["max_pident"]
    if identity < 30:
        return "observed_<30%"
    if identity < 50:
        return "observed_30-<50%"
    if identity < 70:
        return "observed_50-<70%"
    return "observed_>=70%"


def write_tsv(path, fieldnames, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    queries, query_to_pdb = read_queries(args.test_fasta)
    best_hits = read_best_hits(args.alignments, set(queries))
    map_manifest = read_map_manifest(args.map_manifest)

    pdb_to_queries = defaultdict(list)
    for query in queries:
        pdb_to_queries[query_to_pdb[query]].append(query)

    chain_rows = []
    for query in queries:
        hit = best_hits[query]
        row = {
            "query_chain": query,
            "pdb_id": query_to_pdb[query],
            "alignment_status": "hit" if hit else "no_qualifying_hit",
            "observed_max_identity_pct": (
                f"{hit['max_pident']:.4f}" if hit else ""
            ),
            "closest_training_chain": hit["target"] if hit else "",
            "alignment_length": hit["alnlen"] if hit else "",
            "query_coverage": f"{hit['qcov']:.4f}" if hit else "",
            "target_coverage": f"{hit['tcov']:.4f}" if hit else "",
            "evalue": f"{hit['evalue']:.6g}" if hit else "",
            "bitscore": f"{hit['bits']:.4f}" if hit else "",
        }
        chain_rows.append(row)

    chain_fields = list(chain_rows[0])
    write_tsv(args.outdir / "per_chain_identity.tsv", chain_fields, chain_rows)

    map_rows = []
    for emd_id, pdb_id in map_manifest:
        pdb_queries = pdb_to_queries.get(pdb_id)
        if not pdb_queries:
            raise ValueError(
                f"Map manifest PDB {pdb_id} has no chain in the test FASTA"
            )
        hits = [best_hits[query] for query in pdb_queries if best_hits[query]]
        best_map_hit = max(hits, key=lambda item: item["max_pident"]) if hits else None
        row = {
            "emd_id": emd_id,
            "pdb_id": pdb_id,
            "chain_count": len(pdb_queries),
            "chains_with_qualifying_hit": len(hits),
            "alignment_status": "hit" if best_map_hit else "no_qualifying_hit",
            "observed_max_identity_pct": (
                f"{best_map_hit['max_pident']:.4f}" if best_map_hit else ""
            ),
            "identity_stratum": identity_stratum(best_map_hit),
            "query_chain": best_map_hit["query"] if best_map_hit else "",
            "closest_training_chain": best_map_hit["target"] if best_map_hit else "",
            "alignment_length": best_map_hit["alnlen"] if best_map_hit else "",
            "query_coverage": (
                f"{best_map_hit['qcov']:.4f}" if best_map_hit else ""
            ),
            "target_coverage": (
                f"{best_map_hit['tcov']:.4f}" if best_map_hit else ""
            ),
            "evalue": f"{best_map_hit['evalue']:.6g}" if best_map_hit else "",
            "bitscore": f"{best_map_hit['bits']:.4f}" if best_map_hit else "",
        }
        map_rows.append(row)

    map_fields = list(map_rows[0])
    write_tsv(args.outdir / "per_map_identity.tsv", map_fields, map_rows)

    manifest_dir = args.outdir / "manifests"
    manifest_dir.mkdir(exist_ok=True)
    strata = {
        "no_qualifying_hit": "no_qualifying_hit.tsv",
        "observed_<30%": "observed_lt30.tsv",
        "observed_30-<50%": "identity_30_50.tsv",
        "observed_50-<70%": "identity_50_70.tsv",
        "observed_>=70%": "identity_ge70.tsv",
    }
    for stratum, filename in strata.items():
        selected = [
            {"emd_id": row["emd_id"], "pdb_id": row["pdb_id"]}
            for row in map_rows
            if row["identity_stratum"] == stratum
        ]
        write_tsv(manifest_dir / filename, ["emd_id", "pdb_id"], selected)

    counts = defaultdict(int)
    for row in map_rows:
        counts[row["identity_stratum"]] += 1
    with (args.outdir / "summary.txt").open("w", encoding="utf-8") as handle:
        handle.write(f"test_maps\t{len(map_rows)}\n")
        handle.write(f"test_chains\t{len(chain_rows)}\n")
        for stratum in strata:
            handle.write(f"{stratum}\t{counts[stratum]}\n")


if __name__ == "__main__":
    main()
