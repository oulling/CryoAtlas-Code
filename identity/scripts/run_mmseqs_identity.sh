#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  run_mmseqs_identity.sh \
    --test-fasta FILE \
    --train-fasta FILE \
    --map-manifest FILE \
    --outdir DIR \
    [--threads N]
EOF
}

TEST_FASTA=""
TRAIN_FASTA=""
MAP_MANIFEST=""
OUTDIR=""
THREADS=8

while [[ $# -gt 0 ]]; do
  case "$1" in
    --test-fasta)
      TEST_FASTA="$2"
      shift 2
      ;;
    --train-fasta)
      TRAIN_FASTA="$2"
      shift 2
      ;;
    --map-manifest)
      MAP_MANIFEST="$2"
      shift 2
      ;;
    --outdir)
      OUTDIR="$2"
      shift 2
      ;;
    --threads)
      THREADS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

for command_name in mmseqs python3 sha256sum; do
  command -v "$command_name" >/dev/null 2>&1 || {
    echo "Required command not found: $command_name" >&2
    exit 1
  }
done

for required_file in "$TEST_FASTA" "$TRAIN_FASTA" "$MAP_MANIFEST"; do
  [[ -n "$required_file" && -s "$required_file" ]] || {
    echo "Required input is missing or empty: $required_file" >&2
    exit 1
  }
done

[[ "$THREADS" =~ ^[1-9][0-9]*$ ]] || {
  echo "--threads must be a positive integer" >&2
  exit 2
}

[[ -n "$OUTDIR" ]] || {
  echo "--outdir is required" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$OUTDIR/mmseqs_work"
TMPDIR="$WORKDIR/tmp"
mkdir -p "$TMPDIR"

{
  echo "mmseqs_version=$(mmseqs version)"
  echo "parameters=--min-seq-id 0 -c 0.5 --cov-mode 0 -s 7.5 -e 0.001 -a 1"
  echo "threads=$THREADS"
  echo "test_fasta_sha256=$(sha256sum "$TEST_FASTA" | awk '{print $1}')"
  echo "train_fasta_sha256=$(sha256sum "$TRAIN_FASTA" | awk '{print $1}')"
  echo "map_manifest_sha256=$(sha256sum "$MAP_MANIFEST" | awk '{print $1}')"
} > "$OUTDIR/run_metadata.txt"

mmseqs createdb "$TEST_FASTA" "$WORKDIR/testDB" --dbtype 1
mmseqs createdb "$TRAIN_FASTA" "$WORKDIR/trainDB" --dbtype 1

mmseqs search \
  "$WORKDIR/testDB" \
  "$WORKDIR/trainDB" \
  "$WORKDIR/resultDB" \
  "$TMPDIR" \
  --min-seq-id 0 \
  -c 0.5 \
  --cov-mode 0 \
  -s 7.5 \
  -e 0.001 \
  -a 1 \
  --threads "$THREADS"

mmseqs convertalis \
  "$WORKDIR/testDB" \
  "$WORKDIR/trainDB" \
  "$WORKDIR/resultDB" \
  "$OUTDIR/alignments.tsv" \
  --format-output "query,target,pident,alnlen,qcov,tcov,evalue,bits" \
  --threads "$THREADS"

python3 "$SCRIPT_DIR/summarize_identity.py" \
  --test-fasta "$TEST_FASTA" \
  --map-manifest "$MAP_MANIFEST" \
  --alignments "$OUTDIR/alignments.tsv" \
  --outdir "$OUTDIR"

