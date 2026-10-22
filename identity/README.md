# CryoAtlas train–test sequence-identity annotations

This directory contains the sequence-similarity annotations used to describe
the 1,152-map CryoAtlas test pool relative to CryoAtlas training chains.

## Contents

### Original analysis inputs

The exact FASTA inputs are included as losslessly compressed files under
`data/inputs/`:

| Input | Sequence records | Compressed file |
|---|---:|---|
| Test queries | 4,613 | [final_test.fasta.gz](data/inputs/final_test.fasta.gz) |
| Training targets | 60,234 | [final_train.fasta.gz](data/inputs/final_train.fasta.gz) |

[SHA256SUMS](data/inputs/SHA256SUMS) records compressed and original-file hashes.
[input_metadata.json](data/inputs/input_metadata.json) records input sizes,
record counts, the original MMseqs2 build identifier, fixed parameters and
alignment-export columns. Decompression reproduces the original inputs byte
for byte. Counts describe FASTA records, rather than expanded physical-chain
counts.

### Data

- `data/cryoatlas_test_train_identity_annotations.tsv`: map-level annotations
  for all 1,152 test maps.
- `data/per_chain_max_identity_c50.tsv`: best qualifying training-chain match
  for each of the 4,613 test sequence records.
- `data/cryoatlas_original_172_map_manifest.tsv`: membership of the historical
  172-map subset reported in the submitted manuscript, joined to the latest
  annotations for auditability.

The historical 172-map file is not a newly recomputed `<=30%` manifest. Its
membership comes from the submission-time analysis, whereas its identity
columns come from the latest standardized rerun. Because MMseqs2 is actively
maintained, version-dependent changes in its search and alignment behavior can
shift borderline matches; the latest rerun therefore differs slightly from the
original 172-map result. It should be described as the **submission-time
historical subset**, not as a current sequence-identity stratum. For exact
reproducibility, each rerun should report the MMseqs2 version and the input
FASTA checksums recorded by the supplied script.

`no_qualifying_hit` means that no training-chain alignment satisfied all fixed
search criteria. It is not an observed identity of zero and must not be
reported as `<10% identity`.

## Fixed MMseqs2 protocol

The released annotations use an all-versus-all search of test chains against
CryoAtlas training chains:

```text
--min-seq-id 0
-c 0.5
--cov-mode 0
-s 7.5
-e 0.001
-a 1
```

With `--cov-mode 0`, both query and target coverage must satisfy the 0.5
threshold. At chain level, the qualifying alignment with the highest sequence
identity is retained. At map level, maximum identity is the maximum over its
constituent test chains. A map is `no_qualifying_hit` only when none of its
chains has a qualifying training-chain alignment.

For the current 1,152-map annotations, the non-overlapping map-level groups are:

| Stratum | Maps |
|---|---:|
| No qualifying training-chain hit | 124 |
| Observed identity `<30%` | 30 |
| Observed identity `30–<50%` | 89 |
| Observed identity `50–<70%` | 37 |
| Observed identity `>=70%` | 872 |

## Reproduce the analysis

Requirements:

- Linux or WSL2
- MMseqs2 available as `mmseqs`
- Python 3.8 or newer

The test FASTA headers must begin with a chain identifier of the form
`PDB_CHAIN`; text after whitespace or `|` is ignored. The map manifest must be
a TSV containing `emd_id` and `pdb_id` columns.

From the repository root, decompress and verify the inputs first:

```bash
gzip -dk identity/data/inputs/final_test.fasta.gz
gzip -dk identity/data/inputs/final_train.fasta.gz
(cd identity/data/inputs && sha256sum -c SHA256SUMS)
```

Then run the released protocol:

```bash
bash identity/scripts/run_mmseqs_identity.sh \
  --test-fasta identity/data/inputs/final_test.fasta \
  --train-fasta identity/data/inputs/final_train.fasta \
  --map-manifest identity/data/cryoatlas_test_train_identity_annotations.tsv \
  --outdir identity/recomputed \
  --threads 16
```

The command writes raw alignments, per-chain best hits, per-map annotations,
fixed evaluation manifests, and a metadata file containing the MMseqs2 version,
parameters, and SHA-256 checksums of both input FASTA files.

To analyze a future method, replace `--train-fasta` with that method's actual
training-chain FASTA while keeping the remaining protocol fixed.

