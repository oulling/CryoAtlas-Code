# ModelAngelo CNN training and model building

Custom research CNN preprocessing/training plus ModelAngelo CNN/GNN inference, chain assembly, sequence assignment, and refinement. The source reports `1.0.16`; distribution version is `1.0.16.post1`. Data, split manifests, and weights are not bundled. GNN training is not included.

## Changes from official ModelAngelo

Compared with the official [ModelAngelo 1.0.16 source](https://github.com/3dem/model-angelo/tree/v1.0.16), this distribution contains these local additions and modifications:

| Area | Local change |
| --- | --- |
| CNN training | Adds the dataset, arguments, and training loop referenced but not bundled by the official CNN launcher, under `model_angelo/c_alpha/`. Uses channel-weighted focal/Tversky losses, a Tversky warmup/ramp, a late focal-weight ramp, gradient accumulation, and configurable validation crops. |
| Training data | Adds `scripts/prepare_cnn.py` for metadata selection and pickle generation, positive-occupancy polymer CA/P extraction, chain-aware protein-backbone targets, and balanced protein/nucleotide anchor crops. |
| Map coordinates | Corrects the XYZ origin shift when padding a non-cubic ZYX grid; preprocessing also interprets MRC axis labels and origin/start fields before resampling. |
| Noise augmentation | Computes colored-noise amplitudes from all radial shells of each crop rather than indexing a full-map spectrum; the default spectrum multiplier is `6.0`. |
| Logging and resume | Adds per-entry and global focal-loss CSVs, optional W&B logging, and scheduler-state resume; fixes startup when no checkpoint exists. `--scheduler-steps` separates the cosine horizon from the run length. |
| GNN execution | `gnn_output.py` appends layer outputs instead of replacing their lists; `multi_layer_ipa.py` recomputes sequence keys/values across recycling iterations instead of reusing the cache. |
| Runtime compatibility | Updates scheduler serialization for PyTorch 2.7 and casts sequence-attention output to its destination dtype for mixed precision. |
| CNN memory | Changes `Bottleneck`'s default activation checkpointing from enabled to disabled; this trades additional memory for less recomputation. |
| Release usage | Adds pinned dependencies, portable preprocessing/training/export scripts, full-map CNN diagnostics, source checks, and synthetic smoke checks; removes unused modules, debug entry points, and the unused root configuration. The unused experimental relaxation helper and its penalty module are excluded; the CLI `refine` workflow is retained. |

The CNN and GNN inference network architectures remain those of official ModelAngelo 1.0.16. These changes do not introduce a new GNN architecture or trained GNN weights; model building uses separately downloaded pretrained GNN checkpoints.

## Installation

From the repository root, enter this directory with `cd modelangelo`.

Use Python 3.11 and run these Bash commands from the release directory. The dependency pins target PyTorch 2.7.1 with CUDA 12.6:

```bash
python -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cu126
python -m pip install -r requirements.txt
python -m pip install --no-deps -e .
python scripts/check_release.py
python -m model_angelo --help
```

Runtime pins include NumPy 1.26.4, SciPy 1.15.3, Biopython 1.85, mrcfile 1.5.4, fair-esm 2.0.0, Accelerate 1.10.1, and pyhmmer 0.11.2; see `requirements.txt` for all versions. Optional W&B 0.21.1 is in `requirements-optional.txt`. Use a compatible GPU driver; reduce batch/crop sizes for memory limits. Source checks do not establish model accuracy.

Optional synthetic checks exercise CNN forward/backward, colored noise, inference imports, and mixed-polymer forward passes through both GNN definitions. Use `--device cpu` if needed; this does not test end-to-end reconstruction:

```bash
python scripts/smoke_test.py --device cuda:0
```

## CNN preprocessing

Inputs are `DATASET/EMD-<id>/` directories with maps and mmCIF models in the same physical coordinates. Metadata CSV columns:

```text
emdb_id,pdb_id,map_filename,primary_cif_filename,final_status,uncompressed_map_bytes,sample_type,resolution_angstrom
```

Selection requires `final_status=kept`, existing files, and maps within `--max-map-mb`. `sample_type` values containing `rna`/`dna` determine nucleic stratification.

For a raw/processed layout, metadata may specify `map_filename=raw/emd_<id>.map` and `primary_cif_filename=processed/<pdb>_aligned.cif`; `--dataset-root` must contain the EMD folders, for example a single split directory. Filenames and the primary CIF are supplied by metadata.

```bash
python scripts/prepare_cnn.py \
  --dataset-root /path/to/dataset --metadata-csv /path/to/metadata.csv \
  --output-dir /path/to/cnn-pickles \
  --train-entries 512 --val-entries 128 \
  --nucleic-fraction 0.25 --target-voxel 1.5 --workers 8
```

Outputs include pickles, `train.txt`, `val.txt`, selection, and failure summaries. Manifests contain one pickle path per line. Supply independent train/validation entries and any sequence-level separation your protocol requires.

Preprocessing applies MRC axes/origins, resamples to 1.5 Å voxels by default, normalizes density, and extracts positive-occupancy C-alpha/phosphorus anchors with chain boundaries. Training uses 64³ crops, density/local-standard-deviation inputs, and C-alpha/phosphorus/protein-backbone targets. Augmentation includes rotations, B-factor changes, and crop-spectrum colored noise with multiplier `6.0`.

Corner cases: `--all-entries` generates an overlapping diagnostic `val.txt` subset of `train.txt`; replace these with independent manifests. Existing pickles are reused, so use a new directory after changing preprocessing or inputs. Legacy pickles without chain metadata cannot preserve backbone chain boundaries.

## CNN training

The wrapper checks manifest paths and entry IDs for overlap, initializes `RUN_DIR/model.py`, and launches `model_angelo.c_alpha.main`.

```bash
bash scripts/train_cnn.sh \
  /path/to/train.txt /path/to/independent-val.txt /path/to/run \
  --num-steps 25000 --scheduler-steps 135000
```

Environment defaults are `NUM_PROCESSES=1`, `MIXED_PRECISION=bf16`, `BATCH_SIZE=12`, `ACCUMULATION_STEPS=2`. Effective global batch size is `NUM_PROCESSES * BATCH_SIZE * ACCUMULATION_STEPS`. For batch size 96 on two GPUs, prefix the command with `NUM_PROCESSES=2 BATCH_SIZE=12 ACCUMULATION_STEPS=4`. Extra arguments override defaults; W&B is disabled. Full options:

```bash
python -m model_angelo.c_alpha.main --help
```

The loss combines scheduled focal/Tversky terms. Channel order is C-alpha, phosphorus, protein backbone. Wrapper weights are `1.0,1.0,0.5`; the module default `1.0,0.0,0.5` disables phosphorus supervision. Nucleotide supervision also requires nucleotide examples.

Checkpoints are `chkpt_<step>.torch`; reusing a run resumes the latest. `--dont-load` deletes its checkpoints, summaries, and training log. `--scheduler-steps` sets the cosine horizon independently of `--num-steps`.

Focal-loss reports are `entry_focal_rank<N>.csv`, `focal_every_step.csv`, and `focal_every_20_steps.csv`. Validation reports focal/Tversky loss, C-alpha recall at 0.5/0.7, mean probabilities, and signal-to-background ratios.

Export a trusted training checkpoint for CNN inference; export refuses to overwrite an existing file:

```bash
python scripts/export_cnn.py /path/to/run/chkpt_25000.torch /path/to/run/exported.torch
python -m model_angelo.c_alpha.inference \
  --log-dir /path/to/run --model-checkpoint exported.torch \
  --map-path /path/to/map.mrc --output-path /path/to/anchors \
  --device cuda:0 --do-nucleotides
```

For full building, copy a compatible `nucleotides` bundle and replace only the copy's `c_alpha/chkpt.torch` with exported CNN weights. Keep the matching CNN architecture and pretrained GNN checkpoints, then provide `--model-bundle-path`.

## Pretrained model building and refinement

Choose a writable `TORCH_HOME` cache. Setup downloads pretrained weights and the configured ESM model when required:

```bash
python -m model_angelo setup_weights --bundle-name nucleotides
python -m model_angelo setup_weights --bundle-name nucleotides_no_seq
```

With sequences:

```bash
python -m model_angelo build \
  --volume-path /path/to/map.mrc --protein-fasta /path/to/proteins.fasta \
  --rna-fasta /path/to/rna.fasta --dna-fasta /path/to/dna.fasta \
  --output-dir /path/to/build --device 0
```

Omit absent RNA/DNA inputs; `--protein-fasta` remains required. `build` defaults to `nucleotides`. Options include `--mask-path`, `--config-path`, and `--keep-intermediate-results`. Outputs include `<output-dir-name>.cif`, `_raw.cif`, and `entropy_scores/`.

Without sequences:

```bash
python -m model_angelo build_no_seq \
  --volume-path /path/to/map.mrc --output-dir /path/to/build-no-seq --device 0
```

This uses `nucleotides_no_seq` and produces chain HMM profiles. Sequence-free residue assignments may be less accurate.

Refine an existing model:

```bash
python -m model_angelo refine \
  --input-structure /path/to/model.cif --volume-path /path/to/map.mrc \
  --output-dir /path/to/refined --device 0 --write-hmm-profiles
```

Refinement writes `<output-dir-name>_refined.cif`; HMM output is optional. Compatible nucleotide bundles support protein/RNA/DNA inference.

Local bundles require `config.json` and applicable `c_alpha/`/`gnn/` checkpoints. Keep `--model-bundle-name` consistent with the included architecture, even with `--model-bundle-path`.

## Evaluation

Native metrics are backbone/C-alpha errors, lDDT, recall, precision, and residue-type matching. Inputs must share a compatible coordinate frame before spatial matching (default cutoff: 3 Å).

```bash
python -m model_angelo evaluate \
  --predicted-structure /path/to/prediction.cif --target-structure /path/to/target.cif \
  --match-type both --max-dist 3 --output-file /path/to/results/native.log

python -m model_angelo eval_per_resid \
  --predicted-structure /path/to/prediction.cif --target-structure /path/to/target.cif \
  --match-type protein --max-dist 3 --output-file /path/to/results/per-residue.tsv
```

`--match-type` accepts `both`, `protein`, or `nucleotide`. Native errors retain their source definitions; empty models/no correspondences can fail. They do not replace CryoAtlas metrics.

Native correspondence matching shuffles candidates, and CUDA inference can vary slightly between runs. Fixed seeds do not guarantee bitwise-identical outputs. Native distance errors are mean distances, not root-mean-square deviations.

CryoAtlas metric evaluation uses the separate [evaluation directory](../evaluation/README.md) with Phenix, ChimeraX and Q-score. Compile the included `evaluation/USalign.cpp` or supply a US-align executable as described there. Only the target-normalized TM-score is used. The native ModelAngelo evaluation commands above use their own metric definitions.
