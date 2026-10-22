# CryoAtlas evaluation

Evaluate a predicted protein structure against a reference structure and its cryo-EM map. The entry point, `evaluation/evaluate.py`, runs selected metric modules for one target and saves the results as text and JSON. Baseline inference is run separately.

Run the commands below from the **repository root**.

## 1. Install dependencies

The pipeline targets Linux or WSL2. Create the Python environment and compile the included US-align source:

```bash
conda env create -f evaluation/environment.yml
conda activate cryoatlas-eval
g++ -O3 -ffast-math -o evaluation/USalign evaluation/USalign.cpp
```

Install [Phenix](https://phenix-online.org/download/) and [UCSF ChimeraX](https://www.cgl.ucsf.edu/chimerax/download.html) from their official websites under their respective licenses. Initialize Phenix in the shell used for evaluation:

```bash
source /absolute/path/to/phenix/phenix_env.sh

evaluation/USalign -h
command -v phenix.chain_comparison
command -v phenix.molprobity
command -v phenix.map_correlations
chimerax --version
python evaluation/evaluate.py --help
```

ChimeraX must provide the command `qscore #1 toVolume #2 useGui false`. If its executable is not on `PATH`, supply its full path through `--chimerax-bin`. US-align defaults to `evaluation/USalign`; use `--usalign-bin` to select another binary.

The evaluator itself uses the Python standard library. The environment also includes `requests` and `tqdm` for the optional EMDB downloader.

## 2. Prepare the inputs

Download the required files from the [CryoAtlas v1.0 release](https://huggingface.co/datasets/markfaa/CryoAtlas):

- **Benchmark evaluation:** the 618-target test archive contains the reference maps and structures.
- **Released predictions and results:** the [benchmark resource archive](https://huggingface.co/datasets/markfaa/CryoAtlas/tree/main/resources) contains five baseline prediction sets, two ModelAngelo retraining result sets, and ten complete examples.
- **Additional test entries:** the 534-entry archive completes the 1,152-entry test split; these entries are outside the 618-target benchmark.

See the [download guide](https://huggingface.co/datasets/markfaa/CryoAtlas/blob/main/resources/docs/downloads.md) for archive selection and checksums. Training data are not required to evaluate existing predictions.

For one target, provide:

| Argument | Input |
|---|---|
| `--predicted-structure` / `-p` | Predicted atomic structure, PDB or CIF |
| `--target-structure` / `-t` | Matching reference structure, PDB or CIF |
| `--map-file` | Experimental density map, uncompressed MRC or MAP; required for map metrics |
| `--output-file` / `-o` | Output text-file path; its parent directory is created automatically |

Use the matching map–structure pair supplied in the release. Full test maps are named `emd_<id>_fullmap.map`; example filenames may differ. The commands below assume that the selected files have been placed at `data/prediction.cif`, `data/reference.cif`, and `data/density.map`.

## 3. Run evaluation

This command enables all five metric modules:

```bash
python evaluation/evaluate.py \
  --predicted-structure data/prediction.cif \
  --target-structure data/reference.cif \
  --map-file data/density.map \
  --output-file results/target.txt \
  --need-superpose false \
  --usalign true \
  --chain-comparison true \
  --molprobity true \
  --map-cc true \
  --qscore true \
  --chimerax-bin /absolute/path/to/ChimeraX/bin/ChimeraX
```

**Choose the alignment setting from the prediction's coordinate frame.** Keep `--need-superpose false` for models already in the experimental map frame. For predictions in another frame, such as unaligned AF3 or E3-CryoFold outputs, set `--need-superpose true`. US-align then runs first, and subsequent modules use the generated superposed CIF. Check that this file was produced before interpreting map-based scores.

### Metric selection

Each switch takes an explicit `true` or `false` value.

| Switch | Default | Output metrics | Dependency |
|---|---|---|---|
| `--usalign` | `true` | Target-length-normalized TM-score; aligned-residue Cα RMSD | US-align |
| `--chain-comparison` | `true` | Residue coverage, sequence match, sequence recall, completeness | Phenix |
| `--molprobity` | `false` | Ramachandran and rotamer statistics, Cβ deviations, clashscore, bond/angle RMS deviations, MolProbity score | Phenix |
| `--map-cc` | `false` | CC_mask, CC_volume, CC_peaks, CC_box | Phenix + map |
| `--qscore` | `false` | Overall mean atomic Q-score | ChimeraX + map |

For the default structure-comparison metrics only:

```bash
python evaluation/evaluate.py \
  -p data/prediction.cif \
  -t data/reference.cif \
  -o results/target_structure.txt
```

This requires US-align and Phenix, but no map or ChimeraX. Add `--molprobity true` for stereochemical evaluation. Superposition requires US-align even if `--usalign false`.

## 4. Read the outputs

For `--output-file results/target.txt`, the evaluator produces:

```text
results/
├── target.txt           # Readable metric summary
├── target.json          # Input paths, alignment setting, and parsed metrics
├── target_TraceLog.log  # Raw output from external tools
└── target_sup.cif       # Generated when coordinate superposition succeeds
```

The text summary and trace log are appended on repeated runs; JSON is overwritten. Use a distinct output prefix for each target and run. If a metric is missing or a parser fails, inspect the trace log.

**Metric scope:** `ca_rmsd` is the RMSD reported by US-align over its aligned residues; it is distinct from Phenix chain-comparison local RMSD. The script outputs individual metrics and does not compute the paper's GT, Map, or Geom composite scores or aggregate targets. Released per-target tables are available in `Benchmark_outcome/` inside the resource archive and in the [HF results directory](https://huggingface.co/datasets/markfaa/CryoAtlas/tree/main/resources/benchmark).

Record external-tool versions when reporting new evaluations, since versions can affect results.

## Optional: retrieve maps from EMDB

The released archives already contain maps. To retrieve an EMDB primary map separately, use the released `source_accessions.csv` manifest:

```bash
python evaluation/download_emdb_maps.py \
  --manifest /path/to/source_accessions.csv \
  --out data \
  --emd-id EMD-14075
```

The output is `data/EMD-14075/emd_14075.map.gz`. Decompress it before evaluation:

```bash
gzip -dk data/EMD-14075/emd_14075.map.gz
```

Repeat `--emd-id` for several entries, or use `--split test --layout split` for the full test split. Existing nonempty files are skipped. The manifest is available under [HF metadata](https://huggingface.co/datasets/markfaa/CryoAtlas/tree/main/resources/metadata).
