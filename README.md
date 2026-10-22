# CryoAtlas Benchmark

Evaluation tools, ModelAngelo CNN training code and sequence-identity analysis for CryoAtlas.

[Project website](https://oulling.github.io/cryoatlas/) · [Paper](https://openreview.net/forum?id=qHJrv65ncA) · [Dataset on Hugging Face](https://huggingface.co/datasets/markfaa/CryoAtlas)

## Where to start

| Directory | Contents | Instructions |
|---|---|---|
| **evaluation/** | Single-target metric evaluation, US-align source and optional EMDB map downloader | [Evaluation guide](evaluation/README.md) |
| **modelangelo/** | CNN preprocessing/training, model building, export and source checks | [ModelAngelo guide](modelangelo/README.md) |
| **identity/** | Sequence-identity analysis, original train/test FASTA inputs and released annotations | [Identity guide](identity/README.md) |
| **metadata/** | Machine-readable dataset metadata | [Metadata](metadata/README.md) |

The data release contains **16,657 training entries** and **1,152 test entries**, including the **618-target benchmark**. Density maps, structures and sequences are included in the data archives. Benchmark resources also provide baseline predictions and the two released ModelAngelo retraining conditions, with 618 predicted CIFs per condition.

Download data and existing predictions from [Hugging Face](https://huggingface.co/datasets/markfaa/CryoAtlas); see its [download guide](https://huggingface.co/datasets/markfaa/CryoAtlas/blob/main/resources/docs/downloads.md). Large scientific files are hosted with the dataset.

## Evaluation quick start

The metric evaluator targets Linux/WSL2. Its complete metric set uses US-align, Phenix and ChimeraX; installation and examples are in the [evaluation guide](evaluation/README.md).

```bash
conda env create -f evaluation/environment.yml
conda activate cryoatlas-eval
python evaluation/evaluate.py --help
```

For CNN training and inference, use the separate environment described in `modelangelo/README.md`. To reproduce the identity annotations, follow `identity/README.md`; the exact FASTA inputs and SHA-256 files are included under `identity/data/inputs/`.
