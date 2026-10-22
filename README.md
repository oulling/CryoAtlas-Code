# CryoAtlas-Code

Evaluation tools, ModelAngelo CNN training code and sequence-identity analysis for CryoAtlas.

*Accepted at NeurIPS 2026 · Evaluations & Datasets Track*

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

## License

Original CryoAtlas code and documentation are licensed under [CC BY-NC 4.0](LICENSE), unless a file or subdirectory states otherwise. Attribution is required, and the licensed material may be shared and adapted for noncommercial purposes under the license terms.

Third-party components retain their original licenses and attribution:

- ModelAngelo-derived code: see [the upstream MIT license](modelangelo/LICENSE) and [NOTICE](modelangelo/NOTICE).
- US-align: see the permission and attribution notice in [evaluation/USalign.cpp](evaluation/USalign.cpp).

The [CryoAtlas dataset](https://huggingface.co/datasets/markfaa/CryoAtlas) is also distributed under CC BY-NC 4.0. External software and separately obtained pretrained weights remain subject to their distributors' terms.
