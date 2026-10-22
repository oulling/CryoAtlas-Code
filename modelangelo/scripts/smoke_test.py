"""CryoAtlas addition: synthetic release regression checks.

Checks CNN gradients, scale-6 noise, inference imports, and mixed-polymer
forward passes through sequence-guided and sequence-free GNN definitions.
"""
import argparse
import importlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from model_angelo.c_alpha.dataset import CASDataset
from model_angelo.c_alpha.train import compute_losses
from model_angelo.utils.torch_utils import checkpoint_load_latest, get_model_from_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.manual_seed(2026)
    np.random.seed(2026)
    torch.set_num_threads(2)
    for name in ("build", "build_no_seq", "refine", "evaluate", "eval_per_resid", "hmm_search"):
        importlib.import_module(f"model_angelo.apps.{name}")
    importlib.import_module("model_angelo.c_alpha.diag_infer")
    root = Path(__file__).resolve().parents[1]
    model = get_model_from_file(str(root / "model_angelo/model_definitions/nucleotides/c_alpha/model.py"))
    model = model.to(args.device).eval()
    inputs = torch.randn(1, 2, 32, 32, 32, device=args.device)
    logits = model(inputs)
    assert logits.shape == (1, 3, 32, 32, 32), logits.shape
    target = torch.zeros_like(logits)
    target[:, :, 16, 16, 16] = 1
    losses = compute_losses(logits, target, focal_gamma=2, tversky_weight=0.5,
                            channel_weights=torch.tensor([1., 1., .5], device=args.device))
    assert torch.isfinite(losses["loss"])
    losses["loss"].backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    positions = torch.rand(32, 3, device=args.device) * 16 + 8
    prot_mask = torch.arange(32, device=args.device) < 24
    gnn_results = {}
    for bundle in ("nucleotides", "nucleotides_no_seq"):
        gnn = get_model_from_file(str(root / f"model_angelo/model_definitions/{bundle}/gnn/model.py"))
        gnn = gnn.to(args.device).eval()
        kwargs = dict(positions=positions, prot_mask=prot_mask, run_iters=1,
                      cryo_grids=[torch.randn(1, 32, 32, 32, device=args.device)],
                      cryo_global_origins=torch.zeros(1, 3, device=args.device),
                      cryo_voxel_sizes=torch.ones(1, device=args.device),
                      edge_index=None, batch=None)
        if bundle == "nucleotides":
            kwargs.update(sequence=torch.randn(1, 24, 1280, device=args.device),
                          sequence_mask=torch.ones(1, 24, dtype=torch.bool, device=args.device))
        with torch.no_grad():
            predicted = gnn(**kwargs)["pred_positions"][-1]
        assert predicted.shape == positions.shape and torch.isfinite(predicted).all()
        gnn_results[bundle] = "passed"
        del gnn
    with tempfile.TemporaryDirectory() as directory:
        assert checkpoint_load_latest(directory, torch.device("cpu"), match_model=False) == 0
        manifest = Path(directory) / "inputs.txt"
        manifest.write_text("unused.pkl\n")
        dataset = CASDataset(str(manifest))
        assert dataset.noise_spectrum_scale == 6
        crop = np.random.randn(16, 16, 16).astype(np.float32)
        noisy = dataset._add_coloured_noise(crop, None)
        assert noisy.shape == crop.shape and np.isfinite(noisy).all()
    print(json.dumps({"device": args.device, "torch": torch.__version__, "cnn_shape": list(logits.shape),
                      "focal": float(losses["focal"]), "tversky": float(losses["tversky"]),
                      "backward": "passed", "noise_scale": 6, "inference_imports": "passed",
                      "gnn_mixed_polymer_forward": gnn_results}))


if __name__ == "__main__":
    main()
