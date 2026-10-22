"""CryoAtlas addition: full-map CNN predictions and diagnostic export.

Provides the sliding-window prediction helper used by periodic training
exports, plus standalone probability-map and anchor-coordinate output.

Runs the current checkpoint on a preprocessed pickle (or a raw MRC + CIF pair)
and dumps:
  - ca_output_grid.mrc       : raw sigmoid probability map for the Cα channel
  - p_output_grid.mrc        : raw sigmoid probability map for the P  channel
  - bb_output_grid.mrc       : raw sigmoid probability map for backbone trace
  - see_alpha_output_ca.cif  : discrete Cα points after grid_to_points post-processing
  - ground_truth_ca.cif      : ground-truth Cα positions (if available)

Usage:
    python -m model_angelo.c_alpha.diag_infer \
        --log-dir /path/to/run \
        --step 2000 \
        --pickle /path/to/EMD-0008.pkl \
        --out-dir /path/to/predictions
"""
from __future__ import annotations

import argparse
import os
import pickle
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from model_angelo.c_alpha.inference import grid_to_points
from model_angelo.data.dataset_preprocess import decompress_data
from model_angelo.utils.grid import (
    get_local_std,
    load_mrc,
    make_model_angelo_grid,
    save_mrc,
)
from model_angelo.utils.save_pdb_utils import points_to_pdb
from model_angelo.utils.torch_utils import get_model_from_file


def load_checkpoint(log_dir: Path, step: int, device: str):
    ckpt_path = log_dir / f"chkpt_{step}.torch"
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)
    model = get_model_from_file(str(log_dir / "model.py")).to(device)
    state = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    # checkpoint_save stored the module state_dict under key "module".
    for k in ("module", "model"):
        if k in state:
            sd = state[k]
            break
    else:
        sd = state
    model.load_state_dict(sd)
    model.eval()
    return model, state.get("step_num", step)


def load_from_pickle(pkl_path: Path):
    with open(pkl_path, "rb") as f:
        ds = pickle.load(f)
    grid = decompress_data(ds["grid_io"]).astype(np.float32)
    voxel_size = float(ds["voxel_size"])
    global_origin = np.asarray(ds["global_origin"], dtype=np.float32)
    if isinstance(ds["cas"], dict):
        ca_world = np.asarray(ds["cas"].get("CA", np.zeros((0, 3))), dtype=np.float32)
    else:
        ca_world = np.asarray(ds["cas"], dtype=np.float32)
    return grid, voxel_size, global_origin, ca_world


def load_from_mrc(mrc_path: Path, target_voxel: float = 1.0):
    grid, voxel_size, global_origin = load_mrc(str(mrc_path))
    grid, voxel_size, global_origin = make_model_angelo_grid(
        np.copy(grid).astype(np.float32), voxel_size, global_origin, target_voxel
    )
    return grid, voxel_size, global_origin, np.zeros((0, 3), dtype=np.float32)


@torch.no_grad()
def sliding_window_infer(model, grid_np: np.ndarray, device: str,
                         box_size: int = 64, stride: int = 32,
                         batch_size: int = 4, crop: int = 6) -> np.ndarray:
    """Returns (3, Z, Y, X) probability map via sigmoid."""
    # Global normalization (matches inference.py semantics).
    g = (grid_np - grid_np.mean()) / (grid_np.std() + 1e-6)
    grid_t = torch.from_numpy(g).float()
    std_t = get_local_std(grid_t[None, None])[0, 0]
    std_t = std_t / (std_t.max() + 1e-6)

    Z, Y, X = grid_t.shape
    output = torch.zeros(3, Z, Y, X)
    count = torch.zeros(Z, Y, X)

    def span(n):
        xs = list(range(0, max(1, n - box_size), stride))
        if not xs:
            xs = [0]
        if xs[-1] + box_size != n:
            xs.append(max(0, n - box_size))
        return xs

    coords = list(product(span(Z), span(Y), span(X)))
    print(f"[diag] sliding windows: {len(coords)} (box={box_size}, stride={stride})")

    model.to(device)
    i = 0
    while i < len(coords):
        batch_in, batch_coords = [], []
        for _ in range(batch_size):
            if i >= len(coords):
                break
            z0, y0, x0 = coords[i]
            sliced = grid_t[z0:z0 + box_size, y0:y0 + box_size, x0:x0 + box_size]
            sliced_std = std_t[z0:z0 + box_size, y0:y0 + box_size, x0:x0 + box_size]
            # Pad to (box_size)^3 if edge crop.
            if sliced.shape != (box_size,) * 3:
                pad = torch.zeros(box_size, box_size, box_size)
                sd, sh, sw = sliced.shape
                pad[:sd, :sh, :sw] = sliced
                sliced = pad
                pad2 = torch.zeros(box_size, box_size, box_size)
                pad2[:sd, :sh, :sw] = sliced_std
                sliced_std = pad2
            batch_in.append(torch.stack([sliced, sliced_std], dim=0))
            batch_coords.append((z0, y0, x0))
            i += 1
        x = torch.stack(batch_in, dim=0).to(device)
        logits = model(x)
        probs = torch.sigmoid(logits).cpu()
        for b, (z0, y0, x0) in enumerate(batch_coords):
            # Use the centred sub-crop of the window to avoid edge artifacts.
            z1 = min(Z, z0 + box_size - crop)
            y1 = min(Y, y0 + box_size - crop)
            x1 = min(X, x0 + box_size - crop)
            z_a = z0 + crop
            y_a = y0 + crop
            x_a = x0 + crop
            if z_a >= z1 or y_a >= y1 or x_a >= x1:
                continue
            out_slice = probs[
                b, :, crop: crop + (z1 - z_a),
                      crop: crop + (y1 - y_a),
                      crop: crop + (x1 - x_a)
            ]
            output[:, z_a:z1, y_a:y1, x_a:x1] += out_slice
            count[z_a:z1, y_a:y1, x_a:x1] += 1

    count = count.clamp(min=1)
    return (output / count).numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log-dir", required=True)
    ap.add_argument("--step", type=int, required=True)
    ap.add_argument("--pickle", default=None, help="Preprocessed pickle path")
    ap.add_argument("--mrc", default=None, help="Raw MRC path (alternative to --pickle)")
    ap.add_argument("--cif", default=None, help="Ground-truth CIF for overlaying (raw MRC mode)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--box-size", type=int, default=64)
    ap.add_argument("--stride", type=int, default=32)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--target-voxel", type=float, default=1.0,
                    help="Resample voxel size in Angstrom; match the training "
                         "pipeline. Current preprocessing defaults to 1.5.")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Higher default than inference.py (0.05) because "
                         "early-training prob maps are diffuse; low threshold "
                         "yields millions of candidates that make grid_to_points hang.")
    ap.add_argument("--max-candidates", type=int, default=500_000,
                    help="Skip post-processing if above-threshold voxels exceed this.")
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()

    log_dir = Path(args.log_dir).resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[diag] loading checkpoint from {log_dir}/chkpt_{args.step}.torch")
    model, step_num = load_checkpoint(log_dir, args.step, args.device)
    print(f"[diag] loaded model at step {step_num}")

    if args.pickle:
        grid, voxel_size, global_origin, ca_world = load_from_pickle(Path(args.pickle))
        print(f"[diag] loaded pickle: grid={grid.shape} voxel_size={voxel_size} "
              f"n_ca={len(ca_world)}")
    else:
        assert args.mrc, "specify --pickle or --mrc"
        grid, voxel_size, global_origin, ca_world = load_from_mrc(
            Path(args.mrc), target_voxel=args.target_voxel
        )
        if args.cif:
            from model_angelo.utils.pdb_utils import load_cas_ps_from_structure
            ca_world = load_cas_ps_from_structure(args.cif, quiet=True)["CA"]
        print(f"[diag] loaded mrc: grid={grid.shape} voxel_size={voxel_size}")

    probs = sliding_window_infer(
        model, grid, args.device,
        box_size=args.box_size, stride=args.stride, batch_size=args.batch_size,
    )
    print(f"[diag] prob map done: Cα max={probs[0].max():.3f} mean={probs[0].mean():.4f}")

    save_mrc(probs[0].astype(np.float32), voxel_size, global_origin,
             str(out_dir / "ca_output_grid.mrc"))
    save_mrc(probs[1].astype(np.float32), voxel_size, global_origin,
             str(out_dir / "p_output_grid.mrc"))
    save_mrc(probs[2].astype(np.float32), voxel_size, global_origin,
             str(out_dir / "bb_output_grid.mrc"))

    n_above = int((probs[0] > args.threshold).sum())
    print(f"[diag] {n_above} voxels > {args.threshold}")
    if n_above > args.max_candidates:
        print(f"[diag] skipping grid_to_points (> {args.max_candidates} candidates). "
              f"Use a higher --threshold to post-process.")
        ca_pts, ca_raw = np.zeros((0, 3)), np.zeros((0, 3))
    else:
        ca_pts, ca_raw = grid_to_points(
            probs[0], threshold=args.threshold,
            neighbour_distance_threshold=6 / voxel_size,
        )
        print(f"[diag] post-processing: raw={len(ca_raw)} -> merged={len(ca_pts)} Cα "
              f"points (ground truth = {len(ca_world)})")
        points_to_pdb(str(out_dir / "see_alpha_output_ca.cif"),
                      voxel_size * ca_pts + global_origin)
        points_to_pdb(str(out_dir / "see_alpha_output_ca_raw.cif"),
                      voxel_size * ca_raw + global_origin)

    if len(ca_world):
        points_to_pdb(str(out_dir / "ground_truth_ca.cif"), ca_world)

    save_mrc(grid.astype(np.float32), voxel_size, global_origin,
             str(out_dir / "input_grid.mrc"))

    print(f"[diag] wrote outputs to {out_dir}")
    print(f"  - input_grid.mrc            : resampled cryo-EM map")
    print(f"  - ca_output_grid.mrc        : Cα prob (expected: blobs around atoms)")
    print(f"  - p_output_grid.mrc         : P  prob (expected: ~all zero for proteins)")
    print(f"  - bb_output_grid.mrc        : backbone trace prob")
    print(f"  - see_alpha_output_ca.cif   : merged Cα points ({len(ca_pts)} pts)")
    print(f"  - see_alpha_output_ca_raw.cif : all above-threshold voxels "
          f"({len(ca_raw)} pts)")
    print(f"  - ground_truth_ca.cif       : deposited Cα ({len(ca_world)} pts)")


if __name__ == "__main__":
    main()
