"""CryoAtlas addition: CNN losses, training loop, and validation.

Adds channel-weighted focal/Tversky losses, loss-weight schedules, gradient
accumulation, per-entry loss records, and optional full-map predictions.
"""
from __future__ import annotations

import math
import os
import time
import csv
import statistics
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from model_angelo.utils.torch_utils import checkpoint_save


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _parse_channel_weights(spec: str, n_channels: int = 3) -> torch.Tensor:
    parts = [p.strip() for p in (spec or "").split(",") if p.strip()]
    if not parts:
        return torch.ones(n_channels, dtype=torch.float32)
    if len(parts) != n_channels:
        raise ValueError(
            f"--channel-weights expects {n_channels} comma-separated floats, got {len(parts)}: {parts!r}"
        )
    return torch.tensor([float(p) for p in parts], dtype=torch.float32)


def _channel_weight_tensor(
    cw: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """Reshape cw=(C,) to broadcast against target (B, C, Z, Y, X)."""
    cw = cw.to(device=target.device, dtype=torch.float32)
    return cw.view(1, -1, *([1] * (target.dim() - 2)))


# ---------------------------------------------------------------------------
# Losses (fp32-forced, channel-weighted)
# ---------------------------------------------------------------------------
def weighted_focal_loss(
    logits: torch.Tensor, target: torch.Tensor,
    gamma: float = 2.0, eps: float = 1e-6,
    channel_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return weighted_focal_loss_per_sample(
        logits, target, gamma=gamma, eps=eps, channel_weights=channel_weights
    ).mean()


def weighted_focal_loss_per_sample(
    logits: torch.Tensor, target: torch.Tensor,
    gamma: float = 2.0, eps: float = 1e-6,
    channel_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    target = target.clamp(0.0, 1.0)
    prob = torch.sigmoid(logits)
    prob_t = prob * target + (1 - prob) * (1 - target)
    ce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    focal = (1.0 - prob_t).pow(gamma) * ce

    spatial = focal.shape[2:]
    N = 1
    for s in spatial:
        N *= s
    M = target.sum(dim=tuple(range(2, target.dim()))).clamp(min=1.0)
    pos_w = ((N - M) / M).view(M.shape + (1,) * len(spatial))
    weight = pos_w * target + (1 - target)
    voxelwise = weight * focal  # shape (B, C, Z, Y, X)

    if channel_weights is not None:
        per_ch = voxelwise.mean(dim=tuple(range(2, voxelwise.dim())))
        denom = float(channel_weights.sum().item()) or 1.0
        return (per_ch * channel_weights.to(per_ch.device)).sum(dim=1) / denom
    return voxelwise.mean(dim=tuple(range(1, voxelwise.dim())))


def tversky_loss(
    logits: torch.Tensor, target: torch.Tensor,
    alpha: float = 0.3, beta: float = 0.7,
    eps: float = 1e-6,
    channel_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    prob = torch.sigmoid(logits)
    target = target.clamp(0.0, 1.0)
    dims = tuple(range(2, target.dim()))
    tp = (prob * target).sum(dim=dims)
    fp = (prob * (1 - target)).sum(dim=dims)
    fn = ((1 - prob) * target).sum(dim=dims)
    tv = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    loss_per_ch = (1.0 - tv)  # shape (B, C)

    if channel_weights is not None:
        cw = channel_weights.to(loss_per_ch.device)
        per_ch = loss_per_ch.mean(dim=0)        # (C,)
        denom = float(cw.sum().item()) or 1.0
        return (per_ch * cw).sum() / denom
    return loss_per_ch.mean()


def _focal_weight_at(step: int, args) -> float:
    """Cosine ramp focal_weight from 1.0 -> args.focal_late_boost_max
    over [focal_late_boost_start, focal_late_boost_end]."""
    start = int(getattr(args, "focal_late_boost_start", 50_000))
    end   = int(getattr(args, "focal_late_boost_end",   80_000))
    wmax  = float(getattr(args, "focal_late_boost_max", 1.5))
    if wmax <= 1.0 or end <= start:
        return 1.0
    if step <= start:
        return 1.0
    if step >= end:
        return wmax
    progress = (step - start) / (end - start)
    factor = 0.5 * (1.0 - math.cos(math.pi * progress))   # 0->1 cosine S-curve
    return 1.0 + (wmax - 1.0) * factor


def compute_losses(
    logits: torch.Tensor, target: torch.Tensor, *,
    focal_gamma: float, tversky_weight: float, focal_weight: float = 1.0,
    channel_weights: Optional[torch.Tensor] = None,
) -> dict:
    logits = logits.float()
    target = target.float()
    focal_per_sample = weighted_focal_loss_per_sample(
        logits, target, gamma=focal_gamma, channel_weights=channel_weights
    )
    focal = focal_per_sample.mean()
    tv = tversky_loss(logits, target, channel_weights=channel_weights)
    total = focal_weight * focal + tversky_weight * tv
    return {
        "loss": total,
        "focal": focal.detach(),
        "focal_per_sample": focal_per_sample.detach(),
        "tversky": tv.detach(),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
@torch.no_grad()
def run_validation(module, valid_dataloader, accelerator, args,
                   channel_weights: Optional[torch.Tensor] = None) -> dict:
    module.eval()
    a = {k: 0.0 for k in (
        "loss", "focal", "tversky",
        "prob_sum_solvent", "n_solvent",
        "prob_sum_protein", "n_protein",
        "prob_sum_at_ca",   "n_at_ca",
        "n_ca_ge_05", "n_ca_ge_07",
        "n_samples",
    )}

    for x, target, _emdb_ids in valid_dataloader:
        logits = module(x)
        losses = compute_losses(
            logits, target,
            focal_gamma=args.focal_gamma,
            tversky_weight=args.tversky_weight,
            focal_weight=1.0,            # report unweighted focal during valid
            channel_weights=channel_weights,
        )
        probs = torch.sigmoid(logits.float())
        ca_probs   = probs[:, 0]
        target_ca  = target[:, 0]
        input_val  = x[:, 0]
        bs = x.shape[0]

        a["n_samples"] += bs
        a["loss"]    += float(losses["loss"].item())    * bs
        a["focal"]   += float(losses["focal"].item())   * bs
        a["tversky"] += float(losses["tversky"].item()) * bs

        iv_flat = input_val.reshape(bs, -1).float()
        cp_flat = ca_probs.reshape(bs, -1).float()
        tc_flat = target_ca.reshape(bs, -1).float()
        q70 = torch.quantile(iv_flat, 0.70, dim=1, keepdim=True)
        q95 = torch.quantile(iv_flat, 0.95, dim=1, keepdim=True)
        solvent_mask = iv_flat < q70
        protein_mask = iv_flat > q95
        at_ca_mask   = tc_flat > 0.5

        a["prob_sum_solvent"] += float((cp_flat * solvent_mask).sum().item())
        a["n_solvent"]        += float(solvent_mask.sum().item())
        a["prob_sum_protein"] += float((cp_flat * protein_mask).sum().item())
        a["n_protein"]        += float(protein_mask.sum().item())
        a["prob_sum_at_ca"]   += float((cp_flat * at_ca_mask).sum().item())
        a["n_at_ca"]          += float(at_ca_mask.sum().item())
        ca_only = cp_flat[at_ca_mask]
        a["n_ca_ge_05"] += float((ca_only > 0.5).sum().item())
        a["n_ca_ge_07"] += float((ca_only > 0.7).sum().item())

    module.train()

    n = max(1.0, a["n_samples"])
    eps = 1e-6
    solvent_mean = a["prob_sum_solvent"] / max(a["n_solvent"], 1.0)
    protein_mean = a["prob_sum_protein"] / max(a["n_protein"], 1.0)
    at_ca_mean   = a["prob_sum_at_ca"]   / max(a["n_at_ca"],   1.0)
    return {
        "loss":     a["loss"]    / n,
        "focal":    a["focal"]   / n,
        "tversky":  a["tversky"] / n,
        "solvent_mean":  solvent_mean,
        "protein_mean":  protein_mean,
        "at_ca_mean":    at_ca_mean,
        "ca_recall_0.5": a["n_ca_ge_05"] / max(a["n_at_ca"], 1.0),
        "ca_recall_0.7": a["n_ca_ge_07"] / max(a["n_at_ca"], 1.0),
        "snr_vs_solvent": at_ca_mean / (solvent_mean + eps),
        "snr_vs_protein": at_ca_mean / (protein_mean + eps),
    }


# ---------------------------------------------------------------------------
# Tversky ramp (linear | cosine) — same as v2
# ---------------------------------------------------------------------------
def _tversky_weight_at(step: int, args) -> float:
    warmup = int(getattr(args, "tversky_warmup_steps", 10_000))
    ramp = int(getattr(args, "tversky_ramp_steps", 30_000))
    shape = getattr(args, "tversky_ramp_shape", "cosine")
    weight_max = float(getattr(args, "tversky_weight", 1.0))

    if step < warmup:
        return 0.0
    progress = min(1.0, (step - warmup) / max(1, ramp))
    if shape == "linear":
        factor = progress
    elif shape == "cosine":
        factor = 0.5 * (1.0 - math.cos(math.pi * progress))
    else:
        factor = progress
    return weight_max * factor


# ---------------------------------------------------------------------------
# Periodic full-map inference output
# ---------------------------------------------------------------------------
@torch.no_grad()
def dump_full_mrc_predictions(
    module, accelerator, step: int, pickle_paths: List[str],
    out_root: str, box_size: int, stride: int, batch_size: int,
) -> None:
    if not pickle_paths:
        return
    if accelerator.is_main_process:
        from model_angelo.c_alpha.diag_infer import (
            sliding_window_infer, load_from_pickle,
        )
        from model_angelo.utils.grid import save_mrc
        from model_angelo.utils.save_pdb_utils import points_to_pdb

        out_dir = Path(out_root) / "mrc_preds" / f"step_{step}"
        out_dir.mkdir(parents=True, exist_ok=True)

        net = accelerator.unwrap_model(module)
        net.eval()
        device = accelerator.device
        for pkl in pickle_paths:
            try:
                grid, vs, go, ca = load_from_pickle(Path(pkl))
            except Exception as e:
                print(f"[mrc-dump step {step}] failed to load {pkl}: {e}", flush=True)
                continue
            probs = sliding_window_infer(
                net, grid, str(device),
                box_size=box_size, stride=stride, batch_size=batch_size, crop=6,
            )
            name = Path(pkl).stem
            sub = out_dir / name
            sub.mkdir(parents=True, exist_ok=True)
            save_mrc(probs[0].astype(np.float32), vs, go.copy(),
                     str(sub / "ca_output_grid.mrc"))
            if len(ca) and not (sub / "ground_truth_ca.cif").exists():
                points_to_pdb(str(sub / "ground_truth_ca.cif"), ca)
            print(f"[mrc-dump step {step}] wrote {name} -> {sub}", flush=True)
        net.train()
    accelerator.wait_for_everyone()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------
def train(
    *,
    logger,
    step: int,
    module,
    accelerator,
    train_dataloader,
    valid_dataloader,
    opt,
    learning_rate_schedule,
    args,
    summary_writer_wrapper=None,
):
    channel_weights = _parse_channel_weights(getattr(args, "channel_weights", "1,1,1"))
    if accelerator.is_main_process:
        logger.info(
            f"[v2.1] channel_weights = {channel_weights.tolist()}  "
            f"focal_late_boost: start={args.focal_late_boost_start} "
            f"end={args.focal_late_boost_end} max={args.focal_late_boost_max}"
        )

    module.train()
    t0 = time.time()
    last_log_t = t0
    entry_path = Path(args.log_dir) / f"entry_focal_rank{accelerator.process_index}.csv"
    entry_has_data = step > 0 and entry_path.exists() and entry_path.stat().st_size > 0
    entry_file = open(entry_path, "a" if step > 0 else "w", newline="", buffering=1)
    entry_writer = csv.writer(entry_file)
    if not entry_has_data:
        entry_writer.writerow(["step", "emdb_id", "focal_loss"])
    every_step_writer = every_20_writer = None
    every_step_file = every_20_file = None
    if accelerator.is_main_process:
        every_step_path = Path(args.log_dir) / "focal_every_step.csv"
        every_20_path = Path(args.log_dir) / "focal_every_20_steps.csv"
        every_step_has_data = step > 0 and every_step_path.exists() and every_step_path.stat().st_size > 0
        every_20_has_data = step > 0 and every_20_path.exists() and every_20_path.stat().st_size > 0
        every_step_file = open(every_step_path, "a" if step > 0 else "w", newline="", buffering=1)
        every_20_file = open(every_20_path, "a" if step > 0 else "w", newline="", buffering=1)
        every_step_writer = csv.writer(every_step_file)
        every_20_writer = csv.writer(every_20_file)
        if not every_step_has_data:
            every_step_writer.writerow(["step", "global_focal_loss"])
        if not every_20_has_data:
            every_20_writer.writerow(["step", "global_focal_loss"])
    train_iter = iter(train_dataloader)
    total_steps = args.num_steps
    accumulation_steps = max(1, int(getattr(args, "gradient_accumulation_steps", 1)))
    while step < total_steps:
        opt.zero_grad(set_to_none=True)
        current_tv_w    = _tversky_weight_at(step, args)
        current_focal_w = _focal_weight_at(step, args)
        micro_focal, micro_tversky, micro_loss, entry_rows = [], [], [], []
        for _micro_step in range(accumulation_steps):
            try:
                x, target, emdb_ids = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dataloader)
                x, target, emdb_ids = next(train_iter)
            logits = module(x)
            losses = compute_losses(
                logits, target,
                focal_gamma=args.focal_gamma,
                tversky_weight=current_tv_w,
                focal_weight=current_focal_w,
                channel_weights=channel_weights,
            )
            if not torch.isfinite(losses["loss"]):
                raise FloatingPointError(
                    f"non-finite loss at step {step}: {float(losses['loss'])}"
                )
            accelerator.backward(losses["loss"] / accumulation_steps)
            micro_focal.append(float(accelerator.gather(losses["focal"].reshape(1)).mean()))
            micro_tversky.append(float(accelerator.gather(losses["tversky"].reshape(1)).mean()))
            micro_loss.append(float(accelerator.gather(losses["loss"].reshape(1)).mean()))
            entry_rows.extend(
                (emdb_id, float(value))
                for emdb_id, value in zip(emdb_ids, losses["focal_per_sample"])
            )
        grad_norm = torch.tensor(float("nan"), device=accelerator.device)
        if args.grad_clip and args.grad_clip > 0:
            grad_norm = accelerator.clip_grad_norm_(module.parameters(), args.grad_clip)
        opt.step()
        learning_rate_schedule.step()
        step += 1

        global_focal = statistics.mean(micro_focal)
        global_tversky = statistics.mean(micro_tversky)
        global_loss = statistics.mean(micro_loss)
        entry_writer.writerows(
            (step, emdb_id, value) for emdb_id, value in entry_rows
        )
        if accelerator.is_main_process:
            every_step_writer.writerow([step, global_focal])
            if step % 20 == 0:
                every_20_writer.writerow([step, global_focal])

        if accelerator.is_main_process and step % args.log_every == 0:
            now = time.time()
            lr_now = opt.param_groups[0]["lr"]
            samples_per_s = (
                args.log_every * args.batch_size * accelerator.num_processes * accumulation_steps
                / max(now - last_log_t, 1e-6)
            )
            max_mem_gb = (
                torch.cuda.max_memory_allocated(accelerator.device) / (1 << 30)
                if torch.cuda.is_available() else 0.0
            )
            msg = (
                f"step {step}/{total_steps} "
                f"loss={global_loss:.4f} "
                f"focal={global_focal:.4f} "
                f"tversky={global_tversky:.4f} "
                f"tv_w={current_tv_w:.3f} fw={current_focal_w:.3f} "
                f"grad_norm={float(grad_norm):.3f} "
                f"lr={lr_now:.2e} "
                f"samples_s={samples_per_s:.2f} max_mem_gb={max_mem_gb:.2f} "
                f"elapsed={(time.time() - t0) / 60:.1f}m"
            )
            logger.info(msg)
            if summary_writer_wrapper is not None:
                summary_writer_wrapper.log({
                    "train/loss": global_loss,
                    "train/focal": global_focal,
                    "train/tversky": global_tversky,
                    "train/tversky_weight": current_tv_w,
                    "train/focal_weight":  current_focal_w,
                    "train/grad_norm": float(grad_norm),
                    "train/lr": lr_now,
                    "train/samples_per_s": samples_per_s,
                    "train/max_memory_gb": max_mem_gb,
                }, step=step)
            last_log_t = now

        if step % args.valid_every == 0 and valid_dataloader is not None:
            stats = run_validation(module, valid_dataloader, accelerator, args,
                                   channel_weights=channel_weights)
            if accelerator.is_main_process:
                logger.info(
                    f"[valid] step {step} loss={stats['loss']:.4f} "
                    f"focal={stats['focal']:.4f} tversky={stats['tversky']:.4f} "
                    f"at_ca={stats['at_ca_mean']:.3f} solvent={stats['solvent_mean']:.3f} "
                    f"protein={stats['protein_mean']:.3f} "
                    f"snr_solv={stats['snr_vs_solvent']:.2f} snr_prot={stats['snr_vs_protein']:.2f} "
                    f"rec0.5={stats['ca_recall_0.5']:.1%} rec0.7={stats['ca_recall_0.7']:.1%}"
                )
                if summary_writer_wrapper is not None:
                    summary_writer_wrapper.log(
                        {f"valid/{k}": float(v) for k, v in stats.items()},
                        step=step,
                    )

        if accelerator.is_main_process and step % args.checkpoint_every == 0:
            checkpoint_save(
                step_num=step,
                log_dir=args.log_dir,
                accelerator=accelerator,
                module=accelerator.unwrap_model(module),
                optimizer=opt,
                learning_rate_scheduler=learning_rate_schedule,
            )
            logger.info(f"Saved checkpoint at step {step}")

        mrc_every = getattr(args, "mrc_pred_every", 0)
        if mrc_every and step % mrc_every == 0 and step > 0:
            mrc_paths_str = getattr(args, "mrc_pred_pickles", "") or ""
            pickle_paths = [p for p in mrc_paths_str.split(",") if p.strip()]
            if not pickle_paths:
                N = int(getattr(args, "mrc_pred_n_maps", 5))
                try:
                    with open(args.valid_dataset_list) as _f:
                        all_val = [line.strip() for line in _f if line.strip()]
                    sized = []
                    for p in all_val:
                        try:
                            sized.append((os.path.getsize(p), p))
                        except OSError:
                            continue
                    sized.sort()
                    pickle_paths = [p for _, p in sized[:N]]
                except Exception:
                    pickle_paths = []
            dump_full_mrc_predictions(
                module, accelerator, step, pickle_paths,
                out_root=args.log_dir,
                box_size=args.box_size,
                stride=getattr(args, "mrc_pred_stride", 32),
                batch_size=args.batch_size,
            )

    if accelerator.is_main_process:
        checkpoint_save(
            step_num=step,
            log_dir=args.log_dir,
            accelerator=accelerator,
            module=accelerator.unwrap_model(module),
            optimizer=opt,
            learning_rate_scheduler=learning_rate_schedule,
        )
        logger.info("Training complete.")
    entry_file.close()
    if every_step_file is not None:
        every_step_file.close()
        every_20_file.close()
