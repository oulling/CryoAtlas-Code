"""CryoAtlas addition: chain-aware CNN crop loading and augmentation.

Adds protein/nucleotide anchor balancing, chain-safe backbone targets, and
crop-spectrum colored noise with scale 6 to the released inference toolkit.

New pickles may contain ``cas_chains`` with separate protein C-alpha and
nucleotide phosphorus coordinates.  Keeping chains separate prevents false
protein backbone links, and positive crops are balanced between protein and
nucleic-acid anchors.  Legacy pickles still work unchanged.
"""
from __future__ import annotations

import os
import pickle
import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from model_angelo.data.dataset_preprocess import decompress_data
from model_angelo.utils.grid import (
    apply_bfactor_to_map,
    coloured_noise_spectrum,
    get_spectral_indices,
    grid_rot90,
    voxelize_coordinates_numpy,
)


def _trace_backbone(cas: np.ndarray, max_segment_len: float = 4.5,
                    step: float = 0.6) -> np.ndarray:
    if len(cas) < 2:
        return cas.copy() if len(cas) else np.zeros((0, 3), dtype=np.float32)
    diffs = cas[1:] - cas[:-1]
    dists = np.linalg.norm(diffs, axis=-1)
    keep = dists < max_segment_len
    points = [cas]
    for idx in np.where(keep)[0]:
        d = dists[idx]
        n = max(1, int(np.ceil(d / step)))
        ts = np.linspace(0, 1, n + 1)[1:-1]
        if len(ts):
            points.append(cas[idx][None] + ts[:, None] * diffs[idx][None])
    return np.concatenate(points, axis=0)


def _spectral_amplitude_all_shells(fourier: np.ndarray, radius: np.ndarray) -> np.ndarray:
    """Mean Fourier amplitude for every radial shell present in a crop."""
    radius = radius.astype(np.int64, copy=False)
    size = int(radius.max()) + 1
    count = np.bincount(radius.ravel(), minlength=size)
    total = np.bincount(radius.ravel(), weights=np.abs(fourier).ravel(), minlength=size)
    amplitude = np.divide(total, count, out=np.zeros(size), where=count > 0)
    missing = count == 0
    if missing.any():
        present = np.flatnonzero(~missing)
        amplitude[missing] = np.interp(np.flatnonzero(missing), present, amplitude[present])
    return amplitude.astype(np.float32)


class CASDataset(Dataset):
    def __init__(
        self,
        dataset_list: str,
        box_size: int = 64,
        positional_encoding_dim: Optional[int] = None,
        num_cache_elements: int = 5,
        do_data_augmentation: bool = True,
        do_bfactor_augmentation: bool = True,
        global_normalization: bool = False,
        log_path: Optional[str] = None,
        max_noise: float = 0.3,
        bfactor_range: float = 25.0,
        dataset_length: Optional[int] = None,
        shuffle_idx: bool = False,
        target_sigma: float = 0.7,
        random_crop_prob: float = 0.25,
        noise_spectrum_mode: str = "crop",
        # Calibrated on a stratified normal-loss cohort to preserve legacy
        # median difficulty after removing full-map-size dependence.
        noise_spectrum_scale: float = 6.0,
    ):
        super().__init__()
        with open(dataset_list, "r") as f:
            self.paths = [line.strip() for line in f if line.strip()]
        if shuffle_idx:
            random.shuffle(self.paths)

        self.box_size = int(box_size)
        self.positional_encoding_dim = positional_encoding_dim
        self.num_cache_elements = max(1, int(num_cache_elements))
        self.do_data_augmentation = bool(do_data_augmentation)
        self.do_bfactor_augmentation = bool(do_bfactor_augmentation)
        self.global_normalization = bool(global_normalization)
        self.max_noise = float(max_noise)
        self.bfactor_range = float(bfactor_range)
        self.dataset_length = dataset_length
        self.target_sigma = float(target_sigma)
        self.random_crop_prob = float(random_crop_prob)
        if noise_spectrum_mode not in {"crop", "legacy"}:
            raise ValueError("noise_spectrum_mode must be 'crop' or 'legacy'")
        self.noise_spectrum_mode = noise_spectrum_mode
        self.noise_spectrum_scale = float(noise_spectrum_scale)
        if self.noise_spectrum_scale <= 0:
            raise ValueError("noise_spectrum_scale must be positive")

        self._cache: Dict[str, dict] = {}
        self._cache_order: List[str] = []
        self._log_path = log_path

    def __len__(self) -> int:
        if self.dataset_length is not None:
            return int(self.dataset_length)
        return len(self.paths) * 32

    def __getitem__(self, idx: int):
        if not self.paths:
            raise RuntimeError("Dataset list is empty")
        for offset in range(min(32, len(self.paths))):
            path = self.paths[(idx + offset) % len(self.paths)]
            try:
                inputs, targets = self._sample_crop(self._load(path))
                return inputs, targets, os.path.basename(path).removesuffix(".pkl")
            except Exception as e:
                if self._log_path is not None:
                    with open(self._log_path, "a") as f:
                        f.write(f"[CASDataset v2] failed on {path}: {e}\n")
        raise RuntimeError("Failed to load 32 consecutive dataset entries")

    def _load(self, path: str) -> dict:
        if path in self._cache:
            return self._cache[path]
        with open(path, "rb") as f:
            ds = pickle.load(f)
        ds["grid"] = decompress_data(ds["grid_io"]).astype(np.float32)
        origin = np.asarray(ds["global_origin"], dtype=np.float32)
        vs = float(ds["voxel_size"])
        if isinstance(ds["cas"], dict):
            ca = np.asarray(ds["cas"].get("CA", np.zeros((0, 3))), dtype=np.float32)
            p = np.asarray(ds["cas"].get("P", np.zeros((0, 3))), dtype=np.float32)
            ds["ca_vox"] = (ca - origin) / vs if len(ca) else np.zeros((0, 3), np.float32)
            ds["p_vox"] = (p - origin) / vs if len(p) else np.zeros((0, 3), np.float32)
        else:
            ds["ca_vox"] = np.asarray(ds["cas"], dtype=np.float32)
            ds["p_vox"] = np.zeros((0, 3), dtype=np.float32)

        chains = ds.get("cas_chains")
        if chains:
            ds["ca_chains_vox"] = [
                (np.asarray(c["coords"], dtype=np.float32) - origin) / vs
                for c in chains if c.get("kind") == "protein" and len(c.get("coords", ()))
            ]
            ds["p_chains_vox"] = [
                (np.asarray(c["coords"], dtype=np.float32) - origin) / vs
                for c in chains if c.get("kind") in {"rna", "dna"} and len(c.get("coords", ()))
            ]
            ds["ca_vox"] = self._concat_coords(ds["ca_chains_vox"])
            ds["p_vox"] = self._concat_coords(ds["p_chains_vox"])
        else:
            ds["ca_chains_vox"] = [ds["ca_vox"]] if len(ds["ca_vox"]) else []
            ds["p_chains_vox"] = [ds["p_vox"]] if len(ds["p_vox"]) else []
        self._cache[path] = ds
        self._cache_order.append(path)
        if len(self._cache_order) > self.num_cache_elements:
            evict = self._cache_order.pop(0)
            self._cache.pop(evict, None)
        return ds

    def _sample_crop(self, ds: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        grid = ds["grid"]
        ca = ds["ca_vox"]
        p = ds["p_vox"]
        bz = self.box_size
        gz, gy, gx = grid.shape

        has_anchor = len(ca) or len(p)
        use_random = not has_anchor or random.random() < self.random_crop_prob
        if not use_random:
            # Equal modality probability prevents the much larger protein
            # residue pool from starving RNA/DNA supervision.
            if len(ca) and len(p):
                anchors = p if random.random() < 0.5 else ca
            else:
                anchors = ca if len(ca) else p
            anchor = anchors[random.randrange(len(anchors))]
            ax, ay, az = anchor
            jitter = bz // 4
            cz = int(round(az)) + random.randint(-jitter, jitter)
            cy = int(round(ay)) + random.randint(-jitter, jitter)
            cx = int(round(ax)) + random.randint(-jitter, jitter)
        else:
            cz = random.randrange(bz // 2, max(bz // 2 + 1, gz - bz // 2))
            cy = random.randrange(bz // 2, max(bz // 2 + 1, gy - bz // 2))
            cx = random.randrange(bz // 2, max(bz // 2 + 1, gx - bz // 2))

        z0 = int(np.clip(cz - bz // 2, 0, max(0, gz - bz)))
        y0 = int(np.clip(cy - bz // 2, 0, max(0, gy - bz)))
        x0 = int(np.clip(cx - bz // 2, 0, max(0, gx - bz)))

        crop = grid[z0:z0 + bz, y0:y0 + bz, x0:x0 + bz]
        crop = self._pad_to_box(crop, bz)

        voxel_size = float(ds["voxel_size"])
        bfac = 0.0
        if self.do_data_augmentation and self.do_bfactor_augmentation:
            bfac = random.uniform(-self.bfactor_range, self.bfactor_range)
            crop = apply_bfactor_to_map(crop, voxel_size=voxel_size, bfactor=bfac)

        if self.do_data_augmentation and self.max_noise > 0:
            crop = self._add_coloured_noise(crop, ds.get("grid_spectral_amplitude"))

        ca_local = ca - np.array([x0, y0, z0], dtype=np.float32) if len(ca) else ca
        p_local = p - np.array([x0, y0, z0], dtype=np.float32) if len(p) else p
        ca_in = self._in_box(ca_local, bz)
        p_in = self._in_box(p_local, bz)

        ca_target = np.zeros((bz, bz, bz), dtype=np.float32)
        p_target = np.zeros((bz, bz, bz), dtype=np.float32)
        bb_target = np.zeros((bz, bz, bz), dtype=np.float32)

        if len(ca_in):
            voxelize_coordinates_numpy(ca_target, ca_in, clip_output=True)
        if len(p_in):
            voxelize_coordinates_numpy(p_target, p_in, clip_output=True)
        ca_chains_local = [
            chain - np.array([x0, y0, z0], dtype=np.float32)
            for chain in ds["ca_chains_vox"]
        ]
        for chain in ca_chains_local:
            trace_pts = _trace_backbone(self._in_box(chain, bz))
            trace_pts = self._in_box(trace_pts, bz)
            if len(trace_pts):
                voxelize_coordinates_numpy(bb_target, trace_pts, clip_output=True)

        target = np.stack([ca_target, p_target, bb_target], axis=0)

        if self.do_data_augmentation:
            rot_idx = random.randint(0, 23)
            crop = grid_rot90(crop, rot_idx)
            target = grid_rot90(target, rot_idx, has_batch=True)

        crop = np.ascontiguousarray(crop)
        target = np.ascontiguousarray(target)

        if "grid_std" in ds:
            second = self._pad_to_box(ds["grid_std"][z0:z0 + bz, y0:y0 + bz, x0:x0 + bz], bz)
        else:
            second = self._local_std(crop)
        x = np.stack([crop, second], axis=0)

        if self.global_normalization:
            x[0] = (x[0] - x[0].mean()) / (x[0].std() + 1e-6)
            x[1] = x[1] / (x[1].max() + 1e-6)

        return torch.from_numpy(x.astype(np.float32)), torch.from_numpy(target)

    @staticmethod
    def _concat_coords(chains: List[np.ndarray]) -> np.ndarray:
        return (np.concatenate(chains, axis=0) if chains
                else np.zeros((0, 3), dtype=np.float32))

    @staticmethod
    def _pad_to_box(crop: np.ndarray, bz: int) -> np.ndarray:
        if crop.shape == (bz, bz, bz):
            return crop
        padded = np.zeros((bz, bz, bz), dtype=crop.dtype)
        sz = min(bz, crop.shape[0])
        sy = min(bz, crop.shape[1])
        sx = min(bz, crop.shape[2])
        padded[:sz, :sy, :sx] = crop[:sz, :sy, :sx]
        return padded

    @staticmethod
    def _in_box(coords: np.ndarray, bz: int) -> np.ndarray:
        if len(coords) == 0:
            return coords
        mask = (
            (coords[:, 0] >= 0) & (coords[:, 0] < bz)
            & (coords[:, 1] >= 0) & (coords[:, 1] < bz)
            & (coords[:, 2] >= 0) & (coords[:, 2] < bz)
        )
        return coords[mask]

    def _add_coloured_noise(self, crop: np.ndarray, sa: Optional[np.ndarray]) -> np.ndarray:
        crop_ft = np.fft.rfftn(crop)
        R = get_spectral_indices(crop_ft).astype(int)
        if self.noise_spectrum_mode == "crop":
            # Use the crop's physical frequencies. Directly indexing the
            # full-map spectrum makes noise strength depend on full-map size.
            sa = _spectral_amplitude_all_shells(crop_ft, R)
        elif sa is None:
            sa = np.ones(int(R.max()) + 1, dtype=np.float32)
        noise_spec, _, _ = coloured_noise_spectrum(
            sa[: R.max() + 1],
            lres_noise_power=random.uniform(0.0, self.max_noise),
            hres_noise_power=random.uniform(0.0, self.max_noise),
        )
        noise_spec *= self.noise_spectrum_scale
        amp = noise_spec[np.clip(R, 0, len(noise_spec) - 1)]
        noise = np.random.randn(*crop.shape).astype(np.float32)
        noise_ft = np.fft.rfftn(noise)
        noise_ft = noise_ft * amp / (np.abs(noise_ft) + 1e-6)
        noise = np.fft.irfftn(noise_ft, s=crop.shape).astype(np.float32)
        return crop + noise

    @staticmethod
    def _local_std(crop: np.ndarray, k: int = 5) -> np.ndarray:
        from scipy.ndimage import uniform_filter
        mean = uniform_filter(crop, size=k)
        mean_sq = uniform_filter(crop * crop, size=k)
        var = np.maximum(mean_sq - mean * mean, 0.0)
        std = np.sqrt(var)
        m = std.max()
        return std / (m + 1e-6)
