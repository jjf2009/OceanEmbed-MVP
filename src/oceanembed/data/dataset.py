"""Torch Dataset over the processed Zarr.

A sample is one (time, lat, lon) ocean pixel: normalised surface features in,
15-level temperature profile out. Splits are time-blocked -- a random pixel
split would put a pixel's own neighbours (and next day) in the test set.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
import xarray as xr
from torch.utils.data import Dataset

from ..config import Config
from .preprocess import SURFACE_FEATURES

Split = Literal["train", "val", "test"]

FEATURE_NAMES = [
    "sst", "ssh", "sss", "sin_doy", "cos_doy", "lat_norm", "lon_norm",
]
N_FEATURES = len(FEATURE_NAMES)


@dataclass
class NormStats:
    surface: dict
    temp_mean: np.ndarray
    temp_std: np.ndarray
    coords: dict

    @classmethod
    def load(cls, cfg: Config) -> "NormStats":
        raw = json.loads(cfg.path("norm_stats").read_text())
        return cls(
            surface=raw["surface"],
            temp_mean=np.asarray(raw["temperature"]["mean"], dtype="float32"),
            temp_std=np.asarray(raw["temperature"]["std"], dtype="float32"),
            coords=raw["coords"],
        )

    def denormalize(self, arr: np.ndarray) -> np.ndarray:
        return arr * self.temp_std + self.temp_mean


def time_slice(cfg: Config, split: Split) -> slice:
    t = cfg.time
    return {
        "train": slice(t.start, t.train_end),
        "val": slice(t.train_end, t.val_end),
        "test": slice(t.val_end, t.end),
    }[split]


def build_features(
    sst: np.ndarray, ssh: np.ndarray, sss: np.ndarray,
    doy: np.ndarray, lat: np.ndarray, lon: np.ndarray,
    stats: NormStats,
) -> np.ndarray:
    """Stack the 7 model inputs. All arrays broadcast to a common shape."""
    s = stats.surface
    c = stats.coords
    lat_norm = (lat - c["lat_min"]) / max(c["lat_max"] - c["lat_min"], 1e-6) * 2 - 1
    lon_norm = (lon - c["lon_min"]) / max(c["lon_max"] - c["lon_min"], 1e-6) * 2 - 1
    cols = [
        (sst - s["sst"]["mean"]) / s["sst"]["std"],
        (ssh - s["ssh"]["mean"]) / s["ssh"]["std"],
        (sss - s["sss"]["mean"]) / s["sss"]["std"],
        np.sin(2 * np.pi * doy / 365.25),
        np.cos(2 * np.pi * doy / 365.25),
        lat_norm,
        lon_norm,
    ]
    shape = np.broadcast_shapes(*[np.shape(c_) for c_ in cols])
    return np.stack([np.broadcast_to(c_, shape) for c_ in cols], axis=-1).astype("float32")


class ProfileDataset(Dataset):
    def __init__(self, cfg: Config, split: Split, max_samples: int | None = None, seed: int = 42):
        self.cfg = cfg
        self.split = split
        self.stats = NormStats.load(cfg)
        ds = xr.open_zarr(cfg.path("processed")).sel(time=time_slice(cfg, split))
        mask = ds["ocean_mask"].values  # (lat, lon)
        yi, xi = np.nonzero(mask)

        n_t = ds.sizes["time"]
        doy = ds["time"].dt.dayofyear.values.astype("float32")
        lat = ds["latitude"].values.astype("float32")
        lon = ds["longitude"].values.astype("float32")

        surf = {v: ds[v].values[:, yi, xi] for v in SURFACE_FEATURES}  # (time, npix)
        temp = ds["temperature"].values[:, :, yi, xi]                  # (time, level, npix)

        t_idx = np.repeat(np.arange(n_t), len(yi))
        p_idx = np.tile(np.arange(len(yi)), n_t)

        X = build_features(
            surf["sst"].reshape(-1), surf["ssh"].reshape(-1), surf["sss"].reshape(-1),
            doy[t_idx], lat[yi][p_idx], lon[xi][p_idx], self.stats,
        )
        Y = temp.transpose(0, 2, 1).reshape(-1, temp.shape[1])
        Y = (Y - self.stats.temp_mean) / self.stats.temp_std

        keep = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
        X, Y, t_idx, p_idx = X[keep], Y[keep], t_idx[keep], p_idx[keep]

        if max_samples is not None and len(X) > max_samples:
            sel = np.random.default_rng(seed).choice(len(X), max_samples, replace=False)
            X, Y, t_idx, p_idx = X[sel], Y[sel], t_idx[sel], p_idx[sel]

        self.X = torch.from_numpy(X)
        self.Y = torch.from_numpy(Y.astype("float32"))
        self.times = ds["time"].values[t_idx]
        self.lats = lat[yi][p_idx]
        self.lons = lon[xi][p_idx]

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, i: int):
        return self.X[i], self.Y[i]
