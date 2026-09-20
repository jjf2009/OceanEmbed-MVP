"""Load a checkpoint and predict profiles -- one point, or a whole day's field."""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
import xarray as xr

from .config import Config, load_config
from .data.dataset import NormStats, build_features
from .models.embed_mlp import build_model


class Predictor:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.stats = NormStats.load(cfg)
        ckpt = torch.load(cfg.path("checkpoint"), map_location="cpu", weights_only=False)
        self.depths = np.asarray(ckpt["depths"], dtype="float32")
        self.model = build_model(cfg, ckpt["n_features"], ckpt["n_levels"])
        self.model.load_state_dict(ckpt["state_dict"])
        self.model.eval()
        self.ds = xr.open_zarr(cfg.path("processed"))

    # -- inputs -----------------------------------------------------------
    def nearest(self, lat: float, lon: float, date: str) -> xr.Dataset:
        return self.ds.sel(latitude=lat, longitude=lon, method="nearest").sel(time=date, method="nearest")

    def _predict(self, feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with torch.no_grad():
            mu, logvar = self.model(torch.from_numpy(feats))
        mu = self.stats.denormalize(mu.numpy())
        sigma = np.exp(0.5 * logvar.numpy().clip(-7, 7)) * self.stats.temp_std
        return mu, sigma

    # -- public API -------------------------------------------------------
    def profile(self, lat: float, lon: float, date: str) -> dict:
        pt = self.nearest(lat, lon, date)
        if not bool(pt["ocean_mask"]):
            raise ValueError("requested point is land / masked")
        doy = float(np.asarray(pt["time"].dt.dayofyear))
        feats = build_features(
            np.array([float(pt["sst"])]), np.array([float(pt["ssh"])]), np.array([float(pt["sss"])]),
            np.array([doy]), np.array([float(pt["latitude"])]), np.array([float(pt["longitude"])]),
            self.stats,
        )
        mu, sigma = self._predict(feats)
        return {
            "lat": float(pt["latitude"]),
            "lon": float(pt["longitude"]),
            "date": str(np.datetime_as_string(pt["time"].values, unit="D")),
            "depths": self.depths.tolist(),
            "mu": mu[0].tolist(),
            "sigma": sigma[0].tolist(),
            "truth": [float(v) for v in pt["temperature"].values],
            "sst": float(pt["sst"]),
            "ssh": float(pt["ssh"]),
        }

    def field(self, date: str, depth: float) -> dict:
        day = self.ds.sel(time=date, method="nearest")
        mask = day["ocean_mask"].values
        yi, xi = np.nonzero(mask)
        lat = day["latitude"].values.astype("float32")
        lon = day["longitude"].values.astype("float32")
        doy = float(np.asarray(day["time"].dt.dayofyear))
        feats = build_features(
            day["sst"].values[yi, xi], day["ssh"].values[yi, xi], day["sss"].values[yi, xi],
            np.full(len(yi), doy, dtype="float32"), lat[yi], lon[xi], self.stats,
        )
        mu, _ = self._predict(feats)
        k = int(np.argmin(np.abs(self.depths - depth)))
        grid = np.full(mask.shape, np.nan, dtype="float32")
        grid[yi, xi] = mu[:, k]
        return {
            "date": str(np.datetime_as_string(day["time"].values, unit="D")),
            "depth": float(self.depths[k]),
            "lat": lat.tolist(),
            "lon": lon.tolist(),
            "values": [[None if not np.isfinite(v) else float(v) for v in row] for row in grid],
        }


@lru_cache(maxsize=1)
def get_predictor(config_path: str | None = None) -> Predictor:
    return Predictor(load_config(config_path) if config_path else load_config())
