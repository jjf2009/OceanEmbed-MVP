"""Raw cube -> analysis-ready Zarr on the 0.25 deg grid and 15 standard depths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import xarray as xr

from ..config import Config, load_config
from .download import output_path

SURFACE_FEATURES = ["sst", "ssh", "sss"]


def to_target_grid(ds: xr.Dataset, cfg: Config) -> xr.Dataset:
    """Interpolate onto the configured 0.25 deg lat/lon grid."""
    r = cfg.region
    lat = np.arange(r.lat_min, r.lat_max + 1e-9, r.resolution)
    lon = np.arange(r.lon_min, r.lon_max + 1e-9, r.resolution)
    if len(ds.latitude) == len(lat) and np.allclose(ds.latitude.values, lat):
        return ds
    return ds.interp(latitude=lat, longitude=lon, method="linear")


def build(cfg: Config, raw: Path | None = None) -> Path:
    raw = raw or output_path(cfg)
    ds = xr.open_dataset(raw)
    ds = to_target_grid(ds, cfg)

    depths = np.array(cfg.depths, dtype="float32")
    # GLORYS' shallowest level is ~0.5 m, so depth 0 needs extrapolation-by-clamp.
    temp = ds[cfg.source.target_var].interp(
        depth=depths, method="linear", kwargs={"fill_value": "extrapolate"}
    )
    temp = temp.rename({"depth": "level"}).assign_coords(level=depths)

    surface = xr.Dataset(
        {
            "sst": ds["thetao"].isel(depth=0, drop=True),
            "ssh": ds["zos"],
            "sss": ds["so"].isel(depth=0, drop=True),
        }
    )
    # Ocean where every surface input and the whole profile are finite for all times.
    ocean = (
        np.isfinite(surface["sst"]).all("time")
        & np.isfinite(surface["ssh"]).all("time")
        & np.isfinite(surface["sss"]).all("time")
        & np.isfinite(temp).all(("time", "level"))
    )

    out = xr.Dataset(
        {
            "sst": surface["sst"],
            "ssh": surface["ssh"],
            "sss": surface["sss"],
            "temperature": temp.transpose("time", "level", "latitude", "longitude"),
            "ocean_mask": ocean,
        }
    )
    out.attrs["synthetic"] = int(ds.attrs.get("synthetic", 0))

    stats = compute_norm_stats(out, cfg)
    dest = cfg.path("processed")
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    out.chunk({"time": 90}).to_zarr(dest, mode="w")
    stats_path = cfg.path("norm_stats")
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[preprocess] wrote {dest}")
    print(f"[preprocess] dims: {dict(out.sizes)}  ocean pixels: {int(ocean.sum())}")
    print(f"[preprocess] wrote {stats_path}")
    return dest


def compute_norm_stats(ds: xr.Dataset, cfg: Config) -> dict:
    """Mean/std from the TRAIN time block only -- val/test stats would leak."""
    train = ds.sel(time=slice(cfg.time.start, cfg.time.train_end))
    mask = ds["ocean_mask"]
    stats: dict = {"surface": {}, "levels": [float(d) for d in cfg.depths]}
    for var in SURFACE_FEATURES:
        v = train[var].where(mask)
        stats["surface"][var] = {"mean": float(v.mean()), "std": float(v.std()) or 1.0}
    t = train["temperature"].where(mask)
    stats["temperature"] = {
        "mean": [float(x) for x in t.mean(("time", "latitude", "longitude")).values],
        "std": [float(x) or 1.0 for x in t.std(("time", "latitude", "longitude")).values],
    }
    r = cfg.region
    stats["coords"] = {
        "lat_min": r.lat_min, "lat_max": r.lat_max,
        "lon_min": r.lon_min, "lon_max": r.lon_max,
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config) if args.config else load_config()
    build(cfg)


if __name__ == "__main__":
    main()
