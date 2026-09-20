"""Fetch the GLORYS subset for the configured box, or synthesise an equivalent cube.

Real path needs free CMEMS credentials (see .env.example). The --synthetic path
produces a cube with the same dims/vars so every downstream stage is runnable
without credentials.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ..config import Config, load_config

# GLORYS native levels spanning 0-1000 m (subset of the real 50-level grid).
NATIVE_DEPTHS = [
    0.49, 2.65, 5.08, 7.93, 11.41, 15.81, 21.60, 29.44, 40.34, 55.76,
    77.85, 92.33, 109.73, 130.67, 155.85, 186.13, 222.48, 266.04, 318.13,
    380.21, 453.94, 541.09, 643.57, 763.33, 902.34, 1062.44,
]


def output_path(cfg: Config) -> Path:
    return cfg.path("raw_dir") / f"{cfg.name}_glorys.nc"


def download_real(cfg: Config, out: Path) -> Path:
    import copernicusmarine

    out.parent.mkdir(parents=True, exist_ok=True)
    copernicusmarine.subset(
        dataset_id=cfg.source.dataset_id,
        variables=cfg.source.surface_vars,
        minimum_longitude=cfg.region.lon_min,
        maximum_longitude=cfg.region.lon_max,
        minimum_latitude=cfg.region.lat_min,
        maximum_latitude=cfg.region.lat_max,
        start_datetime=f"{cfg.time.start}T00:00:00",
        end_datetime=f"{cfg.time.end}T00:00:00",
        minimum_depth=0.0,
        maximum_depth=1100.0,
        output_filename=out.name,
        output_directory=str(out.parent),
        overwrite=True,
    )
    return out


def synthesise(cfg: Config, out: Path, seed: int = 0) -> Path:
    """Physically-shaped fake cube: exponential thermocline driven by SST and SSH.

    Deliberately learnable but non-trivial -- the thermocline scale depth depends
    on SSH and position, so a model must use more than SST alone.
    """
    rng = np.random.default_rng(seed)
    r = cfg.region
    lat = np.arange(r.lat_min, r.lat_max + 1e-9, r.resolution)
    lon = np.arange(r.lon_min, r.lon_max + 1e-9, r.resolution)
    time = pd.date_range(cfg.time.start, cfg.time.end, freq="D")
    depth = np.array(NATIVE_DEPTHS)

    t = np.arange(len(time))
    doy = time.dayofyear.to_numpy()
    LAT = lat[None, :, None]
    LON = lon[None, None, :]
    T = t[:, None, None]

    # Surface fields -------------------------------------------------------
    seasonal = 1.8 * np.sin(2 * np.pi * (doy[:, None, None] - 100) / 365.25)
    sst = (
        29.5
        - 0.28 * (LAT - r.lat_min)
        + seasonal
        + 0.6 * np.sin(2 * np.pi * (LON - r.lon_min) / 9.0 + 0.02 * T)
        + rng.normal(0, 0.25, (len(time), len(lat), len(lon)))
    )
    # Mesoscale-ish SSH: slow travelling pattern + noise.
    zos = (
        0.12 * np.sin(2 * np.pi * (LON - r.lon_min) / 6.0 - 0.03 * T)
        * np.cos(2 * np.pi * (LAT - r.lat_min) / 7.0)
        + 0.04 * np.sin(2 * np.pi * (doy[:, None, None]) / 365.25)
        + rng.normal(0, 0.015, (len(time), len(lat), len(lon)))
    )
    sss_surf = (
        36.2
        - 0.09 * (LAT - r.lat_min)
        + 2.0 * zos
        + rng.normal(0, 0.05, (len(time), len(lat), len(lon)))
    )

    # Subsurface: T(z) = T_deep + (SST - T_deep) * exp(-z / H), H from SSH.
    H = 55.0 + 260.0 * (zos + 0.2) + 1.5 * (LAT - r.lat_min)  # scale depth (m)
    H = np.clip(H, 25.0, 400.0)
    T_deep = 4.2 + 0.02 * (LAT - r.lat_min)
    Z = depth[None, :, None, None]
    thetao = (
        T_deep[:, None, :, :]
        + (sst - T_deep)[:, None, :, :] * np.exp(-Z / H[:, None, :, :])
        + rng.normal(0, 0.08, (len(time), len(depth), len(lat), len(lon)))
    )
    so = sss_surf[:, None, :, :] - 1.4 * (1 - np.exp(-Z / 300.0))

    # A land mask so the ocean-mask code path is exercised.
    land = (LAT[0] > 18.8) & (LON[0] < 63.0)
    thetao[:, :, land] = np.nan
    so[:, :, land] = np.nan
    sst[:, land] = np.nan
    zos[:, land] = np.nan

    ds = xr.Dataset(
        {
            "thetao": (("time", "depth", "latitude", "longitude"), thetao.astype("float32")),
            "so": (("time", "depth", "latitude", "longitude"), so.astype("float32")),
            "zos": (("time", "latitude", "longitude"), zos.astype("float32")),
        },
        coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
        attrs={"source": "SYNTHETIC -- not real GLORYS data", "synthetic": 1},
    )
    # Surface thetao layer must equal the SST field used to build the profile.
    ds["thetao"][:, 0, :, :] = sst.astype("float32")
    out.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--synthetic", action="store_true", help="skip CMEMS, generate a fake cube")
    ap.add_argument("--force", action="store_true", help="re-download even if the file exists")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else load_config()
    out = output_path(cfg)
    if out.exists() and not args.force:
        print(f"[download] {out} already exists -- skipping (use --force to refetch)")
        return
    if args.synthetic:
        synthesise(cfg, out)
        print(f"[download] wrote SYNTHETIC cube -> {out}")
    else:
        download_real(cfg, out)
        print(f"[download] wrote GLORYS subset -> {out}")


if __name__ == "__main__":
    main()
