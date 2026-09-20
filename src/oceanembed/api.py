"""FastAPI service + static demo UI."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles

from .config import load_config
from .infer import get_predictor

app = FastAPI(title="OceanEmbed", description="Subsurface temperature reconstruction from surface fields")
WEB_DIR = Path(__file__).resolve().parents[2] / "web"


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/meta")
def meta() -> dict:
    cfg = load_config()
    ds = xr.open_zarr(cfg.path("processed"))
    metrics_file = cfg.path("reports_dir") / "metrics.json"
    return {
        "region": vars(cfg.region),
        "depths": cfg.depths,
        "dates": {
            "start": str(np.datetime_as_string(ds["time"].values[0], unit="D")),
            "end": str(np.datetime_as_string(ds["time"].values[-1], unit="D")),
            "test_start": cfg.time.val_end,
        },
        "synthetic": bool(ds.attrs.get("synthetic", 0)),
        "metrics": json.loads(metrics_file.read_text()) if metrics_file.exists() else None,
    }


@app.get("/predict/profile")
def predict_profile(
    lat: float = Query(..., ge=-90, le=90),
    lon: float = Query(..., ge=-180, le=360),
    date: str = Query(...),
) -> dict:
    try:
        return get_predictor().profile(lat, lon, date)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/predict/field")
def predict_field(date: str = Query(...), depth: float = Query(100.0)) -> dict:
    return get_predictor().field(date, depth)


if WEB_DIR.exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
