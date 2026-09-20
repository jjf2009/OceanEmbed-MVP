"""Per-depth skill on the held-out test block, against two baselines.

Baselines matter: a model that cannot beat monthly climatology has learnt
nothing worth claiming.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch
import xarray as xr

from .config import Config, load_config
from .data.dataset import N_FEATURES, ProfileDataset, time_slice
from .models.embed_mlp import build_model


def metrics(pred: np.ndarray, truth: np.ndarray, depths) -> list[dict]:
    out = []
    for k, d in enumerate(depths):
        p, t = pred[:, k], truth[:, k]
        err = p - t
        var = float(np.var(t))
        out.append({
            "depth": float(d),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "bias": float(np.mean(err)),
            "r2": float(1 - np.mean(err ** 2) / var) if var > 0 else float("nan"),
        })
    return out


def climatology_baseline(cfg: Config, test_ds: ProfileDataset) -> np.ndarray:
    """Train-block mean profile per calendar month, looked up per test sample."""
    z = xr.open_zarr(cfg.path("processed"))
    train = z.sel(time=time_slice(cfg, "train"))["temperature"].where(z["ocean_mask"])
    by_month = train.groupby("time.month").mean(("time", "latitude", "longitude"))
    # Index by the month LABEL -- a short train block does not contain all 12.
    table = {int(m): by_month.sel(month=m).values for m in by_month["month"].values}
    overall = train.mean(("time", "latitude", "longitude")).values  # fallback for unseen months
    months = test_ds.times.astype("datetime64[M]").astype(int) % 12 + 1  # 1-based calendar month
    return np.stack([table.get(int(m), overall) for m in months])


def linear_baseline(cfg: Config, train_ds: ProfileDataset, test_ds: ProfileDataset, stats) -> np.ndarray:
    """Least-squares fit of the same 7 features -> profile. The 'is the net earning its keep' check."""
    A = np.hstack([train_ds.X.numpy(), np.ones((len(train_ds), 1), dtype="float32")])
    coef, *_ = np.linalg.lstsq(A, train_ds.Y.numpy(), rcond=None)
    B = np.hstack([test_ds.X.numpy(), np.ones((len(test_ds), 1), dtype="float32")])
    return stats.denormalize(B @ coef)


def evaluate(cfg: Config) -> dict:
    test_ds = ProfileDataset(cfg, "test", max_samples=300_000)
    train_ds = ProfileDataset(cfg, "train", max_samples=300_000)
    stats = test_ds.stats
    truth = stats.denormalize(test_ds.Y.numpy())

    ckpt = torch.load(cfg.path("checkpoint"), map_location="cpu", weights_only=False)
    model = build_model(cfg, ckpt["n_features"], ckpt["n_levels"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_ds), 8192):
            mu, _ = model(test_ds.X[i:i + 8192])
            preds.append(mu.numpy())
    model_pred = stats.denormalize(np.concatenate(preds))

    report = {
        "n_test_samples": int(len(test_ds)),
        "test_period": [cfg.time.val_end, cfg.time.end],
        "model": metrics(model_pred, truth, cfg.depths),
        "climatology": metrics(climatology_baseline(cfg, test_ds), truth, cfg.depths),
        "linear": metrics(linear_baseline(cfg, train_ds, test_ds, stats), truth, cfg.depths),
    }
    report["summary"] = {
        k: {"rmse_mean": float(np.mean([m["rmse"] for m in report[k]]))}
        for k in ("model", "climatology", "linear")
    }
    report["beats_climatology_at_levels"] = int(sum(
        m["rmse"] < c["rmse"] for m, c in zip(report["model"], report["climatology"])
    ))

    out_dir = cfg.path("reports_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "metrics.json").write_text(json.dumps(report, indent=2))
    _plot(report, cfg, out_dir / "rmse_by_depth.png")
    _print(report, cfg)
    return report


def _print(report: dict, cfg: Config) -> None:
    print(f"\n{'depth':>7} {'model':>9} {'clim':>9} {'linear':>9} {'R2':>7}")
    for m, c, l in zip(report["model"], report["climatology"], report["linear"]):
        print(f"{m['depth']:7.0f} {m['rmse']:9.3f} {c['rmse']:9.3f} {l['rmse']:9.3f} {m['r2']:7.3f}")
    s = report["summary"]
    print(f"\nmean RMSE  model {s['model']['rmse_mean']:.3f}  "
          f"climatology {s['climatology']['rmse_mean']:.3f}  linear {s['linear']['rmse_mean']:.3f}")
    print(f"model beats climatology at {report['beats_climatology_at_levels']}/{len(cfg.depths)} levels")


def _plot(report: dict, cfg: Config, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 6))
    for key, style in (("model", "-o"), ("climatology", "--"), ("linear", ":")):
        ax.plot([m["rmse"] for m in report[key]], cfg.depths, style, label=key, markersize=3)
    ax.invert_yaxis()
    ax.set_xlabel("RMSE (degC)")
    ax.set_ylabel("depth (m)")
    ax.set_title("Test-block RMSE by depth")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"[evaluate] wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config) if args.config else load_config())


if __name__ == "__main__":
    main()
