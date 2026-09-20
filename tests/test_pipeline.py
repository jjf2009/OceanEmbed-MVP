import numpy as np
import xarray as xr

from oceanembed.data.dataset import N_FEATURES, ProfileDataset, time_slice


def test_processed_shape(cfg, processed):
    ds = xr.open_zarr(processed)
    assert ds.sizes["level"] == len(cfg.depths) == 15
    assert np.allclose(ds["level"].values, cfg.depths)
    r = cfg.region
    assert ds.sizes["latitude"] == round((r.lat_max - r.lat_min) / r.resolution) + 1
    assert ds["ocean_mask"].values.any()


def test_feature_and_target_shapes(cfg, processed):
    ds = ProfileDataset(cfg, "train", max_samples=5000)
    assert ds.X.shape[1] == N_FEATURES
    assert ds.Y.shape[1] == len(cfg.depths)
    assert np.isfinite(ds.X.numpy()).all() and np.isfinite(ds.Y.numpy()).all()


def test_normalization_roundtrip(cfg, processed):
    ds = ProfileDataset(cfg, "train", max_samples=2000)
    back = ds.stats.denormalize(ds.Y.numpy())
    # Real ocean temperatures, not normalised units.
    assert 0.0 < back.min() < 40.0 and back.max() < 40.0
    assert back[:, 0].mean() > back[:, -1].mean()  # warmer at the surface


def test_splits_do_not_overlap_in_time(cfg, processed):
    parts = {s: ProfileDataset(cfg, s, max_samples=500) for s in ("train", "val", "test")}
    assert parts["train"].times.max() <= parts["val"].times.min()
    assert parts["val"].times.max() <= parts["test"].times.min()


def test_norm_stats_come_from_train_block_only(cfg, processed):
    """Guards against val/test statistics leaking into normalisation."""
    ds = xr.open_zarr(processed)
    train = ds.sel(time=time_slice(cfg, "train"))["temperature"].where(ds["ocean_mask"])
    stats = ProfileDataset(cfg, "train", max_samples=100).stats
    expected = train.mean(("time", "latitude", "longitude")).values
    assert np.allclose(stats.temp_mean, expected, atol=1e-3)
