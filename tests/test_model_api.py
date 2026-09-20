import numpy as np
import torch

from oceanembed.data.dataset import N_FEATURES
from oceanembed.models.base import gaussian_nll
from oceanembed.models.embed_mlp import build_model


def test_model_shapes_and_embedding_dim(cfg):
    model = build_model(cfg, N_FEATURES, len(cfg.depths))
    x = torch.randn(8, N_FEATURES)
    z = model.encode(x)
    mu, logvar = model(x)
    assert z.shape == (8, cfg.model.embedding_dim)
    assert mu.shape == logvar.shape == (8, len(cfg.depths))


def test_nll_penalises_confident_errors():
    target = torch.zeros(4, 3)
    wrong_confident = gaussian_nll(torch.full((4, 3), 5.0), torch.full((4, 3), -3.0), target)
    wrong_unsure = gaussian_nll(torch.full((4, 3), 5.0), torch.full((4, 3), 2.0), target)
    assert wrong_confident > wrong_unsure


def test_train_eval_and_api(cfg, processed, monkeypatch):
    from fastapi.testclient import TestClient

    from oceanembed import api, config, evaluate, infer, train

    train.train(cfg)
    report = evaluate.evaluate(cfg)
    assert len(report["model"]) == len(cfg.depths)
    assert all(np.isfinite(m["rmse"]) for m in report["model"])

    monkeypatch.setattr(config, "load_config", lambda *a, **k: cfg)
    monkeypatch.setattr(api, "load_config", lambda *a, **k: cfg)
    infer.get_predictor.cache_clear()
    monkeypatch.setattr(infer, "get_predictor", lambda *a, **k: infer.Predictor(cfg))
    monkeypatch.setattr(api, "get_predictor", lambda *a, **k: infer.Predictor(cfg))

    client = TestClient(api.app)
    assert client.get("/health").json()["status"] == "ok"

    meta = client.get("/meta").json()
    assert meta["depths"] == cfg.depths and meta["synthetic"] is True

    lat = (cfg.region.lat_min + cfg.region.lat_max) / 2
    lon = (cfg.region.lon_min + cfg.region.lon_max) / 2
    p = client.get(f"/predict/profile?lat={lat}&lon={lon}&date=2021-04-15").json()
    assert len(p["mu"]) == len(p["sigma"]) == len(p["truth"]) == len(cfg.depths)
    assert all(s > 0 for s in p["sigma"])

    f = client.get("/predict/field?date=2021-04-15&depth=100").json()
    assert len(f["values"]) == len(f["lat"]) and len(f["values"][0]) == len(f["lon"])
