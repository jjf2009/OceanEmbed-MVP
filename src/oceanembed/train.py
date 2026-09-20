"""Train the embedding model. Checkpoint carries weights + config + norm stats."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import Config, load_config
from .data.dataset import N_FEATURES, ProfileDataset
from .models.base import gaussian_nll
from .models.embed_mlp import build_model


def run_epoch(model, loader, device, optimizer=None) -> float:
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            mu, logvar = model(x)
            loss = gaussian_nll(mu, logvar, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            total += loss.item() * len(x)
            n += len(x)
    return total / max(n, 1)


def train(cfg: Config) -> Path:
    torch.manual_seed(cfg.train.seed)
    np.random.seed(cfg.train.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = ProfileDataset(cfg, "train", max_samples=cfg.train.max_train_samples, seed=cfg.train.seed)
    val_ds = ProfileDataset(cfg, "val", max_samples=min(cfg.train.max_train_samples, 300_000))
    print(f"[train] device={device} train={len(train_ds)} val={len(val_ds)} levels={len(cfg.depths)}")

    train_dl = DataLoader(train_ds, batch_size=cfg.train.batch_size, shuffle=True, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=cfg.train.batch_size)

    model = build_model(cfg, N_FEATURES, len(cfg.depths)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg.train.epochs)

    best, best_state, bad = float("inf"), None, 0
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr = run_epoch(model, train_dl, device, opt)
        va = run_epoch(model, val_dl, device)
        sched.step()
        flag = ""
        if va < best - 1e-4:
            best, bad = va, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            flag = " *"
        else:
            bad += 1
        print(f"[train] epoch {epoch:3d}  train {tr:8.4f}  val {va:8.4f}  {time.time()-t0:5.1f}s{flag}")
        if bad >= cfg.train.patience:
            print(f"[train] early stop (no val improvement for {cfg.train.patience} epochs)")
            break

    ckpt_path = cfg.path("checkpoint")
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state or model.state_dict(),
            "n_features": N_FEATURES,
            "n_levels": len(cfg.depths),
            "depths": cfg.depths,
            "model_cfg": vars(cfg.model),
            "val_loss": best,
        },
        ckpt_path,
    )
    print(f"[train] best val {best:.4f} -> {ckpt_path}")
    return ckpt_path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    train(load_config(args.config) if args.config else load_config())


if __name__ == "__main__":
    main()
