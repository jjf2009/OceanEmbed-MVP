"""Config loading. One YAML -> nested dataclasses used by every stage."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


@dataclass
class Region:
    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float
    resolution: float


@dataclass
class TimeSpan:
    start: str
    end: str
    train_end: str
    val_end: str


@dataclass
class Source:
    dataset_id: str
    surface_vars: List[str]
    target_var: str


@dataclass
class Paths:
    raw_dir: str
    processed: str
    norm_stats: str
    checkpoint: str
    reports_dir: str


@dataclass
class ModelCfg:
    embedding_dim: int = 64
    encoder_hidden: List[int] = field(default_factory=lambda: [128, 128])
    decoder_hidden: List[int] = field(default_factory=lambda: [128])
    dropout: float = 0.1


@dataclass
class TrainCfg:
    epochs: int = 30
    batch_size: int = 4096
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 5
    seed: int = 42
    max_train_samples: int = 1_500_000


@dataclass
class Config:
    name: str
    region: Region
    time: TimeSpan
    source: Source
    depths: List[float]
    paths: Paths
    model: ModelCfg
    train: TrainCfg
    root: Path = Path(".")

    def path(self, key: str) -> Path:
        """Resolve a configured path relative to the project root."""
        return self.root / getattr(self.paths, key)


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "mvp.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path)
    raw = yaml.safe_load(path.read_text())
    return Config(
        name=raw["name"],
        region=Region(**raw["region"]),
        time=TimeSpan(**raw["time"]),
        source=Source(**raw["source"]),
        depths=[float(d) for d in raw["depths"]],
        paths=Paths(**raw["paths"]),
        model=ModelCfg(**raw.get("model", {})),
        train=TrainCfg(**raw.get("train", {})),
        root=path.resolve().parents[1],
    )
