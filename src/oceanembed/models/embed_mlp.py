"""MVP backbone: per-pixel MLP encoder -> 64-d embedding -> profile decoder."""
from __future__ import annotations

from typing import List, Tuple

import torch
import torch.nn as nn

from .base import ReconstructionModel


def _mlp(sizes: List[int], dropout: float) -> nn.Sequential:
    layers: List[nn.Module] = []
    for a, b in zip(sizes[:-1], sizes[1:]):
        layers += [nn.Linear(a, b), nn.LayerNorm(b), nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class EmbedMLP(ReconstructionModel):
    def __init__(
        self,
        n_features: int,
        n_levels: int,
        embedding_dim: int = 64,
        encoder_hidden: List[int] | None = None,
        decoder_hidden: List[int] | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.n_features = n_features
        self.n_levels = n_levels
        self.embedding_dim = embedding_dim
        enc = encoder_hidden or [128, 128]
        dec = decoder_hidden or [128]

        self.encoder = nn.Sequential(_mlp([n_features, *enc], dropout), nn.Linear(enc[-1], embedding_dim))
        self.decoder = _mlp([embedding_dim, *dec], dropout)
        self.head_mu = nn.Linear(dec[-1], n_levels)
        self.head_logvar = nn.Linear(dec[-1], n_levels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.decoder(z)
        return self.head_mu(h), self.head_logvar(h)


def build_model(cfg, n_features: int, n_levels: int) -> EmbedMLP:
    m = cfg.model
    return EmbedMLP(
        n_features=n_features,
        n_levels=n_levels,
        embedding_dim=m.embedding_dim,
        encoder_hidden=list(m.encoder_hidden),
        decoder_hidden=list(m.decoder_hidden),
        dropout=m.dropout,
    )
