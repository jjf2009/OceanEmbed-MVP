"""Model interface: surface features -> compact embedding -> depth profile.

Any future backbone (ConvLSTM, U-Net, GNN) only has to satisfy encode/decode
so train.py, evaluate.py and infer.py keep working unchanged.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import torch
import torch.nn as nn


class ReconstructionModel(nn.Module, ABC):
    n_features: int
    n_levels: int
    embedding_dim: int

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_features) -> (B, embedding_dim)."""

    @abstractmethod
    def decode(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """(B, embedding_dim) -> (mu, logvar), each (B, n_levels)."""

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode(self.encode(x))


def gaussian_nll(mu: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Heteroscedastic NLL -- lets the model say where it is unsure (TS-Cast idea)."""
    logvar = logvar.clamp(-7.0, 7.0)
    return (0.5 * (logvar + (target - mu) ** 2 / logvar.exp())).mean()
