"""Amostragem do próximo token: temperatura, top-k e top-p.

Responsável: Natanael.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SamplingConfig:
    temperature: float = 1.0
    top_k: int | None = None     # None = desligado
    top_p: float | None = None   # None = desligado


def sample_next(logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator | None = None) -> torch.LongTensor:
    """logits: (B, V) -> (B,) ids amostrados."""
    raise NotImplementedError
