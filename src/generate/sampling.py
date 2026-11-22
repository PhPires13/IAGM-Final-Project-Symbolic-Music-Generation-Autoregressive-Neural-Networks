"""Amostragem do próximo token: temperatura, top-k e top-p (nucleus).

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


def filter_logits(logits: torch.Tensor, cfg: SamplingConfig) -> torch.Tensor:
    """Aplica temperatura, top-k e top-p (nessa ordem); tokens cortados recebem -inf."""
    if cfg.temperature <= 0:
        raise ValueError("temperatura deve ser > 0")
    logits = logits.float() / cfg.temperature

    if cfg.top_k is not None and cfg.top_k < logits.size(-1):
        kth = torch.topk(logits, cfg.top_k, dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if cfg.top_p is not None and cfg.top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        cum = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
        # remove um token se a massa acumulada ANTES dele já atinge top_p (o mais provável fica sempre)
        remove_sorted = (cum - torch.softmax(sorted_logits, dim=-1)) >= cfg.top_p
        remove = remove_sorted.scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))

    return logits


def sample_next(logits: torch.Tensor, cfg: SamplingConfig, generator: torch.Generator | None = None) -> torch.LongTensor:
    """logits: (B, V) -> (B,) ids amostrados."""
    probs = torch.softmax(filter_logits(logits, cfg), dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator).squeeze(-1)
