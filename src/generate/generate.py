"""Rotina de geração condicionada, comum ao Transformer e ao Markov.

A cada `Bar` gerado, força os tokens `Energy_e` e `Tension_t` do próximo compasso da curva e
para quando a curva acaba.

Responsável: Leonardo.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from src.generate.sampling import SamplingConfig
from src.model.interface import NextTokenModel

Curve = Sequence[tuple[int, int]]  # (nível de energia, nível de tensão) por compasso


def generate(
    model: NextTokenModel,
    curves: Sequence[Curve],
    sampling: SamplingConfig,
    vocab: dict[str, int],
    *,
    max_tokens: int,
    generator: torch.Generator | None = None,
) -> list[list[int]]:
    """Gera uma sequência por curva (em lote) e devolve os ids COM os tokens de controle.

    Para decodificar para MIDI: `strip_controls` e depois `tokenizer.decode`.
    """
    raise NotImplementedError
