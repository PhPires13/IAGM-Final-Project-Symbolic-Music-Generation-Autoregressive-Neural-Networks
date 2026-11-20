"""Contrato 4: função única de medição de energia e tensão.

A MESMA função rotula o treino e mede as músicas geradas. Ter duas versões dela invalidaria a
métrica de fidelidade ao controle.

Responsável: Julia M.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Note:
    pitch: int        # MIDI 0-127
    velocity: int     # 1-127
    start: float      # em beats, relativo ao início do compasso
    end: float


@dataclass(frozen=True)
class Key:
    tonic: int                        # classe de altura 0-11 (0 = Dó)
    mode: Literal["major", "minor"]


def measure_bar(notes: Sequence[Note], key: Key) -> tuple[float, float]:
    """Retorna (energia, tensão) contínuas de um compasso, antes da quantização em níveis."""
    raise NotImplementedError


def measure_piece(midi_path: str) -> list[tuple[float, float]]:
    """Estima a tonalidade da peça, divide em compassos e aplica `measure_bar` em cada um.

    É o ponto de entrada usado tanto pelo pré-processamento quanto pela avaliação.
    """
    raise NotImplementedError


def to_levels(values: Sequence[tuple[float, float]], thresholds: dict[str, list[float]]) -> list[tuple[int, int]]:
    """Quantiza (energia, tensão) em níveis 0..K-1 com os limites de `control_thresholds.json`.

    Formato do JSON: {"energy": [t_1, ..., t_{K-1}], "tension": [t_1, ..., t_{K-1}]}, calculado
    só com o conjunto de treino.
    """
    raise NotImplementedError
