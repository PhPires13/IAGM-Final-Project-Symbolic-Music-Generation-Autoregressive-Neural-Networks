"""Contrato 4: função única de medição de energia e tensão.

A MESMA função rotula o treino e mede as músicas geradas. Ter duas versões dela invalidaria a
métrica de fidelidade ao controle. Nos dois casos a entrada é a partitura quantizada pelo REMI
(tokens decodificados), então os compassos medidos batem exatamente com os tokens `Bar`.

Compasso: 4 tempos (o EMOPIA é todo 4/4 a 120 BPM, então um compasso = 2 s). Uma nota pertence ao
compasso em que começa.

Energia (0 a 1) = média de
  - densidade: notas iniciadas no compasso / DENSITY_REF, limitada a 1;
  - intensidade: velocity média das notas do compasso / 127.

Tensão (0 a 1) = média de
  - notas fora da escala: fração das notas cuja classe de altura não pertence à escala da
    tonalidade da peça (maior: diatônica; menor: natural + sensível);
  - dissonância: fração dos pares de notas soando juntas, medidos em cada ataque do compasso, cujo
    intervalo (mod 12) é 2ª menor/maior, trítono ou 7ª menor/maior.

A tonalidade é estimada uma vez por peça (Krumhansl-Schmuckler, ver `src.controls.key`).
Compasso vazio: energia 0 e tensão 0.

Responsável: Julia M.
"""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
from typing import Literal

import symusic

DENSITY_REF = 40.0          # ~percentil 99 de notas por compasso no treino
DISSONANT_INTERVALS = {1, 2, 6, 10, 11}
MAJOR_SCALE = {0, 2, 4, 5, 7, 9, 11}
MINOR_SCALE = {0, 2, 3, 5, 7, 8, 10, 11}  # menor natural + sensível (7ª maior)


@dataclass(frozen=True)
class Note:
    pitch: int        # MIDI 0-127
    velocity: int     # 1-127
    start: float      # em tempos (beats), relativo ao início do compasso
    end: float


@dataclass(frozen=True)
class Key:
    tonic: int                        # classe de altura 0-11 (0 = Dó)
    mode: Literal["major", "minor"]

    def scale(self) -> set[int]:
        base = MAJOR_SCALE if self.mode == "major" else MINOR_SCALE
        return {(self.tonic + pc) % 12 for pc in base}


def _energy(notes: Sequence[Note]) -> float:
    density = min(len(notes) / DENSITY_REF, 1.0)
    intensity = sum(n.velocity for n in notes) / len(notes) / 127.0
    return (density + intensity) / 2


def _out_of_scale(notes: Sequence[Note], key: Key) -> float:
    scale = key.scale()
    return sum(n.pitch % 12 not in scale for n in notes) / len(notes)


def _dissonance(notes: Sequence[Note]) -> float:
    dissonant = total = 0
    for t in sorted({n.start for n in notes}):
        sounding = [n.pitch for n in notes if n.start <= t < n.end]
        for a, b in combinations(sounding, 2):
            total += 1
            dissonant += abs(a - b) % 12 in DISSONANT_INTERVALS
    return dissonant / total if total else 0.0


def measure_bar(notes: Sequence[Note], key: Key) -> tuple[float, float]:
    """Retorna (energia, tensão) contínuas de um compasso, antes da quantização em níveis."""
    if not notes:
        return 0.0, 0.0
    tension = (_out_of_scale(notes, key) + _dissonance(notes)) / 2
    return _energy(notes), tension


def split_bars(score: symusic.Score) -> list[list[Note]]:
    """Notas de todas as faixas agrupadas pelo compasso em que começam (tempos relativos ao compasso).

    O número de compassos vai até o último compasso com nota, igual ao número de tokens `Bar` do
    REMI. Notas que continuam soando além do fim do compasso mantêm a duração completa.
    """
    tpq = score.ticks_per_quarter
    ticks_per_bar = 4 * tpq
    notes = [n for track in score.tracks for n in track.notes]
    if not notes:
        return []
    bars: list[list[Note]] = [[] for _ in range(max(n.time for n in notes) // ticks_per_bar + 1)]
    for n in notes:
        b = n.time // ticks_per_bar
        start = (n.time - b * ticks_per_bar) / tpq
        bars[b].append(Note(n.pitch, n.velocity, start, start + n.duration / tpq))
    return bars


def measure_piece(score: symusic.Score) -> list[tuple[float, float]]:
    """(energia, tensão) de cada compasso de uma partitura quantizada.

    É o ponto de entrada usado tanto pelo pré-processamento quanto pela avaliação.
    """
    from src.controls.key import estimate_key

    key = estimate_key(score)
    return [measure_bar(bar, key) for bar in split_bars(score)]


def to_levels(values: Sequence[tuple[float, float]], thresholds: dict[str, list[float]]) -> list[tuple[int, int]]:
    """Quantiza (energia, tensão) em níveis 0..K-1 com os limites de `control_thresholds.json`.

    Formato do JSON: {"energy": [t_1, ..., t_{K-1}], "tension": [t_1, ..., t_{K-1}]}, calculado
    só com o conjunto de treino. Um valor igual a um limite fica no nível de baixo, o que mantém os
    compassos com tensão exatamente 0 (muito comuns) no nível 0.
    """
    e_thr, t_thr = thresholds["energy"], thresholds["tension"]
    return [(bisect.bisect_left(e_thr, e), bisect.bisect_left(t_thr, t)) for e, t in values]
