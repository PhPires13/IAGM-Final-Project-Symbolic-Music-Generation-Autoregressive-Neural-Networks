"""Fidelidade ao controle: a música gerada segue a curva de energia/tensão pedida?

Cada amostra gerada é medida com a MESMA função do treino (`measure_piece` + `to_levels`) e os níveis
obtidos são comparados, compasso a compasso, com os pedidos. Compassos finais vazios (sem nenhuma
nota) contam como medidos (0, 0).

Métricas, separadas para energia e tensão, agregando todos os compassos de todas as amostras:
  - acurácia exata e acurácia com tolerância de ±1 nível;
  - correlação de Spearman entre nível pedido e nível obtido.

Experimento embaralhado: as mesmas medidas, mas contra a curva de OUTRA amostra (permutação sem
pontos fixos, seed fixa), nos primeiros min(len) compassos. Se a fidelidade à curva pedida não
for maior que à curva não usada, o modelo está ignorando o controle.

Responsável: Júlia B.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import symusic
from scipy.stats import spearmanr

from src.controls.measure import measure_piece, to_levels

Levels = list[tuple[int, int]]


def levels_from_values(values: list[tuple[float, float]], n_bars: int, thresholds: dict) -> Levels:
    values = list(values[:n_bars]) + [(0.0, 0.0)] * (n_bars - len(values))
    return to_levels(values, thresholds)


def measured_levels(midi_path, n_bars: int, thresholds: dict) -> Levels:
    return levels_from_values(measure_piece(symusic.Score(midi_path)), n_bars, thresholds)


def derangement(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    while True:
        perm = rng.permutation(n)
        if n < 2 or not np.any(perm == np.arange(n)):
            return perm


def scores(requested: Sequence[Levels], obtained: Sequence[Levels]) -> dict[str, dict[str, float]]:
    out = {}
    for dim, name in ((0, "energy"), (1, "tension")):
        req = np.array([lv[dim] for r, o in zip(requested, obtained) for lv in r[: len(o)]])
        got = np.array([lv[dim] for r, o in zip(requested, obtained) for lv in o[: len(r)]])
        diff = np.abs(req - got)
        rho = spearmanr(req, got).statistic if np.ptp(req) and np.ptp(got) else float("nan")
        out[name] = {"exact": float(np.mean(diff == 0)), "within_1": float(np.mean(diff <= 1)),
                     "spearman": float(rho), "bars": int(len(req))}
    return out
