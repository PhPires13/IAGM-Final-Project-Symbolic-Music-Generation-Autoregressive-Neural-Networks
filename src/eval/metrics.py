"""Métricas objetivas por música (muspy) e comparação de distribuições no estilo de Yang & Lerch (2020).

Para cada métrica e cada sistema S comparado à referência R (clipes humanos do teste):
  - média ± desvio dos valores;
  - distâncias intra-conjunto de R (|r_i - r_j|) e inter-conjunto R×S (|r_i - s_j|);
  - KDE das duas distribuições de distância, área de sobreposição (OA, maior é melhor) e
    divergência KL(intra_R || inter_RS) (menor é melhor).

Responsável: Júlia B.
"""

from __future__ import annotations

from pathlib import Path

import muspy
import numpy as np
from scipy.stats import gaussian_kde

METRICS = ["pitch_range", "n_pitch_classes_used", "pitch_class_entropy", "scale_consistency",
           "polyphony", "empty_beat_rate", "groove_consistency"]
GRID_POINTS = 1000
EPS = 1e-10


def piece_metrics(path: str | Path) -> dict[str, float]:
    music = muspy.read_midi(path)
    out = {}
    for name in METRICS:
        try:
            if name == "groove_consistency":
                value = muspy.groove_consistency(music, measure_resolution=4 * music.resolution)
            else:
                value = getattr(muspy, name)(music)
            out[name] = float(value)
        except Exception:  # amostra vazia ou degenerada
            out[name] = float("nan")
    return out


def intra_distances(x: np.ndarray) -> np.ndarray:
    i, j = np.triu_indices(len(x), k=1)
    return np.abs(x[i] - x[j])


def inter_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.abs(x[:, None] - y[None, :]).ravel()


def _kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    if np.ptp(values) < 1e-12:  # distribuição degenerada: um pico estreito
        values = values + np.random.default_rng(0).normal(0, 1e-3, len(values))
    density = gaussian_kde(values)(grid)
    return density / max(np.trapezoid(density, grid), EPS)


def overlap_and_kl(p_samples: np.ndarray, q_samples: np.ndarray) -> tuple[float, float]:
    """OA e KL(P || Q) entre as KDEs de duas amostras de valores."""
    lo = min(p_samples.min(), q_samples.min())
    hi = max(p_samples.max(), q_samples.max())
    margin = 0.1 * (hi - lo) + 1e-6
    grid = np.linspace(lo - margin, hi + margin, GRID_POINTS)
    p, q = _kde(p_samples, grid), _kde(q_samples, grid)
    oa = float(np.trapezoid(np.minimum(p, q), grid))
    kl = float(np.trapezoid(p * np.log((p + EPS) / (q + EPS)), grid))
    return oa, kl


def compare(reference: np.ndarray, system: np.ndarray) -> dict[str, float]:
    reference, system = reference[~np.isnan(reference)], system[~np.isnan(system)]
    oa, kl = overlap_and_kl(intra_distances(reference), inter_distances(reference, system))
    return {"mean": float(system.mean()), "std": float(system.std()), "oa": oa, "kl": kl}
