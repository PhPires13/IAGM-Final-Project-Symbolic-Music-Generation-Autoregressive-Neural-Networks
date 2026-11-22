"""Estimativa de tonalidade por peça (algoritmo de Krumhansl-Schmuckler).

Correlaciona o histograma de classes de altura da peça, ponderado pela duração das notas, com os
perfis de Krumhansl-Kessler das 24 tonalidades e fica com a de maior correlação. É o mesmo
algoritmo do `music21.analysis.discrete.KrumhanslSchmuckler`, reescrito com numpy para rodar
direto sobre a partitura do symusic (centenas de vezes mais rápido que converter para music21).
"""

from __future__ import annotations

import numpy as np
import symusic

from src.controls.measure import Key

# Krumhansl & Kessler (1982), perfis a partir da tônica
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


def pitch_class_histogram(score: symusic.Score) -> np.ndarray:
    hist = np.zeros(12)
    for track in score.tracks:
        for n in track.notes:
            hist[n.pitch % 12] += n.duration
    return hist


def estimate_key(score: symusic.Score) -> Key:
    hist = pitch_class_histogram(score)
    if hist.sum() == 0:
        return Key(0, "major")
    best, best_r = Key(0, "major"), -np.inf
    for tonic in range(12):
        for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
            r = np.corrcoef(hist, np.roll(profile, tonic))[0, 1]
            if r > best_r:
                best, best_r = Key(tonic, mode), r
    return best
