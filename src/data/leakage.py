"""Verificação de vazamento entre o Pop1K7 (pré-treino) e a validação/teste do EMOPIA+.

As duas coleções são covers de piano do YouTube reunidos pelo mesmo laboratório, então uma mesma
música pode estar nas duas. Uma música do teste presente no pré-treino inflaria as métricas.

Impressão digital de uma peça: a linha superior (a nota mais aguda de cada ataque, com os ataques
agrupados na grade de 1/4 de beat), reduzida aos intervalos entre notas consecutivas (invariante a
transposição, já que covers mudam de tom), e o conjunto dos seus n-gramas. Para cada clipe C do
EMOPIA+ e cada música S do Pop1K7, a contenção é |ngramas(C) ∩ ngramas(S)| / |ngramas(C)|. Uma
música do Pop1K7 sai do pré-treino se a contenção de algum clipe de validação ou teste nela passar
do limiar. O limiar é calibrado contra a distribuição de contenções entre músicas sabidamente
diferentes (clipes do treino do EMOPIA+ contra clipes do teste, que são de outras músicas).

Responsável: Pedro.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import symusic


def top_line(score: symusic.Score) -> list[int]:
    """Nota mais aguda de cada ataque (ataques agrupados na grade de 1/4 de beat)."""
    q = max(1, score.ticks_per_quarter // 4)
    top: dict[int, int] = {}
    for track in score.tracks:
        for n in track.notes:
            slot = round(n.time / q)
            if n.pitch > top.get(slot, -1):
                top[slot] = n.pitch
    return [top[k] for k in sorted(top)]


def fingerprint(score: symusic.Score, n: int) -> set[tuple[int, ...]]:
    line = np.array(top_line(score))
    if len(line) <= n:
        return set()
    iv = np.diff(line)
    return {tuple(iv[i : i + n]) for i in range(len(iv) - n + 1)}


class Index:
    """Índice invertido n-grama -> músicas, para calcular contenções rapidamente."""

    def __init__(self, prints: dict[str, set]):
        self.postings: dict[tuple, list[str]] = defaultdict(list)
        for song, fp in prints.items():
            for g in fp:
                self.postings[g].append(song)

    def containment(self, query: set) -> dict[str, float]:
        """Contenção da peça `query` em cada música indexada que compartilha algum n-grama."""
        if not query:
            return {}
        counts: dict[str, int] = defaultdict(int)
        for g in query:
            for song in self.postings.get(g, ()):
                counts[song] += 1
        return {song: c / len(query) for song, c in counts.items()}


def max_containment(queries: dict[str, set], index: Index) -> dict[str, tuple[str | None, float]]:
    """Para cada peça de `queries`: (música indexada mais parecida, contenção)."""
    out = {}
    for name, fp in queries.items():
        cont = index.containment(fp)
        best = max(cont.items(), key=lambda kv: kv[1], default=(None, 0.0))
        out[name] = best
    return out
