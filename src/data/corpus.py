"""Leitura uniforme das duas fontes de dados, a partir do nome da partição.

Partições `pretrain_*` vêm do Pop1K7 (sem rótulo de emoção); `train`, `val` e `test` vêm do
EMOPIA+ (emoção = quadrante do clipe). Usado pelo build do dataset, pela calibração e pelas
estatísticas.

Responsável: Pedro.
"""

from __future__ import annotations

from pathlib import Path

import symusic

from src.data import emopia, pop1k7

PRETRAIN_PREFIX = "pretrain_"


def is_pretrain(split: str) -> bool:
    return split.startswith(PRETRAIN_PREFIX)


def roots(cfg: dict, splits: dict) -> dict[str, Path]:
    out = {"emopia": emopia.ensure_extracted(cfg)}
    if any(is_pretrain(s) for s in splits):
        out["pop1k7"] = pop1k7.ensure_extracted(cfg)
    return out


def load_piece(split: str, item: str, root_by_corpus: dict[str, Path]) -> tuple[symusic.Score, int | None]:
    """(partitura de uma faixa, quadrante ou None) de um item de uma partição."""
    if is_pretrain(split):
        return pop1k7.load_score(root_by_corpus["pop1k7"], item), None
    return emopia.load_score(root_by_corpus["emopia"], item), emopia.quadrant(item)
