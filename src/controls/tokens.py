"""Contrato 3: tokens de controle.

Cada música começa com `BOS Emotion_<q>`, em que <q> é o quadrante do EMOPIA (Q1..Q4) ou `None`
para músicas sem rótulo (pré-treino no Pop1K7). Cada compasso começa com `Bar`, seguido de
`Energy_<e>` e `Tension_<t>`, e só depois vêm as notas. Todos entram como tokens especiais do
miditok, então já estão no `tokenizer.json`.

Responsável: Julia M.
"""

from __future__ import annotations

from collections.abc import Sequence

ENERGY = "Energy"
TENSION = "Tension"
EMOTION = "Emotion"
BAR = "Bar_None"
QUADRANTS = (1, 2, 3, 4)


def emotion_token_names() -> list[str]:
    return [f"{EMOTION}_Q{q}" for q in QUADRANTS] + [f"{EMOTION}_None"]


def control_token_names(n_levels: int) -> list[str]:
    """Nomes dos tokens de controle, na ordem em que são registrados no tokenizador."""
    return ([f"{ENERGY}_{i}" for i in range(n_levels)] + [f"{TENSION}_{i}" for i in range(n_levels)]
            + emotion_token_names())


def emotion_id(vocab: dict[str, int], quadrant: int | None) -> int:
    """Id do token de emoção de um quadrante (1..4), ou de `Emotion_None` se `quadrant` for None."""
    return vocab[f"{EMOTION}_None" if quadrant is None else f"{EMOTION}_Q{quadrant}"]


def control_ids(vocab: dict[str, int]) -> set[int]:
    """Ids de todos os tokens forçados (energia, tensão e emoção): nunca são amostrados."""
    return {i for tok, i in vocab.items() if tok.startswith((f"{ENERGY}_", f"{TENSION}_", f"{EMOTION}_"))}


def insert_controls(ids: Sequence[int], levels: Sequence[tuple[int, int]], vocab: dict[str, int]) -> list[int]:
    """Insere `Energy_e`, `Tension_t` logo depois de cada `Bar` da sequência.

    `levels[i]` é o par (energia, tensão) já quantizado do i-ésimo compasso.
    """
    bar = vocab[BAR]
    n_bars = sum(1 for i in ids if i == bar)
    if n_bars != len(levels):
        raise ValueError(f"{n_bars} tokens Bar, mas {len(levels)} pares de níveis")
    out: list[int] = []
    it = iter(levels)
    for i in ids:
        out.append(i)
        if i == bar:
            e, t = next(it)
            out.extend((vocab[f"{ENERGY}_{e}"], vocab[f"{TENSION}_{t}"]))
    return out


def strip_controls(ids: Sequence[int], vocab: dict[str, int]) -> list[int]:
    """Remove todos os tokens de controle (usar antes de decodificar para MIDI)."""
    ctrl = control_ids(vocab)
    return [i for i in ids if i not in ctrl]
