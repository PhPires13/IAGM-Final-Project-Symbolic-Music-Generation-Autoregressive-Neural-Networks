"""Contrato 3: tokens de controle.

Cada compasso começa com `Bar`, seguido de `Energy_<e>` e `Tension_<t>`, e só depois vêm as
notas. Os tokens entram como tokens especiais do miditok, então já estão no `tokenizer.json`.

Responsável: Julia M.
"""

from __future__ import annotations

from collections.abc import Sequence

ENERGY = "Energy"
TENSION = "Tension"
BAR = "Bar_None"


def control_token_names(n_levels: int) -> list[str]:
    """Nomes dos tokens de controle, na ordem em que são registrados no tokenizador."""
    return [f"{ENERGY}_{i}" for i in range(n_levels)] + [f"{TENSION}_{i}" for i in range(n_levels)]


def control_ids(vocab: dict[str, int]) -> set[int]:
    return {i for tok, i in vocab.items() if tok.startswith((f"{ENERGY}_", f"{TENSION}_"))}


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
