"""Contrato 3: tokens de controle.

Cada compasso começa com `Bar`, seguido de `Energy_<e>` e `Tension_<t>`, e só depois vêm as
notas. Os tokens entram como tokens especiais do miditok, então já estão no `tokenizer.json`.

Responsável: Julia M.
"""

from __future__ import annotations

from collections.abc import Sequence

ENERGY = "Energy"
TENSION = "Tension"


def control_token_names(n_levels: int) -> list[str]:
    """Nomes dos tokens de controle, na ordem em que são registrados no tokenizador."""
    return [f"{ENERGY}_{i}" for i in range(n_levels)] + [f"{TENSION}_{i}" for i in range(n_levels)]


def insert_controls(ids: Sequence[int], levels: Sequence[tuple[int, int]], vocab: dict[str, int]) -> list[int]:
    """Insere `Energy_e`, `Tension_t` logo depois de cada `Bar` da sequência.

    `levels[i]` é o par (energia, tensão) já quantizado do i-ésimo compasso.
    """
    raise NotImplementedError


def strip_controls(ids: Sequence[int], vocab: dict[str, int]) -> list[int]:
    """Remove todos os tokens de controle (usar antes de decodificar para MIDI)."""
    raise NotImplementedError
