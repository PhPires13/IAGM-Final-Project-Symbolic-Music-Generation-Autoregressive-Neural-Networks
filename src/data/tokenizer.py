"""Contrato 2: tokenizador REMI congelado em `configs/tokenizer.json`.

O EMOPIA+ e o Pop1K7 estão alinhados ao beat e são todos em 4/4: o modelo trabalha em beats, e os
tokens de fórmula de compasso ficam desligados. Os de andamento também: o andamento varia beat a
beat (o mapa vem da sincronização com o áudio) e não faz parte do que o modelo aprende; a
renderização usa um andamento fixo, igual para as músicas humanas e as geradas.

A grade de posições é de 1/12 de beat: no EMOPIA+, 85,5% dos inícios de nota caem na grade de 1/4
de beat e 92,5% na de 1/12 (há tercinas). A grade de 1/8 não acrescenta nada à de 1/4.

Os tokens de controle (energia, tensão e emoção) entram como tokens especiais, já reservados no
vocabulário.

Uso: uv run python -m src.data.tokenizer [--force]

Responsável: Pedro.
"""

from __future__ import annotations

import argparse

import symusic
from miditok import REMI, TokenizerConfig

from src.common import load_config, resolve
from src.controls.tokens import control_token_names

PAD, BOS, EOS = "PAD_None", "BOS_None", "EOS_None"


def build_tokenizer(n_levels: int) -> REMI:
    config = TokenizerConfig(
        pitch_range=(21, 109),               # teclado do piano
        beat_res={(0, 4): 12, (4, 12): 4},   # posições em 1/12 de beat (semicolcheias e tercinas)
        num_velocities=32,                   # a velocity entra na medida de energia
        use_chords=False,
        use_rests=False,
        use_tempos=False,
        use_time_signatures=False,
        use_programs=False,
        special_tokens=["PAD", "BOS", "EOS", *control_token_names(n_levels)],
    )
    return REMI(config)


def load_tokenizer(path=None) -> REMI:
    path = resolve(path or load_config()["paths"]["tokenizer"])
    return REMI(params=path)


def encode(tokenizer: REMI, score: symusic.Score) -> list[int]:
    """Ids REMI de uma partitura de piano (uma faixa), sem BOS/EOS nem tokens de controle."""
    seqs = tokenizer(score)
    seq = seqs[0] if isinstance(seqs, list) else seqs
    return list(seq.ids)


def decode(tokenizer: REMI, ids: list[int]) -> symusic.Score:
    """Ids (sem tokens de controle) -> partitura. BOS/EOS/PAD são ignorados pelo miditok."""
    return tokenizer.decode([ids])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="sobrescreve um tokenizer.json existente")
    args = parser.parse_args()

    cfg = load_config()
    out = resolve(cfg["paths"]["tokenizer"])
    if out.exists() and not args.force:
        raise SystemExit(f"{out} já existe e está congelado; use --force só se o grupo combinar a mudança")
    tokenizer = build_tokenizer(cfg["controls"]["n_levels"])
    tokenizer.save(out)
    print(f"vocabulário: {len(tokenizer)} tokens | salvo em {out}")


if __name__ == "__main__":
    main()
