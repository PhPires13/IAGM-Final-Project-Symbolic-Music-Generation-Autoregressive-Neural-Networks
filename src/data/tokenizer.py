"""Contrato 2: tokenizador REMI congelado em `configs/tokenizer.json`.

Os MIDIs do EMOPIA são transcrições de áudio, todos com um único andamento (120 BPM) e 4/4, então
tokens de andamento e de fórmula de compasso não carregam informação e ficam desligados. Um
"compasso" corresponde a 4 tempos da grade de 120 BPM, ou seja, 2 segundos.

Os tokens de controle entram como tokens especiais, já reservados no vocabulário.

Uso: uv run python -m src.data.tokenizer [--force]
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
        beat_res={(0, 4): 8, (4, 12): 4},    # posições em 1/8 de tempo (62,5 ms a 120 BPM)
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
