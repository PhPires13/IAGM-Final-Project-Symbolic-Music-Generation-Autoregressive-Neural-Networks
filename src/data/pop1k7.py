"""Acesso ao Pop1K7 (Hsiao et al., 2021), usado no pré-treino.

1747 covers de piano pop transcritos e sincronizados com o beat pelo mesmo laboratório do EMOPIA.
Usamos `midi_synchronized/src_00k/<n>.mid`: uma faixa de piano, 4/4, com o mapa de andamento
alinhado aos beats. Não há rótulo de emoção (as sequências recebem `Emotion_None`).

Cada música é identificada por `src_00k/<n>`.

Responsável: Pedro.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import symusic

from src.common import resolve
from src.data.emopia import merge_piano

FOLDER = "Pop1K7"
SUBDIR = "midi_synchronized"


def ensure_extracted(cfg: dict) -> Path:
    """Extrai só `midi_synchronized/` do zip em `paths.raw_dir` e devolve a pasta `Pop1K7`."""
    root = resolve(cfg["paths"]["raw_dir"]) / FOLDER
    if (root / SUBDIR).is_dir():
        return root
    zip_path = resolve(cfg["paths"]["pop1k7_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} não encontrado: rode scripts/download_data.sh")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{FOLDER}/{SUBDIR}/") and m.endswith(".mid")]
        zf.extractall(root.parent, members=members)
    return root


def list_songs(root: Path) -> list[str]:
    paths = (root / SUBDIR).glob("*/*.mid")
    return sorted((f"{p.parent.name}/{p.stem}" for p in paths), key=lambda s: (s.split("/")[0], int(s.split("/")[1])))


def midi_path(root: Path, song: str) -> Path:
    return root / SUBDIR / f"{song}.mid"


def load_score(root: Path, song: str) -> symusic.Score:
    return merge_piano(symusic.Score(midi_path(root, song)))


def song_split(songs: list[str], val_fraction: float, seed: int) -> tuple[list[str], list[str]]:
    """Separa `val_fraction` das músicas para a validação do pré-treino, com seed fixa."""
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(songs))
    n_val = max(1, round(val_fraction * len(songs)))
    val = sorted(songs[i] for i in order[:n_val])
    train = sorted(songs[i] for i in order[n_val:])
    return train, val
