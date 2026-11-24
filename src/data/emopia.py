"""Acesso ao EMOPIA+ (Huang et al., 2024): extração do zip, leitura dos clipes e divisão oficial.

O EMOPIA+ tem os mesmos clipes do EMOPIA 1.0, com os mesmos nomes, mas quantizados no beat (o
`midis/` do EMOPIA 1.0 é transcrição crua, sem grade; ver docs/pivo_pretreino.md). Cada MIDI tem as
faixas `Melody`, `Texture`, `Bass` e `Chord`; a `Chord` é anotação de acordes e fica de fora, as
outras são juntadas numa única faixa de piano.

Um clipe se chama `Q<quadrante>_<id da música no YouTube (11 caracteres)>_<índice do clipe>`, ex.:
`Q1_0vLPYiPN7qY_0`. Como alguns ids do YouTube têm `_`, o id da música é extraído por posição.

Responsável: Pedro.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import symusic

from src.common import resolve

FOLDER = "EMOPIA+"
ANNOTATION_TRACKS = {"Chord"}


def ensure_extracted(cfg: dict) -> Path:
    """Extrai o zip em `paths.raw_dir` (só na primeira vez) e devolve a pasta `EMOPIA+`."""
    root = resolve(cfg["paths"]["raw_dir"]) / FOLDER
    if (root / "midis").is_dir():
        return root
    zip_path = resolve(cfg["paths"]["emopia_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} não encontrado: rode scripts/download_data.sh")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if m.startswith(f"{FOLDER}/midis/") or m.startswith(f"{FOLDER}/split/")]
        zf.extractall(root.parent, members=members)
    return root


def list_clips(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "midis").glob("*.mid"))


def midi_path(root: Path, clip: str) -> Path:
    return root / "midis" / f"{clip}.mid"


def merge_piano(score: symusic.Score, skip: set[str] = frozenset()) -> symusic.Score:
    """Junta as faixas (menos as de `skip`) numa única faixa de piano, mantendo andamentos e compassos."""
    out = symusic.Score(score.ticks_per_quarter)
    out.tempos = score.tempos
    out.time_signatures = score.time_signatures
    piano = symusic.Track("piano", 0, False)
    for track in score.tracks:
        if track.name not in skip:
            piano.notes.extend(track.notes)
    piano.notes.sort(key=lambda n: (n.time, n.pitch))
    out.tracks.append(piano)
    return out


def load_score(root: Path, clip: str) -> symusic.Score:
    return merge_piano(symusic.Score(midi_path(root, clip)), ANNOTATION_TRACKS)


def official_split(root: Path) -> dict[str, list[str]]:
    """Divisão por música publicada com o EMOPIA+ (`split/{train,val,test}_clip.csv`)."""
    out = {}
    for name in ("train", "val", "test"):
        clips = pd.read_csv(root / "split" / f"{name}_clip.csv")["clip_name"]
        out[name] = sorted(c.removesuffix(".mid") for c in clips)
    return out


def song_id(clip: str) -> str:
    return clip[3:14]


def quadrant(clip: str) -> int:
    return int(clip[1])
