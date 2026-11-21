"""Acesso ao EMOPIA 1.0: extração do zip e utilidades sobre os ids de clipe.

Um clipe se chama `Q<quadrante>_<id da música no YouTube (11 caracteres)>_<índice do clipe>`, ex.:
`Q1_0vLPYiPN7qY_0`. Como alguns ids do YouTube têm `_`, o id da música é extraído por posição.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from src.common import resolve


def ensure_extracted(cfg: dict) -> Path:
    """Extrai o zip em `paths.raw_dir` (só na primeira vez) e devolve a pasta `EMOPIA_1.0`."""
    root = resolve(cfg["paths"]["raw_dir"]) / "EMOPIA_1.0"
    if (root / "midis").is_dir():
        return root
    zip_path = resolve(cfg["paths"]["emopia_zip"])
    if not zip_path.exists():
        raise FileNotFoundError(f"{zip_path} não encontrado: baixe o EMOPIA 1.0 e coloque o zip na raiz do repositório")
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if not m.startswith("__MACOSX/") and not m.endswith(".DS_Store")]
        zf.extractall(root.parent, members=members)
    return root


def list_clips(root: Path) -> list[str]:
    return sorted(p.stem for p in (root / "midis").glob("*.mid"))


def midi_path(root: Path, clip: str) -> Path:
    return root / "midis" / f"{clip}.mid"


def song_id(clip: str) -> str:
    return clip[3:14]


def quadrant(clip: str) -> int:
    return int(clip[1])
