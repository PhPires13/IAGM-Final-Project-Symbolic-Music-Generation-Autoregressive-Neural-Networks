"""Utilidades compartilhadas: raiz do repositório, leitura do config e seed global."""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "default.yaml"


def resolve(path: str | os.PathLike) -> Path:
    """Caminho relativo à raiz do repositório (absolutos passam direto)."""
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike | None = None) -> dict[str, Any]:
    """Lê o YAML de config. Sem argumento, lê configs/default.yaml."""
    with open(resolve(path) if path else DEFAULT_CONFIG, encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    """Fixa a seed de random, numpy e torch (CPU e CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
