"""Dataset em janelas para o treino do modelo autoregressivo.

Cada sequência de `data/processed/{split}.pt` é recortada em janelas de `max_len + 1` tokens com
passo `stride` (a última janela de cada sequência termina no fim dela). Sequências mais curtas
viram uma única janela completada com PAD. Cada item devolve (entrada, alvo) deslocados de um
token, com -100 no alvo das posições de PAD (ignoradas pela cross-entropy).
"""

from __future__ import annotations

import torch
from torch.utils.data import Dataset

from src.common import load_config, resolve

IGNORE_INDEX = -100


def load_split(split: str, cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    return torch.load(resolve(cfg["paths"]["processed_dir"]) / f"{split}.pt", weights_only=False)


class WindowDataset(Dataset):
    def __init__(self, sequences: list[torch.Tensor], max_len: int, stride: int, pad_id: int):
        self.sequences = sequences
        self.max_len = max_len
        self.pad_id = pad_id
        self.index: list[tuple[int, int]] = []
        span = max_len + 1
        for i, seq in enumerate(sequences):
            n = len(seq)
            starts = list(range(0, max(n - span, 0) + 1, stride))
            if n > span and starts[-1] != n - span:
                starts.append(n - span)
            self.index.extend((i, s) for s in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, start = self.index[idx]
        chunk = self.sequences[i][start : start + self.max_len + 1]
        x = torch.full((self.max_len,), self.pad_id, dtype=torch.long)
        y = torch.full((self.max_len,), IGNORE_INDEX, dtype=torch.long)
        x[: len(chunk) - 1] = chunk[:-1]
        y[: len(chunk) - 1] = chunk[1:]
        return x, y
