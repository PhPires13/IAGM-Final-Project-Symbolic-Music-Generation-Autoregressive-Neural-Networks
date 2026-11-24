"""Dataset em janelas para o treino do modelo autoregressivo.

Cada sequência de `data/processed/{split}.pt` é `BOS Emotion ... EOS`. Com `keep_prefix = k`, os
k primeiros tokens (o prefixo `BOS Emotion`) são repetidos no início de TODA janela, e o resto da
sequência (o corpo) é recortado em pedaços de `max_len + 1 - k` tokens com passo `stride` (o último
pedaço termina no fim do corpo). Assim o token de emoção nunca sai do contexto, como na geração.
Sem prefixo (k = 0), é o recorte simples.

Cada item devolve (entrada, alvo) deslocados de um token, com -100 no alvo das posições ignoradas
pela cross-entropy: PAD, o próprio token de emoção (é sempre dado, nunca previsto) e, nas janelas
que não começam no início da música, o primeiro token do corpo (previsto sem nenhum contexto).

Responsável: Pedro.
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
    def __init__(self, sequences: list[torch.Tensor], max_len: int, stride: int, pad_id: int, keep_prefix: int = 0):
        self.sequences = sequences
        self.max_len = max_len
        self.pad_id = pad_id
        self.k = keep_prefix
        self.index: list[tuple[int, int]] = []
        span = max_len + 1 - keep_prefix
        for i, seq in enumerate(sequences):
            n = len(seq) - keep_prefix
            starts = list(range(0, max(n - span, 0) + 1, stride))
            if n > span and starts[-1] != n - span:
                starts.append(n - span)
            self.index.extend((i, s) for s in starts)

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        i, start = self.index[idx]
        seq, k = self.sequences[i], self.k
        body = seq[k + start : k + start + self.max_len + 1 - k]
        chunk = torch.cat([seq[:k], body]).long()
        x = torch.full((self.max_len,), self.pad_id, dtype=torch.long)
        y = torch.full((self.max_len,), IGNORE_INDEX, dtype=torch.long)
        x[: len(chunk) - 1] = chunk[:-1]
        y[: len(chunk) - 1] = chunk[1:]
        if k:
            y[: k - 1] = IGNORE_INDEX          # o token de emoção é dado, não previsto
            if start > 0:
                y[k - 1] = IGNORE_INDEX        # primeiro token do pedaço, sem contexto
        return x, y
