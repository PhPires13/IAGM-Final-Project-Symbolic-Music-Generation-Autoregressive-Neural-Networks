"""Rotina de geração condicionada, comum ao Transformer e ao Markov.

Cada amostra começa com `BOS Bar Energy_e0 Tension_t0`. A cada passo, todas as linhas do lote
avançam exatamente um token (o lote fica sempre retangular):
  - se a linha tem tokens forçados na fila, usa o próximo da fila;
  - senão, amostra do modelo, com tokens de controle, PAD e BOS proibidos, e EOS proibido até a
    curva acabar;
  - se o token amostrado é `Bar` e ainda há compassos na curva, enfileira `Energy_e Tension_t` do
    próximo compasso; se a curva já acabou, a linha termina (o `Bar` é trocado por EOS).
Linhas que terminam passam a receber PAD, que é descartado no fim.

Responsável: Leonardo.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from src.controls.tokens import BAR, ENERGY, TENSION, control_ids
from src.data.tokenizer import BOS, EOS, PAD
from src.generate.sampling import SamplingConfig, sample_next
from src.model.interface import NextTokenModel

Curve = Sequence[tuple[int, int]]  # (nível de energia, nível de tensão) por compasso


@dataclass
class Generated:
    ids: list[int]          # BOS ... EOS, COM os tokens de controle
    finished: bool          # False se bateu em max_tokens antes de terminar a curva


def _context_len(model) -> int | None:
    """Quantos tokens do prefixo o modelo realmente usa (None = todos)."""
    if hasattr(model, "context_len"):
        return model.context_len
    cfg = getattr(model, "cfg", None)
    return getattr(cfg, "max_len", None)


def _model_device(model) -> torch.device:
    if isinstance(model, torch.nn.Module):
        return next(model.parameters()).device
    return torch.device("cpu")


@torch.no_grad()
def generate(
    model: NextTokenModel,
    curves: Sequence[Curve],
    sampling: SamplingConfig,
    vocab: dict[str, int],
    *,
    max_tokens: int,
    generator: torch.Generator | None = None,
) -> list[Generated]:
    """Gera uma sequência por curva, em lote."""
    device = _model_device(model)
    bos, eos, pad, bar = vocab[BOS], vocab[EOS], vocab[PAD], vocab[BAR]

    def controls(level: tuple[int, int]) -> list[int]:
        return [vocab[f"{ENERGY}_{level[0]}"], vocab[f"{TENSION}_{level[1]}"]]

    banned = torch.zeros(len(vocab), dtype=torch.bool)
    banned[list(control_ids(vocab)) + [pad, bos]] = True
    banned = banned.to(device)

    b = len(curves)
    seqs = [[bos, bar] for _ in range(b)]
    queues = [deque(controls(c[0])) for c in curves]
    next_bar = [1] * b                  # índice do próximo compasso da curva a ser aberto
    done = [False] * b
    finished = [False] * b
    use_amp = device.type == "cuda"
    ctx = _context_len(model)

    while not all(done):
        if len(seqs[0]) >= max_tokens:
            break
        need_model = [i for i in range(b) if not done[i] and not queues[i]]
        sampled = {}
        if need_model:
            prefix = torch.tensor([seqs[i][-ctx:] if ctx else seqs[i] for i in need_model],
                                  dtype=torch.long, device=device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_amp):
                logits = model.next_token_logits(prefix).float().to(device)
            mask = banned.expand(len(need_model), -1).clone()
            for row, i in enumerate(need_model):
                if next_bar[i] < len(curves[i]):
                    mask[row, eos] = True
            logits = logits.masked_fill(mask, float("-inf"))
            gen_device = generator.device if generator is not None else device
            tokens = sample_next(logits.to(gen_device), sampling, generator)
            sampled = dict(zip(need_model, tokens.tolist()))

        for i in range(b):
            if done[i]:
                seqs[i].append(pad)
                continue
            if queues[i]:
                seqs[i].append(queues[i].popleft())
                continue
            tok = sampled[i]
            if tok == bar and next_bar[i] >= len(curves[i]):
                tok = eos
            seqs[i].append(tok)
            if tok == bar:
                queues[i].extend(controls(curves[i][next_bar[i]]))
                next_bar[i] += 1
            elif tok == eos:
                done[i] = finished[i] = True

    return [Generated([t for t in s if t != pad], f) for s, f in zip(seqs, finished)]
