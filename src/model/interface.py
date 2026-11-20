"""Contrato 6: interface comum do Transformer e da cadeia de Markov.

A rotina de geração só conhece este método, então a amostragem e o forçamento dos tokens de
controle são o mesmo código para os dois sistemas.

Responsáveis: Natanael e Leonardo.
"""

from __future__ import annotations

from typing import Protocol

import torch


class NextTokenModel(Protocol):
    vocab_size: int

    def next_token_logits(self, prefix: torch.LongTensor) -> torch.Tensor:
        """prefix: (B, T) ids de tokens -> (B, vocab_size) logits (ou log-probabilidades) do próximo token.

        Todas as linhas do lote têm o mesmo comprimento T. O Transformer trunca o prefixo para a
        própria janela de contexto; a cadeia de Markov usa só os últimos n-1 tokens.
        """
        ...
