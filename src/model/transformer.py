"""Transformer decoder-only (estilo GPT) escrito com camadas nativas do PyTorch.

Embedding de tokens + embedding posicional aprendido -> N x nn.TransformerEncoderLayer com máscara
causal (pre-LayerNorm) -> LayerNorm final -> cabeça linear (pesos amarrados ao embedding).

Implementa a interface comum `next_token_logits` (contrato 6), usada pela rotina de geração.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn


@dataclass
class TransformerConfig:
    vocab_size: int
    max_len: int = 1024
    d_model: int = 512
    n_layers: int = 6
    n_heads: int = 8
    d_ff: int = 2048
    dropout: float = 0.1
    tie_weights: bool = True
    pad_id: int = 0


class MusicTransformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = cfg.vocab_size
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=cfg.n_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.head.weight = self.tok_emb.weight
        self.register_buffer("causal_mask", nn.Transformer.generate_square_subsequent_mask(cfg.max_len), persistent=False)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.LongTensor) -> torch.Tensor:
        """ids: (B, T) com T <= max_len -> logits (B, T, V).

        O padding fica sempre à direita, então a máscara causal basta: posições reais nunca
        enxergam PAD, e a perda dos PADs é ignorada no treino.
        """
        _, t = ids.shape
        if t > self.cfg.max_len:
            raise ValueError(f"sequência de {t} tokens > contexto de {self.cfg.max_len}")
        pos = torch.arange(t, device=ids.device)
        x = self.drop(self.tok_emb(ids) + self.pos_emb(pos))
        x = self.blocks(x, mask=self.causal_mask[:t, :t], is_causal=True)
        return self.head(self.norm(x))

    @torch.no_grad()
    def next_token_logits(self, prefix: torch.LongTensor) -> torch.Tensor:
        """prefix: (B, T) -> (B, V). Usa só os últimos `max_len` tokens do prefixo."""
        prefix = prefix[:, -self.cfg.max_len :].to(self.pos_emb.weight.device)
        return self.forward(prefix)[:, -1, :].float()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def config_dict(self) -> dict:
        return asdict(self.cfg)


def load_checkpoint(path, device: str | torch.device = "cpu") -> MusicTransformer:
    """Carrega um checkpoint salvo por `src.model.train` em modo de avaliação."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MusicTransformer(TransformerConfig(**ckpt["model_config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()
