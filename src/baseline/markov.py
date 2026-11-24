"""Baseline: cadeia de Markov de ordem n sobre os mesmos tokens do Transformer.

Probabilidades com desconto absoluto interpolado (Ney et al., 1994), uma forma de backoff
normalizada:

    P_k(w | h_k) = max(c(h_k, w) - D, 0) / c(h_k) + D * N1+(h_k) / c(h_k) * P_{k-1}(w | h_{k-1})

onde h_k são os últimos k tokens, c(.) são contagens no treino e N1+(h_k) é o número de tokens
distintos vistos depois de h_k. Contexto nunca visto (c(h_k) = 0) recai na ordem de baixo. A base
é a distribuição de unigramas com add-one. Uma cadeia de ordem n usa contextos de até n-1 tokens.

Como o vocabulário tem menos de 512 tokens, um contexto de até 7 tokens cabe exatamente em um
inteiro de 63 bits (9 bits por token), sem colisões; as contagens ficam em arrays numpy ordenados.

A ordem n e o desconto D são escolhidos pela log-verossimilhança na validação do EMOPIA+. (Testamos
também Witten-Bell: a ordem 3 já piorava em relação à 2, e a melhor NLL ficou um pouco acima.)

Dados: os mesmos que o Transformer vê nas duas fases, para a comparação ser justa: o treino do
EMOPIA+ (com transposição) mais o Pop1K7 sem transposição (`--no-pretrain` desliga), que cabe na
memória do job sem perder nenhuma música.

Uso:
  uv run python -m src.baseline.markov                 # ajusta n = 2..8 e D, escolhe pela validação e salva
  uv run python -m src.baseline.markov --no-pretrain   # só com o treino do EMOPIA+
Saída: outputs/markov/markov.npz e outputs/markov/selection.json

Responsável: Leonardo.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from src.common import load_config, resolve
from src.data.dataset import load_split

BITS = 9
MAX_CONTEXT = 7  # 7 * 9 = 63 bits
DISCOUNTS = (0.5, 0.75, 0.9)


def _interpolate(c_hw, c_h, n1, p_lower, discount: float):
    return np.maximum(c_hw - discount, 0) / c_h + discount * n1 / c_h * p_lower


def _context_keys(seqs: list[np.ndarray], k: int) -> tuple[np.ndarray, np.ndarray]:
    """Para cada posição t >= k de cada sequência: (chave do contexto x[t-k:t], próximo token x[t])."""
    keys, nexts = [], []
    for s in seqs:
        if len(s) <= k:
            continue
        key = np.zeros(len(s) - k, dtype=np.int64)
        for j in range(k):
            key = (key << BITS) | s[j : len(s) - k + j]
        keys.append(key)
        nexts.append(s[k:])
    if not keys:
        return np.zeros(0, np.int64), np.zeros(0, np.int64)
    return np.concatenate(keys), np.concatenate(nexts)


class _OrderTable:
    """Contagens de uma ordem k: contextos únicos ordenados e, para cada um, (próximo, contagem)."""

    def __init__(self, ctx: np.ndarray, total: np.ndarray, distinct: np.ndarray,
                 offsets: np.ndarray, next_tok: np.ndarray, count: np.ndarray, vocab_size: int):
        self.ctx, self.total, self.distinct = ctx, total, distinct
        self.offsets, self.next_tok, self.count = offsets, next_tok, count
        self.pair_key = np.repeat(np.arange(len(ctx), dtype=np.int64), np.diff(offsets)) * vocab_size + next_tok
        self.vocab_size = vocab_size

    @classmethod
    def fit(cls, seqs: list[np.ndarray], k: int, vocab_size: int) -> "_OrderTable":
        keys, nexts = _context_keys(seqs, k)
        pairs = np.stack([keys, nexts], axis=1)
        uniq, count = np.unique(pairs, axis=0, return_counts=True)  # ordenado por (contexto, próximo)
        ctx, start = np.unique(uniq[:, 0], return_index=True)
        offsets = np.append(start, len(uniq))
        total = np.add.reduceat(count, start)
        distinct = np.diff(offsets)
        return cls(ctx, total, distinct, offsets, uniq[:, 1].copy(), count, vocab_size)

    def lookup(self, keys: np.ndarray) -> np.ndarray:
        """Índice de cada contexto na tabela, ou -1 se nunca foi visto."""
        idx = np.searchsorted(self.ctx, keys)
        idx = np.minimum(idx, len(self.ctx) - 1)
        return np.where(self.ctx[idx] == keys, idx, -1)

    def pair_count(self, ctx_idx: np.ndarray, tokens: np.ndarray) -> np.ndarray:
        """c(h, w) para pares (contexto já encontrado, token)."""
        q = ctx_idx * self.vocab_size + tokens
        pos = np.minimum(np.searchsorted(self.pair_key, q), len(self.pair_key) - 1)
        return np.where(self.pair_key[pos] == q, self.count[pos], 0)

    def arrays(self, prefix: str) -> dict[str, np.ndarray]:
        return {f"{prefix}_{name}": getattr(self, name)
                for name in ("ctx", "total", "distinct", "offsets", "next_tok", "count")}


class MarkovModel:
    def __init__(self, unigram: np.ndarray, tables: list[_OrderTable], order: int, discount: float = 0.75):
        self.unigram = unigram          # P_0, add-one
        self.tables = tables            # tables[k-1] = contexto de k tokens
        self.order = order              # usa contextos de até order-1 tokens
        self.discount = discount
        self.vocab_size = len(unigram)
        self.context_len = order - 1

    # ---------- treino ----------
    @classmethod
    def fit(cls, seqs: list[np.ndarray], vocab_size: int, max_order: int) -> "MarkovModel":
        if max_order - 1 > MAX_CONTEXT:
            raise ValueError(f"ordem máxima suportada: {MAX_CONTEXT + 1}")
        counts = np.bincount(np.concatenate(seqs), minlength=vocab_size).astype(np.float64)
        unigram = (counts + 1) / (counts.sum() + vocab_size)
        tables = [_OrderTable.fit(seqs, k, vocab_size) for k in range(1, max_order)]
        return cls(unigram, tables, max_order)

    # ---------- avaliação ----------
    def token_log_probs(self, seqs: list[np.ndarray]) -> dict[int, np.ndarray]:
        """log P(x_t | x_<t) de todos os tokens t >= 1, para cada ordem n = 1..self.order."""
        targets = np.concatenate([s[1:] for s in seqs])
        p = self.unigram[targets]
        out = {1: np.log(p)}
        for k, table in enumerate(self.tables[: self.order - 1], start=1):
            # chave do contexto de k tokens antes de cada alvo; posições com menos de k tokens antes ficam -1
            keys = np.concatenate([
                np.concatenate([np.full(min(k - 1, len(s) - 1), -1, np.int64), _context_keys([s], k)[0]])
                for s in seqs])
            idx = np.where(keys >= 0, table.lookup(np.maximum(keys, 0)), -1)
            seen = idx >= 0
            c_hw = np.zeros_like(p)
            c_hw[seen] = table.pair_count(idx[seen], targets[seen])
            c_h = np.where(seen, table.total[np.maximum(idx, 0)], 1)
            n1 = np.where(seen, table.distinct[np.maximum(idx, 0)], 0)
            p = np.where(seen, _interpolate(c_hw, c_h, n1, p, self.discount), p)
            out[k + 1] = np.log(p)
        return out

    # ---------- interface comum (contrato 6) ----------
    def next_token_logits(self, prefix: torch.LongTensor) -> torch.Tensor:
        """prefix: (B, T) -> (B, V) log-probabilidades do próximo token."""
        rows = prefix.detach().cpu().numpy()
        out = np.empty((len(rows), self.vocab_size))
        for b, row in enumerate(rows):
            p = self.unigram.copy()
            for k, table in enumerate(self.tables[: self.order - 1], start=1):
                if len(row) < k:
                    break
                key = 0
                for tok in row[-k:]:
                    key = (key << BITS) | int(tok)
                idx = int(table.lookup(np.array([key], np.int64))[0])
                if idx < 0:
                    continue
                lo, hi = table.offsets[idx], table.offsets[idx + 1]
                c_hw = np.zeros(self.vocab_size)
                c_hw[table.next_tok[lo:hi]] = table.count[lo:hi]
                p = _interpolate(c_hw, table.total[idx], table.distinct[idx], p, self.discount)
            out[b] = np.log(p)
        return torch.from_numpy(out).float()

    # ---------- persistência ----------
    def save(self, path) -> None:
        arrays = {"unigram": self.unigram, "order": np.array(self.order), "discount": np.array(self.discount)}
        for k, t in enumerate(self.tables[: self.order - 1], start=1):
            arrays.update(t.arrays(f"k{k}"))
        np.savez_compressed(path, **arrays)

    @classmethod
    def load(cls, path) -> "MarkovModel":
        z = np.load(path)
        order, unigram = int(z["order"]), z["unigram"]
        tables = [_OrderTable(*(z[f"k{k}_{n}"] for n in ("ctx", "total", "distinct", "offsets", "next_tok", "count")),
                              vocab_size=len(unigram))
                  for k in range(1, order)]
        return cls(unigram, tables, order, float(z["discount"]))


def _as_numpy(seqs: list[torch.Tensor]) -> list[np.ndarray]:
    return [s.numpy().astype(np.int64) for s in seqs]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pretrain", action="store_true", help="não usa o Pop1K7")
    args = parser.parse_args()

    cfg = load_config()
    max_order = MAX_CONTEXT + 1
    train, val = load_split("train", cfg), load_split("val", cfg)
    vocab_size = train["vocab_size"]
    seqs = _as_numpy(train["sequences"])
    sources = [f"EMOPIA+ train ({len(seqs)} sequências)"]
    pre_path = Path(resolve(cfg["paths"]["processed_dir"])) / "pretrain_train.pt"
    if not args.no_pretrain and pre_path.exists():
        pre = load_split("pretrain_train", cfg)
        extra = [s for s, t in zip(pre["sequences"], pre["transpose"]) if t == 0]
        seqs += _as_numpy(extra)
        sources.append(f"Pop1K7 sem transposição ({len(extra)} sequências)")
        del pre
    print(f"ajustando ordens até {max_order} em {sum(len(s) for s in seqs)} tokens: {' + '.join(sources)}", flush=True)
    full = MarkovModel.fit(seqs, vocab_size, max_order)
    del seqs

    val_seqs = _as_numpy(val["sequences"])
    selection = []
    for d in DISCOUNTS:
        full.discount = d
        for n, lp in full.token_log_probs(val_seqs).items():
            if n >= 2:
                selection.append({"order": n, "discount": d, "val_nll": float(-lp.mean()),
                                  "val_ppl": float(math.exp(-lp.mean()))})
        print(f"  D = {d}: " + " | ".join(f"n={s['order']} {s['val_nll']:.4f}" for s in selection if s["discount"] == d))
    best = min(selection, key=lambda s: s["val_nll"])
    print(f"escolhido: n = {best['order']}, D = {best['discount']} | NLL {best['val_nll']:.4f} "
          f"(perplexidade {best['val_ppl']:.2f})")

    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "markov"
    out_dir.mkdir(parents=True, exist_ok=True)
    n = best["order"]
    MarkovModel(full.unigram, full.tables[: n - 1], n, best["discount"]).save(out_dir / "markov.npz")
    (out_dir / "selection.json").write_text(json.dumps({"best": best, "all": selection, "data": sources}, indent=2) + "\n")
    print(f"salvo em {out_dir / 'markov.npz'}")


if __name__ == "__main__":
    main()
