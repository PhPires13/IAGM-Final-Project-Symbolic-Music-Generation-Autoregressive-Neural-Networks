"""Estatísticas do dataset para o artigo.

Por partição: músicas, clipes (total e por quadrante), duração e número de notas dos clipes,
comprimento das sequências de tokens (só as originais, sem transposição) e número de janelas.

Saída: `outputs/stats/dataset_stats.json` e uma tabela em markdown no terminal.

Uso: uv run python -m src.data.stats
"""

from __future__ import annotations

import json
from collections import Counter

import numpy as np
import symusic

from src.common import load_config, resolve
from src.data.dataset import WindowDataset, load_split
from src.data.emopia import ensure_extracted, midi_path, quadrant, song_id
from src.data.tokenizer import PAD, load_tokenizer


def _mean_std(values) -> dict:
    a = np.asarray(values, dtype=float)
    return {"mean": round(float(a.mean()), 1), "std": round(float(a.std()), 1),
            "min": round(float(a.min()), 1), "max": round(float(a.max()), 1)}


def main() -> None:
    cfg = load_config()
    root = ensure_extracted(cfg)
    tokenizer = load_tokenizer()
    splits = json.loads(resolve(cfg["paths"]["splits"]).read_text(encoding="utf-8"))

    stats = {"vocab_size": len(tokenizer), "splits": {}}
    for split, clips in splits.items():
        scores = [symusic.Score(midi_path(root, c)) for c in clips]
        data = load_split(split, cfg)
        original = [s for s, t in zip(data["sequences"], data["transpose"]) if t == 0]
        windows = WindowDataset(data["sequences"], cfg["data"]["max_seq_len"], cfg["data"]["stride"],
                                tokenizer.vocab[PAD])
        stats["splits"][split] = {
            "songs": len({song_id(c) for c in clips}),
            "clips": len(clips),
            "clips_per_quadrant": dict(sorted(Counter(quadrant(c) for c in clips).items())),
            "clip_seconds": _mean_std([s.to("second").end() for s in scores]),
            "notes_per_clip": _mean_std([sum(len(t.notes) for t in s.tracks) for s in scores]),
            "tokens_per_clip": _mean_std([len(s) for s in original]),
            "sequences_with_augmentation": len(data["sequences"]),
            "windows": len(windows),
            "controls": data["controls"],
        }

    out = resolve(cfg["paths"]["outputs_dir"]) / "stats" / "dataset_stats.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(stats, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"Vocabulário: {stats['vocab_size']} tokens\n")
    print("| Partição | Músicas | Clipes | Duração (s) | Notas | Tokens | Seq. c/ aumento | Janelas |")
    print("|---|---|---|---|---|---|---|---|")
    for split, s in stats["splits"].items():
        f = lambda d: f"{d['mean']:.0f} ± {d['std']:.0f}"  # noqa: E731
        print(f"| {split} | {s['songs']} | {s['clips']} | {f(s['clip_seconds'])} | {f(s['notes_per_clip'])} | "
              f"{f(s['tokens_per_clip'])} | {s['sequences_with_augmentation']} | {s['windows']} |")
    print(f"\nsalvo em {out}")


if __name__ == "__main__":
    main()
