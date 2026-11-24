"""Contrato 5: partições do EMOPIA+ e do Pop1K7, salvas em `configs/splits.json`.

- `train`, `val`, `test`: divisão por música publicada com o EMOPIA+ (`split/*_clip.csv`). É a
  mesma divisão do `scripts/prepare_split.ipynb` do EMOPIA 1.0 (8 músicas por quadrante dominante
  no teste e na validação, `random_state=1`): conferimos que as músicas de cada partição coincidem.
- `pretrain_train`, `pretrain_val`: músicas do Pop1K7, `pretrain.val_fraction` na validação do
  pré-treino, depois de remover as que vazam para a validação/teste do EMOPIA+ (`src.data.leakage`).
  O relatório do vazamento fica em `configs/pop1k7_leakage.json`.

Uso: uv run python -m src.data.split
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict

from tqdm import tqdm

from src.common import load_config, resolve
from src.data import emopia, pop1k7
from src.data.leakage import Index, fingerprint, max_containment


def leakage_report(cfg: dict, emopia_root, pop_root, splits: dict) -> dict:
    n, threshold = cfg["pretrain"]["leakage_ngram"], cfg["pretrain"]["leakage_threshold"]
    songs = pop1k7.list_songs(pop_root)
    pop = {s: fingerprint(pop1k7.load_score(pop_root, s), n) for s in tqdm(songs, desc="impressões do Pop1K7")}
    held_out = {c: fingerprint(emopia.load_score(emopia_root, c), n) for c in splits["val"] + splits["test"]}
    matches = max_containment(held_out, Index(pop))
    leaked = defaultdict(list)
    for clip, (song, value) in matches.items():
        if song is not None and value > threshold:
            leaked[song].append({"clip": clip, "containment": round(value, 3)})

    # referência: o mesmo cálculo entre clipes de músicas diferentes do próprio EMOPIA+
    by_song = defaultdict(set)
    for c in splits["train"]:
        by_song[emopia.song_id(c)] |= fingerprint(emopia.load_score(emopia_root, c), n)
    base = sorted(v for _, v in max_containment(held_out, Index(dict(by_song))).values())
    pop_values = sorted(v for _, v in matches.values())

    def pct(values, q):
        return round(values[min(len(values) - 1, int(q * len(values)))], 3)

    return {
        "ngram": n, "threshold": threshold,
        "removed_songs": dict(sorted(leaked.items())),
        "n_removed": len(leaked),
        "clips_matched": sum(len(v) for v in leaked.values()),
        "containment_pop1k7": {"p50": pct(pop_values, 0.5), "p95": pct(pop_values, 0.95), "max": round(pop_values[-1], 3)},
        "containment_emopia_train_vs_heldout": {"p50": pct(base, 0.5), "p95": pct(base, 0.95), "max": round(base[-1], 3)},
    }


def main() -> None:
    cfg = load_config()
    emopia_root = emopia.ensure_extracted(cfg)
    pop_root = pop1k7.ensure_extracted(cfg)

    splits = emopia.official_split(emopia_root)
    song_sets = {k: {emopia.song_id(c) for c in v} for k, v in splits.items()}
    assert not (song_sets["train"] & song_sets["val"] or song_sets["train"] & song_sets["test"]
                or song_sets["val"] & song_sets["test"]), "música em mais de uma partição"

    report = leakage_report(cfg, emopia_root, pop_root, splits)
    clean = [s for s in pop1k7.list_songs(pop_root) if s not in report["removed_songs"]]
    splits["pretrain_train"], splits["pretrain_val"] = pop1k7.song_split(clean, cfg["pretrain"]["val_fraction"], cfg["seed"])

    out = resolve(cfg["paths"]["splits"])
    out.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")
    leak_path = resolve(cfg["paths"]["leakage_report"])
    leak_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    for name in ("train", "val", "test"):
        per_q = Counter(emopia.quadrant(c) for c in splits[name])
        print(f"{name:14s}: {len(song_sets[name]):4d} músicas, {len(splits[name]):4d} clipes | por quadrante: {dict(sorted(per_q.items()))}")
    for name in ("pretrain_train", "pretrain_val"):
        print(f"{name:14s}: {len(splits[name]):4d} músicas do Pop1K7")
    print(f"vazamento: {report['n_removed']} músicas do Pop1K7 removidas ({report['clips_matched']} clipes de val/teste "
          f"com contenção > {report['threshold']}) | relatório em {leak_path}")
    print(f"salvo em {out}")


if __name__ == "__main__":
    main()
