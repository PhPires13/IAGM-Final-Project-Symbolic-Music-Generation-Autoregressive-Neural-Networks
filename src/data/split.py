"""Contrato 5: split oficial do EMOPIA, por música.

Reproduz `scripts/prepare_split.ipynb` do EMOPIA 1.0: a partir de `metadata_by_song.csv`, sorteia
8 músicas de cada quadrante dominante para o teste e, do restante, mais 8 por quadrante para a
validação, sempre com `random_state=1`. O que sobra é treino.

Uso: uv run python -m src.data.split
"""

from __future__ import annotations

import json
from collections import Counter

import pandas as pd

from src.common import load_config, resolve
from src.data.emopia import ensure_extracted, list_clips, quadrant, song_id

SONGS_PER_QUADRANT = 8
RANDOM_STATE = 1  # fixo no notebook oficial


def _sample_by_quadrant(songs: pd.DataFrame) -> pd.DataFrame:
    return pd.concat(
        [songs[songs["DominantQ"] == q].sample(SONGS_PER_QUADRANT, random_state=RANDOM_STATE) for q in (1, 2, 3, 4)]
    )


def official_song_split(metadata_csv) -> dict[str, list[str]]:
    data = pd.read_csv(metadata_csv).set_index("songID")
    test = _sample_by_quadrant(data)
    train_val = data.drop(labels=test.index)
    val = _sample_by_quadrant(train_val)
    train = train_val.drop(labels=val.index)
    return {"train": list(train.index), "val": list(val.index), "test": list(test.index)}


def main() -> None:
    cfg = load_config()
    root = ensure_extracted(cfg)
    songs = official_song_split(root / "metadata_by_song.csv")

    song_to_split = {s: name for name, ss in songs.items() for s in ss}
    assert len(song_to_split) == sum(len(ss) for ss in songs.values()), "música em mais de uma partição"

    splits: dict[str, list[str]] = {"train": [], "val": [], "test": []}
    for clip in list_clips(root):
        splits[song_to_split[song_id(clip)]].append(clip)

    out = resolve(cfg["paths"]["splits"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(splits, indent=2) + "\n", encoding="utf-8")

    for name, clips in splits.items():
        per_q = Counter(quadrant(c) for c in clips)
        print(f"{name:5s}: {len(songs[name]):3d} músicas, {len(clips):4d} clipes | por quadrante: {dict(sorted(per_q.items()))}")
    print(f"salvo em {out}")


if __name__ == "__main__":
    main()
