"""Calcula os limites dos níveis (tercis para K = 3) de energia e tensão SÓ no conjunto de treino.

Mede todos os compassos dos clipes de treino (originais, sem transposição) na partitura quantizada
pelo REMI e salva os quantis em `configs/control_thresholds.json`. Também salva os valores brutos
em `outputs/controls/train_values.json` para as verificações de sanidade.

Uso: uv run python -m src.controls.calibrate
"""

from __future__ import annotations

import json

import numpy as np
import symusic
from tqdm import tqdm

from src.common import load_config, resolve
from src.controls.measure import measure_piece
from src.data.emopia import ensure_extracted, midi_path
from src.data.tokenizer import decode, encode, load_tokenizer


def main() -> None:
    cfg = load_config()
    root = ensure_extracted(cfg)
    tokenizer = load_tokenizer()
    k = cfg["controls"]["n_levels"]
    train = json.loads(resolve(cfg["paths"]["splits"]).read_text(encoding="utf-8"))["train"]

    values = {}
    for clip in tqdm(train, desc="medindo treino"):
        score = decode(tokenizer, encode(tokenizer, symusic.Score(midi_path(root, clip))))
        values[clip] = measure_piece(score)

    energy = np.array([e for v in values.values() for e, _ in v])
    tension = np.array([t for v in values.values() for _, t in v])
    qs = [i / k for i in range(1, k)]
    thresholds = {"energy": [float(x) for x in np.quantile(energy, qs)],
                  "tension": [float(x) for x in np.quantile(tension, qs)]}

    out = resolve(cfg["paths"]["control_thresholds"])
    out.write_text(json.dumps(thresholds, indent=2) + "\n", encoding="utf-8")
    raw = resolve(cfg["paths"]["outputs_dir"]) / "controls" / "train_values.json"
    raw.parent.mkdir(parents=True, exist_ok=True)
    raw.write_text(json.dumps(values), encoding="utf-8")
    print(f"{len(energy)} compassos de {len(train)} clipes | limites: {thresholds} | salvo em {out}")


if __name__ == "__main__":
    main()
