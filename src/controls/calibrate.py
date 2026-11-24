"""Calcula os limites dos níveis (tercis para K = 3) de energia e tensão SÓ no conjunto de treino.

Mede todos os compassos dos clipes de treino do EMOPIA+ (originais, sem transposição) na partitura
quantizada pelo REMI e salva os quantis em `configs/control_thresholds.json`. Os mesmos limites valem
para o Pop1K7 no pré-treino, para que um nível signifique a mesma coisa nas duas fases. Também salva
os valores brutos em `outputs/controls/train_values.json` e, com `--pretrain`, os do Pop1K7 em
`outputs/controls/pretrain_values.json`, para as verificações de sanidade.

Uso: uv run python -m src.controls.calibrate [--pretrain]

Responsável: Julia M.

"""

from __future__ import annotations

import argparse
import json

import numpy as np
from tqdm import tqdm

from src.common import load_config, resolve
from src.controls.measure import measure_piece
from src.data.corpus import load_piece, roots
from src.data.tokenizer import decode, encode, load_tokenizer


def measure_split(split: str, items: list[str], root_by_corpus: dict, tokenizer) -> dict[str, list]:
    values = {}
    for item in tqdm(items, desc=f"medindo {split}", mininterval=5):
        score = load_piece(split, item, root_by_corpus)[0]
        values[item] = measure_piece(decode(tokenizer, encode(tokenizer, score)))
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrain", action="store_true", help="mede também o Pop1K7 (só para a sanidade)")
    args = parser.parse_args()

    cfg = load_config()
    tokenizer = load_tokenizer()
    k = cfg["controls"]["n_levels"]
    splits = json.loads(resolve(cfg["paths"]["splits"]).read_text(encoding="utf-8"))
    wanted = {"train": splits["train"], **({"pretrain_train": splits["pretrain_train"]} if args.pretrain else {})}
    root_by_corpus = roots(cfg, wanted)
    values = measure_split("train", wanted["train"], root_by_corpus, tokenizer)

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
    print(f"{len(energy)} compassos de {len(values)} clipes | limites: {thresholds} | salvo em {out}")
    if args.pretrain:
        pre = measure_split("pretrain_train", wanted["pretrain_train"], root_by_corpus, tokenizer)
        (raw.parent / "pretrain_values.json").write_text(json.dumps(pre), encoding="utf-8")
        print(f"Pop1K7: {sum(len(v) for v in pre.values())} compassos de {len(pre)} músicas")


if __name__ == "__main__":
    main()
