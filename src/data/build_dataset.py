"""Tokeniza o EMOPIA com os tokens de controle e salva uma sequência por clipe (e por transposição).

Para cada clipe:
  1. tokeniza com o REMI congelado;
  2. decodifica de volta para uma partitura quantizada e mede (energia, tensão) por compasso com
     `measure_piece`; medir a versão quantizada garante que os compassos batem com os tokens `Bar`
     e é exatamente o que acontece com as músicas geradas na avaliação;
  3. quantiza em níveis com os tercis do treino e insere `Energy_e Tension_t` depois de cada `Bar`;
  4. no treino, repete para cada transposição de `data.transpose_range` (os níveis medidos no
     original valem para todas: energia não depende da altura e a tensão é relativa à tonalidade,
     que transpõe junto). Transposições que tiram alguma nota do teclado são descartadas.

Saída: `data/processed/{train,val,test}.pt`, cada um com
  {"sequences": [LongTensor], "clip_ids": [str], "transpose": [int], "levels": [[(e, t), ...]],
   "controls": bool, "vocab_size": int}
Cada sequência é `BOS + tokens + EOS`, inteira; o recorte em janelas é feito por `src.data.dataset`.

Uso:
  uv run python -m src.data.build_dataset               # precisa de configs/control_thresholds.json
  uv run python -m src.data.build_dataset --no-controls # sem tokens de controle (teste do pipeline)
"""

from __future__ import annotations

import argparse
import json
from collections import Counter

import symusic
import torch
from tqdm import tqdm

from src.common import load_config, resolve, set_seed
from src.controls.measure import measure_piece, to_levels
from src.controls.tokens import insert_controls
from src.data.emopia import ensure_extracted, midi_path
from src.data.tokenizer import BOS, EOS, decode, encode, load_tokenizer

PIANO_LOW, PIANO_HIGH = 21, 108


def _transpositions(score: symusic.Score, shifts: list[int]):
    pitches = [n.pitch for t in score.tracks for n in t.notes]
    lo, hi = min(pitches), max(pitches)
    for s in shifts:
        if PIANO_LOW <= lo + s and hi + s <= PIANO_HIGH:
            yield s, (score if s == 0 else score.shift_pitch(s))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-controls", action="store_true", help="não mede nem insere tokens de controle")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    root = ensure_extracted(cfg)
    tokenizer = load_tokenizer()
    vocab = tokenizer.vocab
    splits = json.loads(resolve(cfg["paths"]["splits"]).read_text(encoding="utf-8"))

    thresholds = None
    if not args.no_controls:
        thr_path = resolve(cfg["paths"]["control_thresholds"])
        if not thr_path.exists():
            raise SystemExit(f"{thr_path} não existe: rode a calibração dos tercis (src.controls) ou use --no-controls")
        thresholds = json.loads(thr_path.read_text(encoding="utf-8"))

    lo, hi = cfg["data"]["transpose_range"]
    out_dir = resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    for split, clips in splits.items():
        shifts = list(range(lo, hi + 1)) if split == "train" else [0]
        data = {"sequences": [], "clip_ids": [], "transpose": [], "levels": [],
                "controls": not args.no_controls, "vocab_size": len(tokenizer)}
        skipped = Counter()
        for clip in tqdm(clips, desc=split):
            score = symusic.Score(midi_path(root, clip))
            levels = []
            if thresholds is not None:
                levels = to_levels(measure_piece(decode(tokenizer, encode(tokenizer, score))), thresholds)
            for shift, s in _transpositions(score, shifts):
                ids = encode(tokenizer, s)
                if thresholds is not None:
                    ids = insert_controls(ids, levels, vocab)
                data["sequences"].append(torch.tensor([vocab[BOS], *ids, vocab[EOS]], dtype=torch.long))
                data["clip_ids"].append(clip)
                data["transpose"].append(shift)
                data["levels"].append(levels)
            skipped["transposições fora do teclado"] += len(shifts) - sum(1 for _ in _transpositions(score, shifts))
        torch.save(data, out_dir / f"{split}.pt")
        lens = torch.tensor([len(x) for x in data["sequences"]], dtype=torch.float)
        print(f"{split}: {len(data['sequences'])} sequências de {len(clips)} clipes | "
              f"comprimento médio {lens.mean():.0f} ± {lens.std():.0f} | {dict(skipped)}")


if __name__ == "__main__":
    main()
