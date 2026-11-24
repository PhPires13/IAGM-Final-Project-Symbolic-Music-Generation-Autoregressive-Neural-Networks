"""Tokeniza o EMOPIA+ e o Pop1K7 com os tokens de controle e salva uma sequência por peça (e por transposição).

Para cada peça:
  1. tokeniza com o REMI congelado;
  2. decodifica de volta para uma partitura quantizada e mede (energia, tensão) por compasso com
     `measure_piece`; medir a versão quantizada garante que os compassos batem com os tokens `Bar`
     e é exatamente o que acontece com as músicas geradas na avaliação;
  3. quantiza em níveis com os tercis do treino do EMOPIA+ e insere `Energy_e Tension_t` depois de
     cada `Bar`;
  4. prefixa `BOS Emotion_<q>` (quadrante do EMOPIA+, ou `Emotion_None` no Pop1K7) e termina com `EOS`;
  5. nas partições de treino, repete para cada transposição (`data.transpose_range` no EMOPIA+,
     `pretrain.transpose_range` no Pop1K7). Os níveis medidos no original valem para todas: energia
     não depende da altura e a tensão é relativa à tonalidade, que transpõe junto. Transposições
     que tiram alguma nota do teclado são descartadas.

Saída: `data/processed/<partição>.pt` para train, val, test (EMOPIA+) e pretrain_train, pretrain_val
(Pop1K7), cada um com
  {"sequences": [ShortTensor], "clip_ids": [str], "transpose": [int], "levels": [[(e, t), ...]],
   "emotion": [int | None], "controls": bool, "vocab_size": int}
Os tokens ficam em int16 (o vocabulário tem menos de 32768 tokens) para caber na memória com o
Pop1K7 transposto; `WindowDataset` converte para long. O recorte em janelas é feito por
`src.data.dataset`.

Uso:
  uv run python -m src.data.build_dataset                       # todas as partições
  uv run python -m src.data.build_dataset --splits train val    # só algumas
  uv run python -m src.data.build_dataset --no-controls         # sem tokens de controle (teste)

Responsável: Pedro.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from multiprocessing import Pool

import symusic
import torch
from tqdm import tqdm

from src.common import load_config, resolve, set_seed
from src.controls.measure import measure_piece, to_levels
from src.controls.tokens import emotion_id, insert_controls
from src.data.corpus import is_pretrain, load_piece, roots
from src.data.tokenizer import BOS, EOS, decode, encode, load_tokenizer

PIANO_LOW, PIANO_HIGH = 21, 108
ALL_SPLITS = ("train", "val", "test", "pretrain_train", "pretrain_val")

_state: dict = {}


def _transpositions(score: symusic.Score, shifts: list[int]):
    pitches = [n.pitch for t in score.tracks for n in t.notes]
    if not pitches:
        return
    lo, hi = min(pitches), max(pitches)
    for s in shifts:
        if PIANO_LOW <= lo + s and hi + s <= PIANO_HIGH:
            yield s, (score if s == 0 else score.shift_pitch(s))


def _init(roots_by_corpus, thresholds) -> None:
    _state.update(tokenizer=load_tokenizer(), roots=roots_by_corpus, thresholds=thresholds)


def _process(job: tuple[str, str, list[int]]):
    """Uma peça -> lista de (transposição, ids, níveis, emoção), mais quantas transposições caíram."""
    split, item, shifts = job
    tokenizer, thresholds = _state["tokenizer"], _state["thresholds"]
    vocab = tokenizer.vocab
    score, quadrant = load_piece(split, item, _state["roots"])
    levels = []
    if thresholds is not None:
        levels = to_levels(measure_piece(decode(tokenizer, encode(tokenizer, score))), thresholds)
    out = []
    for shift, s in _transpositions(score, shifts):
        ids = encode(tokenizer, s)
        if thresholds is not None:
            ids = insert_controls(ids, levels, vocab)
        out.append((shift, [vocab[BOS], emotion_id(vocab, quadrant), *ids, vocab[EOS]], levels, quadrant))
    return item, out, len(shifts) - len(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", nargs="+", default=list(ALL_SPLITS), choices=ALL_SPLITS)
    parser.add_argument("--no-controls", action="store_true", help="não mede nem insere tokens de controle")
    parser.add_argument("--workers", type=int, default=None, help="processos (padrão: CPUs do job)")
    args = parser.parse_args()

    cfg = load_config()
    set_seed(cfg["seed"])
    splits = json.loads(resolve(cfg["paths"]["splits"]).read_text(encoding="utf-8"))
    splits = {k: splits[k] for k in args.splits}
    roots_by_corpus = roots(cfg, splits)
    vocab_size = len(load_tokenizer())

    thresholds = None
    if not args.no_controls:
        thr_path = resolve(cfg["paths"]["control_thresholds"])
        if not thr_path.exists():
            raise SystemExit(f"{thr_path} não existe: rode src.controls.calibrate ou use --no-controls")
        thresholds = json.loads(thr_path.read_text(encoding="utf-8"))

    out_dir = resolve(cfg["paths"]["processed_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))

    with Pool(workers, initializer=_init, initargs=(roots_by_corpus, thresholds)) as pool:
        for split, items in splits.items():
            section = cfg["pretrain"] if is_pretrain(split) else cfg["data"]
            lo, hi = section["transpose_range"]
            shifts = list(range(lo, hi + 1)) if split.endswith("train") else [0]
            data = {"sequences": [], "clip_ids": [], "transpose": [], "levels": [], "emotion": [],
                    "controls": thresholds is not None, "vocab_size": vocab_size}
            skipped = 0
            jobs = [(split, item, shifts) for item in items]
            for item, results, n_skipped in tqdm(pool.imap(_process, jobs, chunksize=4), total=len(jobs),
                                                 desc=split, mininterval=10):
                skipped += n_skipped
                for shift, ids, levels, quadrant in results:
                    data["sequences"].append(torch.tensor(ids, dtype=torch.int16))
                    data["clip_ids"].append(item)
                    data["transpose"].append(shift)
                    data["levels"].append(levels)
                    data["emotion"].append(quadrant)
            torch.save(data, out_dir / f"{split}.pt")
            lens = torch.tensor([len(x) for x in data["sequences"]], dtype=torch.float)
            print(f"{split}: {len(data['sequences'])} sequências de {len(items)} peças | "
                  f"{int(lens.sum())} tokens | comprimento médio {lens.mean():.0f} ± {lens.std():.0f} | "
                  f"transposições fora do teclado: {skipped} | por emoção: {dict(Counter(data['emotion']))}",
                  flush=True)


if __name__ == "__main__":
    main()
