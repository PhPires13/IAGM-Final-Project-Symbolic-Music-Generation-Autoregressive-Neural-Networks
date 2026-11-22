"""Gera amostras condicionadas pelas curvas reais de uma partição e salva MIDIs + manifesto.

Sistemas:
  transformer  checkpoint de `src.model.train` (--checkpoint)
  markov       outputs/markov/markov.npz (ou --checkpoint)
  human        os próprios clipes da partição, passados pelo REMI (tokeniza e decodifica), para a
               referência humana ter exatamente a mesma quantização das amostras geradas

As curvas vêm de `data/processed/<split>.pt` (uma por clipe, sem transposição). A amostra i usa a
curva do clipe i % n_clipes, então --n 100 no teste (88 clipes) repete 12 curvas.

Saída: outputs/samples/<nome>/<split>/{000.mid, ..., manifest.json, ids.pt}, com <nome> = sistema
ou --name. O manifesto tem, por amostra: arquivo, clipe da curva, níveis pedidos, se terminou.

Uso:
  uv run python -m src.generate.run --system transformer --checkpoint outputs/runs/large/best.pt --top-p 0.95
  uv run python -m src.generate.run --system markov
  uv run python -m src.generate.run --system human
  # ajuste de amostragem: sempre na validação
  uv run python -m src.generate.run --system transformer --checkpoint ... --split val --temperature 0.9 --name tf_t09

Responsável: Leonardo.
"""

from __future__ import annotations

import argparse
import json
import time

import torch
from tqdm import tqdm

from src.baseline.markov import MarkovModel
from src.common import load_config, resolve, set_seed
from src.controls.tokens import strip_controls
from src.data.dataset import load_split
from src.data.tokenizer import decode, load_tokenizer
from src.generate.generate import generate
from src.generate.sampling import SamplingConfig
from src.model.transformer import load_checkpoint


def real_curves(split: str, cfg: dict) -> tuple[list[str], list[list[tuple[int, int]]], list[torch.Tensor]]:
    data = load_split(split, cfg)
    if not data["controls"]:
        raise SystemExit(f"data/processed/{split}.pt foi gerado com --no-controls; rode o build_dataset completo")
    keep = [i for i, t in enumerate(data["transpose"]) if t == 0]
    return ([data["clip_ids"][i] for i in keep], [[tuple(lv) for lv in data["levels"][i]] for i in keep],
            [data["sequences"][i] for i in keep])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", choices=["transformer", "markov", "human"], required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--n", type=int, default=None, help="número de amostras (padrão: generation.n_samples)")
    parser.add_argument("--name", help="nome da pasta de saída (padrão: o sistema)")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--max-tokens-per-bar", type=int, default=150, help="limite de segurança do comprimento")
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config()
    seed = cfg["seed"] if args.seed is None else args.seed
    set_seed(seed)
    tokenizer = load_tokenizer()
    vocab = tokenizer.vocab
    clips, curves, sequences = real_curves(args.split, cfg)
    n = args.n or (len(clips) if args.system == "human" else cfg["generation"]["n_samples"])
    order = [i % len(clips) for i in range(n)]

    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "samples" / (args.name or args.system) / args.split
    out_dir.mkdir(parents=True, exist_ok=True)
    sampling = SamplingConfig(args.temperature, args.top_k, args.top_p)
    manifest = {"system": args.system, "split": args.split, "checkpoint": args.checkpoint, "seed": seed,
                "sampling": vars(sampling), "samples": []}
    all_ids: list[list[int]] = []
    t0 = time.time()

    if args.system == "human":
        results = [(sequences[j].tolist(), True) for j in order]
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.system == "transformer":
            if not args.checkpoint:
                raise SystemExit("--checkpoint é obrigatório para o transformer")
            model = load_checkpoint(resolve(args.checkpoint), device)
        else:
            path = resolve(args.checkpoint or resolve(cfg["paths"]["outputs_dir"]) / "markov" / "markov.npz")
            model = MarkovModel.load(path)
            device = torch.device("cpu")
        generator = torch.Generator(device=device).manual_seed(seed)
        results = []
        for start in tqdm(range(0, n, args.batch_size), desc=f"gerando ({args.system})"):
            batch = [curves[j] for j in order[start : start + args.batch_size]]
            max_tokens = 4 + args.max_tokens_per_bar * max(len(c) for c in batch)
            results += [(g.ids, g.finished) for g in
                        generate(model, batch, sampling, vocab, max_tokens=max_tokens, generator=generator)]

    for k, (j, (ids, finished)) in enumerate(zip(order, results)):
        fname = f"{k:03d}.mid"
        decode(tokenizer, strip_controls(ids, vocab)).dump_midi(out_dir / fname)
        all_ids.append(ids)
        manifest["samples"].append({"file": fname, "curve_clip": clips[j], "requested_levels": curves[j],
                                    "finished": finished, "n_tokens": len(ids)})

    manifest["seconds"] = round(time.time() - t0, 1)
    manifest["unfinished"] = sum(not s["finished"] for s in manifest["samples"])
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    torch.save(all_ids, out_dir / "ids.pt")
    print(f"{n} amostras em {out_dir} | {manifest['seconds']}s | não terminadas: {manifest['unfinished']}")


if __name__ == "__main__":
    main()
