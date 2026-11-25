"""Resumo do ajuste de amostragem: ordena as configurações avaliadas na validação.

Critério principal: OA média entre as métricas do muspy (maior = distribuições mais próximas das
músicas humanas). Ao lado, a fidelidade ao controle (acurácia exata de energia e de tensão) e o
ganho sobre a curva embaralhada, para não escolher uma configuração que ignore o controle.

A escolhida é a de maior OA média entre as que têm ganho positivo em energia E em tensão (se
nenhuma tiver, a de maior OA, com um aviso). Ela vai para `outputs/eval/<split>/best_sampling.json`,
com a temperatura, o top-k e o top-p lidos do manifesto das amostras, e é usada pelo
`jobs/pipeline.sh` na geração final.

Uso: python -m src.model.sampling_summary --split val [--prefix finetune_large]
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    parser.add_argument("--prefix", default="", help="só considera sistemas cujo nome começa assim")
    args = parser.parse_args()

    outputs = resolve(load_config()["paths"]["outputs_dir"])
    path = outputs / "eval" / args.split / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for system, metrics in results["distribution"].items():
        if system == results["reference"] or not system.startswith(args.prefix):
            continue
        fid = results["fidelity"][system]
        req, shuf = fid["requested"], fid["shuffled"]
        rows.append({
            "system": system,
            "oa": np.mean([m["oa"] for m in metrics.values()]),
            "kl": np.mean([m["kl"] for m in metrics.values()]),
            "energy": req["energy"]["exact"],
            "energy_gain": req["energy"]["exact"] - shuf["energy"]["exact"],
            "tension": req["tension"]["exact"],
            "tension_gain": req["tension"]["exact"] - shuf["tension"]["exact"],
            "unfinished": fid["unfinished"],
        })
    rows.sort(key=lambda r: r["oa"], reverse=True)
    print(f"| Configuração | OA média ↑ | KL médio ↓ | Energia exata (ganho) | Tensão exata (ganho) | Não terminadas |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['system']} | {r['oa']:.3f} | {r['kl']:.3f} | {r['energy']:.3f} ({r['energy_gain']:+.3f}) | "
              f"{r['tension']:.3f} ({r['tension_gain']:+.3f}) | {r['unfinished']} |")
    if not rows:
        raise SystemExit("nenhum sistema avaliado com esse prefixo")

    controlled = [r for r in rows if r["energy_gain"] > 0 and r["tension_gain"] > 0]
    best = (controlled or rows)[0]
    if not controlled:
        print("\nAVISO: nenhuma configuração tem ganho positivo nas duas curvas; escolhida só pela OA")
    manifest = json.loads((outputs / "samples" / best["system"] / args.split / "manifest.json").read_text(encoding="utf-8"))
    choice = {**{k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in best.items()},
              **manifest["sampling"], "checkpoint": manifest.get("checkpoint"), "follows_controls": bool(controlled)}
    out = outputs / "eval" / args.split / "best_sampling.json"
    out.write_text(json.dumps(choice, indent=2) + "\n", encoding="utf-8")
    print(f"\nescolhida: {best['system']} (temperatura {choice['temperature']}, top-p {choice['top_p']}, "
          f"top-k {choice['top_k']}) -> {out}")


if __name__ == "__main__":
    main()
