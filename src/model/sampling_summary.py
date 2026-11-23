"""Resumo do ajuste de amostragem: ordena as configurações avaliadas na validação.

Critério principal: OA média entre as métricas do muspy (maior = distribuições mais próximas das
músicas humanas). Ao lado, a fidelidade ao controle (acurácia exata de energia e de tensão) e o
ganho sobre a curva embaralhada, para não escolher uma configuração que ignore o controle.

Uso: uv run python -m src.model.sampling_summary --split val
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src.common import load_config, resolve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    path = resolve(load_config()["paths"]["outputs_dir"]) / "eval" / args.split / "results.json"
    results = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for system, metrics in results["distribution"].items():
        if system == results["reference"]:
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


if __name__ == "__main__":
    main()
