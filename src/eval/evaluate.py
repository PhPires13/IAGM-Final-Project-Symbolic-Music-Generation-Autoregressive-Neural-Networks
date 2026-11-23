"""Avaliação final: métricas objetivas (distribuições) e fidelidade ao controle.

Lê as pastas geradas por `src.generate.run` (outputs/samples/<sistema>/<split>/) e compara cada
sistema com a referência humana. Um sistema pode ser dado como `nome:split`, por exemplo
`human:val`, que serve de verificação de sanidade: validação contra teste deve dar OA alta.

Saída em outputs/eval/<split>/:
  per_piece.csv      métricas por música e sistema
  results.json       média ± desvio, OA e KL por métrica; fidelidade pedida x embaralhada
  tables.md          tabelas prontas para o artigo
  metrics_violin.pdf distribuições das métricas
  curves.pdf         curva pedida x obtida de algumas amostras

Uso:
  uv run python -m src.eval.evaluate --systems transformer markov human:val
  uv run python -m src.eval.evaluate --split val --systems tf_t09 tf_t10 markov   # ajuste de amostragem

Responsável: Júlia B.
"""

from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402
from tqdm import tqdm  # noqa: E402

from src.common import load_config, resolve  # noqa: E402
from src.eval.fidelity import derangement, measured_levels, scores  # noqa: E402
from src.eval.metrics import METRICS, compare, piece_metrics  # noqa: E402


def load_system(spec: str, default_split: str, cfg: dict) -> tuple[str, dict, list]:
    name, _, split = spec.partition(":")
    split = split or default_split
    folder = resolve(cfg["paths"]["outputs_dir"]) / "samples" / name / split
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    return f"{name}:{split}" if split != default_split else name, manifest, [folder / s["file"] for s in manifest["samples"]]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--reference", default="human")
    parser.add_argument("--systems", nargs="+", required=True)
    args = parser.parse_args()

    cfg = load_config()
    thresholds = json.loads(resolve(cfg["paths"]["control_thresholds"]).read_text(encoding="utf-8"))
    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "eval" / args.split
    out_dir.mkdir(parents=True, exist_ok=True)

    rows, fidelity, curves_for_plot = [], {}, {}
    for spec in [args.reference, *args.systems]:
        label, manifest, files = load_system(spec, args.split, cfg)
        requested = [[tuple(lv) for lv in s["requested_levels"]] for s in manifest["samples"]]
        obtained = []
        for path, req in tqdm(list(zip(files, requested)), desc=label):
            rows.append({"system": label, "file": path.name, **piece_metrics(path)})
            obtained.append(measured_levels(path, len(req), thresholds))
        perm = derangement(len(requested), cfg["generation"]["shuffle_seed"])
        fidelity[label] = {"requested": scores(requested, obtained),
                           "shuffled": scores([requested[p] for p in perm], obtained),
                           "unfinished": manifest.get("unfinished", 0), "n": len(files)}
        curves_for_plot[label] = (requested[0], obtained[0])

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "per_piece.csv", index=False)
    ref_label = load_system(args.reference, args.split, cfg)[0]
    ref = df[df.system == ref_label]
    labels = list(dict.fromkeys(df.system))

    distribution = {label: {m: compare(ref[m].to_numpy(), df[df.system == label][m].to_numpy()) for m in METRICS}
                    for label in labels if label != ref_label}
    distribution[ref_label] = {m: {"mean": float(ref[m].mean()), "std": float(ref[m].std(ddof=0))} for m in METRICS}
    nan_counts = {label: int(df[df.system == label][METRICS].isna().any(axis=1).sum()) for label in labels}
    (out_dir / "results.json").write_text(json.dumps(
        {"reference": ref_label, "distribution": distribution, "fidelity": fidelity, "pieces_with_nan": nan_counts},
        indent=2) + "\n", encoding="utf-8")

    # ---------- tabelas ----------
    others = [label for label in labels if label != ref_label]
    lines = [f"## Métricas objetivas ({args.split}; referência: {ref_label})", "",
             "| Métrica | " + " | ".join([f"{ref_label} (média ± dp)"] + [f"{s} (média ± dp) | OA ↑ | KL ↓" for s in others]) + " |",
             "|---|" + "---|" * (1 + 3 * len(others))]
    for m in METRICS:
        r = distribution[ref_label][m]
        cells = [f"{r['mean']:.3f} ± {r['std']:.3f}"]
        for s in others:
            d = distribution[s][m]
            cells += [f"{d['mean']:.3f} ± {d['std']:.3f}", f"{d['oa']:.3f}", f"{d['kl']:.3f}"]
        lines.append(f"| {m} | " + " | ".join(cells) + " |")
    lines += ["", "## Fidelidade ao controle (compasso a compasso)", "",
              "| Sistema | Curva | Energia exata | Energia ±1 | Energia ρ | Tensão exata | Tensão ±1 | Tensão ρ |",
              "|---|---|---|---|---|---|---|---|"]
    for label in labels:
        for kind, name in (("requested", "pedida"), ("shuffled", "embaralhada")):
            e, t = fidelity[label][kind]["energy"], fidelity[label][kind]["tension"]
            lines.append(f"| {label} | {name} | {e['exact']:.3f} | {e['within_1']:.3f} | {e['spearman']:.3f} | "
                         f"{t['exact']:.3f} | {t['within_1']:.3f} | {t['spearman']:.3f} |")
    lines += ["", "Amostras com métrica indefinida (NaN): " + ", ".join(f"{k}: {v}" for k, v in nan_counts.items()),
              "Amostras que não terminaram a curva: " + ", ".join(f"{k}: {fidelity[k]['unfinished']}" for k in labels)]
    (out_dir / "tables.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---------- figuras ----------
    fig, axes = plt.subplots(2, (len(METRICS) + 1) // 2, figsize=(7, 3.6))
    for ax, m in zip(axes.ravel(), METRICS):
        data = [df[df.system == label][m].dropna().to_numpy() for label in labels]
        ax.violinplot(data, showmedians=True)
        ax.set_xticks(range(1, len(labels) + 1), labels, rotation=45, fontsize=6)
        ax.set_title(m, fontsize=7)
        ax.tick_params(axis="y", labelsize=6)
    for ax in axes.ravel()[len(METRICS):]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out_dir / "metrics_violin.pdf")
    fig.savefig(out_dir / "metrics_violin.png", dpi=200)

    fig, axes = plt.subplots(len(others), 2, figsize=(7, 1.5 * max(len(others), 1)), squeeze=False)
    for row, label in enumerate(others):
        req, got = curves_for_plot[label]
        for col, (dim, name) in enumerate(((0, "energia"), (1, "tensão"))):
            ax = axes[row, col]
            ax.step(range(len(req)), [lv[dim] for lv in req], where="post", label="pedida")
            ax.step(range(len(got)), [lv[dim] + 0.05 for lv in got], where="post", label="obtida")
            ax.set_yticks(range(cfg["controls"]["n_levels"]))
            ax.xaxis.set_major_locator(MaxNLocator(integer=True))
            ax.set_title(f"{label}: {name}", fontsize=7)
            ax.tick_params(labelsize=6)
    for ax in axes[-1]:
        ax.set_xlabel("compasso", fontsize=7)
    axes[0, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out_dir / "curves.pdf")
    fig.savefig(out_dir / "curves.png", dpi=200)
    print(f"\nresultados em {out_dir}")


if __name__ == "__main__":
    main()
