"""Figura das curvas de treino e validação (loss por época) a partir de `metrics.csv`.

Uso:
  uv run python -m src.model.plot_curves pretrain_large pretrain_small --out pretrain_curves
  uv run python -m src.model.plot_curves scratch_large finetune_large scratch_small finetune_small --out emopia_curves
Saída: outputs/figures/<out>.pdf (e .png). Execuções sem metrics.csv são puladas com um aviso.
Junte na mesma figura só execuções avaliadas na mesma validação (Pop1K7 ou EMOPIA+).
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

from src.common import load_config, resolve  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", default="training_curves", help="nome do arquivo em outputs/figures/")
    args = parser.parse_args()

    out_root = resolve(load_config()["paths"]["outputs_dir"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))  # largura de uma coluna do template ISMIR
    for i, run in enumerate(args.runs):
        path = out_root / "runs" / run / "metrics.csv"
        if not path.exists():
            print(f"aviso: {path} não existe; pulando {run}")
            continue
        df = pd.read_csv(path)
        color = f"C{i}"
        ax.plot(df["epoch"], df["train_loss"], color=color, linestyle="--", label=f"{run} (treino)")
        ax.plot(df["epoch"], df["val_loss"], color=color, label=f"{run} (validação)")
        best = df["val_loss"].idxmin()
        ax.scatter(df["epoch"][best], df["val_loss"][best], color=color, zorder=3, s=12)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("época")
    ax.set_ylabel("cross-entropy")
    ax.legend(fontsize=6, frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    fig_dir = out_root / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(fig_dir / f"{args.out}.{ext}", dpi=200)
    print(f"salvo em {fig_dir / args.out}.pdf")


if __name__ == "__main__":
    main()
