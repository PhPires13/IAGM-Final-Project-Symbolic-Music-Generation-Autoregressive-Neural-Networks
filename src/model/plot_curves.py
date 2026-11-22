"""Figura das curvas de treino e validação (loss por época) a partir de `metrics.csv`.

Uso: uv run python -m src.model.plot_curves large small   # nomes das runs em outputs/runs/
Saída: outputs/figures/training_curves.pdf (e .png)
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
    args = parser.parse_args()

    out_root = resolve(load_config()["paths"]["outputs_dir"])
    fig, ax = plt.subplots(figsize=(3.4, 2.4))  # largura de uma coluna do template ISMIR
    for i, run in enumerate(args.runs):
        df = pd.read_csv(out_root / "runs" / run / "metrics.csv")
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
        fig.savefig(fig_dir / f"training_curves.{ext}", dpi=200)
    print(f"salvo em {fig_dir / 'training_curves.pdf'}")


if __name__ == "__main__":
    main()
