"""Figuras compactas do artigo, a partir de resultados_artigo/ (rodada do cluster de 25/11).

Uso (na raiz do repositório): python latex/figuras/make_figures.py
Requer matplotlib, pandas e pillow.
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "resultados_artigo"
OUT = Path(__file__).resolve().parent
plt.rcParams.update({"font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7,
                     "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
                     "font.family": "serif"})

NAMES = {"human": "Humano", "finetune_large": "Pré-treino+FT", "scratch_large": "Do zero",
         "markov": "Markov"}
COLORS = {"human": "#555555", "finetune_large": "#1f77b4", "scratch_large": "#ff7f0e", "markov": "#2ca02c"}


def training_curves() -> None:
    fig, ax = plt.subplots(figsize=(3.3, 1.55))
    for run, color in [("finetune_large", COLORS["finetune_large"]), ("scratch_large", COLORS["scratch_large"])]:
        m = pd.read_csv(RES / "treino" / run / "metrics.csv")
        ax.plot(m.epoch, m.train_loss, "--", color=color, lw=1)
        ax.plot(m.epoch, m.val_loss, "-", color=color, lw=1.2, label=f"{NAMES[run]}")
        best = m.loc[m.val_loss.idxmin()]
        ax.plot(best.epoch, best.val_loss, "o", color=color, ms=3)
    ax.plot([], [], "k--", lw=1, label="treino")
    ax.plot([], [], "k-", lw=1, label="validação")
    ax.set_xlabel("época (EMOPIA+)")
    ax.set_ylabel("entropia cruzada")
    ax.grid(alpha=0.3)
    ax.legend(ncol=2, loc="upper right", frameon=False)
    fig.tight_layout(pad=0.2)
    fig.savefig(OUT / "treino.pdf")


def violins() -> None:
    df = pd.read_csv(RES / "teste" / "per_piece.csv")
    metrics = [("n_pitch_classes_used", "classes de altura"), ("pitch_class_entropy", "entropia de classes"),
               ("scale_consistency", "consistência de escala"), ("groove_consistency", "consistência rítmica")]
    systems = list(NAMES)
    fig, axes = plt.subplots(2, 2, figsize=(3.3, 2.3))
    for ax, (col, title) in zip(axes.flat, metrics):
        data = [df[df.system == s][col].dropna().values for s in systems]
        parts = ax.violinplot(data, showmedians=True, showextrema=False, widths=0.8)
        for body, s in zip(parts["bodies"], systems):
            body.set_facecolor(COLORS[s])
            body.set_alpha(0.6)
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(0.8)
        ax.set_title(title)
        ax.set_xticks(range(1, len(systems) + 1))
        ax.set_xticklabels(["Hum.", "PT+AF", "Zero", "Markov"] if ax in axes[1] else [], fontsize=5.5)
        ax.grid(axis="y", alpha=0.3)
    fig.tight_layout(pad=0.2, w_pad=0.6, h_pad=0.4)
    fig.savefig(OUT / "violinos.pdf")


def control_curves() -> None:
    """Recorta, da figura gerada pelo pipeline, as linhas do Transformer com pré-treino e do Markov."""
    img = Image.open(RES / "teste" / "curves.png")
    w, _ = img.size
    # faixas de pixels das linhas finetune_large e markov (cortes nos espaços em branco entre os painéis)
    rows = [img.crop((0, top, w, bottom)) for top, bottom in [(0, 303), (1172, 1462)]]
    out = Image.new("RGB", (w, sum(r.size[1] for r in rows)), "white")
    y = 0
    for r in rows:
        out.paste(r, (0, y))
        y += r.size[1]
    out.save(OUT / "curvas_controle.png")


if __name__ == "__main__":
    training_curves()
    violins()
    control_curves()
