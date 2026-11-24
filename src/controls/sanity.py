"""Verificações de sanidade dos atributos de controle (sobre o treino).

- balanceamento dos níveis de energia e de tensão;
- correlação de Spearman entre energia e tensão (se for muito alta, os dois controles são redundantes);
- diferença por quadrante de emoção do EMOPIA+ (esperado: energia maior em Q1/Q2, de alta excitação);
- balanceamento dos níveis no Pop1K7 com os limites do EMOPIA+ (se houver `pretrain_values.json`);
- gráfico das curvas de alguns clipes.

Requer `src.controls.calibrate` (com `--pretrain` para a parte do Pop1K7). Saída em `outputs/controls/`.

Responsável: Julia M.

Uso: uv run python -m src.controls.sanity
"""

from __future__ import annotations

import json
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

from src.common import load_config, resolve  # noqa: E402
from src.controls.measure import to_levels  # noqa: E402
from src.data.emopia import quadrant  # noqa: E402


def main() -> None:
    cfg = load_config()
    out_dir = resolve(cfg["paths"]["outputs_dir"]) / "controls"
    values = json.loads((out_dir / "train_values.json").read_text(encoding="utf-8"))
    thresholds = json.loads(resolve(cfg["paths"]["control_thresholds"]).read_text(encoding="utf-8"))

    levels = {c: to_levels(v, thresholds) for c, v in values.items()}
    e_lv = Counter(e for lv in levels.values() for e, _ in lv)
    t_lv = Counter(t for lv in levels.values() for _, t in lv)
    n = sum(e_lv.values())
    report = {
        "bars": n,
        "energy_level_share": {k: round(e_lv[k] / n, 3) for k in sorted(e_lv)},
        "tension_level_share": {k: round(t_lv[k] / n, 3) for k in sorted(t_lv)},
    }
    energy = np.array([e for v in values.values() for e, _ in v])
    tension = np.array([t for v in values.values() for _, t in v])
    report["spearman_energy_tension"] = round(float(spearmanr(energy, tension).statistic), 3)
    report["mean_by_quadrant"] = {
        f"Q{q}": {
            "energy": round(float(np.mean([e for c, v in values.items() if quadrant(c) == q for e, _ in v])), 3),
            "tension": round(float(np.mean([t for c, v in values.items() if quadrant(c) == q for _, t in v])), 3),
        }
        for q in (1, 2, 3, 4)
    }
    pre_path = out_dir / "pretrain_values.json"
    if pre_path.exists():
        pre = json.loads(pre_path.read_text(encoding="utf-8"))
        pre_lv = [lv for v in pre.values() for lv in to_levels(v, thresholds)]
        pe, pt = Counter(e for e, _ in pre_lv), Counter(t for _, t in pre_lv)
        report["pop1k7"] = {
            "bars": len(pre_lv),
            "energy_level_share": {k: round(pe[k] / len(pre_lv), 3) for k in sorted(pe)},
            "tension_level_share": {k: round(pt[k] / len(pre_lv), 3) for k in sorted(pt)},
        }
    (out_dir / "sanity.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))

    clips = sorted(values)[:: max(1, len(values) // 4)][:4]
    fig, axes = plt.subplots(len(clips), 1, figsize=(6, 1.6 * len(clips)), sharex=True)
    for ax, clip in zip(np.atleast_1d(axes), clips):
        e, t = zip(*levels[clip])
        ax.step(range(len(e)), e, where="post", label="energia")
        ax.step(range(len(t)), np.array(t) + 0.05, where="post", label="tensão")
        ax.set_yticks(range(cfg["controls"]["n_levels"]))
        ax.set_title(clip, fontsize=8)
    np.atleast_1d(axes)[0].legend(fontsize=7, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("compasso")
    fig.tight_layout()
    fig.savefig(out_dir / "curves_examples.png", dpi=150)
    print(f"figura em {out_dir / 'curves_examples.png'}")


if __name__ == "__main__":
    main()
