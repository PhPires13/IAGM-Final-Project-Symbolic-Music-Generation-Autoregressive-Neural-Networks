"""Etapas do pipeline de experimentos que rodam depois do treino (chamadas pelo jobs/pipeline.sh).

  select  lê os metrics.csv de todas as execuções, grava outputs/runs/summary.{json,md} (melhor
          val_loss de cada uma, ganho do pré-treino sobre o do zero) e escolhe o melhor fine-tune.
  plot    figuras das curvas de treino: pré-treino (validação no Pop1K7) e EMOPIA+ (do zero x fine-tune).
  tune    na VALIDAÇÃO: gera com cada combinação de temperatura x top-p usando o checkpoint escolhido,
          avalia contra os clipes humanos e grava outputs/eval/val/best_sampling.json.
  final   no TESTE (único uso do teste): ajusta o Markov, gera com a amostragem escolhida para todas
          as execuções do EMOPIA+ (fine-tune e do zero), o Markov e a referência humana, e avalia.
          Com --gen all, gera também todas as combinações da grade com o melhor checkpoint.

Uso (normalmente via jobs/pipeline.sh, um job por etapa):
  python -m src.pipeline select
  python -m src.pipeline plot
  python -m src.pipeline tune [--temperatures 0.8 0.9 1.0] [--top-ps 0.9 0.95 1.0] [--n 100]
  python -m src.pipeline final [--gen best|all] [--n 100]

Responsável: Leonardo.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys

import pandas as pd

from src.common import load_config, resolve

SIZES = ("large", "small")
EMOPIA_RUNS = [f"{kind}_{size}" for size in SIZES for kind in ("finetune", "scratch")]
PRETRAIN_RUNS = [f"pretrain_{size}" for size in SIZES]
SUFFIX = ""  # "_smoke" no teste local (--suffix): usa as execuções de `train --smoke`
DEFAULT_TEMPERATURES = (0.8, 0.9, 1.0)
DEFAULT_TOP_PS = (0.9, 0.95, 1.0)


def _outputs():
    return resolve(load_config()["paths"]["outputs_dir"])


def _run(*args: str) -> None:
    cmd = [sys.executable, "-m", *args]
    print(f"\n$ {' '.join(cmd[2:])}", flush=True)
    subprocess.run(cmd, check=True)


def _grid_name(run: str, t: float, p: float) -> str:
    return f"{run}_t{t}_p{p}"


def _summary() -> dict:
    path = _outputs() / "runs" / "summary.json"
    if not path.exists():
        raise SystemExit(f"{path} não existe: rode a etapa select antes")
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- select
def select() -> None:
    runs_dir = _outputs() / "runs"
    rows = {}
    for run in [r + SUFFIX for r in PRETRAIN_RUNS + EMOPIA_RUNS]:
        path = runs_dir / run / "metrics.csv"
        if not path.exists():
            print(f"aviso: {run} não tem metrics.csv (treino não rodou ou falhou)")
            continue
        df = pd.read_csv(path)
        if df.empty:
            continue
        best = df.loc[df["val_loss"].idxmin()]
        cfg = json.loads((runs_dir / run / "run_config.json").read_text(encoding="utf-8"))
        rows[run] = {"best_epoch": int(best["epoch"]), "epochs": int(df["epoch"].max()),
                     "val_loss": float(best["val_loss"]), "val_ppl": float(math.exp(best["val_loss"])),
                     "train_loss_at_best": float(best["train_loss"]), "params_M": round(cfg["num_params"] / 1e6, 2),
                     "seconds_per_epoch": float(df["seconds"].mean()), "device": cfg.get("device"),
                     "val_set": cfg.get("data", {}).get("val", "val")}
    finetunes = {r: v for r, v in rows.items() if r.startswith("finetune_")}
    if not finetunes:
        raise SystemExit("nenhum fine-tune terminou; veja os logs de finetune_*")
    best_run = min(finetunes, key=lambda r: finetunes[r]["val_loss"])
    gains = {size: rows[f"scratch_{size}{SUFFIX}"]["val_loss"] - rows[f"finetune_{size}{SUFFIX}"]["val_loss"]
             for size in SIZES if f"scratch_{size}{SUFFIX}" in rows and f"finetune_{size}{SUFFIX}" in rows}
    summary = {"runs": rows, "best_finetune": best_run, "pretrain_gain_nats": gains}
    (runs_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = ["| Execução | Validação | Parâmetros (M) | Melhor época / total | val loss | Perplexidade | train loss na melhor | s/época |",
             "|---|---|---|---|---|---|---|---|"]
    for run, r in rows.items():
        mark = " **(escolhida)**" if run == best_run else ""
        lines.append(f"| {run}{mark} | {r['val_set']} | {r['params_M']} | {r['best_epoch']} / {r['epochs']} | "
                     f"{r['val_loss']:.4f} | {r['val_ppl']:.2f} | {r['train_loss_at_best']:.4f} | {r['seconds_per_epoch']:.0f} |")
    lines += ["", "Ganho do pré-treino (val loss do zero − fine-tune, em nats): "
              + (", ".join(f"{s}: {g:+.4f}" for s, g in gains.items()) or "—"),
              f"GPU: {next(iter(rows.values()))['device']}"]
    (runs_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nmelhor fine-tune: {best_run} -> {runs_dir / 'summary.json'}")


# ---------------------------------------------------------------- plot
def plot() -> None:
    _run("src.model.plot_curves", *[r + SUFFIX for r in PRETRAIN_RUNS], "--out", "pretrain_curves")
    _run("src.model.plot_curves", *[r + SUFFIX for r in EMOPIA_RUNS], "--out", "emopia_curves")


# ---------------------------------------------------------------- tune
def tune(temperatures, top_ps, n: int) -> None:
    run = _summary()["best_finetune"]
    ckpt = f"outputs/runs/{run}/best.pt"
    _run("src.generate.run", "--system", "human", "--split", "val")
    names = []
    for t in temperatures:
        for p in top_ps:
            name = _grid_name(run, t, p)
            names.append(name)
            _run("src.generate.run", "--system", "transformer", "--checkpoint", ckpt, "--split", "val",
                 "--n", str(n), "--temperature", str(t), "--top-p", str(p), "--name", name)
    _run("src.eval.evaluate", "--split", "val", "--systems", *names)
    _run("src.model.sampling_summary", "--split", "val", "--prefix", run)


# ---------------------------------------------------------------- final
def final(gen: str, n: int, temperatures, top_ps) -> None:
    summary = _summary()
    best_path = _outputs() / "eval" / "val" / "best_sampling.json"
    if not best_path.exists():
        raise SystemExit(f"{best_path} não existe: rode a etapa tune antes")
    best = json.loads(best_path.read_text(encoding="utf-8"))
    sampling = ["--temperature", str(best["temperature"])]
    if best.get("top_p") is not None:
        sampling += ["--top-p", str(best["top_p"])]
    if best.get("top_k") is not None:
        sampling += ["--top-k", str(best["top_k"])]
    print(f"amostragem escolhida na validação: {best['system']} -> {' '.join(sampling)}")

    _run("src.baseline.markov")
    _run("src.generate.run", "--system", "human", "--split", "test")
    _run("src.generate.run", "--system", "human", "--split", "val")
    systems = []
    for run in [r + SUFFIX for r in EMOPIA_RUNS]:
        if run in summary["runs"]:
            _run("src.generate.run", "--system", "transformer", "--checkpoint", f"outputs/runs/{run}/best.pt",
                 "--split", "test", "--n", str(n), "--name", run, *sampling)
            systems.append(run)
    _run("src.generate.run", "--system", "markov", "--split", "test", "--n", str(n), *sampling)
    systems.append("markov")
    if gen == "all":
        run = summary["best_finetune"]
        for t in temperatures:
            for p in top_ps:
                name = _grid_name(run, t, p)
                _run("src.generate.run", "--system", "transformer", "--checkpoint", f"outputs/runs/{run}/best.pt",
                     "--split", "test", "--n", str(n), "--temperature", str(t), "--top-p", str(p), "--name", name)
                systems.append(name)
    _run("src.eval.evaluate", "--split", "test", "--systems", *systems, "human:val")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["select", "plot", "tune", "final"])
    parser.add_argument("--temperatures", nargs="+", type=float, default=list(DEFAULT_TEMPERATURES))
    parser.add_argument("--top-ps", nargs="+", type=float, default=list(DEFAULT_TOP_PS))
    parser.add_argument("--n", type=int, default=None, help="amostras por sistema (padrão: generation.n_samples)")
    parser.add_argument("--gen", choices=["best", "all"], default="best")
    parser.add_argument("--suffix", default="", help="sufixo dos nomes das execuções (teste local: _smoke)")
    args = parser.parse_args()
    global SUFFIX
    SUFFIX = args.suffix
    n = args.n or load_config()["generation"]["n_samples"]

    if args.stage == "select":
        select()
    elif args.stage == "plot":
        plot()
    elif args.stage == "tune":
        tune(args.temperatures, args.top_ps, n)
    else:
        final(args.gen, n, args.temperatures, args.top_ps)


if __name__ == "__main__":
    main()
