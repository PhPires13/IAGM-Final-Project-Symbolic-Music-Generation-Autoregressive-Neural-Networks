"""Laço de treino do Transformer.

- cross-entropy do próximo token (teacher forcing), ignorando PAD;
- AdamW (weight decay só nas matrizes), warmup linear + decaimento cosseno, gradient clipping;
- bf16 automático em GPU (A100/L40S), fp32 em CPU;
- avaliação na validação a cada época, early stopping por paciência;
- checkpoints `last.pt` (para retomar) e `best.pt` (menor loss de validação);
- log em `metrics.csv` (vira a figura de curvas de treino do artigo).

Partições: `data.train` e `data.val` no YAML da execução (padrão `train`/`val`, o EMOPIA+; o
pré-treino usa `pretrain_train`/`pretrain_val`, o Pop1K7). Com `init_from`, os pesos iniciais vêm
de outro checkpoint (fine-tune), com otimizador e scheduler novos; a arquitetura tem que ser igual.

Saída em `outputs/runs/<name>/`.

Uso:
  python -m src.model.train --config configs/pretrain_large.yaml
  python -m src.model.train --config configs/finetune_large.yaml          # depois do pré-treino
  python -m src.model.train --config configs/scratch_small.yaml --resume  # do zero, retomando
  python -m src.model.train --config configs/pretrain_small.yaml --smoke  # teste rápido do pipeline
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.common import load_config, resolve, set_seed
from src.data.dataset import IGNORE_INDEX, WindowDataset, load_split
from src.data.tokenizer import PAD, load_tokenizer
from src.model.transformer import MusicTransformer, TransformerConfig

SMOKE_OVERRIDES = {
    "model": {"max_len": 128, "d_model": 64, "n_layers": 1, "n_heads": 2, "d_ff": 128},
    "train": {"stride": 128, "batch_size": 8, "warmup_steps": 5, "max_epochs": 2, "num_workers": 0},
}
SMOKE_MAX_SEQS = 32  # sequências por partição no modo --smoke


def lr_lambda(warmup: int, total: int, min_ratio: float):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = min(1.0, (step - warmup) / max(1, total - warmup))
        return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
    return f


def make_optimizer(model: torch.nn.Module, tcfg: dict) -> torch.optim.AdamW:
    params = {n: p for n, p in model.named_parameters() if p.requires_grad}
    decay = [p for p in params.values() if p.dim() >= 2]
    no_decay = [p for p in params.values() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": tcfg["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=tcfg["lr"], betas=tuple(tcfg["betas"]))


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1), ignore_index=IGNORE_INDEX)


@torch.no_grad()
def evaluate(model, loader, device, amp_dtype) -> float:
    """Loss média por token (não por lote) no conjunto inteiro."""
    model.eval()
    total, count = 0.0, 0
    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
            logits = model(x)
        n = int((y != IGNORE_INDEX).sum())
        total += loss_fn(logits, y).item() * n
        count += n
    model.train()
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true", help="continua de outputs/runs/<name>/last.pt")
    parser.add_argument("--smoke", action="store_true", help="modelo e dados minúsculos, 1 época")
    parser.add_argument("--init-from", help="sobrescreve o init_from do YAML (ex.: checkpoint de smoke)")
    args = parser.parse_args()
    args.smoke_skip_init = False

    cfg = load_config()
    run_cfg = load_config(args.config)
    if args.smoke:
        run_cfg["name"] = f"{run_cfg['name']}_smoke"
        for section, values in SMOKE_OVERRIDES.items():
            run_cfg[section].update(values)
        # no smoke a arquitetura muda, então o init_from do YAML só vale se vier de outro smoke
        args.smoke_skip_init = bool(run_cfg.get("init_from")) and not args.init_from
    if args.init_from:
        run_cfg["init_from"] = args.init_from
    mcfg, tcfg = run_cfg["model"], run_cfg["train"]
    set_seed(cfg["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp_dtype = torch.bfloat16 if device.type == "cuda" and torch.cuda.is_bf16_supported() else None
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    tokenizer = load_tokenizer()
    pad_id = tokenizer.vocab[PAD]
    dcfg = run_cfg.get("data", {})
    train_split, val_split = dcfg.get("train", "train"), dcfg.get("val", "val")
    train_data, val_data = load_split(train_split, cfg), load_split(val_split, cfg)
    train_seqs, val_seqs = train_data["sequences"], val_data["sequences"]
    if args.smoke:
        train_seqs, val_seqs = train_seqs[:SMOKE_MAX_SEQS], val_seqs[:SMOKE_MAX_SEQS]
    keep = cfg["data"]["keep_prefix"]
    train_ds = WindowDataset(train_seqs, mcfg["max_len"], tcfg["stride"], pad_id, keep)
    val_ds = WindowDataset(val_seqs, mcfg["max_len"], mcfg["max_len"] - keep + 1, pad_id, keep)
    loader_kw = {"num_workers": tcfg["num_workers"], "pin_memory": device.type == "cuda"}
    train_loader = DataLoader(train_ds, batch_size=tcfg["batch_size"], shuffle=True, drop_last=True,
                              generator=torch.Generator().manual_seed(cfg["seed"]), **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=tcfg["batch_size"], shuffle=False, **loader_kw)

    model_cfg = TransformerConfig(vocab_size=len(tokenizer), pad_id=pad_id, **mcfg)
    model = MusicTransformer(model_cfg).to(device)
    init_from = run_cfg.get("init_from")
    if init_from and not args.smoke_skip_init:
        src = torch.load(resolve(init_from), map_location=device, weights_only=False)
        if src["model_config"] != model.config_dict():
            raise SystemExit(f"arquitetura de {init_from} difere da configuração: {src['model_config']} x {model.config_dict()}")
        model.load_state_dict(src["model"])
        print(f"pesos iniciais de {init_from} (época {src['state']['epoch']}, val {src['state']['best_val']:.4f})")
    optimizer = make_optimizer(model, tcfg)
    total_steps = tcfg["max_epochs"] * len(train_loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda(tcfg["warmup_steps"], total_steps, tcfg["min_lr_ratio"]))

    run_dir = resolve(cfg["paths"]["outputs_dir"]) / "runs" / run_cfg["name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = run_dir / "metrics.csv"
    state = {"epoch": 0, "step": 0, "best_val": float("inf"), "bad_epochs": 0}

    if args.resume and (run_dir / "last.pt").exists():
        ckpt = torch.load(run_dir / "last.pt", map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        state = ckpt["state"]
        print(f"retomando da época {state['epoch']} (passo {state['step']})")
    else:
        with open(metrics_path, "w", newline="") as f:
            csv.writer(f).writerow(["epoch", "step", "train_loss", "val_loss", "val_ppl", "lr", "seconds"])

    (run_dir / "run_config.json").write_text(json.dumps(
        {"run": run_cfg, "model_config": model.config_dict(), "num_params": model.num_params(),
         "train_windows": len(train_ds), "val_windows": len(val_ds), "steps_per_epoch": len(train_loader),
         "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
         "amp_dtype": str(amp_dtype), "controls": train_data["controls"],
         "data": {"train": train_split, "val": val_split}}, indent=2) + "\n")
    print(f"{run_cfg['name']}: {model.num_params() / 1e6:.2f}M parâmetros | {len(train_ds)} janelas de treino | "
          f"{len(train_loader)} passos/época | {device} ({amp_dtype or 'fp32'}) | controles: {train_data['controls']} | "
          f"dados: {train_split}/{val_split}")

    def save(name: str) -> None:
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "state": state,
                    "model_config": model.config_dict(), "run_config": run_cfg}, run_dir / name)

    if state["bad_epochs"] >= tcfg["patience"] or state["epoch"] >= tcfg["max_epochs"]:
        print(f"execução já terminada (época {state['epoch']}, early stopping ou máximo de épocas); nada a fazer")
    model.train()
    while state["epoch"] < tcfg["max_epochs"] and state["bad_epochs"] < tcfg["patience"]:
        t0, running, n_batches = time.time(), 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                logits = model(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            state["step"] += 1
            running += loss.item()
            n_batches += 1
            if state["step"] % 100 == 0:
                print(f"  passo {state['step']}: loss {running / n_batches:.4f} | lr {scheduler.get_last_lr()[0]:.2e}", flush=True)

        state["epoch"] += 1
        train_loss = running / max(n_batches, 1)
        val_loss = evaluate(model, val_loader, device, amp_dtype)
        lr = scheduler.get_last_lr()[0]
        seconds = time.time() - t0
        with open(metrics_path, "a", newline="") as f:
            csv.writer(f).writerow([state["epoch"], state["step"], f"{train_loss:.5f}", f"{val_loss:.5f}",
                                    f"{math.exp(val_loss):.3f}", f"{lr:.3e}", f"{seconds:.1f}"])

        improved = val_loss < state["best_val"]
        if improved:
            state["best_val"], state["bad_epochs"] = val_loss, 0
        else:
            state["bad_epochs"] += 1
        save("last.pt")
        if improved:
            save("best.pt")
        print(f"época {state['epoch']}: train {train_loss:.4f} | val {val_loss:.4f} (ppl {math.exp(val_loss):.2f})"
              f"{' *' if improved else ''} | {seconds:.0f}s", flush=True)
        if state["bad_epochs"] >= tcfg["patience"]:
            print(f"early stopping: {tcfg['patience']} épocas sem melhorar")
            break

    print(f"melhor val loss: {state['best_val']:.4f} | checkpoints em {run_dir}")


if __name__ == "__main__":
    main()
