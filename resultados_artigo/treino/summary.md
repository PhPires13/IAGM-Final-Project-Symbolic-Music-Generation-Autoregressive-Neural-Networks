| Execução | Validação | Parâmetros (M) | Melhor época / total | val loss | Perplexidade | train loss na melhor | s/época |
|---|---|---|---|---|---|---|---|
| pretrain_large | pretrain_val | 19.61 | 4 / 7 | 1.5373 | 4.65 | 1.4750 | 275 |
| pretrain_small | pretrain_val | 3.37 | 18 / 20 | 1.5740 | 4.83 | 1.5996 | 79 |
| finetune_large **(escolhida)** | val | 19.61 | 1 / 4 | 1.5613 | 4.76 | 1.6724 | 41 |
| scratch_large | val | 19.61 | 5 / 10 | 2.1267 | 8.39 | 1.7961 | 41 |
| finetune_small | val | 3.37 | 2 / 5 | 1.6141 | 5.02 | 1.6199 | 13 |
| scratch_small | val | 3.37 | 5 / 10 | 1.9057 | 6.72 | 1.8519 | 13 |

Ganho do pré-treino (val loss do zero − fine-tune, em nats): large: +0.5655, small: +0.2916
GPU: NVIDIA L40S
