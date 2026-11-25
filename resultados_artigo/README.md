# Resultados da rodada final (cluster, 25/11)

Cópia do `outputs/` do cluster usada no artigo, sem os checkpoints. Todos os números do artigo saem
daqui; a análise comentada está em [`../docs/notas_resultados.md`](../docs/notas_resultados.md).

| Pasta | Conteúdo | Origem no `outputs/` do cluster |
|---|---|---|
| `treino/` | `summary.md` (tabela das 6 execuções) e, por execução, a configuração e a perda por época | `runs/` |
| `validacao/` | grade de temperatura × top-p e a amostragem escolhida (`best_sampling.json`) | `eval/val/` |
| `teste/` | tabelas finais (`tables.md`), métricas por peça, violinos e curvas pedida × obtida | `eval/test/` |
| `figuras/` | curvas de pré-treino e de treino no EMOPIA+, exemplo das curvas de controle | `figures/` |
| `dados/` | sanidade dos níveis de controle e escolha da ordem do Markov | `controls/`, `markov/` |
| `musicas/` | as 100 peças geradas por sistema no teste e a grade da validação, com `manifest.json` (ver [`musicas/LEIA-ME.md`](musicas/LEIA-ME.md)) | `samples/` |

Os limites dos níveis e o relatório de vazamento estão em [`../configs/`](../configs/).
