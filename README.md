# Geração de piano com curvas de energia e tensão

Trabalho final de DCC831 – IA Generativa para Música (UFMG, 2026/2).

Geração de música para piano condicionada, compasso a compasso, por níveis de **energia** e **tensão**,
usando o dataset [EMOPIA](https://annahung31.github.io/EMOPIA/), representação REMI e um Transformer
autoregressivo escrito em PyTorch, comparado com uma cadeia de Markov de ordem n.

> Em construção.

## Instalação (local)

Requer [uv](https://docs.astral.sh/uv/). O projeto usa Python 3.10, a mesma versão do cluster.

```bash
uv sync
```

Coloque o `EMOPIA_1.0.zip` na raiz do repositório.

Depois de adicionar uma dependência (`uv add <pacote>`), regenere o `requirements.txt`:

```bash
./scripts/export_requirements.sh
```

## Execução no cluster (SLURM)

Passo a passo completo, do treino às tabelas finais: [`docs/cluster.md`](docs/cluster.md).

O `jobs/run.slurm` carrega o módulo `pytorch/2.5.1-py3.10-cuda12.8`, monta um venv com o
`requirements.txt` e roda o comando passado. Para o fluxo completo (dados -> treino dos dois
modelos, encadeados por dependência do SLURM), com o `EMOPIA_1.0.zip` na raiz:

```bash
./jobs/train_all.sh              # ou --no-data para reaproveitar data/processed; --resume para continuar
./jobs/generate.sh               # depois do treino: músicas do large, small e markov (um job por modelo)
```

Jobs avulsos:

```bash
mkdir -p logs
sbatch -J large jobs/run.slurm python -m src.model.train --config configs/model_large.yaml
sbatch -J small jobs/run.slurm python -m src.model.train --config configs/model_small.yaml
```

## Reprodução

```bash
# dados (o split e o tokenizador já estão versionados em configs/; os dois primeiros só regeneram)
uv run python -m src.data.split           # split oficial do EMOPIA por música -> configs/splits.json
uv run python -m src.data.tokenizer       # REMI congelado -> configs/tokenizer.json
uv run python -m src.controls.calibrate   # tercis de energia/tensão no treino -> configs/control_thresholds.json
uv run python -m src.controls.sanity      # balanceamento, correlação, curvas -> outputs/controls/
uv run python -m src.data.build_dataset   # tokens + controles + transposição -> data/processed/*.pt
uv run python -m src.data.stats           # estatísticas -> outputs/stats/dataset_stats.json
```

```bash
# treino do Transformer (no cluster; data/processed/ precisa ter sido gerado lá com o build_dataset)
mkdir -p logs
sbatch -J large jobs/run.slurm python -m src.model.train --config configs/model_large.yaml
sbatch -J small jobs/run.slurm python -m src.model.train --config configs/model_small.yaml
# se o job cair, o mesmo comando com --resume continua do último checkpoint
uv run python -m src.model.plot_curves large small   # -> outputs/figures/training_curves.pdf
```

Para testar o pipeline localmente em segundos: `uv run python -m src.model.train --config configs/model_small.yaml --smoke`.

```bash
# baseline Markov (n e desconto escolhidos pela verossimilhança na validação)
uv run python -m src.baseline.markov                # -> outputs/markov/markov.npz

# geração condicionada pelas curvas reais do teste (100 amostras por sistema)
uv run python -m src.generate.run --system transformer --checkpoint outputs/runs/large/best.pt --top-p 0.95
uv run python -m src.generate.run --system markov
uv run python -m src.generate.run --system human    # referência: clipes de teste com a mesma quantização
# o ajuste de temperatura/top-k/top-p é feito com --split val, nunca no teste

# áudio: 5 amostras sorteadas por sistema -> samples/
uv run python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 transformer markov human
```

```bash
# avaliação no teste: métricas do muspy (OA/KL de Yang & Lerch), fidelidade ao controle e
# experimento embaralhado; human:val é a verificação de sanidade (validação x teste)
uv run python -m src.generate.run --system human --split val
uv run python -m src.eval.evaluate --systems transformer markov human:val   # -> outputs/eval/test/
```

## Estrutura

```
configs/        default.yaml (config compartilhada), tokenizer.json, control_thresholds.json
jobs/           run.slurm (job genérico) e scripts que submetem as etapas (treino, geração, ajuste, avaliação)
scripts/        utilitários (export do requirements.txt)
src/data/       split, tokenização REMI, janelas, estatísticas
src/controls/   medição de energia e tensão, quantização, tokens de controle
src/model/      Transformer, laço de treino
src/baseline/   cadeia de Markov
src/generate/   amostragem, geração condicionada, renderização de áudio
src/eval/       métricas, OA/KL, fidelidade ao controle
samples/        MIDIs e áudios selecionados de cada sistema
```

## Contratos entre as etapas

| # | Contrato | Onde | Responsável |
|---|---|---|---|
| 1 | Estrutura do repositório e seed global (`seed` em `configs/default.yaml`, `src.common.set_seed`) | `src/common.py` | Leonardo |
| 2 | Tokenizador REMI congelado, com os tokens de controle reservados como tokens especiais | `configs/tokenizer.json` | Pedro |
| 3 | Tokens `Energy_0..K-1` e `Tension_0..K-1` (K = 3); ordem `Bar`, `Energy`, `Tension`, notas; limites = tercis do treino | `src/controls/tokens.py`, `configs/control_thresholds.json` | Julia M. |
| 4 | Função única de medição `measure_bar(notas, tonalidade) -> (energia, tensão)`, usada no rótulo do treino e na avaliação | `src/controls/measure.py` | Julia M. |
| 5 | Split oficial do EMOPIA (por música), salvo como `{"train": [...], "val": [...], "test": [...]}` com ids de clipe | `configs/splits.json` | Pedro |
| 6 | Interface comum `next_token_logits(prefixo (B, T)) -> logits (B, V)` | `src/model/interface.py` | Natanael e Leonardo |
| 7 | Curvas de avaliação: curvas reais do teste e versões embaralhadas entre músicas (seed fixa), 100 ou mais amostras por sistema | `src/eval/` | Júlia B. |

O conjunto de teste só é usado na avaliação final.

## Contribuições

| Integrante | Contribuição |
|---|---|
| Pedro | _a preencher_ |
| Julia M. | _a preencher_ |
| Natanael | _a preencher_ |
| Leonardo | _a preencher_ |
| Júlia B. | _a preencher_ |
