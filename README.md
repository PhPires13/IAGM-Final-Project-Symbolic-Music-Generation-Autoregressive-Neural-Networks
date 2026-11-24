# Geração de piano com curvas de energia e tensão

Trabalho final de DCC831 – IA Generativa para Música (UFMG, 2026/2).

Geração de música para piano condicionada por **emoção** (quadrante de Russell, um token por música) e,
compasso a compasso, por níveis de **energia** e **tensão**. Um Transformer autoregressivo escrito em
PyTorch é pré-treinado no [Pop1K7](https://zenodo.org/records/13167761) e ajustado no
[EMOPIA+](https://zenodo.org/records/13122742), com representação REMI, e comparado com uma cadeia de
Markov de ordem n treinada nos mesmos dados.

Por que EMOPIA+ e pré-treino (e não o EMOPIA 1.0 do plano original): [`docs/pivo_pretreino.md`](docs/pivo_pretreino.md).

## Instalação (local)

Requer [uv](https://docs.astral.sh/uv/). O projeto usa Python 3.10, a mesma versão do cluster.

```bash
uv sync
./scripts/download_data.sh     # EMOPIA+.zip e Pop1K7.zip (Zenodo, CC-BY 4.0) na raiz
```

Depois de adicionar uma dependência (`uv add <pacote>`), regenere o `requirements.txt`:

```bash
./scripts/export_requirements.sh
```

## Execução no cluster (SLURM)

Todos os experimentos, do dado cru às tabelas do artigo, com um comando:

```bash
./jobs/pipeline.sh --dry-run   # mostra os jobs e as dependências
./jobs/pipeline.sh             # dados -> pré-treino -> fine-tune (+ ablação do zero) -> seleção -> figuras -> ajuste -> teste
```

Guia do pipeline: [`docs/pipeline.md`](docs/pipeline.md). Ambiente, etapas avulsas e problemas
comuns: [`docs/cluster.md`](docs/cluster.md).

## Reprodução, etapa por etapa

```bash
export PYTHONPATH=.
# dados (split, tokenizador, limites e relatório de vazamento já estão versionados em configs/)
uv run python -m src.data.split                # divisão oficial do EMOPIA+ + Pop1K7 sem vazamento -> configs/splits.json
uv run python -m src.data.tokenizer --force    # REMI com grade de 1/12 de beat -> configs/tokenizer.json
uv run python -m src.controls.calibrate --pretrain   # tercis no treino do EMOPIA+ -> configs/control_thresholds.json
uv run python -m src.controls.sanity           # balanceamento, correlação, energia por quadrante -> outputs/controls/
uv run python -m src.data.build_dataset        # tokens + emoção + controles + transposição -> data/processed/*.pt
uv run python -m src.data.stats                # estatísticas -> outputs/stats/dataset_stats.json

# treino (no cluster): pré-treino, fine-tune e ablação do zero
uv run python -m src.model.train --config configs/pretrain_large.yaml
uv run python -m src.model.train --config configs/finetune_large.yaml
uv run python -m src.model.train --config configs/scratch_large.yaml
# teste rápido local: --smoke (e --init-from outputs/runs/pretrain_small_smoke/best.pt no fine-tune)

# seleção, figuras, ajuste de amostragem (validação) e avaliação final (teste)
uv run python -m src.pipeline select
uv run python -m src.pipeline plot
uv run python -m src.pipeline tune
uv run python -m src.pipeline final            # Markov + geração + avaliação -> outputs/eval/test/

# áudio: 5 amostras sorteadas por sistema -> samples/
uv run python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 finetune_large markov human
```

## Estrutura

```
configs/        default.yaml, YAMLs das execuções (pretrain/finetune/scratch), splits, tokenizador, limites, vazamento
jobs/           run.slurm (job genérico), pipeline.sh (tudo encadeado) e scripts das etapas avulsas
scripts/        download dos dados, export do requirements.txt
docs/           pivô do projeto, guia do pipeline, guia do cluster
src/data/       EMOPIA+, Pop1K7, split, vazamento, tokenização REMI, janelas, estatísticas
src/controls/   medição de energia e tensão, quantização, tokens de controle
src/model/      Transformer, laço de treino
src/baseline/   cadeia de Markov
src/generate/   amostragem, geração condicionada, renderização de áudio
src/eval/       métricas, OA/KL, fidelidade ao controle, emoção por quadrante
src/pipeline.py etapas pós-treino do pipeline (seleção, figuras, ajuste, avaliação final)
samples/        MIDIs e áudios selecionados de cada sistema
```

## Contratos entre as etapas

| # | Contrato | Onde | Responsável |
|---|---|---|---|
| 1 | Estrutura do repositório e seed global (`seed` em `configs/default.yaml`, `src.common.set_seed`) | `src/common.py` | Leonardo |
| 2 | Tokenizador REMI congelado, com os tokens de controle reservados como tokens especiais | `configs/tokenizer.json` | Pedro |
| 3 | `BOS Emotion_Q1..Q4/None` no início; tokens `Energy_0..K-1` e `Tension_0..K-1` (K = 3); ordem `Bar`, `Energy`, `Tension`, notas; limites = tercis do treino do EMOPIA+ | `src/controls/tokens.py`, `configs/control_thresholds.json` | Julia M. |
| 4 | Função única de medição `measure_bar(notas, tonalidade) -> (energia, tensão)`, usada no rótulo do treino e na avaliação | `src/controls/measure.py` | Julia M. |
| 5 | Split oficial do EMOPIA+ (por música) e Pop1K7 sem as músicas que vazam para val/teste, em `{"train", "val", "test", "pretrain_train", "pretrain_val"}` | `configs/splits.json` | Pedro |
| 6 | Interface comum `next_token_logits(prefixo (B, T)) -> logits (B, V)` | `src/model/interface.py` | Natanael e Leonardo |
| 7 | Curvas de avaliação: curvas e quadrantes reais do teste e curvas embaralhadas entre músicas (seed fixa), 100 ou mais amostras por sistema | `src/eval/` | Júlia B. |

O conjunto de teste só é usado na avaliação final.

## Contribuições

| Integrante | Contribuição |
|---|---|
| Pedro | _a preencher_ |
| Julia M. | _a preencher_ |
| Natanael | _a preencher_ |
| Leonardo | _a preencher_ |
| Júlia B. | _a preencher_ |
