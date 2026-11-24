# Guia: pipeline completo dos experimentos (`jobs/pipeline.sh`)

Um único comando submete todos os experimentos do artigo no cluster, do dado cru às tabelas finais.
Cada etapa é um job do SLURM, com o próprio log, e só começa quando a anterior termina. O desenho e
a motivação de cada etapa estão em [`pivo_pretreino.md`](pivo_pretreino.md); os detalhes do
ambiente no cluster, em [`cluster.md`](cluster.md).

## 0. Pré-requisitos (uma vez)

Na raiz do repositório, no nó de login do cluster:

```bash
git pull
./scripts/download_data.sh     # baixa EMOPIA+.zip (15 MB) e Pop1K7.zip (300 MB) do Zenodo
```

Se o nó de login não tiver acesso à internet, rode o mesmo script na sua máquina e copie os zips:

```bash
scp EMOPIA+.zip Pop1K7.zip <usuario>@<login-do-cluster>:~/<pasta-do-repositorio>/
```

O split, o tokenizador, os limites dos níveis e o relatório de vazamento já estão versionados em
`configs/`; o pipeline só reconstrói os tensores em `data/processed/`.

## 1. Rodar

```bash
./jobs/pipeline.sh --dry-run    # confere os jobs e as dependências, sem submeter nada
./jobs/pipeline.sh              # submete tudo
```

O script imprime o id de cada job. A cadeia é esta:

```
data ─┬─> pretrain_large ──> finetune_large ─┐
      ├─> pretrain_small ──> finetune_small ─┤
      ├─> scratch_large ─────────────────────┼─> pipe-select ──> pipe-tune ──> pipe-final
      └─> scratch_small ─────────────────────┘   pipe-plot
```

| Etapa | Job | O que faz | Duração esperada (L40S/A100) |
|---|---|---|---|
| dados | `data` | extrai os zips, tokeniza, mede energia/tensão, transpõe o treino | ~5 min |
| pré-treino | `pretrain_large`, `pretrain_small` | Pop1K7 (57M tokens com transposição), `Emotion_None` + curvas | ~5 min/época (large, até 12 épocas); ~1,5 min/época (small) |
| fine-tune | `finetune_large`, `finetune_small` | EMOPIA+ a partir do pré-treino, lr 1e-4, emoção + curvas | ~1 min/época (large) |
| ablação | `scratch_large`, `scratch_small` | EMOPIA+ do zero, sem pré-treino | ~1 min/época (large) |
| seleção | `pipe-select` | tabela de val loss de todas as execuções, escolhe o melhor fine-tune | < 1 min |
| figuras | `pipe-plot` | curvas de treino (pré-treino; EMOPIA+ do zero x fine-tune) | < 1 min |
| ajuste | `pipe-tune` | grade temperatura {0,8; 0,9; 1,0} × top-p {0,9; 0,95; 1,0} na **validação**, 100 amostras cada | ~20–30 min |
| final | `pipe-final` | Markov, 100 amostras no **teste** de cada execução e do Markov com a amostragem escolhida, avaliação | ~20–30 min |

Todos os treinos param sozinhos por early stopping (`patience` épocas sem melhorar a validação).

## 2. Opções

| Opção | Efeito |
|---|---|
| `--dry-run` | só mostra os `sbatch` que seriam submetidos, com as dependências |
| `--from <etapa>` | começa de `train` (padrão), `select`, `tune` ou `final`, usando o que já existe em `outputs/` |
| `--no-data` | não reconstrói `data/processed/` |
| `--resume` | os treinos continuam do `last.pt` (execuções já terminadas não treinam de novo) |
| `--sizes "large"` | só um tamanho de modelo (padrão `"large small"`) |
| `--no-scratch` | sem a ablação do zero |
| `--gen all` | no teste, gera também com **todas** as combinações da grade (melhor checkpoint), além da melhor |
| `--n 100` | amostras por sistema no ajuste e no teste |
| `--temperatures "0.8 0.9 1.0"`, `--top-ps "0.9 0.95 1.0"` | grade do ajuste (e do `--gen all`) |

## 3. Acompanhar

```bash
squeue -u $USER                          # o que está na fila e rodando
tail -f logs/pretrain_large-<id>.out     # treino: loss a cada 100 passos e resumo por época
tail -f logs/pipe-tune-<id>.out          # etapas do pipeline: cada comando aparece com "$ ..."
sacct -j <id> --format=JobID,JobName,State,Elapsed,NodeList
```

No começo de cada log de treino aparecem a GPU, o número de parâmetros, as janelas, os passos por
época e as partições usadas (`dados: pretrain_train/pretrain_val` no pré-treino, `train/val` nos
demais). No fine-tune aparece também `pesos iniciais de outputs/runs/pretrain_<t>/best.pt`.

## 4. Se algo falhar

| Situação | O que fazer |
|---|---|
| Um treino caiu (tempo, OOM) | `./jobs/pipeline.sh --no-data --resume` (refaz só o que falta) |
| Um pré-treino falhou | o fine-tune correspondente fica preso em `DependencyNeverSatisfied`: `scancel` dele e do `pipe-*`, corrija e rode `--no-data --resume` |
| Um treino da ablação falhou, o resto terminou | `scancel` do que ficou preso e `./jobs/pipeline.sh --from select` |
| `pipe-tune` ou `pipe-final` falhou | veja o último `$ ...` no log, corrija e `./jobs/pipeline.sh --from tune` (ou `--from final`) |
| Quer outra grade de amostragem | `./jobs/pipeline.sh --from tune --temperatures "0.7 0.8" --top-ps "0.9 0.95"` |

O `pipe-select` e o `pipe-plot` esperam todos os treinos terminarem, com ou sem erro (`afterany`), e
usam o que existir: uma execução que falhou só some da tabela. O `pipe-tune` e o `pipe-final` só
rodam se a etapa anterior deu certo (`afterok`).

## 5. Onde ficam os resultados (e o que vai para o artigo)

| Arquivo | Conteúdo | Seção do artigo |
|---|---|---|
| `outputs/runs/summary.md` | val loss e perplexidade de cada execução, melhor época, s/época, GPU, ganho do pré-treino | Experimentos (hiperparâmetros, hardware), ablação |
| `outputs/figures/pretrain_curves.pdf`, `emopia_curves.pdf` | curvas de treino e validação | Experimentos |
| `outputs/eval/val/best_sampling.json` | amostragem escolhida (temperatura, top-p) e por quê | Experimentos (protocolo) |
| `outputs/eval/val/tables.md` | tabela do ajuste na validação | material suplementar |
| `outputs/eval/test/tables.md` | métricas (média ± dp, OA, KL), fidelidade pedida x embaralhada, emoção por quadrante | Resultados |
| `outputs/eval/test/metrics_violin.pdf`, `curves.pdf` | distribuições das métricas e curva pedida x obtida | Resultados |
| `outputs/markov/selection.json` | ordem e desconto do Markov, perplexidade de validação | Método (baseline) |
| `configs/pop1k7_leakage.json` | músicas do Pop1K7 removidas por vazamento | Método (dados) |
| `outputs/samples/<sistema>/test/` | os MIDIs gerados, com `manifest.json` (curva e quadrante pedidos) | áudios |

Na tabela do teste, `human:val` é a verificação de sanidade (validação contra teste: OA alta) e a
linha `human` da emoção mostra a separação esperada entre quadrantes.

## 6. Depois do pipeline: áudios

Na sua máquina (precisa de FluidSynth, ffmpeg e um soundfont livre):

```bash
rsync -av <usuario>@<login-do-cluster>:~/<pasta-do-repositorio>/outputs/ outputs/ --exclude '*.pt'
uv run python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 finetune_large markov human
```

O `render` sorteia 5 músicas por sistema (seed fixa) e grava `.mid` e `.mp3` em `samples/`, que vai
para o repositório.

## Teste local (CPU, sem cluster)

As mesmas etapas rodam localmente com modelos minúsculos, para conferir o encadeamento:

```bash
export PYTHONPATH=.
uv run python -m src.data.build_dataset
uv run python -m src.model.train --config configs/pretrain_small.yaml --smoke
uv run python -m src.model.train --config configs/finetune_small.yaml --smoke --init-from outputs/runs/pretrain_small_smoke/best.pt
uv run python -m src.model.train --config configs/scratch_small.yaml --smoke
uv run python -m src.pipeline select --suffix _smoke
uv run python -m src.pipeline plot --suffix _smoke
uv run python -m src.pipeline tune --suffix _smoke --n 4 --temperatures 0.9 --top-ps 0.95
uv run python -m src.pipeline final --suffix _smoke --n 4
```
