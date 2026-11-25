# Geração de piano com curvas de energia e tensão

Trabalho final de DCC831 – IA Generativa para Música (UFMG, 2026/2).

Geração de música para piano condicionada por **emoção** (quadrante de Russell, um token por música) e,
compasso a compasso, por níveis de **energia** e **tensão**. Um Transformer autoregressivo escrito em
PyTorch é pré-treinado no [Pop1K7](https://zenodo.org/records/13167761) e ajustado no
[EMOPIA+](https://zenodo.org/records/13122742), com representação REMI, e comparado com uma cadeia de
Markov de ordem n treinada nos mesmos dados.

Enunciado do trabalho: [`docs/enunciado.md`](docs/enunciado.md). Por que EMOPIA+ e pré-treino (e não o EMOPIA 1.0 do
plano original): [`docs/pivo_pretreino.md`](docs/pivo_pretreino.md).

## Resultados, amostras e artigo

- **Artigo (ISMIR, 4 páginas):** [`entregas/IAGM_Projeto_Final_Artigo.pdf`](entregas/IAGM_Projeto_Final_Artigo.pdf); fonte em [`latex/main.tex`](latex/main.tex)
  (compilar com `cd latex && tectonic main.tex`; as figuras saem de `python latex/figuras/make_figures.py`).
- **Resultados da rodada final no cluster:** [`resultados_artigo/`](resultados_artigo/) (tabelas e figuras de treino, validação e
  teste, e as 100 músicas geradas por sistema). Notas e números comentados em
  [`docs/notas_resultados.md`](docs/notas_resultados.md).
- **Amostras para ouvir:** [`samples/`](samples/), com MIDI e áudio (`.m4a`). São 5 pedidos (curva + quadrante de um clipe de
  teste) sorteados com seed fixa e gerados pelos dois sistemas; `finetune_large_014` e `markov_014` receberam o mesmo
  pedido, e `samples/selection.json` liga cada arquivo ao clipe humano de origem (`samples/human/`).

| Pedido (clipe de teste) | Quadrante | Transformer (pré-treino + ajuste fino) | Markov | Clipe humano |
|---|---|---|---|---|
| `Q1_e838BE_MH6s_1` | Q1 | `finetune_large_014` | `markov_014` | `human_014` |
| `Q2_AWNbcxXDmt0_0` | Q2 | `finetune_large_031` | `markov_031` | `human_031` |
| `Q2_DYBBfpEx-JI_0` | Q2 | `finetune_large_035` | `markov_035` | `human_035` |
| `Q4_jCLgDtk695I_3` | Q4 | `finetune_large_081` | `markov_081` | `human_081` |
| `Q1_GY3f6ckBVkA_1` | Q1 | `finetune_large_091` | `markov_091` | `human_003` |

## Instalação (local)

Requer **Python 3.10**, a mesma versão do cluster (o `torch==2.5.1` não tem pacote para versões mais
novas em todas as plataformas).

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
./scripts/download_data.sh     # EMOPIA+.zip e Pop1K7.zip (Zenodo, CC-BY 4.0) na raiz
```

No Linux ou Windows sem GPU, acrescente `--extra-index-url https://download.pytorch.org/whl/cpu` ao
`pip install` para baixar o torch só de CPU, em vez dos ~2,5 GB de bibliotecas CUDA.

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
python -m src.data.split                # divisão oficial do EMOPIA+ + Pop1K7 sem vazamento -> configs/splits.json
python -m src.data.tokenizer --force    # REMI com grade de 1/12 de beat -> configs/tokenizer.json
python -m src.controls.calibrate --pretrain   # tercis no treino do EMOPIA+ -> configs/control_thresholds.json
python -m src.controls.sanity           # balanceamento, correlação, energia por quadrante -> outputs/controls/
python -m src.data.build_dataset        # tokens + emoção + controles + transposição -> data/processed/*.pt
python -m src.data.stats                # estatísticas -> outputs/stats/dataset_stats.json

# treino (no cluster): pré-treino, fine-tune e ablação do zero
python -m src.model.train --config configs/pretrain_large.yaml
python -m src.model.train --config configs/finetune_large.yaml
python -m src.model.train --config configs/scratch_large.yaml
# teste rápido local: --smoke (e --init-from outputs/runs/pretrain_small_smoke/best.pt no fine-tune)

# seleção, figuras, ajuste de amostragem (validação) e avaliação final (teste)
python -m src.pipeline select
python -m src.pipeline plot
python -m src.pipeline tune
python -m src.pipeline final            # Markov + geração + avaliação -> outputs/eval/test/

# áudio: 5 pedidos sorteados (seed fixa), os mesmos em todos os sistemas -> samples/
python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 finetune_large markov human
# alternativa no macOS, sem soundfont nem ffmpeg (banco GS do sistema, saída .m4a), a partir das músicas do cluster:
python -m src.generate.render --backend macos --samples-dir resultados_artigo/musicas/teste finetune_large markov human
```

## Estrutura

```
entregas/          resumo e artigo em PDF
samples/           5 pedidos sorteados: MIDI + áudio do Transformer, do Markov e do clipe humano
resultados_artigo/ tabelas, figuras e músicas da rodada final no cluster (base do artigo)
latex/             fonte do artigo (template ISMIR) e script das figuras
src/               código (ver abaixo)
configs/           default.yaml, YAMLs das execuções, split, tokenizador, limites dos níveis, vazamento
jobs/              run.slurm (job genérico), pipeline.sh (tudo encadeado) e scripts das etapas avulsas
scripts/           download dos dados, render de áudio no macOS
docs/              enunciado, roadmap, pivô, guias do pipeline e do cluster, notas dos resultados
```

| Código | O que faz |
|---|---|
| `src/data/` | EMOPIA+, Pop1K7, split, vazamento, tokenização REMI, janelas, estatísticas |
| `src/controls/` | medição de energia e tensão, quantização, tokens de controle e de emoção |
| `src/model/` | Transformer, laço de treino, figuras de treino, resumo da grade de amostragem |
| `src/baseline/` | cadeia de Markov |
| `src/generate/` | amostragem (temperatura, top-k, top-p), geração condicionada, renderização de áudio |
| `src/eval/` | métricas do MusPy, OA/KL, fidelidade ao controle, emoção por quadrante |
| `src/pipeline.py` | etapas pós-treino (seleção, figuras, ajuste da amostragem, avaliação final) |

| Documento | Conteúdo |
|---|---|
| [`docs/enunciado.md`](docs/enunciado.md) | enunciado do trabalho |
| [`docs/roadmap.md`](docs/roadmap.md) | plano original: divisão de tarefas, contratos, cronograma e checklist |
| [`docs/pivo_pretreino.md`](docs/pivo_pretreino.md) | mudança para EMOPIA+ e pré-treino no Pop1K7, com o desenho dos experimentos |
| [`docs/pipeline.md`](docs/pipeline.md) | como rodar todos os experimentos com um comando |
| [`docs/cluster.md`](docs/cluster.md) | ambiente no cluster do DCC e etapas avulsas |
| [`docs/notas_resultados.md`](docs/notas_resultados.md) | resultados comentados da rodada final |

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
| Pedro Henrique Gonçalves Pires | Dados e representação: split oficial do EMOPIA+, Pop1K7 e verificação de vazamento, tokenizador REMI (grade de 1/12), dataset em janelas e estatísticas; registro do pivô; artigo, resumo e áudios das amostras |
| Julia Souza Moura | Atributos de controle: medição de energia e tensão por compasso, quantização por tercis, tokens de controle e de emoção, verificações de sanidade dos níveis |
| Natanael dos Santos | Transformer: modelo, laço de treino e amostragem; pré-treino, ajuste fino e ablação no cluster (SLURM); roadmap do projeto |
| Leonardo Cesar Cota de Castro | Estrutura do repositório e contratos; baseline Markov; rotina de geração condicionada comum; pipeline único dos experimentos; renderização |
| Júlia Borba Fonseca de Souza | Avaliação: métricas do MusPy, OA/KL, fidelidade ao controle, experimento com curvas embaralhadas, emoção por quadrante, escolha da amostragem e notas dos resultados |

## Uso de IA

O trabalho contou com o auxílio de IA durante o seu desenvolvimento. Cada integrante usou as ferramentas abaixo:

- **Pedro — Claude Code (Claude Opus 5.5, esforço médio)**
  - Planejamento do projeto, ajuda teórica, suporte de código na parte de dados (ensinar, melhorias, bugs), registro do pivô, melhoria da escrita e da formatação do artigo e organização do repositório
  - Transcrição: [`transcricoes/transcricao_sessao_pedro_pires.md`](transcricoes/transcricao_sessao_pedro_pires.md)
- **Julia M. — Hotmart Chat (GPT Sol)**
  - Ajuda teórica (tonalidade, escala, dissonância), suporte de código na medição de energia e tensão e nos tokens de controle, dúvidas na geração das amostras
  - Transcrição: [`transcricoes/transcricao_sessao_julia_moura.md`](transcricoes/transcricao_sessao_julia_moura.md)
- **Natanael — Claude Code (Claude Opus 5.5, esforço alto)**
  - Roadmap do projeto, suporte de código no Transformer, no laço de treino e na amostragem, scripts do cluster (SLURM) e correção de erros do ambiente
  - Transcrição: [`transcricoes/transcricao_sessao_natan.md`](transcricoes/transcricao_sessao_natan.md)
- **Leonardo — Kiro (plano gratuito, Claude Sonnet 4.5)**
  - Estrutura do repositório e contratos, suporte de código no baseline Markov, na geração condicionada e no pipeline, README
  - Transcrição: [`transcricoes/transcricao_sessao_leonardo.md`](transcricoes/transcricao_sessao_leonardo.md)
- **Júlia B. — Hotmart Chat (GPT Sol)**
  - Ajuda teórica (OA, KL e KDE no estilo de Yang & Lerch), suporte de código nas métricas e na fidelidade ao controle, interpretação dos resultados do cluster
  - Transcrição: [`transcricoes/transcricao_sessao_julia_borba.md`](transcricoes/transcricao_sessao_julia_borba.md)

As sessões dessas ferramentas ficam salvas só localmente ou na conta de cada pessoa, sem opção de compartilhar por link. Com isso, os arquivos estão em [`transcricoes/`](transcricoes/).
