# Guia: rodando o TP2 no cluster do DCC

Todos os jobs usam a partição `high_performance` (A100 ou L40S, as duas com bf16) por meio do
`jobs/run.slurm`, que carrega o módulo `pytorch/2.5.1-py3.10-cuda12.8` e monta um venv em
`/tmp/venv_tp2_$USER` no nó. Os comandos abaixo rodam no nó de login, na raiz do repositório.

**O caminho recomendado é o pipeline completo, com um único comando: [`pipeline.md`](pipeline.md).**
Este guia cobre a preparação, o ambiente e como rodar as etapas uma a uma.

## 0. Preparação (uma vez)

```bash
git clone git@github.com:PhPires13/IAGM-TP-Final-Symbolic-Music-Generation-Autoregressive-Neural-Networks.git
cd IAGM-TP-Final-Symbolic-Music-Generation-Autoregressive-Neural-Networks
./scripts/download_data.sh      # EMOPIA+.zip e Pop1K7.zip do Zenodo
```

Sem internet no nó de login: rode o `download_data.sh` na sua máquina e copie os zips com
`scp EMOPIA+.zip Pop1K7.zip <usuario>@<login-do-cluster>:~/<pasta-do-repositorio>/` (atenção aos `:`
antes do caminho; sem eles o `scp` faz uma cópia local).

Nas próximas vezes basta `git pull`.

> O EMOPIA 1.0 (`EMOPIA_1.0.zip`) não é mais usado: os MIDIs dele não estão alinhados ao beat.
> Ver [`pivo_pretreino.md`](pivo_pretreino.md).

## 1. O job genérico (`jobs/run.slurm`)

Ele roda o comando passado depois dele:

```bash
mkdir -p logs
sbatch -J <nome> jobs/run.slurm python -m <módulo> [args...]    # log em logs/<nome>-<id>.out
```

Dentro do job: pede 1 GPU, 12 CPUs e 16 GB; registra nó, commit e GPU no log; carrega o módulo;
cria (se preciso) o venv com `--system-site-packages` e instala o `requirements.txt`, com o venv à
frente do `PYTHONPATH` do módulo; e executa o comando com `srun`.

## 2. Etapas uma a uma

```bash
./jobs/train_all.sh                 # dados -> pré-treino -> fine-tune, e a ablação do zero
./jobs/tune_sampling.sh             # grade de amostragem na validação (finetune_large)
./jobs/final_eval.sh --temperature 0.9 --top-p 0.95    # avaliação no teste
./jobs/generate.sh --temperature 0.9 --top-p 0.95      # só gerar músicas (um job por modelo)
```

O `pipeline.sh` faz o mesmo encadeado e escolhe o checkpoint e a amostragem sozinho; os scripts
avulsos servem para refazer uma parte. Opções de cada um estão no cabeçalho do arquivo.

### Acompanhando

```bash
squeue -u $USER
tail -f logs/<nome>-<id>.out
sacct -j <id> --format=JobID,JobName,State,Elapsed,NodeList
scancel <id>
sinfo -p high_performance -o "%P %l"     # limite de tempo da partição
```

Anote a GPU e o tempo por época (estão no log e em `outputs/runs/summary.md`): vão para a seção de
hardware do artigo.

## 3. Trazendo os resultados

```bash
# na SUA máquina, na raiz do repositório
rsync -av <usuario>@<login-do-cluster>:~/<pasta-do-repositorio>/outputs/ outputs/ --exclude '*.pt'
python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 finetune_large markov human
```

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Job termina na hora sem log | a pasta `logs/` não existia | os scripts `jobs/*.sh` já criam; com `sbatch` avulso, `mkdir -p logs` antes |
| `EMOPIA+.zip não encontrado` / `Pop1K7.zip não encontrado` | zips não baixados | passo 0 |
| `foi gerado com --no-controls` ou `é de antes do EMOPIA+` | `data/processed` antigo | `./jobs/train_all.sh` (sem `--no-data`) reconstrói |
| `arquitetura de ... difere da configuração` | `finetune_*.yaml` com modelo diferente do `pretrain_*.yaml` | deixar a seção `model` igual nos dois |
| `CUDA out of memory` | lote grande demais para a GPU | reduzir `batch_size` em `configs/*.yaml` e rodar com `--resume` |
| Erro de import ou pacote quebrado no venv | venv corrompido no nó | o job recria sozinho se o pip quebrar; senão `srun -p high_performance --qos=normal --gpus-per-node=1 rm -rf /tmp/venv_tp2_$USER` |
| Job preso em `DependencyNeverSatisfied` | um job anterior da cadeia falhou | `scancel` e ver [`pipeline.md`](pipeline.md), seção 4 |
| Treino parou cedo demais | early stopping | ver `metrics.csv`; se a validação ainda caía, aumentar `patience` e rodar com `--resume` |
