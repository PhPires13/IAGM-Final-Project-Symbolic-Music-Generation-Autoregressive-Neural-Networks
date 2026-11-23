# Guia: rodando o TP2 no cluster do DCC

Este guia leva do repositório vazio no cluster até as tabelas finais do artigo. Todos os jobs usam a
partição `high_performance` (A100 ou L40S, as duas com bf16) por meio do `jobs/run.slurm`, que
carrega o módulo `pytorch/2.5.1-py3.10-cuda12.8` e monta um venv em `/tmp/venv_tp2_$USER` no nó.

Os comandos abaixo rodam no nó de login, na raiz do repositório.

## 0. Preparação (uma vez)

```bash
git clone git@github.com:PhPires13/IAGM-TP-Final-Symbolic-Music-Generation-Autoregressive-Neural-Networks.git tp2
cd tp2
```

O dataset não está no repositório. Copie o zip da sua máquina para a raiz do clone:

```bash
# na SUA máquina
scp EMOPIA_1.0.zip <usuario>@<login-do-cluster>:~/tp2/
```

Nas próximas vezes basta `git pull` dentro de `tp2/`.

## 1. Treino (dados + modelo grande + modelo pequeno)

```bash
./jobs/train_all.sh
```

Isso submete três jobs encadeados:

| Job | O que faz | Duração esperada |
|---|---|---|
| `data` | extrai o EMOPIA, tokeniza, mede energia/tensão, insere controles, transpõe o treino -> `data/processed/` | 2 a 5 min |
| `large` | Transformer de 6 camadas, d_model 512, contexto 1024 (19,6M parâmetros) | ver o tempo por época no log |
| `small` | Transformer de 4 camadas, d_model 256, contexto 512 (3,4M); plano B | idem |

Os dois treinos só começam se o `data` terminar sem erro (`afterok`) e rodam em paralelo. Cada um para
sozinho por early stopping (5 épocas sem melhorar a validação) ou em 40 épocas.

### Acompanhando

```bash
squeue -u $USER                      # jobs na fila / rodando
tail -f logs/large-<id>.out          # loss a cada 100 passos e resumo de cada época
sacct -j <id> --format=JobID,JobName,State,Elapsed,NodeList   # estado e duração depois de terminar
scancel <id>                         # cancelar
```

No começo de cada log aparecem o nó, a GPU (`nvidia-smi`), a versão do torch e o commit usado.
Anote a GPU e o tempo por época: eles vão para a seção de hardware do artigo.

### Se um treino cair

```bash
./jobs/train_all.sh --no-data --resume
```

Os treinos continuam do último `last.pt` (o `data/processed/` existente é reaproveitado).

### Saída

`outputs/runs/<large|small>/`: `best.pt` (menor loss de validação, é o que se usa daqui em diante),
`last.pt`, `metrics.csv` e `run_config.json`. A figura das curvas de treino:

```bash
sbatch -J curves jobs/run.slurm python -m src.model.plot_curves large small   # -> outputs/figures/
```

## 2. Escolha do checkpoint

O `best.pt` de cada modelo já é o checkpoint de menor loss de validação. Entre o grande e o pequeno,
compare `val_loss` no fim do `metrics.csv` de cada um (ou o resumo no fim do log) e fique com o menor.
Nos passos seguintes, troque `outputs/runs/large/best.pt` pelo do pequeno se for ele o escolhido.

## 3. Ajuste de amostragem (na validação)

```bash
./jobs/tune_sampling.sh outputs/runs/large/best.pt large
# ou, para já deixar na fila esperando o treino:
./jobs/tune_sampling.sh outputs/runs/large/best.pt large <id-do-job-large>
```

Gera 100 amostras na validação para cada combinação de temperatura {0,9; 1,0; 1,1} × top-p
{0,9; 0,95; 1,0} e avalia todas contra os clipes humanos da validação. O fim de `logs/tune-<id>.out`
traz uma tabela ordenada por OA média, com a fidelidade ao controle ao lado. Escolha a configuração com
OA alta que **não** perca fidelidade (o ganho sobre a curva embaralhada precisa continuar positivo).

## 4. Avaliação final (no teste)

Com a amostragem escolhida:

```bash
./jobs/final_eval.sh --temperature 1.0 --top-p 0.95
```

Ajusta o Markov (n e desconto pela validação), gera 100 amostras do Transformer e 100 do Markov com as
curvas reais do teste, exporta a referência humana e roda a avaliação. Este é o **único** passo que usa
o conjunto de teste.

Resultados em `outputs/eval/test/`:

- `tables.md`: tabela das métricas (média ± dp, OA, KL) e tabela de fidelidade (curva pedida x embaralhada);
- `metrics_violin.pdf` e `curves.pdf`: figuras para o artigo;
- `results.json` e `per_piece.csv`: números completos.

O `human:val` na tabela é a verificação de sanidade: validação contra teste tem que dar OA alta.

## 5. Trazendo os resultados e gerando os áudios

O áudio é renderizado localmente (precisa de FluidSynth, ffmpeg e um soundfont livre, por exemplo
FluidR3_GM ou GeneralUser GS):

```bash
# na SUA máquina, na raiz do repositório
rsync -av <usuario>@<login-do-cluster>:~/tp2/outputs/ outputs/ --exclude '*.pt'
rsync -av <usuario>@<login-do-cluster>:~/tp2/outputs/runs/ outputs/runs/ --include '*/' --include 'metrics.csv' --include 'run_config.json' --exclude '*'
uv run python -m src.generate.render --soundfont /caminho/FluidR3_GM.sf2 transformer markov human
```

O `render` sorteia 5 amostras por sistema (seed fixa) e grava `.mid` + `.mp3` em `samples/`, que é
versionado e citado no artigo.

## Problemas comuns

| Sintoma | Causa provável | O que fazer |
|---|---|---|
| Job termina na hora sem log | a pasta `logs/` não existia | os scripts `jobs/*.sh` já criam; com `sbatch` avulso, `mkdir -p logs` antes |
| `EMOPIA_1.0.zip não encontrado` | zip não copiado para a raiz do clone | ver passo 0 |
| `foi gerado com --no-controls` | `data/processed` de um teste antigo | `./jobs/train_all.sh` (sem `--no-data`) reconstrói |
| `CUDA out of memory` | lote grande demais para a GPU | reduzir `batch_size` em `configs/model_*.yaml` e rodar com `--resume` |
| Erro de import ou pacote quebrado no venv | venv corrompido no nó | `srun -p high_performance --qos=normal --gpus-per-node=1 rm -rf /tmp/venv_tp2_$USER` e resubmeter |
| Treino parou cedo demais | early stopping | ver `metrics.csv`; se a validação ainda caía, aumentar `patience` e rodar com `--resume` |
