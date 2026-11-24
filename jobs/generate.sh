#!/usr/bin/env bash
# Gera as músicas de cada modelo com as curvas reais de uma partição, um job por modelo (em paralelo).
#
#   <execução>     -> Transformer de outputs/runs/<execução>/best.pt (ex.: finetune_large, scratch_small)
#   markov         -> outputs/markov/markov.npz (ajustado antes, se não existir ou se os dados forem mais novos)
#   human          -> os clipes humanos da partição passados pelo REMI (referência da avaliação)
#
# Uso, na raiz do repositório no cluster, depois do treino:
#   ./jobs/generate.sh                                          # finetune_large finetune_small markov, teste, 100 amostras
#   ./jobs/generate.sh --temperature 1.0 --top-p 0.95           # com a amostragem escolhida na validação
#   ./jobs/generate.sh --models "finetune_large scratch_large" --split val --n 20
#   ./jobs/generate.sh --after <id-large>:<id-small>            # espera os treinos terminarem sem erro
#
# Saída: outputs/samples/<modelo>/<split>/{000.mid, ..., manifest.json, ids.pt}.
# Para avaliar depois: python -m src.eval.evaluate --split <split> --systems finetune_large finetune_small markov
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

MODELS="finetune_large finetune_small markov"
SPLIT="test"
N=""
AFTER=""
sampling=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --models) MODELS="$2"; shift 2 ;;
        --split) SPLIT="$2"; shift 2 ;;
        --n) N="$2"; shift 2 ;;
        --after) AFTER="$2"; shift 2 ;;
        --temperature|--top-p|--top-k|--seed) sampling+=("$1" "$2"); shift 2 ;;
        -h|--help) sed -n '2,16p' "$0"; exit 0 ;;
        *) echo "opção desconhecida: $1" >&2; exit 1 ;;
    esac
done
case "$SPLIT" in val|test) ;; *) echo "--split precisa ser val ou test" >&2; exit 1 ;; esac

common="--split $SPLIT ${N:+--n $N} ${sampling[*]:-}"
dep=()
[ -n "$AFTER" ] && dep=(--dependency="afterok:$AFTER")

for model in $MODELS; do
    case "$model" in
        markov)
            cmd="python -m src.generate.run --system markov $common"
            # reajusta se não existe ou se os dados foram reconstruídos depois (ex.: Markov do EMOPIA 1.0)
            if [ ! -f outputs/markov/markov.npz ] || [ data/processed/train.pt -nt outputs/markov/markov.npz ]; then
                cmd="python -m src.baseline.markov && $cmd"
            fi
            ;;
        human)
            cmd="python -m src.generate.run --system human --split $SPLIT ${N:+--n $N}"
            ;;
        *)
            ckpt="outputs/runs/$model/best.pt"
            if [ -z "$AFTER" ] && [ ! -f "$ckpt" ]; then
                echo "checkpoint não encontrado: $ckpt (o treino de '$model' já terminou?)" >&2
                exit 1
            fi
            cmd="python -m src.generate.run --system transformer --checkpoint $ckpt --name $model $common"
            ;;
    esac
    id=$(sbatch --parsable "${dep[@]}" -J "gen-$model" jobs/run.slurm bash -c "$cmd")
    echo "$model: job $id | log: logs/gen-$model-$id.out | saída: outputs/samples/$model/$SPLIT/"
done
