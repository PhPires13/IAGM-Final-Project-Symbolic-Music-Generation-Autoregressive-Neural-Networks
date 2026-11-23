#!/usr/bin/env bash
# Avaliação final NO TESTE, com a amostragem escolhida na validação, em um único job:
# ajusta o Markov, gera 100 amostras do Transformer e do Markov com as curvas reais do teste,
# exporta a referência humana (teste e validação) e roda a avaliação completa.
#
# Uso, na raiz do repositório no cluster:
#   ./jobs/final_eval.sh --temperature 1.0 --top-p 0.95
#   ./jobs/final_eval.sh --checkpoint outputs/runs/small/best.pt --temperature 1.0 --top-p 0.95
# Resultado: outputs/eval/test/ (tables.md, results.json, figuras) e outputs/samples/.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CKPT="outputs/runs/large/best.pt"
sampling=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --checkpoint) CKPT="$2"; shift 2 ;;
        --temperature|--top-p|--top-k) sampling+=("$1" "$2"); shift 2 ;;
        *) echo "opção desconhecida: $1" >&2; exit 1 ;;
    esac
done

cmds="python -m src.baseline.markov"
cmds+=" && python -m src.generate.run --system transformer --checkpoint $CKPT ${sampling[*]:-}"
cmds+=" && python -m src.generate.run --system markov ${sampling[*]:-}"
cmds+=" && python -m src.generate.run --system human"
cmds+=" && python -m src.generate.run --system human --split val"
cmds+=" && python -m src.eval.evaluate --systems transformer markov human:val"

id=$(sbatch --parsable -J final jobs/run.slurm bash -c "$cmds")
echo "avaliação final: job $id | log: logs/final-$id.out"
