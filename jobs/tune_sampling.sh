#!/usr/bin/env bash
# Ajuste de amostragem NA VALIDAÇÃO (nunca no teste): gera amostras com várias combinações de
# temperatura/top-p e avalia todas contra os clipes humanos da validação, em um único job.
#
# Uso, na raiz do repositório no cluster, depois do treino:
#   ./jobs/tune_sampling.sh                                   # usa outputs/runs/large/best.pt
#   ./jobs/tune_sampling.sh outputs/runs/small/best.pt small  # outro checkpoint / prefixo dos nomes
#   ./jobs/tune_sampling.sh outputs/runs/large/best.pt large <id-do-job-de-treino>   # espera o treino
# Resultado: outputs/eval/val/tables.md e o resumo no fim do log (logs/tune-<id>.out).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

CKPT="${1:-outputs/runs/large/best.pt}"
PREFIX="${2:-large}"
AFTER="${3:-}"
TEMPERATURES=(0.9 1.0 1.1)
TOP_PS=(0.9 0.95 1.0)

cmds="python -m src.generate.run --system human --split val"
names=()
for t in "${TEMPERATURES[@]}"; do
    for p in "${TOP_PS[@]}"; do
        name="${PREFIX}_t${t}_p${p}"
        names+=("$name")
        cmds+=" && python -m src.generate.run --system transformer --checkpoint $CKPT --split val --temperature $t --top-p $p --name $name"
    done
done
cmds+=" && python -m src.eval.evaluate --split val --systems ${names[*]}"
cmds+=" && python -m src.model.sampling_summary --split val"

dep=()
[ -n "$AFTER" ] && dep=(--dependency="afterok:$AFTER")
id=$(sbatch --parsable "${dep[@]}" -J tune jobs/run.slurm bash -c "$cmds")
echo "ajuste de amostragem: job $id (${#names[@]} combinações) | log: logs/tune-$id.out"
