#!/usr/bin/env bash
# Dispara no SLURM, encadeados: preparo dos dados -> treino do modelo grande e do pequeno (em paralelo).
#
# Uso, na raiz do repositório no cluster (com o EMOPIA_1.0.zip na raiz):
#   ./jobs/train_all.sh             # reconstrói data/processed e depois treina
#   ./jobs/train_all.sh --no-data   # usa o data/processed que já existe
#   ./jobs/train_all.sh --resume    # (com ou sem --no-data) continua os treinos do último checkpoint
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

build_data=1
train_args=()
for arg in "$@"; do
    case "$arg" in
        --no-data) build_data=0 ;;
        --resume) train_args+=(--resume) ;;
        *) echo "opção desconhecida: $arg" >&2; exit 1 ;;
    esac
done

dep=()
if [ "$build_data" -eq 1 ]; then
    data_id=$(sbatch --parsable -J data jobs/run.slurm python -m src.data.build_dataset)
    echo "dados:   job $data_id"
    dep=(--dependency="afterok:$data_id")
fi

large_id=$(sbatch --parsable "${dep[@]}" -J large jobs/run.slurm \
    python -m src.model.train --config configs/model_large.yaml "${train_args[@]}")
small_id=$(sbatch --parsable "${dep[@]}" -J small jobs/run.slurm \
    python -m src.model.train --config configs/model_small.yaml "${train_args[@]}")
echo "grande:  job $large_id"
echo "pequeno: job $small_id"
echo
echo "acompanhar: squeue -u \$USER"
echo "            tail -f logs/large-$large_id.out"
