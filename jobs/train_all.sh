#!/usr/bin/env bash
# Dispara no SLURM, encadeados por dependência:
#   data  ->  pretrain_large, pretrain_small, scratch_large, scratch_small   (em paralelo)
#   pretrain_<t>  ->  finetune_<t>
#
# Uso, na raiz do repositório no cluster (com EMOPIA+.zip e Pop1K7.zip na raiz; ver scripts/download_data.sh):
#   ./jobs/train_all.sh                 # reconstrói data/processed e depois treina tudo
#   ./jobs/train_all.sh --no-data       # usa o data/processed que já existe
#   ./jobs/train_all.sh --resume        # (com ou sem --no-data) continua os treinos do último checkpoint
#   ./jobs/train_all.sh --sizes small   # só o tamanho pequeno (padrão: "large small")
#   ./jobs/train_all.sh --no-scratch    # sem a ablação do zero
# Imprime, na última linha, "IDS <finetune_large> <finetune_small> ..." para o jobs/pipeline.sh.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

build_data=1
scratch=1
sizes="large small"
train_args=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --no-data) build_data=0; shift ;;
        --no-scratch) scratch=0; shift ;;
        --resume) train_args+=(--resume); shift ;;
        --sizes) sizes="$2"; shift 2 ;;
        *) echo "opção desconhecida: $1" >&2; exit 1 ;;
    esac
done

submit() {  # submit <nome> <dependência ou ""> <comando...>
    local name="$1" dep="$2"; shift 2
    local opts=(--parsable -J "$name")
    [ -n "$dep" ] && opts+=(--dependency="afterok:$dep")
    sbatch "${opts[@]}" jobs/run.slurm "$@"
}

data_id=""
if [ "$build_data" -eq 1 ]; then
    data_id=$(submit data "" python -m src.data.build_dataset)
    echo "dados:              job $data_id"
fi

final_ids=()
for size in $sizes; do
    pre_id=$(submit "pretrain_$size" "$data_id" python -m src.model.train --config "configs/pretrain_$size.yaml" "${train_args[@]}")
    ft_id=$(submit "finetune_$size" "$pre_id" python -m src.model.train --config "configs/finetune_$size.yaml" "${train_args[@]}")
    echo "pré-treino $size:   job $pre_id  ->  fine-tune $size: job $ft_id"
    final_ids+=("$ft_id")
    if [ "$scratch" -eq 1 ]; then
        sc_id=$(submit "scratch_$size" "$data_id" python -m src.model.train --config "configs/scratch_$size.yaml" "${train_args[@]}")
        echo "do zero $size:      job $sc_id"
        final_ids+=("$sc_id")
    fi
done
echo
echo "acompanhar: squeue -u \$USER   |   tail -f logs/<nome>-<id>.out"
echo "IDS ${final_ids[*]}"
