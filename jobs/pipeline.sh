#!/usr/bin/env bash
# Pipeline completo dos experimentos no SLURM, um job por etapa, encadeados por dependência:
#
#   train   data -> pretrain_{large,small} -> finetune_{large,small}, e scratch_{large,small}   (jobs/train_all.sh)
#   select  tabela de val_loss de todas as execuções e escolha do melhor fine-tune   (depois de todos os treinos)
#   plot    figuras das curvas de treino                                            (em paralelo com o select)
#   tune    grade temperatura x top-p na VALIDAÇÃO com o checkpoint escolhido       (depois do select)
#   final   geração no TESTE com a melhor amostragem (ou com todas: --gen all) + avaliação   (depois do tune)
#
# Logs: logs/<etapa>-<id>.out (os de treino com o nome de cada execução; os demais como pipe-<etapa>).
# Guia: docs/pipeline.md
#
# Uso, na raiz do repositório no cluster (com EMOPIA+.zip e Pop1K7.zip na raiz):
#   ./jobs/pipeline.sh                          # tudo, do zero
#   ./jobs/pipeline.sh --dry-run                # só mostra os jobs e as dependências, sem submeter
#   ./jobs/pipeline.sh --from tune              # retoma de uma etapa (train | select | tune | final)
#   ./jobs/pipeline.sh --no-data --resume       # retreina reaproveitando data/processed e os checkpoints
#   ./jobs/pipeline.sh --gen all                # no teste, gera também todas as combinações da grade
# Outras opções: --sizes "large small", --no-scratch, --n 100,
#   --temperatures "0.8 0.9 1.0", --top-ps "0.9 0.95 1.0"
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs

from="train"
gen="best"
dry=0
train_opts=()
stage_opts=()
while [ "$#" -gt 0 ]; do
    case "$1" in
        --from) from="$2"; shift 2 ;;
        --gen) gen="$2"; shift 2 ;;
        --dry-run) dry=1; shift ;;
        --no-data|--no-scratch|--resume) train_opts+=("$1"); shift ;;
        --sizes) train_opts+=(--sizes "$2"); shift 2 ;;
        --n) stage_opts+=(--n "$2"); shift 2 ;;
        --temperatures) read -ra v <<< "$2"; stage_opts+=(--temperatures "${v[@]}"); shift 2 ;;
        --top-ps) read -ra v <<< "$2"; stage_opts+=(--top-ps "${v[@]}"); shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "opção desconhecida: $1" >&2; exit 1 ;;
    esac
done
case "$from" in train|select|tune|final) ;; *) echo "--from: train, select, tune ou final" >&2; exit 1 ;; esac
case "$gen" in best|all) ;; *) echo "--gen: best ou all" >&2; exit 1 ;; esac

if [ "$dry" -eq 1 ]; then  # sbatch falso: imprime o comando e devolve um id fictício
    fake=$(mktemp -d)
    trap 'rm -rf "$fake"' EXIT
    printf '#!/usr/bin/env bash\nn=$(( $(cat "%s/n" 2>/dev/null || echo 1000) + 1 )); echo $n > "%s/n"\necho "  [dry-run] sbatch $*  -> job $n" >&2; echo $n\n' \
        "$fake" "$fake" > "$fake/sbatch"
    chmod +x "$fake/sbatch"
    export PATH="$fake:$PATH"
fi

submit() {  # submit <nome> <dependência ou ""> <comando...>
    local name="$1" dep="$2"; shift 2
    local opts=(--parsable -J "$name")
    [ -n "$dep" ] && opts+=(--dependency="$dep")
    sbatch "${opts[@]}" jobs/run.slurm "$@"
}

order=(train select tune final)
started=0
dep_train=""
dep_select=""
dep_tune=""

for stage in "${order[@]}"; do
    [ "$stage" = "$from" ] && started=1
    [ "$started" -eq 0 ] && continue
    case "$stage" in
        train)
            out=$(./jobs/train_all.sh "${train_opts[@]}")
            echo "$out" | grep -v '^IDS '
            ids=$(echo "$out" | sed -n 's/^IDS //p' | tr ' ' ':')
            [ -n "$ids" ] && dep_train="afterany:$ids"
            ;;
        select)
            sel=$(submit pipe-select "$dep_train" python -m src.pipeline select)
            plt=$(submit pipe-plot "$dep_train" python -m src.pipeline plot)
            echo "select: job $sel | plot: job $plt"
            dep_select="afterok:$sel"
            ;;
        tune)
            tun=$(submit pipe-tune "$dep_select" python -m src.pipeline tune "${stage_opts[@]}")
            echo "tune:   job $tun"
            dep_tune="afterok:$tun"
            ;;
        final)
            fin=$(submit pipe-final "$dep_tune" python -m src.pipeline final --gen "$gen" "${stage_opts[@]}")
            echo "final:  job $fin"
            ;;
    esac
done

echo
echo "acompanhar: squeue -u \$USER   |   tail -f logs/pipe-<etapa>-<id>.out"
echo "resultados: outputs/runs/summary.md, outputs/figures/, outputs/eval/val/best_sampling.json, outputs/eval/test/tables.md"
