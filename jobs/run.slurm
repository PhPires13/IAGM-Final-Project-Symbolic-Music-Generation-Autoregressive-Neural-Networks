#!/bin/bash
# Job genérico: roda no cluster do DCC o comando passado como argumento.
#
# Uso (da raiz do repositório; a pasta logs/ precisa existir antes do sbatch):
#   mkdir -p logs
#   sbatch -J large jobs/run.slurm python -m src.model.train --config configs/model_large.yaml
#   sbatch -J small jobs/run.slurm python -m src.model.train --config configs/model_small.yaml
#
# Opções passadas ao sbatch na linha de comando (-J, --mem, --time...) sobrescrevem as de baixo.

#SBATCH --qos=normal
#SBATCH --mem=16G
#SBATCH --cpus-per-task=12
#SBATCH --partition=high_performance
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --job-name=tp2
#SBATCH --output=logs/%x-%j.out

set -eo pipefail

if [ "$#" -eq 0 ]; then
    echo "Uso: sbatch jobs/run.slurm <comando> [args...]" >&2
    exit 1
fi

cd "${SLURM_SUBMIT_DIR:-.}"

node="$(hostname -s).sense.dcc.ufmg.br"
echo "[job] $USER @ $node | job $SLURM_JOB_ID | partição $SLURM_JOB_PARTITION | $(date)"
echo "[job] diretório: $(pwd) | commit: $(git rev-parse --short HEAD 2>/dev/null || echo '?')"
echo "[job] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-?}"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader ${CUDA_VISIBLE_DEVICES:+-i $CUDA_VISIBLE_DEVICES} || true
echo ""

module purge
module load pytorch/2.5.1-py3.10-cuda12.8

# Venv por nó, separado de outros projetos. --system-site-packages reaproveita o torch do módulo,
# então o `torch==2.5.1` do requirements.txt já sai satisfeito e não é baixado de novo.
# O módulo exporta PYTHONPATH apontando para o site-packages dele, e o PYTHONPATH vem antes do venv
# no sys.path: sem o ajuste abaixo, o pip e o numpy do módulo escondem as versões do venv.
# O flock serializa a criação/instalação: dois jobs no mesmo nó não instalam no venv ao mesmo tempo.
VENV="/tmp/venv_tp2_${USER}"
echo "[job] PYTHONPATH do módulo: ${PYTHONPATH:-(vazio)}"
export PYTHONPATH="$VENV/lib/python3.10/site-packages${PYTHONPATH:+:$PYTHONPATH}"
(
    flock 9
    if [ -d "$VENV" ] && ! "$VENV/bin/python" -m pip --version >/dev/null 2>&1; then
        echo "[job] venv quebrado em $VENV; recriando"
        rm -rf "$VENV"
    fi
    if [ ! -d "$VENV" ]; then
        echo "[job] criando venv em $VENV"
        python3.10 -m venv --system-site-packages "$VENV"
    fi
    "$VENV/bin/python" -m pip install --quiet --disable-pip-version-check -r requirements.txt
) 9>"$VENV.lock"
source "$VENV/bin/activate"
python -c "import torch, numpy; print('[job] torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0), '| numpy', numpy.__version__)"

echo "[job] comando: $*"
echo "----------------------------------------------------"
srun "$@"
