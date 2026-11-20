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
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
echo ""

module purge
module load pytorch/2.5.1-py3.10-cuda12.8

# Venv por nó, separado de outros projetos. --system-site-packages reaproveita o torch do módulo,
# então o `torch==2.5.1` do requirements.txt já sai satisfeito e não é baixado de novo.
VENV="/tmp/venv_tp2_${USER}"
if [ ! -d "$VENV" ]; then
    echo "[job] criando venv em $VENV"
    python3.10 -m venv --system-site-packages "$VENV"
fi
source "$VENV/bin/activate"
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt
python -c "import torch; print('[job] torch', torch.__version__, '| cuda', torch.version.cuda, '| gpu', torch.cuda.get_device_name(0))"

echo "[job] comando: $*"
echo "----------------------------------------------------"
srun "$@"
