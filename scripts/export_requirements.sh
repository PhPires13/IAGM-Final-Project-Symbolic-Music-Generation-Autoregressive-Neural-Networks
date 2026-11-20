#!/usr/bin/env bash
# Gera requirements.txt a partir do uv.lock (rodar depois de qualquer `uv add`/`uv lock`).
# As linhas do torch são trocadas por um `torch==2.5.1` sem marcador nem sufixo +cpu:
# no cluster ele é satisfeito pelo torch do módulo (venv com --system-site-packages),
# e em outra máquina o pip baixa o torch padrão do PyPI.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "# Gerado por scripts/export_requirements.sh a partir do uv.lock. Não editar à mão."
  uv export --format requirements-txt --no-hashes --no-emit-project --no-header --no-annotate \
    | grep -v '^torch=='
  echo "torch==2.5.1"
} > requirements.txt
echo "requirements.txt: $(grep -vc '^#' requirements.txt) pacotes"
