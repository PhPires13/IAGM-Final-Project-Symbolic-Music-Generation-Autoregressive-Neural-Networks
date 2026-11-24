#!/usr/bin/env bash
# Baixa os dois conjuntos de dados (CC-BY 4.0, Zenodo) para a raiz do repositório.
#   EMOPIA+ (Huang et al., 2024): https://zenodo.org/records/13122742   ~15 MB
#   Pop1K7  (Hsiao et al., 2021): https://zenodo.org/records/13167761   ~300 MB
# Uso: ./scripts/download_data.sh   (pula o que já existe com o tamanho certo)
set -euo pipefail
cd "$(dirname "$0")/.."

fetch() {
    local file="$1" url="$2" size="$3"
    if [ -f "$file" ] && [ "$(stat -c %s "$file")" = "$size" ]; then
        echo "$file já existe"
        return
    fi
    echo "baixando $file ..."
    curl -fL --retry 3 -o "$file.part" "$url"
    mv "$file.part" "$file"
    [ "$(stat -c %s "$file")" = "$size" ] || { echo "tamanho inesperado em $file" >&2; exit 1; }
}

fetch "EMOPIA+.zip" "https://zenodo.org/api/records/13122742/files/EMOPIA+.zip/content" 14835865
fetch "Pop1K7.zip"  "https://zenodo.org/api/records/13167761/files/Pop1K7.zip/content"  302666839
echo "ok"
