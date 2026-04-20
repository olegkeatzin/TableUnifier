#!/usr/bin/env bash
# Миграция data/ и output/ в namespace по токен-модели.
#
# До:   data/embeddings/<ds>/row_embeddings_*.npy      (rubert)
#       data/embeddings/<ds>/column_embeddings_*.npz   (qwen3)
#       data/graphs/<ds>/...
#       data/graphs/unified/..., v3_unified/..., v14_mrl/..., v14_mrl_cross/...
#       output/*.pt
#
# После: data/embeddings/columns/<ds>/                 (shared qwen3)
#        data/embeddings/rows/<TAG>/<ds>/              (per-model)
#        data/graphs/<TAG>/<ds>/..., data/graphs/<TAG>/{unified,v3_unified,v14_mrl,...}/
#        output/<TAG>/*.pt
#
# Запуск:
#   bash scripts/migrate_to_model_tag.sh --dry-run   # посмотреть, что двинется
#   bash scripts/migrate_to_model_tag.sh             # выполнить
#   bash scripts/migrate_to_model_tag.sh --tag my-tag
#
# Идемпотентен: если что-то уже переложено, повторный запуск ничего не сломает.

set -euo pipefail

TAG="rubert-tiny2"
DRY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run) DRY=1; shift ;;
        --tag)     TAG="$2"; shift 2 ;;
        -h|--help)
            sed -n '2,22p' "$0"; exit 0 ;;
        *) echo "Unknown arg: $1" >&2; exit 1 ;;
    esac
done

if [[ ! -d data ]] || [[ ! -d output ]]; then
    echo "Запусти из корня TableUnifier (где лежат data/ и output/)" >&2
    exit 1
fi

run() {
    if [[ $DRY -eq 1 ]]; then
        echo "DRY: $*"
    else
        eval "$@"
    fi
}

echo "=== TAG=$TAG  DRY=$DRY ==="

# -------------------------------------------------------------
# 1) Row embeddings -> data/embeddings/rows/<TAG>/<ds>/
# -------------------------------------------------------------
echo "--- [1/4] row embeddings ---"
run "mkdir -p 'data/embeddings/rows/$TAG'"
shopt -s nullglob
for ds in data/embeddings/*/; do
    name=$(basename "$ds")
    case "$name" in columns|rows) continue ;; esac
    [[ -f "$ds/row_embeddings_a.npy" ]] || continue
    run "mkdir -p 'data/embeddings/rows/$TAG/$name'"
    run "mv '$ds/row_embeddings_a.npy' 'data/embeddings/rows/$TAG/$name/'"
    run "mv '$ds/row_embeddings_b.npy' 'data/embeddings/rows/$TAG/$name/'"
done
shopt -u nullglob

# -------------------------------------------------------------
# 2) Column embeddings + columns_*.csv -> data/embeddings/columns/<ds>/  (shared)
# -------------------------------------------------------------
echo "--- [2/4] column embeddings ---"
run "mkdir -p data/embeddings/columns"
shopt -s nullglob
for ds in data/embeddings/*/; do
    name=$(basename "$ds")
    case "$name" in columns|rows) continue ;; esac
    run "mkdir -p 'data/embeddings/columns/$name'"
    for f in column_embeddings_a.npz column_embeddings_b.npz columns_a.csv columns_b.csv; do
        [[ -f "$ds/$f" ]] || continue
        run "mv '$ds/$f' 'data/embeddings/columns/$name/'"
    done
    # пустой каталог можно снести
    [[ -z "$(ls -A "$ds" 2>/dev/null || true)" ]] && run "rmdir '$ds'" || true
done
shopt -u nullglob

# -------------------------------------------------------------
# 3) Графы: data/graphs/<*> -> data/graphs/<TAG>/<*>
# -------------------------------------------------------------
echo "--- [3/4] graphs ---"
if [[ -d "data/graphs/$TAG" ]] && [[ $DRY -eq 0 ]]; then
    echo "  data/graphs/$TAG уже существует — считаю, что миграция графов уже сделана"
else
    TMP="data/graphs/.${TAG}_tmp"
    run "mkdir -p '$TMP'"
    shopt -s nullglob
    for g in data/graphs/*/; do
        name=$(basename "$g")
        case "$name" in ".${TAG}_tmp"|"$TAG") continue ;; esac
        run "mv '$g' '$TMP/$name'"
    done
    shopt -u nullglob
    run "mv '$TMP' 'data/graphs/$TAG'"
fi

# -------------------------------------------------------------
# 4) Обученные модели: output/*.pt -> output/<TAG>/
# -------------------------------------------------------------
echo "--- [4/4] output checkpoints ---"
run "mkdir -p 'output/$TAG'"
shopt -s nullglob
for f in output/*.pt output/*.pth output/*.config.json; do
    [[ -f "$f" ]] || continue
    run "mv '$f' 'output/$TAG/'"
done
# hpo_*.json специфичны для rubert — их тоже в тег
for f in output/hpo_*.json; do
    [[ -f "$f" ]] || continue
    run "mv '$f' 'output/$TAG/'"
done
shopt -u nullglob

echo "=== done ==="
