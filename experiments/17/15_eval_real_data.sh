#!/usr/bin/env bash
# Exp 17 — Оценка обученной GAT-модели на реальных данных.
#
# Шаг 1: exp 07 — построить кэш реального графа под bge-m3 + посчитать
#        unsupervised метрики (category consistency, reciprocal, separation).
# Шаг 2: exp 15 — supervised метрики на gold (398 ручных) + silver (26K LLM) парах.
#
# Результаты:
#   output/07_real_data_test_bge-m3/results.json                — unsupervised
#   output/bge-m3/v17_views_gat_model_real_trusted_eval.json    — supervised
#
# Использование:
#   bash experiments/17/15_eval_real_data.sh                       # ntxent (дефолт)
#   bash experiments/17/15_eval_real_data.sh bce                   # bce-модель
#   bash experiments/17/15_eval_real_data.sh ntxent --skip-step1   # пропустить exp 07
#                                                                  # если кэш уже есть

set -euo pipefail

LOSS="${1:-ntxent}"
shift || true

MODEL_TAG="bge-m3"
ROW_MODEL="BAAI/bge-m3"
TARGET_DIM=1024
GRAPH_CACHE_DIR="output/07_real_data_test_${MODEL_TAG}"

if [[ "$LOSS" == "bce" ]]; then
    MODEL_PATH="output/${MODEL_TAG}/v17_views_gat_bce_model.pt"
    BCE_FLAG="--bce"
else
    MODEL_PATH="output/${MODEL_TAG}/v17_views_gat_model.pt"
    BCE_FLAG=""
fi

SKIP_STEP1=false
for arg in "$@"; do
    case "$arg" in
        --skip-step1) SKIP_STEP1=true ;;
    esac
done

if [[ ! -f "$MODEL_PATH" ]]; then
    echo "ОШИБКА: модель не найдена: $MODEL_PATH" >&2
    echo "Сначала запусти: uv run python -m experiments.17.12_train${LOSS:+ --loss $LOSS}" >&2
    exit 1
fi

echo "=== Exp 17 / Real-Data Eval ==="
echo "Модель:     $MODEL_PATH"
echo "Loss:       $LOSS"
echo "Граф-кэш:   $GRAPH_CACHE_DIR"
echo "Skip step1: $SKIP_STEP1"
echo

# -------------------------------------------------------------------- #
# Шаг 1 — Реальный граф + unsupervised метрики
# -------------------------------------------------------------------- #
if [[ "$SKIP_STEP1" == "false" ]]; then
    echo "[1/2] exp 07: реальный граф (bge-m3) + unsupervised метрики..."
    uv run python -m experiments.07_real_data_test \
        --row-model-name "$ROW_MODEL" \
        --target-col-dim "$TARGET_DIM" \
        --output-dir "$GRAPH_CACHE_DIR" \
        --model-path "$MODEL_PATH" \
        --trust-remote-code
else
    echo "[1/2] пропускаем exp 07 (--skip-step1)"
    if [[ ! -f "$GRAPH_CACHE_DIR/graph.pt" ]]; then
        echo "ОШИБКА: нет кэшированного графа в $GRAPH_CACHE_DIR" >&2
        echo "Запусти без --skip-step1." >&2
        exit 1
    fi
fi

# -------------------------------------------------------------------- #
# Шаг 2 — Supervised метрики на gold/silver
# -------------------------------------------------------------------- #
echo
echo "[2/2] exp 15: supervised eval на gold + silver метках..."
uv run python -m experiments.15_test_on_real_labels \
    --model "$MODEL_PATH" \
    --mrl --target-dim "$TARGET_DIM" --no-input-projection \
    --graph-dir "$GRAPH_CACHE_DIR" \
    $BCE_FLAG

echo
echo "=== Готово ==="
echo "Unsupervised: $GRAPH_CACHE_DIR/results.json"
echo "Supervised:   ${MODEL_PATH%.pt}_real_trusted_eval.json"
echo "Смотреть:     experiments/17/16_view_real_eval.ipynb"
