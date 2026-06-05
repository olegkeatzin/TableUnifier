# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**TableUnifier** — система автоматического объединения таблиц с использованием сопоставления схем (schema matching) и разрешения сущностей (entity resolution). Написана как дипломная работа. Pipeline: скачивание датасетов → генерация эмбеддингов → построение графов → обучение моделей.

## Commands

### Environment & Dependencies
```bash
# Package manager: uv (not pip)
uv sync                      # Install all dependencies
uv add <package>             # Add new dependency
source .venv/bin/activate    # Activate venv (Linux/Mac)
```

### Running Tests
```bash
uv run pytest                          # All unit tests
uv run pytest tests/test_losses.py     # Single test file
uv run pytest -k "test_triplet"        # Single test by name
uv run python _integration_test.py    # End-to-end integration test (requires Ollama)
```

### Веб-приложение (инференс)
```bash
# FastAPI + статика фронта на http://localhost:8000
uv run python -m table_unifier.server.main --host 0.0.0.0 --port 8000
```
Требует Ollama по адресу `OllamaConfig.host` (для column embeddings qwen3-embedding:8b)
и чекпоинт `output/bge-m3/v14_mrl_gat_model.pt` (+ опц. `.config.json` рядом с ним).
Фронт раздаётся из `src/table_unifier/server/static/` (HTML+React через CDN, без сборки).
Серверный код изолирован в `src/table_unifier/server/`, остальной pipeline не трогает.

### Running Experiments
```bash
# Основные эксперименты
uv run python experiments/01_data_exploration.py
uv run python experiments/02_build_graphs.py
uv run python experiments/14_build_unified_graph_mrl.py
uv run python experiments/14_train_gat_mrl.py
uv run python experiments/16_benchmark_row_models.py --dry-run   # бенчмарк 8 токен-моделей
uv run python experiments/13_label_with_agent.py   # требует gemma4:26b + интернет

# Exp 18 — запускается как модуль (есть дефис в имени файла)
uv run python -m experiments.18_ga_hdbscan_bge_m3

# Подготовка русских датасетов (exp 17)
cd experiments/17 && uv run python prepare.py          # все датасеты
cd experiments/17 && uv run python prepare.py --only lamoda cars_ru  # выборочно
# Ноутбуки exp 17 (запускать на nvidia-server, данных локально нет):
#   02_prepare.ipynb  — обзор подготовленных датасетов
#   03_eda_v2.ipynb   — EDA: качество natural labels, format divergence, column dropout
cd experiments/17 && uv run python 04_synth_pairs.py                        # natural pairs (auto_ru, ozon) + базовая синтетика
cd experiments/17 && uv run python 04_synth_pairs.py --only auto_ru ozon    # выборочно
cd experiments/17 && uv run python 05_generate_synonyms.py    # LLM-синонимы колонок и значений (требует Ollama)
cd experiments/17 && uv run python 06_build_views.py          # N=4 supplier views → C(4,2)=6 датасетов на источник
cd experiments/17 && uv run python 06_build_views.py --only lamoda cars_ru  # выборочно

# Exp 17 — обучение bge-m3 на view-pair данных (запускать на nvidia-server):
# Шаг 1: генерация эмбеддингов для exp17 датасетов + Magellan (bge-m3 row + qwen3 column)
uv run python -m experiments.17.10_gen_embeddings                        # всё
uv run python -m experiments.17.10_gen_embeddings --skip-magellan        # только exp17
uv run python -m experiments.17.10_gen_embeddings --skip-columns         # только row emb
# Шаг 2: построение unified-графа (Magellan in-domain + exp17 view-pairs, без devices)
uv run python -m experiments.17.11_build_graph                           # bge-m3, target-dim=1024
uv run python -m experiments.17.11_build_graph --include-devices         # включить trivial devices
# Шаг 3: обучение GAT (ntxent или bce)
uv run python -m experiments.17.12_train                                 # ntxent, 500 эпох
uv run python -m experiments.17.12_train --loss bce --patience 30
# Eval: real data (требует пересборки графа с bge-m3)
uv run python -m experiments.07_real_data_test \
    --row-model-name BAAI/bge-m3 --target-col-dim 1024 \
    --output-dir output/07_real_data_test_bge-m3 --trust-remote-code
uv run python -m experiments.15_test_on_real_labels \
    --model output/bge-m3/v17_views_gat_model.pt \
    --mrl --target-dim 1024 --no-input-projection \
    --graph-dir output/07_real_data_test_bge-m3
```

### Миграция существующих данных в namespace-раскладку
```bash
bash scripts/migrate_to_model_tag.sh --dry-run   # посмотреть, что двинется
bash scripts/migrate_to_model_tag.sh             # выполнить (идемпотентен)
```

### Experiment Tracking
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # View results at http://localhost:5000
```

### Data Sync (между устройствами — rclone → Яндекс Диск)
```bash
# После работы — загрузить
rclone sync data/ yandex:TableUnifier/data/ -P
rclone sync output/ yandex:TableUnifier/output/ -P
rclone copy mlflow.db yandex:TableUnifier/ -P

# Перед работой — скачать
rclone sync yandex:TableUnifier/data/ data/ -P
rclone sync yandex:TableUnifier/output/ output/ -P
rclone copy yandex:TableUnifier/mlflow.db . -P
```
Код — через git. Данные (`data/`, `output/`, `mlflow.db`) — через rclone. Подробнее: `SYNC.md`.

### External Services Required
- **Ollama** must be reachable at `OllamaConfig.host` for column embedding generation. В этом проекте Ollama/GPU живёт на удалённом nvidia-server — не запускай локально.
- Default LLM: `qwen3.5:9b`, default embedding model: `qwen3-embedding:8b`

## Architecture

### Three-Stage Pipeline

```
CSV Tables
  → [embedding_generation.py]  TokenEmbedder (row/token, HF-модель, pooling=cls|mean) +
                               Ollama qwen3-embedding:8b (column, 4096-dim)
  → [graph_builder.py]         HeteroData: row nodes, token nodes, token→row edges (col embeddings as attr)
  → [schema_trainer.py]        SchemaProjector: 4096→256-dim column metric space
  → [er_trainer.py]            EntityResolutionGNN: row embeddings → cosine similarity → duplicate detection
```

Дефолтная токен-модель — `cointegrated/rubert-tiny2` (312-dim). В exp 16 сравниваются 8 моделей (MiniLM-384, e5-base-768, e5-large-1024, LaBSE-en-ru-768, sbert_large_nlu_ru-1024, gte-multilingual-768, bge-m3-1024, rubert-tiny2-312) на MRL + NT-Xent пайплайне. Лучшая модель по результатам — **bge-m3**.

### Model Architecture

**SchemaProjector** (`models/schema_matching.py`): MLP проецирует 4096-dim column embeddings в 256-dim метрическое пространство с L2-нормализацией. Обучается на Triplet Loss по парам столбцов.

**EntityResolutionGNN** (`models/entity_resolution.py`): Гетерогенный GNN с `L` слоями `GNNLayer`. Узлы: `row` (hidden size токен-модели, pooled из `last_hidden_state`) и `token` (vocabulary embeddings). Рёбра `token→row` несут column embeddings (4096-dim, либо MRL-обрезка до hidden) как атрибуты. Финальные row embeddings используются для поиска дубликатов через косинусное сходство.

**EntityResolutionGAT** (`models/entity_resolution.py`): Альтернативная архитектура — то же, но использует GATv2Conv вместо EdgeMeanConv. Эксперименты 09, 11 тренируют эту модель.

**GNNLayer / EdgeMeanConv** (`models/gnn_layer.py`): `GNNLayer` — основной слой GNN (token→row + опционально row→token), внутри использует `EdgeMeanConv` (MessagePassing с mean агрегацией и edge features).

**GATLayer** (`models/gat_layer.py`): GATv2Conv с edge features и multi-head attention. Использует GraphNorm вместо LayerNorm. Поддерживает bidirectional (token→row + row→token).

**Losses** (`models/losses.py`): TripletLoss с полу-жёстким майнингом (`mine_semi_hard`) + NT-Xent (InfoNCE) loss. ER trainer поддерживает оба.

### Key Design Decisions

- **Два типа эмбеддингов строк**: pooled `last_hidden_state` токен-модели (hidden size: 312/384/768/1024) для узлов row; её же `get_input_embeddings()` для узлов token
- **Column embeddings как атрибуты рёбер**: Позволяет модели учитывать контекст столбца при агрегации
- **Unified graph (winning approach)**: все датасеты объединяются в один большой HeteroData с глобальным token-вокабуляром (exp 08 → exp 14 с MRL). Это даёт **лучшие результаты, чем round-robin** обучение по per-dataset графам. Round-robin (`train_entity_resolution_multidataset()`) остаётся в коде как baseline, но новые эксперименты строят unified.
- **IDF-фильтрация токенов**: В `graph_builder.py` токены фильтруются по IDF для снижения шума и размера графа
- **Namespace по токен-модели**: row-эмбеддинги, графы и чекпоинты лежат под `<model_tag>/`, чтобы одновременно хранить артефакты нескольких моделей (см. `src/table_unifier/paths.py`). Column-эмбеддинги qwen3 — shared, не зависят от row-модели.
- **HDBSCAN + GA (exp 18)**: вместо порогового поиска на косинусном сходстве — кластеризация HDBSCAN с параметрами, подобранными генетическим алгоритмом (DEAP) по F1 на val pairs.

### Data Flow Details

- **Синтетические датасеты**: Magellan Data Repository (20+ бенчмарков — beer, bikes, movies, books и др.), хранятся в `data/synthetic/`
- **Русские датасеты** (exp 17): 6 датасетов с Kaggle (Lamoda, cars.ru, Ozon, auto.ru ×2, DeviceStatus 15K). Скачиваются через `kagglehub`, подготовленные parquet-файлы — `data/raw_ru/<name>/clean.parquet`. Скрипт подготовки: `experiments/17/prepare.py`

  **Natural labels** (результаты `03_eda_v2.ipynb`):
  - `auto_ru × auto_ru_2020`: 16,955 sell_id пар — два независимых скрейпа auto.ru с разными парсерами. **sell_id обязательно дропать из признаков** (trivial leakage — 100% совпадает в парах). Форматные расхождения: `color` (текст vs HEX), `bodyType` (регистр). 170/365 общих колонок совпадают точно, медиана divergence 0.1% — реалистичная, но умеренная задача.
  - `ozon`: 318,850 cross-file URL пар (один URL в 34 разных месячных выгрузках). 100% пар cross-file, 95.6% имеют разные счётчики избранного → валидные natural labels.
  - `lamoda`: `about.Артикул` уникален (0 дублей из 10,318) — натуральных пар нет. Title+Brand FP rate 65.3% — мусор. Только синтетика.
  - `cars_ru`: нет VIN, description дублей 2.6% — natural pairs нет. Только синтетика.
  - `devices`: синтетический датасет (37% model_id разделяют >1 производитель). Только синтетика.

  **column_dropout в реальных парах** (auto_ru × auto_ru_2020): null-divergence медиана 0.0% — колонки не выпадают. Значит агрессивный dropout нужно симулировать в `04_synth_pairs.py`.
- Предвычисленные данные хранятся в `data/` (исключены из git, синхронизируются через rclone)
- Обученные модели сохраняются в `output/<model_tag>/` (исключены из git, `*.pt`/`*.pth`)
- Эксперименты трекируются через MLflow (`mlflow.db`)

Раскладка `data/` после миграции (см. `scripts/migrate_to_model_tag.sh`):

```
data/
├── synthetic/<dataset>/           # таблицы A/B + train/valid/test — shared
├── raw_ru/<name>/clean.parquet    # русские датасеты (exp 17)
├── embeddings/
│   ├── columns/<dataset>/         # column_embeddings_{a,b}.npz + columns_{a,b}.csv (qwen3, shared)
│   └── rows/<model_tag>/<dataset>/  # row_embeddings_{a,b}.npy (per-модель)
└── graphs/<model_tag>/
    ├── <dataset>/                 # per-dataset графы (exp 02)
    ├── v3_unified/                # unified, exp 08
    ├── v14_mrl/                   # MRL unified, exp 14
    └── v14_mrl_cross/<dataset>/   # cross-domain, exp 14
output/<model_tag>/
    ├── v14_mrl_gat_model.pt
    ├── v18_ga_hdbscan_results.json  # exp 18
    └── *.config.json
```

`model_tag` задаётся через `EntityResolutionConfig.token_model_tag` или CLI-флаг `--model-tag`. Дефолт — `rubert-tiny2`; текущая лучшая модель — `bge-m3`.

### Config System

Все параметры — в датаклассах `src/table_unifier/config.py`:
- `OllamaConfig`: хост, модели LLM и embedding
- `SchemaMatchingConfig`: размерности, lr, epochs для SchemaProjector
- `EntityResolutionConfig`: размеры слоёв, число GNN-слоёв, параметры обучения

### Module Map

All paths relative to `src/table_unifier/`:

| Модуль | Назначение |
|--------|-----------|
| `config.py` | Все конфиги как датаклассы |
| `paths.py` | Namespaced-раскладка: `columns_dir`, `rows_dir`, `unified_dir`, `output_dir_for` |
| `ollama_client.py` | Обёртка над Ollama API |
| `dataset/download.py` | Скачивание с Magellan Data Repository |
| `dataset/embedding_generation.py` | `TokenEmbedder` (HF-модель, cls/mean pooling, префикс) + Ollama column embeddings |
| `dataset/graph_builder.py` | Построение HeteroData графов |
| `dataset/pair_sampling.py` | Генерация пар/триплетов для обучения |
| `dataset/data_split.py` | Стратифицированный split по связным компонентам |
| `dataset/schema_augmentation.py` | LLM-синонимы столбцов (аугментация) |
| `dataset/value_corruption.py` | Порча значений ячеек (typo, format, drop) |
| `models/schema_matching.py` | SchemaProjector |
| `models/entity_resolution.py` | EntityResolutionGNN + EntityResolutionGAT |
| `models/gnn_layer.py` | `GNNLayer` (основной) + `EdgeMeanConv` (inner MessagePassing) |
| `models/gat_layer.py` | GATv2 layer с edge features |
| `models/losses.py` | TripletLoss + полу-жёсткий майнинг |
| `evaluation/clustering.py` | Threshold sweep (F1), ROC-AUC, HDBSCAN clustering |
| `evaluation/ga_hdbscan.py` | HDBSCAN + DEAP GA подбор параметров (exp 18); backends cpu/gpu/auto |
| `evaluation/ga_cc.py` | Connected-components кластеризация + DEAP GA (гены `tau`, `top_k`) |
| `evaluation/ga_common.py` | Общая F-beta фитнес-функция + pair-метрики для обоих ГА-кластеризаторов |
| `training/schema_trainer.py` | Обучение SchemaProjector |
| `training/er_trainer.py` | Обучение ER модели: `train_entity_resolution` (single), `train_entity_resolution_multidataset` (round-robin baseline), `train_entity_resolution_minibatch` (unified graph, exps 14/18, основной путь), `train_entity_resolution_bce` (BCE variant); `get_row_embeddings` + `find_duplicates` для инференса |

### Server (инференс-API)

`src/table_unifier/server/` изолирован от pipeline. `main.py` (`create_app()`) собирает FastAPI и монтирует фронт из `static/`. Структура:
- `routes/` — эндпоинты: `sources` (загрузка таблиц), `graph` (построение/просмотр графа), `infer` (запуск инференса), `clusters` (результаты дедупликации), `export` (выгрузка).
- `services/` — `inference.py` (прогон чекпоинта), `progress.py` (SSE-шина `bus` для прогресса), `storage.py` (хранение загруженных данных).
- `models.py` — Pydantic-модели запросов/ответов.

### Testing

Тесты в `tests/` не требуют Ollama и работают на синтетических данных. Общие фикстуры (`conftest.py`):
- `table_a`, `table_b`, `labels_df` — мини-таблицы (3 строки) для unit-тестов
- `column_embeddings` — случайные 64-dim эмбеддинги столбцов
- `small_hetero_data` — минимальный HeteroData граф (6 row, 10 token, 20 edges) с маленькими размерностями (row_dim=32, token_dim=32, col_dim=64) — используется в тестах моделей

При создании новых тестов модели используй `small_hetero_data` и передавай соответствующие размерности (32/32/64), а не дефолтные из конфига (312/312/4096).
