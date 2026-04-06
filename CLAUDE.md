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
.venv\Scripts\activate       # Activate venv (Windows)
```

### Running Tests
```bash
uv run pytest                          # All unit tests
uv run pytest tests/test_losses.py     # Single test file
uv run pytest -k "test_triplet"        # Single test by name
uv run python _integration_test.py    # End-to-end integration test (requires Ollama)
```

### Running Experiments
```bash
# Experiment scripts are in experiments/ — run with uv
uv run python experiments/01_data_exploration.py
uv run python experiments/02_build_graphs.py
uv run python experiments/03_train_er_without_sm.py
uv run python experiments/05_hpo.py
uv run python experiments/06_train_final.py
uv run python experiments/07_real_data_test.py
uv run python experiments/08_build_unified_graph.py
uv run python experiments/09_train_gat.py
uv run python experiments/10_evaluate.py
uv run python experiments/11_train_gat_bce.py
uv run python experiments/12_label_real_data.py
uv run python experiments/13_label_with_agent.py   # требует gemma4:26b + интернет
```

### Experiment Tracking
```bash
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db  # View results at http://localhost:5000
```

### External Services Required
- **Ollama** must be running locally for column embedding generation (`OllamaConfig.host = "http://localhost:11434"`)
- Default LLM: `qwen3.5:9b`, default embedding model: `qwen3-embedding:8b`

## Architecture

### Three-Stage Pipeline

```
CSV Tables
  → [embedding_generation.py]  rubert-tiny2 (row/token, 312-dim) + Ollama qwen3-embedding:8b (column, 4096-dim)
  → [graph_builder.py]         HeteroData: row nodes, token nodes, token→row edges (col embeddings as attr)
  → [schema_trainer.py]        SchemaProjector: 4096→256-dim column metric space
  → [er_trainer.py]            EntityResolutionGNN: row embeddings → cosine similarity → duplicate detection
```

### Model Architecture

**SchemaProjector** (`models/schema_matching.py`): MLP проецирует 4096-dim column embeddings в 256-dim метрическое пространство с L2-нормализацией. Обучается на Triplet Loss по парам столбцов.

**EntityResolutionGNN** (`models/entity_resolution.py`): Гетерогенный GNN с `L` слоями `EdgeMeanConv`. Узлы: `row` (312-dim CLS) и `token` (312-dim vocab). Рёбра `token→row` несут column embeddings (4096-dim) как атрибуты. Финальные row embeddings используются для поиска дубликатов через косинусное сходство.

**EdgeMeanConv** (`models/gnn_layer.py`): Message passing с mean агрегацией, использует edge features (token→row и опционально обратное направление).

**GATLayer** (`models/gat_layer.py`): GATv2Conv с edge features и multi-head attention. Использует GraphNorm вместо LayerNorm. Поддерживает bidirectional (token→row + row→token).

### Key Design Decisions

- **Два типа эмбеддингов строк**: CLS-токен rubert-tiny2 (312-dim) для узлов row, vocabulary embeddings того же rubert для узлов token
- **Column embeddings как атрибуты рёбер**: Позволяет модели учитывать контекст столбца при агрегации
- **Multi-dataset training**: `train_entity_resolution_multidataset()` обучает модель round-robin на нескольких датасетах — ключевой механизм transfer learning
- **IDF-фильтрация токенов**: В `graph_builder.py` токены фильтруются по IDF для снижения шума и размера графа

### Data Flow Details

- Датасеты: Magellan Data Repository (20+ бенчмарков — beer, bikes, movies, books и др.)
- Предвычисленные данные хранятся в `data/embeddings/` и `data/graphs/` (исключены из git)
- Обученные модели сохраняются в `output/` (исключены из git, `*.pt`/`*.pth`)
- Эксперименты трекируются через MLflow (`mlflow.db`)

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
| `ollama_client.py` | Обёртка над Ollama API |
| `dataset/download.py` | Скачивание с Magellan Data Repository |
| `dataset/embedding_generation.py` | rubert-tiny2 + Ollama embeddings |
| `dataset/graph_builder.py` | Построение HeteroData графов |
| `dataset/pair_sampling.py` | Генерация пар/триплетов для обучения |
| `dataset/data_split.py` | Стратифицированный split по связным компонентам |
| `dataset/schema_augmentation.py` | LLM-синонимы столбцов (аугментация) |
| `dataset/value_corruption.py` | Порча значений ячеек (typo, format, drop) |
| `models/schema_matching.py` | SchemaProjector |
| `models/entity_resolution.py` | EntityResolutionGNN |
| `models/gnn_layer.py` | EdgeMeanConv layer |
| `models/gat_layer.py` | GATv2 layer с edge features |
| `models/losses.py` | TripletLoss + полу-жёсткий майнинг |
| `evaluation/clustering.py` | Threshold sweep (F1), ROC-AUC, HDBSCAN clustering |
| `training/schema_trainer.py` | Обучение SchemaProjector |
| `training/er_trainer.py` | Обучение ER модели (single/multi-dataset) |
