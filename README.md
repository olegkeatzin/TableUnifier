<div align="center">

# 🔄 TableUnifier

**Интеллектуальная система для Schema Matching и Entity Resolution с применением GNN и LLM**

[![Python 3.12+](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.8+-ee4c2c.svg)](https://pytorch.org)
[![PyG](https://img.shields.io/badge/PyG-2.5+-3C2179.svg)](https://pyg.org)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20LLM-orange.svg)](https://ollama.ai/)
[![Tests](https://img.shields.io/badge/tests-67%20passed-brightgreen.svg)](tests/)

</div>

---

TableUnifier — система для автоматического объединения таблиц с разными схемами и поиска дубликатов строк между ними. Основана на локальных LLM (Ollama), метрическом обучении (Triplet Loss) и граф-нейросетях (GNN).

### Основные возможности

- 🔗 **Schema Matching** — сопоставление столбцов с разными названиями через семантические эмбеддинги и Triplet Loss
- 🔍 **Entity Resolution** — поиск дубликатов строк между таблицами с разными схемами через GNN (TransformerConv)
- 🏭 **Генерация данных** — единый Faker-based генератор синтетических датасетов с Ground Truth для обоих модулей
- 🖥️ **Локальный LLM** — Ollama (не требует API-ключей и интернета)
- ✅ **Тесты** — 67 автоматических тестов покрывают все 4 пайплайна

---

## Архитектура

```
┌──────────────────────────────────────────────────────────────────┐
│                         data_generation/                         │
│          EntityPool → ER пары + SM standalone-таблицы           │
│         LLM описания → Ollama embeddings → SM dataset            │
└─────────────────────────────┬────────────────────────────────────┘
                               │
              ┌────────────────┴────────────────┐
              ▼                                 ▼
┌─────────────────────────┐    ┌──────────────────────────────────┐
│    schema_matching/     │    │        entity_resolution/        │
│                         │    │                                  │
│  Triplet Loss Projector │    │  Row nodes: FastText BoW         │
│  [D_raw] → [256]        │    │  Token nodes: FastText (frozen)  │
│       ↓                 │    │           ↓                      │
│  Hungarian Matching     │    │  TransformerConv + edge_attr     │
│       ↓                 │    │  (column_embeddings из SM)       │
│  Column Mapping         │    │           ↓                      │
│                         │    │  Triplet Loss (row-level)        │
│  column_embeddings ─────│────│→  edge_attr → Zero-Shot ER      │
└─────────────────────────┘    │           ↓                      │
                                │  Cosine Similarity               │
                                │           ↓                      │
                                │  Duplicate Pairs                 │
                                └──────────────────────────────────┘
```

**Ключевая связь:** column embeddings, обученные модулем Schema Matching, используются как **edge attributes** в GNN для Entity Resolution. Это обеспечивает **Zero-Shot** обобщение на новые типы столбцов без переобучения.

---

## Установка

### Требования

- Python 3.9+
- [Ollama](https://ollama.ai/) (локальный LLM-сервер)
- [uv](https://astral.sh/uv) (рекомендуется) или pip

### Через uv

```bash
git clone https://github.com/olegkeatzin/TableUnifier.git
cd TableUnifier

# Базовая установка (Schema Matching + DataGen)
uv sync

# С модулем Entity Resolution (PyTorch Geometric)
uv sync --extra er

# Все зависимости
uv sync --extra all
```

### Через pip

```bash
pip install -e .          # базовая
pip install -e ".[er]"    # + Entity Resolution
pip install -e ".[all]"   # всё
```

### Настройка Ollama

```bash
# Установить: https://ollama.ai/
ollama pull qwen3-embedding:8b    # модель эмбеддингов (для column embeddings)
ollama pull qwen3.5:9b            # модель генерации описаний
```

### FastText-модель для Entity Resolution

```bash
# Скачать предобученную FastText-модель (русский язык, 300-мерная)
# https://fasttext.cc/docs/en/crawl-vectors.html
# Файл: cc.ru.300.bin (~4.5 ГБ в .bin, ~600 МБ в .gz}
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.ru.300.bin.gz
gzip -d cc.ru.300.bin.gz

# Указать путь в конфигурации:
# er_experiment_config.json → "fasttext_model_path": "cc.ru.300.bin"
```

---

## Быстрый старт

### Schema Matching

```python
import pandas as pd
from table_unifier import TableUnifier, AppConfig

config = AppConfig()               # или AppConfig.from_file("my_config.json")
config.ollama.host = "http://127.0.0.1:11434"

unifier = TableUnifier(config)

reference_df = pd.DataFrame({
    "ID": [1, 2, 3],
    "Имя": ["Иван", "Мария", "Пётр"],
    "Возраст": [25, 30, 35],
})

target_df = pd.DataFrame({
    "Номер": [1, 2, 3],
    "ФИО": ["Анна", "Сергей", "Елена"],
    "Лет": [28, 32, 45],
})

unifier.set_reference_schema(reference_df)
unified_df, metrics = unifier.unify_table(target_df)

print(unified_df)
print(f"Сопоставлено: {metrics['matched_columns']}/{metrics['total_reference_columns']}")
```

### Entity Resolution

```python
from table_unifier.entity_resolution import DuplicateDetector, ERConfig, FastTextEmbedder

config = ERConfig.load("pipelines/entity_resolution/er_experiment_config.json")

# FastText-модель загружается один раз
ft = FastTextEmbedder(config.fasttext_model_path)

detector = DuplicateDetector.from_checkpoint(
    checkpoint_path="er_dataset/checkpoints/best_model.pt",
    config=config,
    row_input_dim=300,   # FastText dim
    col_embed_dim=768,   # Ollama embedding dim
    fasttext_embedder=ft,
)

matches = detector.find_duplicates(table_a_df, table_b_df, threshold=0.7)

for match in matches:
    print(f"A[{match.row_idx_a}] ↔ B[{match.row_idx_b}]  sim={match.similarity:.3f}")
```

### Генерация датасета

```bash
python -m table_unifier --config pipelines/config.json
```

Или через Jupyter:

```bash
jupyter notebook pipelines/01_dataset_generation.ipynb
```

---

## Структура проекта

```
TableUnifier/
├── src/table_unifier/
│   ├── __init__.py              # Публичный API
│   ├── __main__.py              # CLI → UnifiedDatasetGenerator
│   ├── config.py                # AppConfig, OllamaConfig, EmbeddingConfig
│   ├── core.py                  # TableUnifier, EmbSerialV2
│   ├── models.py                # OllamaLLM, OllamaEmbedding
│   ├── data_generation/         # Генерация данных SM + ER
│   │   ├── config.py            #   DataGenConfig
│   │   ├── columns.py           #   COLUMN_TEMPLATES, EntityPool, TableGenerator
│   │   ├── ollama_utils.py      #   OllamaAccelerator
│   │   └── generator.py         #   UnifiedDatasetGenerator
│   ├── schema_matching/         # Schema Matching (Triplet Loss)
│   │   ├── config.py            #   SMConfig
│   │   ├── data_generator.py    #   SMDatasetGenerator
│   │   ├── dataset.py           #   SMDataset, load_and_split_dataset
│   │   ├── model.py             #   SchemaMatchingModel, ProjectionHead
│   │   ├── trainer.py           #   SMTrainer
│   │   └── inference.py         #   SchemaMatcherInference
│   └── entity_resolution/       # Entity Resolution (GNN)
│       ├── config.py            #   ERConfig
│       ├── token_embedder.py    #   FastTextEmbedder
│       ├── graph_builder.py     #   TablePairGraphBuilder
│       ├── gnn_model.py         #   EntityResolutionGNN (TransformerConv)
│       ├── er_data_generator.py #   ERTablePairGenerator, EntityPool
│       ├── er_dataset.py        #   ERGraphDataset, build_and_save_graphs
│       ├── trainer.py           #   ERTrainer
│       └── inference.py         #   DuplicateDetector, evaluate_on_test
├── pipelines/
│   ├── 01_dataset_generation.ipynb   # Генерация объединённого датасета
│   ├── 02_schema_matching_training.ipynb  # Обучение SM-модели
│   ├── 03_entity_resolution_training.ipynb  # Обучение GNN
│   ├── 04_full_inference.ipynb       # Инференс: SM + ER на новых парах
│   └── config.json                   # Единый конфиг для всех пайплайнов
├── tests/
│   ├── conftest.py                   # Общие фикстуры
│   ├── test_pipeline_01_config.py    # Тесты конфигурации и DataGenConfig
│   ├── test_pipeline_02_schema_matching.py  # Тесты SM (dataset, model, trainer)
│   ├── test_pipeline_03_entity_resolution.py  # Тесты ER (GNN, dataset, trainer)
│   └── test_pipeline_04_inference.py  # Тесты инференса + кросс-пайплайн
├── unified_dataset/                  # Сгенерированные данные (git-ignored)
├── pyproject.toml
└── README.md
```

---

## Модуль 1: Data Generation

Единый генератор создаёт данные для обоих модулей за один проход.

### Как это работает

1. **EntityPool** — пул канонических сущностей (товары с атрибутами)
2. **ER пары** — таблицы с контролируемым пересечением строк (дубликаты + пертурбации)
3. **SM таблицы** — standalone-таблицы с вариативными названиями столбцов
4. **OllamaAccelerator** — автоподбор `batch_size`, `keep_alive` (GPU residency), параллельные LLM-запросы

```python
from table_unifier.data_generation import DataGenConfig, UnifiedDatasetGenerator

config = DataGenConfig(
    ollama_host="http://127.0.0.1:11434",
    llm_model="qwen3.5:9b",
    embedding_model="qwen3-embedding:8b",
    auto_batch_size=True,
    num_train_pairs=300,
    num_val_pairs=50,
    num_test_pairs=50,
    num_extra_sm_tables=200,
)
generator = UnifiedDatasetGenerator(config)
generator.generate()
```

Конфигурация: [`pipelines/unified/unified_experiment_config.json`](pipelines/unified/unified_experiment_config.json).

---

## Модуль 2: Schema Matching

### Архитектура

```
Столбец: "Цена за ед."
     ↓
LLM (qwen3.5:9b) — генерация описания
     ↓
Embedding (qwen3-embedding:8b) — сырой вектор [D_raw]
     ↓
ProjectionHead (MLP) — [D_raw] → [256]  (Triplet Loss)
     ↓
L2-нормализация
     ↓
Hungarian Matching — оптимальное сопоставление столбцов
     ↓
Column Embeddings → GNN edge_attr (Entity Resolution)
```

### Запуск

```bash
jupyter notebook pipelines/02_schema_matching_training.ipynb
```

Конфигурация: [`pipelines/config.json`](pipelines/config.json) — секция `schema_matching`.

### Параметры модели

| Параметр | Значение |
|----------|---------|
| Input dim | авто-определение из данных |
| Projection dims | [512, 384] |
| Output dim | 256 |
| Dropout | 0.1 |
| Batch norm | ✅ |
| Batch size | 64 |
| Learning rate | 1e-3 |
| Margin | 0.2 |
| Mining | semihard |

---

## Модуль 3: Entity Resolution

### Архитектура GNN

```
  Таблица A          Таблица B
 ┌──────────┐       ┌──────────┐
 │ Row Nodes│       │ Row Nodes│   ← FastText BoW (среднее векторов токенов строки)
 └────┬─────┘       └─────┬────┘
      │   edge_attr =     │
      │   column_emb      │
      │   (из SM)         │
      └────────┬──────────┘
               │
        ┌──────┴──────┐
        │ Token Nodes │             ← FastText embeddings (предобученные, заморожены)
        └──────┬──────┘
               │
   ┌───────────┴───────────┐
   │ GNN (TransformerConv) │   × num_gnn_layers
   │ + Residual            │
   │ + LayerNorm           │
   └───────────┬───────────┘
               │
        JumpingKnowledge
               │
         L2 Normalize
               │
        Row Embeddings [128]
               │
        Cosine Similarity
               │
        Duplicate Pairs
```

### Ключевые компоненты

| Компонент | Описание |
|-----------|----------|
| **FastTextEmbedder** | Предобученные FastText-векторы (300-мерные). OOV через субсловные n-граммы: `sim("цена", "ценна") ≈ high`. Не требует обучения. |
| **Bag-of-Words (BoW)** | Стартовый эмбеддинг строки = среднее FastText-векторов всех токенов. Не требует LLM и Ollama. |
| **TransformerConv** | Multi-head attention с edge attributes. Двунаправленный: Token→Row, Row→Token |
| **Edge Attributes** | Column embeddings из SM — Zero-Shot обобщение на новые типы столбцов |
| **JumpingKnowledge** | Конкатенация представлений всех слоёв GNN (борьба с over-smoothing) |
| **Triplet Loss** | Semihard-майнинг на уровне строк: дубликаты притягиваются, не-дубликаты отталкиваются |

### Параметры модели

| Параметр | Значение |
|----------|---------|
| Token embed dim | 300 (FastText) |
| Hidden dim | 256 |
| Output dim | 128 |
| Num heads | 4 |
| Edge dim | 128 |
| GNN layers | 2 |
| Dropout | 0.1 |
| Margin | 0.3 |
| Mining | semihard |

### Запуск

```bash
jupyter notebook pipelines/03_entity_resolution_training.ipynb
```

Конфигурация: [`pipelines/config.json`](pipelines/config.json) — секция `entity_resolution`.

### Этапы пайплайна

| Ноутбук | Описание |
|---------|----------|
| `01_dataset_generation.ipynb` | Генерация синтетических пар таблиц с дубликатами + SM-таблиц |
| `02_schema_matching_training.ipynb` | Обучение `SchemaMatchingModel` (ProjectionHead + Triplet Loss) |
| `03_entity_resolution_training.ipynb` | Построение графов (FastText) + обучение GNN (TransformerConv) |
| `04_full_inference.ipynb` | SM: Hungarian Matching → column embeddings → ER: Duplicate Detection |

---

## Конфигурации

Все пайплайны используют единый файл [`pipelines/config.json`](pipelines/config.json) со следующими секциями:

| Секция | Описание |
|--------|----------|
| `data_generation` | Параметры генерации: модели Ollama, кол-во пар, пертурбации, шаблоны столбцов |
| `schema_matching` | Параметры SM-модели: архитектура, обучение, пути к данным |
| `entity_resolution` | Параметры GNN: hidden_dim, num_layers, стратегия майнинга, порог |
| `inference` | Параметры инференса: пороги косинусного сходства |

### AppConfig

```python
from table_unifier import AppConfig

config = AppConfig()
config.ollama.host = "http://127.0.0.1:11434"
config.ollama.embedding_model = "qwen3-embedding:8b"
config.ollama.llm_model = "qwen3.5:9b"
config.embedding.batch_size = 50
config.classifier.min_similarity = 0.6
config.to_file("my_config.json")
```

### DataGenConfig

```python
from table_unifier.data_generation import DataGenConfig

config = DataGenConfig(
    ollama_host="http://127.0.0.1:11434",
    llm_model="qwen3.5:9b",
    embedding_model="qwen3-embedding:8b",
    auto_batch_size=True,
    num_train_pairs=300,
    num_extra_sm_tables=200,
    # Расширение датасета:
    extra_product_groups=["Медикаменты", "Химреактивы"],
    extra_units=["т", "м2", "рулон"],
    custom_columns={
        "delivery_date": {
            "description": "Дата поставки",
            "variants": {"ru": ["Дата поставки", "Срок поставки"], "en": ["Delivery Date", "ETA"]},
            "data_type": "str",
            "values": ["01.04.2026", "По согласованию"],
        }
    }
)
config.save("my_datagen_config.json")
```

### ERConfig

```python
from table_unifier.entity_resolution import ERConfig

config = ERConfig(hidden_dim=256, num_gnn_layers=3, margin=0.5)
config.save("my_datagen_config.json")
```

### ERConfig

```python
from table_unifier.entity_resolution import ERConfig

config = ERConfig(hidden_dim=256, num_gnn_layers=3, margin=0.5)
config.save("my_er_config.json")
```

---

## Тесты

Проект покрыт **67 автоматическими тестами**, охватывающими все 4 пайплайна:

| Файл | Тестов | Покрытие |
|------|--------|----------|
| `test_pipeline_01_config.py` | 8 | Загрузка конфига, `DataGenConfig`: создание, сохранение, пути |
| `test_pipeline_02_schema_matching.py` | 26 | `SMDataset`, `ProjectionHead`, `SchemaMatchingModel`, `SMTrainer` |
| `test_pipeline_03_entity_resolution.py` | 17 | `GNNLayer`, `EntityResolutionGNN`, `ERGraphDataset`, `ERTrainer` |
| `test_pipeline_04_inference.py` | 16 | Загрузка SM/ER моделей, метрики, кросс-пайплайн консистентность |

### Запуск тестов

```bash
# Активировать окружение
.venv\Scripts\Activate.ps1          # Windows (PowerShell)
source .venv/bin/activate           # Linux / macOS

# Запустить все тесты
pytest tests/ -v

# Запустить конкретный модуль
pytest tests/test_pipeline_02_schema_matching.py -v
```

> **Примечание:** `pyproject.toml` настроен с `--cov` флагами (требует `pytest-cov`).  
> Без него запускайте через `pytest tests/ -v --override-ini="addopts=-v"`.

---

## Зависимости

**Базовые:**
```
pandas, numpy, scikit-learn, scipy
torch>=2.8.0, pytorch-metric-learning>=2.9.0
faiss-cpu>=1.12.0
ollama>=0.1.0
faker>=18.0.0
matplotlib, seaborn, tqdm
```

**Entity Resolution (`pip install -e ".[er]"`):**
```
torch-geometric>=2.5.0
gensim>=4.3.0
```

> **Примечание:** эмбеддинги строк (BoW) теперь вычисляются локально через FastText — Ollama для строк **не требуется**. Ollama используется только для генерации описаний столбцов.

---

## Лицензия

MIT License — см. [LICENSE](LICENSE).

---

<div align="center">

Дипломная работа — *Entity Resolution на основе GNN* | Курсовой проект — *Schema Matching*

</div>
