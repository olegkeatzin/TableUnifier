"""
Унифицированная генерация данных для Schema Matching и Entity Resolution.

Модуль объединяет процесс генерации данных: один проход создаёт
датасет, пригодный для обучения обоих модулей.

Архитектура:
    ┌─────────────────────────────────────────────────────────────────┐
    │                 EntityPool (канонические сущности)              │
    │                          ↓                                      │
    │    ┌──────────── TableGenerator ────────────┐                   │
    │    │                                        │                   │
    │    │  ER пары таблиц        Extra SM таблицы│                   │
    │    │  (train/val/test)      (standalone)     │                   │
    │    └────────┬──────────────────┬─────────────┘                  │
    │             │                  │                                 │
    │             ▼                  ▼                                 │
    │    ┌─── OllamaAccelerator ──────────────────┐                   │
    │    │  • auto batch_size calibration          │                   │
    │    │  • keep_alive (GPU residency)           │                   │
    │    │  • parallel LLM descriptions            │                   │
    │    │  • num_predict / num_ctx optimization   │                   │
    │    └────────┬──────────────────┬─────────────┘                  │
    │             │                  │                                 │
    │    SM Dataset (JSON)     ER Dataset (CSV + meta)                │
    │    column embeddings     table pairs + ground truth             │
    └─────────────────────────────────────────────────────────────────┘

Компоненты:
    - DataGenConfig:            конфигурация всех параметров
    - COLUMN_TEMPLATES:         единый реестр шаблонов столбцов
    - TableGenerator:           генератор таблиц (standalone + ER пары)
    - EntityPool:               пул сущностей для ER дубликатов
    - ColumnNameVariator:       вариатор названий столбцов
    - OllamaAccelerator:        ускорение Ollama (auto batch, concurrency)
    - UnifiedDatasetGenerator:  главный генератор

Использование:
    from table_unifier.data_generation import DataGenConfig, UnifiedDatasetGenerator

    config = DataGenConfig(
        ollama_host="http://100.74.62.22:11434",
        auto_batch_size=True,
        num_train_pairs=300,
        num_extra_sm_tables=200,
    )
    generator = UnifiedDatasetGenerator(config)
    generator.generate()
"""

from .config import DataGenConfig
from .columns import (
    COLUMN_TEMPLATES,
    MANDATORY_COLUMNS,
    OPTIONAL_COLUMNS,
    ER_REQUIRED_ENTITY_COLUMNS,
    build_column_templates,
    build_column_lists,
    get_entity_columns,
    perturb_string,
    perturb_number,
    perturb_value,
    ColumnNameVariator,
    EntityPool,
    TablePairData,
    TableGenerator,
)
from .ollama_utils import OllamaAccelerator
from .generator import UnifiedDatasetGenerator
from .magellan_loader import MagellanConfig, MagellanDatasetLoader, MAGELLAN_DATASETS

__all__ = [
    # Config
    'DataGenConfig',
    'MagellanConfig',
    
    # Column templates
    'COLUMN_TEMPLATES',
    'MANDATORY_COLUMNS',
    'OPTIONAL_COLUMNS',
    'ER_REQUIRED_ENTITY_COLUMNS',
    'build_column_templates',
    'build_column_lists',
    'get_entity_columns',
    
    # Perturbation
    'perturb_string',
    'perturb_number',
    'perturb_value',
    
    # Generators
    'ColumnNameVariator',
    'EntityPool',
    'TablePairData',
    'TableGenerator',
    
    # Ollama
    'OllamaAccelerator',
    
    # Main
    'UnifiedDatasetGenerator',
    'MagellanDatasetLoader',
    'MAGELLAN_DATASETS',
]
