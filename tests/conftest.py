"""
Общие фикстуры и утилиты для тестов пайплайнов TableUnifier.
"""

import json
import os
import tempfile
import shutil

import numpy as np
import pandas as pd
import pytest
import torch


# ─────────────────── Размерности (как в реальном проекте) ───────────────────

EMBED_DIM = 768          # размерность raw embedding (условная, для тестов)
SM_OUTPUT_DIM = 256       # output проекции SM
ER_OUTPUT_DIM = 128       # output GNN ER
FASTTEXT_DIM = 300        # FastText dim
ER_HIDDEN_DIM = 256
ER_EDGE_DIM = 128
NUM_CLASSES = 5           # количество классов (base_name) в тестовых данных
SAMPLES_PER_CLASS = 10    # сэмплов на класс


# ─────────────────── Генерация синтетических данных ───────────────────

@pytest.fixture
def sm_dataset_entries():
    """Синтетический SM датасет: записи в формате JSON (как из генератора)."""
    rng = np.random.RandomState(42)
    class_names = [f"class_{i}" for i in range(NUM_CLASSES)]
    entries = []

    for cls in class_names:
        for j in range(SAMPLES_PER_CLASS):
            embedding = rng.randn(EMBED_DIM).tolist()
            entries.append({
                "base_name": cls,
                "column_name": f"{cls}_variant_{j}",
                "description": f"Description for {cls} variant {j}",
                "data_type": "str",
                "embedding": embedding,
                "source": "test",
            })

    return entries


@pytest.fixture
def sm_dataset_path(sm_dataset_entries, tmp_path):
    """Сохранённый SM датасет в JSON-файле."""
    path = tmp_path / "sm_dataset.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sm_dataset_entries, f)
    return str(path)


@pytest.fixture
def sm_metadata(tmp_path):
    """Минимальная SM метаданные."""
    meta = {
        "stats": {
            "unique_base_names": NUM_CLASSES,
            "tables_from_er": 10,
            "tables_from_extra": 5,
        }
    }
    path = tmp_path / "sm_metadata.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return str(path)


@pytest.fixture
def sample_table_a():
    """Тестовая таблица A."""
    return pd.DataFrame({
        "Артикул": ["ART-001", "ART-002", "ART-003", "ART-004", "ART-005"],
        "Наименование": ["Болт М8", "Гайка М8", "Шайба М8", "Винт М6", "Анкер М10"],
        "Цена": [10.5, 5.2, 3.1, 7.8, 25.0],
        "Количество": [100, 200, 300, 150, 50],
    })


@pytest.fixture
def sample_table_b():
    """Тестовая таблица B (другие имена столбцов, частично те же данные)."""
    return pd.DataFrame({
        "Код товара": ["ART-001", "ART-003", "ART-006", "ART-007"],
        "Название": ["Болт M8", "Шайба M8", "Саморез 3x30", "Дюбель 8x40"],
        "Стоимость": [10.5, 3.1, 2.0, 4.5],
        "Кол-во": [100, 300, 500, 250],
    })


@pytest.fixture
def er_pair_metadata():
    """Ground truth метаданные для ER пары."""
    return {
        "num_rows_a": 5,
        "num_rows_b": 4,
        "num_duplicates": 2,
        "duplicate_pairs": [[0, 0], [2, 1]],
        "column_mapping_a": {
            "Артикул": "article",
            "Наименование": "product_name",
            "Цена": "base_price",
            "Количество": "quantity",
        },
        "column_mapping_b": {
            "Код товара": "article",
            "Название": "product_name",
            "Стоимость": "base_price",
            "Кол-во": "quantity",
        },
        "entity_ids_a": [0, 1, 2, 3, 4],
        "entity_ids_b": [0, 2, 5, 6],
    }


@pytest.fixture
def er_test_dir(tmp_path, sample_table_a, sample_table_b, er_pair_metadata):
    """Директория с тестовой ER парой (CSV + meta.json)."""
    test_dir = tmp_path / "er_raw" / "test"
    test_dir.mkdir(parents=True)

    sample_table_a.to_csv(test_dir / "pair_0000_table_a.csv", index=False)
    sample_table_b.to_csv(test_dir / "pair_0000_table_b.csv", index=False)

    with open(test_dir / "pair_0000_meta.json", "w", encoding="utf-8") as f:
        json.dump(er_pair_metadata, f)

    return str(test_dir)


@pytest.fixture
def full_config():
    """Полный конфиг в формате config.json (минимальный для тестов)."""
    return {
        "data_generation": {
            "ollama_host": "http://127.0.0.1:11434",
            "llm_model": "test-llm",
            "embedding_model": "test-embed",
            "auto_batch_size": False,
            "initial_batch_size": 2,
            "max_batch_size": 4,
            "min_batch_size": 1,
            "keep_alive": "5m",
            "num_predict": 50,
            "num_ctx": 512,
            "warmup": False,
            "locales": ["en_US"],
            "min_rows_per_table": 3,
            "max_rows_per_table": 5,
            "min_optional_columns": 1,
            "max_optional_columns": 3,
            "column_name_variation_level": 0.5,
            "include_typos": False,
            "include_abbreviations": False,
            "sample_size": 3,
            "num_entity_pool": 20,
            "num_train_pairs": 3,
            "num_val_pairs": 1,
            "num_test_pairs": 1,
            "min_common_entities": 2,
            "max_common_entities": 3,
            "min_unique_entities": 1,
            "max_unique_entities": 2,
            "perturbation_prob": 0.1,
            "missing_value_prob": 0.05,
            "num_extra_sm_tables": 2,
            "output_dir": "unified_dataset",
            "sm_dataset_file": "sm_dataset.json",
            "sm_metadata_file": "sm_metadata.json",
            "er_raw_dir": "raw",
        },
        "schema_matching": {
            "projection_dims": [512, 384],
            "output_dim": SM_OUTPUT_DIM,
            "dropout": 0.1,
            "use_batch_norm": True,
            "batch_size": 8,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_epochs": 2,
            "margin": 0.2,
            "mining_strategy": "semihard",
            "scheduler_patience": 2,
            "early_stopping_patience": 3,
            "train_ratio": 0.7,
            "val_ratio": 0.15,
            "test_ratio": 0.15,
            "similarity_threshold": 0.6,
            "model_dir": "sm_models",
        },
        "entity_resolution": {
            "fasttext_model_path": "cc.ru.300.bin",
            "token_embed_dim": FASTTEXT_DIM,
            "min_token_length": 2,
            "max_token_length": 50,
            "max_token_doc_freq": 0.8,
            "cell_separator": " | ",
            "hidden_dim": ER_HIDDEN_DIM,
            "edge_dim": ER_EDGE_DIM,
            "num_gnn_layers": 2,
            "num_heads": 4,
            "dropout": 0.1,
            "use_jumping_knowledge": True,
            "output_dim": ER_OUTPUT_DIM,
            "batch_size": 2,
            "learning_rate": 0.001,
            "weight_decay": 1e-5,
            "num_epochs": 2,
            "margin": 0.3,
            "mining_strategy": "semihard",
            "scheduler_patience": 2,
            "early_stopping_patience": 3,
            "graphs_dir": "er_graphs",
            "model_dir": "er_models",
        },
        "inference": {
            "sm_similarity_threshold": 0.6,
            "er_similarity_threshold": 0.7,
        },
    }


@pytest.fixture
def config_json_path(full_config, tmp_path):
    """Сохранённый config.json."""
    path = tmp_path / "config.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(full_config, f, ensure_ascii=False)
    return str(path)
