"""Эксперимент 2 — Обучение Entity Resolution без Schema Matching.

Эмбеддинги столбцов берутся напрямую из qwen3-embedding:8b
(без проекционной модели Schema Matching).

Использование:
    python -m experiments.02_train_er_without_sm --dataset Beer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.dataset.download import DATASETS, download_dataset, load_dataset
from table_unifier.dataset.embedding_generation import (
    TokenEmbedder,
    generate_column_embeddings,
)
from table_unifier.dataset.graph_builder import build_graph
from table_unifier.dataset.pair_sampling import (
    build_triplet_indices,
    mine_hard_negatives,
    split_labeled_pairs,
)
from table_unifier.ollama_client import OllamaClient
from table_unifier.training.er_trainer import (
    find_duplicates,
    get_row_embeddings,
    train_entity_resolution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="ER без Schema Matching")
    parser.add_argument(
        "--dataset", default="beer",
        help=f"Имя датасета. Доступные: {list(DATASETS.keys())}",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
    er_cfg = config.entity_resolution
    er_cfg.epochs = args.epochs
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # ================================================================ #
    #  1. Загрузка данных
    # ================================================================ #
    logger.info("=== 1. Загрузка данных ===")
    ds_path = download_dataset(args.dataset, config.data_dir)
    tables = load_dataset(ds_path, name=args.dataset)
    table_a, table_b = tables["tableA"], tables["tableB"]
    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    # ================================================================ #
    #  2. Эмбеддинги столбцов (напрямую из Ollama, без SM-проекции)
    # ================================================================ #
    logger.info("=== 2. Эмбеддинги столбцов ===")
    client = OllamaClient(config.ollama)

    col_emb_a = generate_column_embeddings(client, table_a, columns_a)
    col_emb_b = generate_column_embeddings(client, table_b, columns_b)
    column_embeddings = {**col_emb_a, **col_emb_b}

    # ================================================================ #
    #  3. Инициализация TokenEmbedder (rubert-tiny2)
    # ================================================================ #
    logger.info("=== 3. TokenEmbedder ===")
    token_embedder = TokenEmbedder(
        model_name=er_cfg.token_model_name, device=device,
    )
    er_cfg.row_dim = token_embedder.hidden_dim
    er_cfg.token_dim = token_embedder.hidden_dim

    # Установить col_dim из фактических эмбеддингов
    sample_col_emb = next(iter(column_embeddings.values()))
    er_cfg.col_dim = len(sample_col_emb)

    # ================================================================ #
    #  4. Построение графа
    # ================================================================ #
    logger.info("=== 4. Построение графа ===")
    data, id_to_global_a, id_to_global_b = build_graph(
        table_a, table_b, column_embeddings, token_embedder,
        columns_a=columns_a, columns_b=columns_b,
    )

    # ================================================================ #
    #  5. Формирование триплетов
    # ================================================================ #
    logger.info("=== 5. Формирование триплетов ===")

    # Train
    train_labels = tables["train"]
    positives_train, _ = split_labeled_pairs(train_labels)
    row_emb_a_train = token_embedder.embed_sentences(
        [f"{r.to_dict()}" for _, r in table_a.iterrows()]
    )
    row_emb_b_train = token_embedder.embed_sentences(
        [f"{r.to_dict()}" for _, r in table_b.iterrows()]
    )
    id_to_idx_a = {int(row["id"]): i for i, (_, row) in enumerate(table_a.iterrows())}
    id_to_idx_b = {int(row["id"]): i for i, (_, row) in enumerate(table_b.iterrows())}

    hard_triplets_train = mine_hard_negatives(
        row_emb_a_train, row_emb_b_train,
        positives_train, id_to_idx_a, id_to_idx_b, top_k=5,
    )
    train_triplet_indices = build_triplet_indices(
        hard_triplets_train, id_to_global_a, id_to_global_b,
    )

    # Validation
    val_triplet_indices = None
    if "valid" in tables:
        positives_val, _ = split_labeled_pairs(tables["valid"])
        hard_triplets_val = mine_hard_negatives(
            row_emb_a_train, row_emb_b_train,
            positives_val, id_to_idx_a, id_to_idx_b, top_k=3,
        )
        if hard_triplets_val:
            val_triplet_indices = build_triplet_indices(
                hard_triplets_val, id_to_global_a, id_to_global_b,
            )

    logger.info("Train triplets: %d, Val triplets: %s",
                len(train_triplet_indices),
                len(val_triplet_indices) if val_triplet_indices is not None else "нет")

    # ================================================================ #
    #  6. Обучение
    # ================================================================ #
    logger.info("=== 6. Обучение GNN ===")
    save_path = config.output_dir / "er_model.pt"

    model = train_entity_resolution(
        data=data,
        triplet_indices=train_triplet_indices,
        config=er_cfg,
        val_triplet_indices=val_triplet_indices,
        device=device,
        save_path=save_path,
    )

    # ================================================================ #
    #  7. Оценка на тестовых данных
    # ================================================================ #
    logger.info("=== 7. Оценка ===")
    embeddings = get_row_embeddings(model, data, device=device)

    if "test" in tables:
        test_labels = tables["test"]
        positives_test, negatives_test = split_labeled_pairs(test_labels)

        # Метрики через cosine similarity
        correct = 0
        total = len(positives_test) + len(negatives_test)
        threshold = 0.5

        for a_id, b_id in positives_test:
            ga = id_to_global_a.get(a_id)
            gb = id_to_global_b.get(b_id)
            if ga is not None and gb is not None:
                sim = (embeddings[ga] @ embeddings[gb]).item()
                if sim >= threshold:
                    correct += 1

        for a_id, b_id in negatives_test:
            ga = id_to_global_a.get(a_id)
            gb = id_to_global_b.get(b_id)
            if ga is not None and gb is not None:
                sim = (embeddings[ga] @ embeddings[gb]).item()
                if sim < threshold:
                    correct += 1

        accuracy = correct / max(total, 1)
        logger.info("Test accuracy (threshold=%.2f): %.4f (%d/%d)",
                     threshold, accuracy, correct, total)

    # Примеры найденных дубликатов
    duplicates = find_duplicates(
        embeddings, id_to_global_a, id_to_global_b, threshold=0.7,
    )
    logger.info("Найдено пар с similarity >= 0.7: %d", len(duplicates))
    for a_id, b_id, sim in duplicates[:10]:
        row_a = table_a[table_a["id"] == a_id].iloc[0] if a_id in table_a["id"].values else None
        row_b = table_b[table_b["id"] == b_id].iloc[0] if b_id in table_b["id"].values else None
        logger.info("  sim=%.3f | A: %s | B: %s",
                     sim,
                     row_a.to_dict() if row_a is not None else "?",
                     row_b.to_dict() if row_b is not None else "?")


if __name__ == "__main__":
    main()
