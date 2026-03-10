"""Эксперимент 2 — Обучение Entity Resolution без Schema Matching.

Эмбеддинги столбцов берутся напрямую из qwen3-embedding:8b
(без проекционной модели Schema Matching).

Поддерживает обучение на одном датасете или на ВСЕХ сразу
(единый граф по всем доменам).

Использование:
    # Один датасет
    python -m experiments.02_train_er_without_sm --dataset beer

    # Все датасеты
    python -m experiments.02_train_er_without_sm --all
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


def run_single_dataset(
    name: str,
    config: Config,
    er_cfg: EntityResolutionConfig,
    client: OllamaClient,
    token_embedder: TokenEmbedder,
    device: str,
) -> dict | None:
    """Полный пайплайн ER для одного датасета. Возвращает метрики или None."""
    logger.info("=" * 60)
    logger.info("Датасет: %s", name)
    logger.info("=" * 60)

    # 1. Загрузка
    try:
        ds_path = download_dataset(name, config.data_dir)
        tables = load_dataset(ds_path, name=name)
    except Exception:
        logger.exception("Не удалось загрузить %s — пропуск", name)
        return None

    if "tableA" not in tables or "tableB" not in tables:
        logger.warning("%s: нет tableA/tableB — пропуск", name)
        return None
    if "train" not in tables:
        logger.warning("%s: нет train split — пропуск", name)
        return None

    table_a, table_b = tables["tableA"], tables["tableB"]
    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    # 2. Эмбеддинги столбцов
    col_emb_a = generate_column_embeddings(client, table_a, columns_a)
    col_emb_b = generate_column_embeddings(client, table_b, columns_b)
    column_embeddings = {**col_emb_a, **col_emb_b}

    er_cfg.row_dim = token_embedder.hidden_dim
    er_cfg.token_dim = token_embedder.hidden_dim
    sample_col_emb = next(iter(column_embeddings.values()))
    er_cfg.col_dim = len(sample_col_emb)

    # 3. Построение графа
    data, id_to_global_a, id_to_global_b = build_graph(
        table_a, table_b, column_embeddings, token_embedder,
        columns_a=columns_a, columns_b=columns_b,
    )

    # 4. Триплеты
    positives_train, _ = split_labeled_pairs(tables["train"])
    row_emb_a = token_embedder.embed_sentences(
        [f"{r.to_dict()}" for _, r in table_a.iterrows()]
    )
    row_emb_b = token_embedder.embed_sentences(
        [f"{r.to_dict()}" for _, r in table_b.iterrows()]
    )
    id_to_idx_a = {str(row["id"]): i for i, (_, row) in enumerate(table_a.iterrows())}
    id_to_idx_b = {str(row["id"]): i for i, (_, row) in enumerate(table_b.iterrows())}

    hard_triplets_train = mine_hard_negatives(
        row_emb_a, row_emb_b,
        positives_train, id_to_idx_a, id_to_idx_b, top_k=5,
    )
    train_triplet_indices = build_triplet_indices(
        hard_triplets_train, id_to_global_a, id_to_global_b,
    )

    val_triplet_indices = None
    if "valid" in tables:
        positives_val, _ = split_labeled_pairs(tables["valid"])
        hard_triplets_val = mine_hard_negatives(
            row_emb_a, row_emb_b,
            positives_val, id_to_idx_a, id_to_idx_b, top_k=3,
        )
        if hard_triplets_val:
            val_triplet_indices = build_triplet_indices(
                hard_triplets_val, id_to_global_a, id_to_global_b,
            )

    logger.info("[%s] Train triplets: %d, Val triplets: %s",
                name, len(train_triplet_indices),
                len(val_triplet_indices) if val_triplet_indices is not None else "нет")

    # 5. Обучение
    save_path = config.output_dir / f"er_model_{name}.pt"
    model = train_entity_resolution(
        data=data,
        triplet_indices=train_triplet_indices,
        config=er_cfg,
        val_triplet_indices=val_triplet_indices,
        device=device,
        save_path=save_path,
    )

    # 6. Оценка
    embeddings = get_row_embeddings(model, data, device=device)
    result = {"name": name, "domain": DATASETS[name]["domain"]}

    if "test" in tables:
        test_labels = tables["test"]
        positives_test, negatives_test = split_labeled_pairs(test_labels)

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
        result["accuracy"] = accuracy
        result["correct"] = correct
        result["total"] = total
        logger.info("[%s] Test accuracy: %.4f (%d/%d)", name, accuracy, correct, total)

    duplicates = find_duplicates(
        embeddings, id_to_global_a, id_to_global_b, threshold=0.7,
    )
    result["duplicates_found"] = len(duplicates)
    logger.info("[%s] Дубликатов (sim>=0.7): %d", name, len(duplicates))

    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="ER без Schema Matching")
    parser.add_argument(
        "--dataset", default=None,
        help=f"Имя одного датасета. Доступные: {list(DATASETS.keys())}",
    )
    parser.add_argument("--all", action="store_true",
                        help="Обучить на ВСЕХ доступных датасетах")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.all and not args.dataset:
        parser.error("Укажите --dataset <name> или --all")

    config = Config(
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
    )
    config.output_dir.mkdir(parents=True, exist_ok=True)
    er_cfg = config.entity_resolution
    er_cfg.epochs = args.epochs
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    dataset_names = list(DATASETS.keys()) if args.all else [args.dataset]

    # Общие ресурсы
    client = OllamaClient(config.ollama)
    token_embedder = TokenEmbedder(
        model_name=er_cfg.token_model_name, device=device,
    )

    all_results: list[dict] = []
    failed: list[str] = []

    for name in dataset_names:
        try:
            result = run_single_dataset(
                name, config, er_cfg, client, token_embedder, device,
            )
            if result is not None:
                all_results.append(result)
            else:
                failed.append(name)
        except Exception:
            logger.exception("Ошибка при обработке %s", name)
            failed.append(name)

    # Сводка
    logger.info("=" * 60)
    logger.info("ИТОГО: обработано %d / %d датасетов", len(all_results), len(dataset_names))
    if failed:
        logger.warning("Не удалось: %s", failed)
    for r in all_results:
        acc = r.get("accuracy")
        acc_str = f"{acc:.4f}" if acc is not None else "N/A"
        logger.info("  %s (%s): accuracy=%s, duplicates=%d",
                     r["name"], r["domain"], acc_str, r.get("duplicates_found", 0))


if __name__ == "__main__":
    main()
