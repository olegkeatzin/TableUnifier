"""Эксперимент 3 — Обучение Entity Resolution без Schema Matching.

Загружает предварительно построенные графы из data/graphs/ (шаг 02_build_graphs.py).
OllamaClient при инференсе **не нужен**.

Поддерживает два режима:
- Одиночный: обучение на графе одного датасета.
- Единый (--all): обучение на unified графе (все датасеты).

Использование:
    # Один датасет
    python -m experiments.03_train_er_without_sm --dataset beer

    # Единое обучение на unified графе
    python -m experiments.03_train_er_without_sm --all
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.dataset.download import DATASETS
from table_unifier.dataset.pair_sampling import split_labeled_pairs
from table_unifier.training.er_trainer import (
    find_duplicates,
    get_row_embeddings,
    train_entity_resolution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Оценка на тестовом наборе
# ------------------------------------------------------------------ #

def evaluate_test(
    embeddings: torch.Tensor,
    test_df: pd.DataFrame,
    id_to_global_a: dict[str, int],
    id_to_global_b: dict[str, int],
) -> dict:
    """Вычислить развёрнутые метрики на тестовом наборе (пары A×B)."""
    positives, negatives = split_labeled_pairs(test_df)

    scores: list[float] = []
    labels: list[int] = []

    for a_id, b_id in positives:
        ga = id_to_global_a.get(a_id)
        gb = id_to_global_b.get(b_id)
        if ga is not None and gb is not None:
            scores.append((embeddings[ga] @ embeddings[gb]).item())
            labels.append(1)

    for a_id, b_id in negatives:
        ga = id_to_global_a.get(a_id)
        gb = id_to_global_b.get(b_id)
        if ga is not None and gb is not None:
            scores.append((embeddings[ga] @ embeddings[gb]).item())
            labels.append(0)

    if not labels:
        return {}

    scores_arr = np.array(scores)
    labels_arr = np.array(labels)

    best_f1, best_thresh = 0.0, 0.5
    for thr in np.linspace(scores_arr.min(), scores_arr.max(), 100):
        preds = (scores_arr >= thr).astype(int)
        f1 = f1_score(labels_arr, preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = float(thr)

    preds_best = (scores_arr >= best_thresh).astype(int)

    return {
        "threshold": best_thresh,
        "precision": float(precision_score(labels_arr, preds_best, zero_division=0)),
        "recall": float(recall_score(labels_arr, preds_best, zero_division=0)),
        "f1": float(best_f1),
        "roc_auc": float(roc_auc_score(labels_arr, scores_arr)),
        "avg_precision": float(average_precision_score(labels_arr, scores_arr)),
        "n_pos": int(labels_arr.sum()),
        "n_neg": int((1 - labels_arr).sum()),
    }


# ------------------------------------------------------------------ #
#  Обучение на одном датасете (из pre-built графа)
# ------------------------------------------------------------------ #

def run_single_dataset(
    name: str,
    graphs_dir: Path,
    er_cfg: EntityResolutionConfig,
    output_dir: Path,
    device: str,
) -> dict | None:
    """Обучение ER на одном pre-built графе."""
    ds_dir = graphs_dir / name

    if not (ds_dir / "graph.pt").exists():
        logger.warning("[%s] Граф не найден в %s — пропуск", name, ds_dir)
        return None

    logger.info("=" * 60)
    logger.info("Обучение: %s", name)
    logger.info("=" * 60)

    # Загрузка pre-built данных
    graph = torch.load(ds_dir / "graph.pt", weights_only=False)
    train_triplets = torch.load(ds_dir / "train_triplets.pt", weights_only=False)
    val_triplets = torch.load(ds_dir / "val_triplets.pt", weights_only=False)

    with open(ds_dir / "id_to_global_a.json") as f:
        id_to_global_a = json.load(f)
    with open(ds_dir / "id_to_global_b.json") as f:
        id_to_global_b = json.load(f)
    with open(ds_dir / "stats.json") as f:
        stats = json.load(f)

    # Обновить размерности конфига
    er_cfg.row_dim = int(graph["row"].x.shape[1])
    er_cfg.token_dim = int(graph["token"].x.shape[1])
    er_cfg.col_dim = int(graph["token", "in_row", "row"].edge_attr.shape[1])

    val_triplets_arg = val_triplets if len(val_triplets) > 0 else None

    logger.info("[%s] rows=%d, train_triplets=%d, val_triplets=%s",
                name, graph["row"].x.shape[0], len(train_triplets),
                len(val_triplets) if val_triplets_arg is not None else "нет")

    # Обучение
    save_path = output_dir / f"er_model_{name}.pt"
    model, history = train_entity_resolution(
        data=graph,
        triplet_indices=train_triplets,
        config=er_cfg,
        val_triplet_indices=val_triplets_arg,
        device=device,
        save_path=save_path,
    )

    # Оценка
    embeddings = get_row_embeddings(model, graph, device=device)

    result: dict = {
        "name": name,
        "domain": stats.get("domain", ""),
        "n_train_triplets": len(train_triplets),
        "n_val_triplets": len(val_triplets),
        "history": history,
    }

    test_path = ds_dir / "test.csv"
    if test_path.exists():
        test_df = pd.read_csv(test_path)
        metrics = evaluate_test(embeddings, test_df, id_to_global_a, id_to_global_b)
        result.update(metrics)
        logger.info(
            "[%s] Test — F1=%.4f, P=%.4f, R=%.4f, ROC-AUC=%.4f (thr=%.3f)",
            name,
            metrics.get("f1", 0), metrics.get("precision", 0),
            metrics.get("recall", 0), metrics.get("roc_auc", 0),
            metrics.get("threshold", 0.5),
        )

    duplicates = find_duplicates(embeddings, id_to_global_a, id_to_global_b, threshold=0.7)
    result["duplicates_found"] = len(duplicates)

    return result


# ------------------------------------------------------------------ #
#  Единое обучение на unified графе
# ------------------------------------------------------------------ #

def run_unified(
    graphs_dir: Path,
    er_cfg: EntityResolutionConfig,
    output_dir: Path,
    device: str,
) -> list[dict]:
    """Обучение на unified графе (все датасеты)."""
    unified_dir = graphs_dir / "unified"

    if not (unified_dir / "graph.pt").exists():
        logger.error("Unified граф не найден — сначала запустите 02_build_graphs.py --all")
        return []

    logger.info("=" * 60)
    logger.info("Загрузка unified графа")
    logger.info("=" * 60)

    graph = torch.load(unified_dir / "graph.pt", weights_only=False)
    train_triplets = torch.load(unified_dir / "train_triplets.pt", weights_only=False)
    val_triplets = torch.load(unified_dir / "val_triplets.pt", weights_only=False)

    with open(unified_dir / "stats.json") as f:
        u_stats = json.load(f)

    # Обновить размерности конфига
    er_cfg.row_dim = int(graph["row"].x.shape[1])
    er_cfg.token_dim = int(graph["token"].x.shape[1])
    er_cfg.col_dim = int(graph["token", "in_row", "row"].edge_attr.shape[1])

    val_triplets_arg = val_triplets if len(val_triplets) > 0 else None

    logger.info("Unified: rows=%d, tokens=%d, edges=%d, train=%d, val=%s",
                graph["row"].x.shape[0], graph["token"].x.shape[0],
                graph["token", "in_row", "row"].edge_index.shape[1],
                len(train_triplets),
                len(val_triplets) if val_triplets_arg is not None else "нет")

    # Обучение
    save_path = output_dir / "er_model_unified.pt"
    model, history = train_entity_resolution(
        data=graph,
        triplet_indices=train_triplets,
        config=er_cfg,
        val_triplet_indices=val_triplets_arg,
        device=device,
        save_path=save_path,
    )

    # Оценка по каждому датасету
    embeddings = get_row_embeddings(model, graph, device=device)

    all_results: list[dict] = []
    failed: list[str] = u_stats.get("failed", [])

    for ds_info in u_stats["datasets"]:
        ds_name = ds_info["name"]
        ds_dir = unified_dir / ds_name

        result: dict = {
            "name": ds_name,
            "domain": ds_info.get("domain", ""),
        }

        # Загрузить per-dataset маппинги
        with open(ds_dir / "id_to_global_a.json") as f:
            id_to_global_a = json.load(f)
        with open(ds_dir / "id_to_global_b.json") as f:
            id_to_global_b = json.load(f)

        test_path = ds_dir / "test.csv"
        if test_path.exists():
            test_df = pd.read_csv(test_path)
            metrics = evaluate_test(embeddings, test_df, id_to_global_a, id_to_global_b)
            result.update(metrics)
            logger.info(
                "[%s] Test — F1=%.4f, P=%.4f, R=%.4f, ROC-AUC=%.4f (thr=%.3f)",
                ds_name,
                metrics.get("f1", 0), metrics.get("precision", 0),
                metrics.get("recall", 0), metrics.get("roc_auc", 0),
                metrics.get("threshold", 0.5),
            )

        duplicates = find_duplicates(embeddings, id_to_global_a, id_to_global_b, threshold=0.7)
        result["duplicates_found"] = len(duplicates)
        all_results.append(result)

    # Сводка
    summary = {
        "mode": "unified",
        "total_datasets": len(all_results),
        "failed": failed,
        "total_train_triplets": int(len(train_triplets)),
        "total_val_triplets": int(len(val_triplets)),
        "history": history,
        "results": [
            {k: v for k, v in r.items() if k != "history"}
            for r in all_results
        ],
    }
    summary_path = output_dir / "er_results_unified.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Сводка сохранена: %s", summary_path)

    logger.info("=" * 60)
    logger.info("ИТОГО (unified): обучено на %d датасетах", len(all_results))
    if failed:
        logger.warning("Не удалось загрузить: %s", failed)
    for r in all_results:
        logger.info(
            "  %s | F1=%.4f | P=%.4f | R=%.4f | AUC=%.4f",
            r["name"],
            r.get("f1", float("nan")),
            r.get("precision", float("nan")),
            r.get("recall", float("nan")),
            r.get("roc_auc", float("nan")),
        )

    return all_results


# ------------------------------------------------------------------ #
#  Точка входа
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="ER без Schema Matching (pre-built графы)")
    parser.add_argument(
        "--dataset", default=None,
        help=f"Имя одного датасета. Доступные: {list(DATASETS.keys())}",
    )
    parser.add_argument("--all", action="store_true",
                        help="Единое обучение на unified графе")
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

    graphs_dir = config.data_dir / "graphs"

    if args.all:
        run_unified(graphs_dir, er_cfg, config.output_dir, device)
    else:
        result = run_single_dataset(
            args.dataset, graphs_dir, er_cfg, config.output_dir, device,
        )
        if result is not None:
            summary = {"results": [{k: v for k, v in result.items() if k != "history"}]}
            summary_path = config.output_dir / f"er_results_{args.dataset}.json"
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, ensure_ascii=False, indent=2)
            logger.info("Результат сохранён: %s", summary_path)


if __name__ == "__main__":
    main()
