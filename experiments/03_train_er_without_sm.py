"""Эксперимент 3 — Обучение Entity Resolution без Schema Matching.

Загружает предварительно построенные графы из data/graphs/ (шаг 02_build_graphs.py).
OllamaClient при инференсе **не нужен**.

Режимы:
- Одиночный: обучение на графе одного датасета.
- Multi-dataset (--all): round-robin обучение на всех датасетах.
  По умолчанию 5 датасетов выделяются как hold-out для cross-domain теста.

Использование:
    # Один датасет
    python -m experiments.03_train_er_without_sm --dataset beer

    # Все датасеты с holdout (по умолчанию)
    python -m experiments.03_train_er_without_sm --all

    # Все датасеты без holdout
    python -m experiments.03_train_er_without_sm --all --no-holdout
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.dataset.download import DATASETS, HOLDOUT_DATASETS
from table_unifier.dataset.pair_sampling import split_labeled_pairs
from table_unifier.training.er_trainer import (
    get_row_embeddings,
    train_entity_resolution,
    train_entity_resolution_multidataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Оценка на тестовом наборе (threshold-independent)
# ------------------------------------------------------------------ #

def evaluate_ranking(
    embeddings: torch.Tensor,
    df: pd.DataFrame,
    id_to_global_a: dict[str, int],
    id_to_global_b: dict[str, int],
) -> dict:
    """Вычислить threshold-independent метрики (ROC-AUC, Average Precision)."""
    positives, negatives = split_labeled_pairs(df)

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

    return {
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
    er_cfg.col_dim = int(graph.col_embeddings.shape[1])

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
        metrics = evaluate_ranking(embeddings, pd.read_csv(test_path), id_to_global_a, id_to_global_b)
        result.update(metrics)
        logger.info(
            "[%s] Test — ROC-AUC=%.4f, AP=%.4f",
            name, metrics.get("roc_auc", 0), metrics.get("avg_precision", 0),
        )

    return result


# ------------------------------------------------------------------ #
#  Загрузка графов с диска
# ------------------------------------------------------------------ #

def _load_graph_datasets(
    names: list[str],
    graphs_dir: Path,
) -> tuple[list[dict], list[str]]:
    """Загрузить графы по списку имён. Возвращает (datasets, failed)."""
    datasets: list[dict] = []
    failed: list[str] = []
    for name in names:
        ds_dir = graphs_dir / name
        if not (ds_dir / "graph.pt").exists():
            logger.warning("[%s] Граф не найден — пропуск", name)
            failed.append(name)
            continue
        try:
            graph = torch.load(ds_dir / "graph.pt", weights_only=False)
            train_tri = torch.load(ds_dir / "train_triplets.pt", weights_only=False)
            val_tri = torch.load(ds_dir / "val_triplets.pt", weights_only=False)

            with open(ds_dir / "stats.json") as f:
                stats = json.load(f)

            datasets.append({
                "name": name,
                "domain": stats.get("domain", ""),
                "graph": graph,
                "train_triplets": train_tri,
                "val_triplets": val_tri if len(val_tri) > 0 else None,
            })
            logger.info(
                "  [%s] rows=%d, train_tri=%d, val_tri=%d",
                name, graph["row"].x.shape[0], len(train_tri), len(val_tri),
            )
        except Exception:
            logger.exception("Ошибка загрузки графа %s", name)
            failed.append(name)
    return datasets, failed


# ------------------------------------------------------------------ #
#  Оценка модели на списке датасетов
# ------------------------------------------------------------------ #

def _evaluate_datasets(
    model,
    datasets: list[dict],
    graphs_dir: Path,
    device: str,
    eval_type: str = "in-domain",
) -> list[dict]:
    """Оценить модель на списке датасетов (ROC-AUC, AP)."""
    results: list[dict] = []
    for ds in datasets:
        name = ds["name"]
        ds_dir = graphs_dir / name

        result: dict = {
            "name": name,
            "domain": ds["domain"],
            "eval_type": eval_type,
        }

        embeddings = get_row_embeddings(model, ds["graph"], device=device)

        with open(ds_dir / "id_to_global_a.json") as f:
            id_to_global_a = json.load(f)
        with open(ds_dir / "id_to_global_b.json") as f:
            id_to_global_b = json.load(f)

        # Для cross-domain: используем ВСЕ размеченные пары (train+valid+test),
        # т.к. модель не видела ни одной строки этого датасета.
        # Для in-domain: только test.csv (train/valid использовались при обучении).
        if eval_type == "cross-domain":
            all_dfs = []
            for split in ("train", "valid", "test"):
                p = ds_dir / f"{split}.csv"
                if p.exists():
                    all_dfs.append(pd.read_csv(p))
            eval_df = pd.concat(all_dfs, ignore_index=True) if all_dfs else None
        else:
            test_path = ds_dir / "test.csv"
            eval_df = pd.read_csv(test_path) if test_path.exists() else None

        if eval_df is not None:
            metrics = evaluate_ranking(embeddings, eval_df, id_to_global_a, id_to_global_b)
            result.update(metrics)
            logger.info(
                "[%s] %s — ROC-AUC=%.4f, AP=%.4f (n=%d)",
                name, eval_type,
                metrics.get("roc_auc", 0), metrics.get("avg_precision", 0),
                metrics.get("n_pos", 0) + metrics.get("n_neg", 0),
            )

        results.append(result)

    return results


# ------------------------------------------------------------------ #
#  Multi-dataset обучение (round-robin) с holdout
# ------------------------------------------------------------------ #

def run_multidataset(
    graphs_dir: Path,
    er_cfg: EntityResolutionConfig,
    output_dir: Path,
    device: str,
    holdout_names: list[str] | None = None,
) -> list[dict]:
    """Round-robin обучение с разделением на train/holdout.

    Args:
        holdout_names: датасеты для cross-domain тестирования.
            Модель на них НЕ обучается, но оценивается.
    """
    holdout_set = set(holdout_names or [])

    # Собираем все доступные графы
    available = sorted([
        d.name for d in graphs_dir.iterdir()
        if d.is_dir() and d.name != "unified" and (d / "graph.pt").exists()
    ])

    if not available:
        logger.error("Нет готовых per-dataset графов в %s", graphs_dir)
        return []

    train_names = [n for n in available if n not in holdout_set]
    holdout_names_actual = [n for n in available if n in holdout_set]

    logger.info("=" * 60)
    logger.info("Multi-dataset обучение с holdout")
    logger.info("  Train (%d): %s", len(train_names), train_names)
    logger.info("  Holdout (%d): %s", len(holdout_names_actual), holdout_names_actual)
    logger.info("=" * 60)

    # Загрузка
    train_datasets, train_failed = _load_graph_datasets(train_names, graphs_dir)
    holdout_datasets, holdout_failed = _load_graph_datasets(holdout_names_actual, graphs_dir)

    if not train_datasets:
        logger.error("Ни один обучающий граф не удалось загрузить")
        return []

    # Обучение (только на train_datasets)
    save_path = output_dir / "er_model_multidataset.pt"
    model, history = train_entity_resolution_multidataset(
        datasets=train_datasets,
        config=er_cfg,
        device=device,
        save_path=save_path,
    )

    # ---- Оценка: in-domain (train датасеты, test split) ---- #
    logger.info("=" * 60)
    logger.info("Оценка: in-domain (%d датасетов)", len(train_datasets))
    logger.info("=" * 60)
    in_domain_results = _evaluate_datasets(
        model, train_datasets, graphs_dir, device, eval_type="in-domain",
    )

    # ---- Оценка: cross-domain (holdout датасеты, все пары) ---- #
    cross_domain_results: list[dict] = []
    if holdout_datasets:
        logger.info("=" * 60)
        logger.info("Оценка: cross-domain (%d датасетов)", len(holdout_datasets))
        logger.info("=" * 60)
        cross_domain_results = _evaluate_datasets(
            model, holdout_datasets, graphs_dir, device, eval_type="cross-domain",
        )

    all_results = in_domain_results + cross_domain_results

    # Сводка
    total_train = sum(len(ds["train_triplets"]) for ds in train_datasets)
    total_val = sum(
        len(ds["val_triplets"]) for ds in train_datasets
        if ds["val_triplets"] is not None
    )

    summary = {
        "mode": "multidataset_holdout",
        "train_datasets": [ds["name"] for ds in train_datasets],
        "holdout_datasets": [ds["name"] for ds in holdout_datasets],
        "total_train_triplets": total_train,
        "total_val_triplets": total_val,
        "failed": train_failed + holdout_failed,
        "history": history,
        "in_domain_results": [
            {k: v for k, v in r.items() if k != "history"}
            for r in in_domain_results
        ],
        "cross_domain_results": [
            {k: v for k, v in r.items() if k != "history"}
            for r in cross_domain_results
        ],
    }
    summary_path = output_dir / "er_results_multidataset.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Сводка сохранена: %s", summary_path)

    # Итого
    def _print_results(label: str, results: list[dict]) -> None:
        logger.info("--- %s ---", label)
        for r in results:
            logger.info(
                "  %s | ROC-AUC=%.4f | AP=%.4f",
                r["name"],
                r.get("roc_auc", float("nan")),
                r.get("avg_precision", float("nan")),
            )
        aucs = [r.get("roc_auc", float("nan")) for r in results]
        valid = [a for a in aucs if not np.isnan(a)]
        if valid:
            logger.info("  Среднее ROC-AUC: %.4f", np.mean(valid))

    logger.info("=" * 60)
    _print_results("IN-DOMAIN (test split)", in_domain_results)
    if cross_domain_results:
        _print_results("CROSS-DOMAIN (все пары)", cross_domain_results)
    logger.info("=" * 60)

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
                        help="Обучение на всех датасетах (round-robin)")
    parser.add_argument("--holdout", action="store_true",
                        help="Выделить hold-out датасеты для cross-domain теста "
                             f"(по умолчанию: {HOLDOUT_DATASETS})")
    parser.add_argument("--no-holdout", action="store_true",
                        help="Обучать на ВСЕХ датасетах без hold-out")
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
        # По умолчанию --all включает holdout, если не указано --no-holdout
        if args.no_holdout:
            holdout = None
        else:
            holdout = HOLDOUT_DATASETS
        run_multidataset(graphs_dir, er_cfg, config.output_dir, device,
                         holdout_names=holdout)
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
