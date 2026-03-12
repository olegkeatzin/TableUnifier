"""Эксперимент 2 — Обучение Entity Resolution без Schema Matching.

Использует синтетический датасет из data/synthetic/ и предварительно
вычисленные эмбеддинги из data/embeddings/.

Эмбеддинги столбцов берутся напрямую из qwen3-embedding:8b (предвычислены).
Граф строится с предвычисленными row-эмбеддингами от ruBERT-tiny2.
OllamaClient при инференсе **не нужен**.

Поддерживает обучение на одном датасете или на ВСЕХ сразу.

Использование:
    # Один датасет
    python -m experiments.02_train_er_without_sm --dataset beer

    # Все датасеты
    python -m experiments.02_train_er_without_sm --all
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
from table_unifier.dataset.embedding_generation import TokenEmbedder
from table_unifier.dataset.graph_builder import build_graph
from table_unifier.dataset.pair_sampling import (
    build_triplet_indices,
    mine_hard_negatives,
    split_labeled_pairs,
)
from table_unifier.training.er_trainer import (
    find_duplicates,
    get_row_embeddings,
    train_entity_resolution,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Загрузка синтетического датасета + предвычисленных эмбеддингов
# ------------------------------------------------------------------ #

def load_synth_dataset(
    name: str,
    synth_dir: Path,
    emb_dir: Path,
) -> dict | None:
    """Загрузить синтетические таблицы, сплиты и предвычисленные эмбеддинги."""
    ds_synth = synth_dir / name
    ds_emb = emb_dir / name

    required = [
        ds_synth / "tableA_synth.csv",
        ds_synth / "tableB_synth.csv",
        ds_synth / "train.csv",
        ds_emb / "column_embeddings_a.npz",
        ds_emb / "column_embeddings_b.npz",
        ds_emb / "row_embeddings_a.npy",
        ds_emb / "row_embeddings_b.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.warning("[%s] Отсутствуют файлы: %s — пропуск", name, [p.name for p in missing])
        return None

    table_a = pd.read_csv(ds_synth / "tableA_synth.csv")
    table_b = pd.read_csv(ds_synth / "tableB_synth.csv")

    splits: dict[str, pd.DataFrame] = {}
    for split in ("train", "valid", "test"):
        p = ds_synth / f"{split}.csv"
        if p.exists():
            splits[split] = pd.read_csv(p)

    col_emb_a = dict(np.load(ds_emb / "column_embeddings_a.npz"))
    col_emb_b = dict(np.load(ds_emb / "column_embeddings_b.npz"))
    row_emb_a = np.load(ds_emb / "row_embeddings_a.npy")
    row_emb_b = np.load(ds_emb / "row_embeddings_b.npy")

    return {
        "table_a": table_a,
        "table_b": table_b,
        "splits": splits,
        "col_emb_a": col_emb_a,
        "col_emb_b": col_emb_b,
        "row_emb_a": row_emb_a,
        "row_emb_b": row_emb_b,
    }


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

    # Подбор оптимального порога по F1 на тестовых данных
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
#  Обучение на одном датасете
# ------------------------------------------------------------------ #

def run_single_dataset(
    name: str,
    config: Config,
    er_cfg: EntityResolutionConfig,
    token_embedder: TokenEmbedder,
    device: str,
) -> dict | None:
    """Полный пайплайн ER для одного датасета. Возвращает метрики или None."""
    logger.info("=" * 60)
    logger.info("Датасет: %s", name)
    logger.info("=" * 60)

    # 1. Загрузка синтетического датасета
    data_dict = load_synth_dataset(name, config.data_dir / "synthetic", config.data_dir / "embeddings")
    if data_dict is None:
        return None

    table_a = data_dict["table_a"]
    table_b = data_dict["table_b"]
    splits = data_dict["splits"]
    col_emb_a: dict[str, np.ndarray] = data_dict["col_emb_a"]
    col_emb_b: dict[str, np.ndarray] = data_dict["col_emb_b"]
    row_emb_a: np.ndarray = data_dict["row_emb_a"]
    row_emb_b: np.ndarray = data_dict["row_emb_b"]

    if "train" not in splits:
        logger.warning("%s: нет train split — пропуск", name)
        return None

    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]
    column_embeddings = {**col_emb_a, **col_emb_b}

    # 2. Обновление размерностей конфига по реальным данным
    er_cfg.row_dim = row_emb_a.shape[1]
    er_cfg.token_dim = token_embedder.hidden_dim
    er_cfg.col_dim = next(iter(column_embeddings.values())).shape[0]

    # 3. Индексы для поиска по ID
    id_to_idx_a = {str(row["id"]): i for i, (_, row) in enumerate(table_a.iterrows())}
    id_to_idx_b = {str(row["id"]): i for i, (_, row) in enumerate(table_b.iterrows())}

    # 4. Hard-negative triplets (используем предвычисленные row-эмбеддинги)
    positives_train, _ = split_labeled_pairs(splits["train"])
    hard_triplets_train = mine_hard_negatives(
        row_emb_a, row_emb_b, positives_train, id_to_idx_a, id_to_idx_b, top_k=5,
    )

    # 5. Построение графа с предвычисленными row-эмбеддингами
    graph, id_to_global_a, id_to_global_b = build_graph(
        table_a, table_b, column_embeddings, token_embedder,
        columns_a=columns_a, columns_b=columns_b,
        precomputed_row_embeddings_a=row_emb_a,
        precomputed_row_embeddings_b=row_emb_b,
    )

    train_triplet_indices = build_triplet_indices(
        hard_triplets_train, id_to_global_a, id_to_global_b,
    )

    val_triplet_indices = None
    if "valid" in splits:
        positives_val, _ = split_labeled_pairs(splits["valid"])
        hard_triplets_val = mine_hard_negatives(
            row_emb_a, row_emb_b, positives_val, id_to_idx_a, id_to_idx_b, top_k=3,
        )
        if hard_triplets_val:
            val_triplet_indices = build_triplet_indices(
                hard_triplets_val, id_to_global_a, id_to_global_b,
            )

    logger.info("[%s] Train triplets: %d, Val triplets: %s",
                name, len(train_triplet_indices),
                len(val_triplet_indices) if val_triplet_indices is not None else "нет")

    # 6. Обучение
    save_path = config.output_dir / f"er_model_{name}.pt"
    model, history = train_entity_resolution(
        data=graph,
        triplet_indices=train_triplet_indices,
        config=er_cfg,
        val_triplet_indices=val_triplet_indices,
        device=device,
        save_path=save_path,
    )

    # 7. Оценка
    embeddings = get_row_embeddings(model, graph, device=device)

    result: dict = {
        "name": name,
        "domain": DATASETS[name]["domain"],
        "n_train_triplets": len(train_triplet_indices),
        "n_val_triplets": len(val_triplet_indices) if val_triplet_indices is not None else 0,
        "history": history,
    }

    if "test" in splits:
        metrics = evaluate_test(embeddings, splits["test"], id_to_global_a, id_to_global_b)
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
#  Точка входа
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="ER без Schema Matching (синтетический датасет)")
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

    # TokenEmbedder нужен только для построения токен-графа (vocab embeddings)
    token_embedder = TokenEmbedder(model_name=er_cfg.token_model_name, device=device)

    all_results: list[dict] = []
    failed: list[str] = []

    for name in dataset_names:
        try:
            result = run_single_dataset(name, config, er_cfg, token_embedder, device)
            if result is not None:
                all_results.append(result)
            else:
                failed.append(name)
        except Exception:
            logger.exception("Ошибка при обработке %s", name)
            failed.append(name)

    # Сохранить сводный JSON
    summary = {
        "total": len(all_results),
        "failed": failed,
        "results": [
            {k: v for k, v in r.items() if k != "history"}
            for r in all_results
        ],
    }
    summary_path = config.output_dir / "er_results.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Сводка сохранена: %s", summary_path)

    # Вывод итогов в лог
    logger.info("=" * 60)
    logger.info("ИТОГО: обработано %d / %d", len(all_results), len(dataset_names))
    if failed:
        logger.warning("Не удалось: %s", failed)
    for r in all_results:
        logger.info(
            "  %s | F1=%.4f | P=%.4f | R=%.4f | AUC=%.4f",
            r["name"],
            r.get("f1", float("nan")),
            r.get("precision", float("nan")),
            r.get("recall", float("nan")),
            r.get("roc_auc", float("nan")),
        )


if __name__ == "__main__":
    main()
