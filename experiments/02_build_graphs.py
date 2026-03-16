"""Эксперимент 2 — Построение графов для Entity Resolution.

Строит HeteroData-графы из синтетических таблиц и предвычисленных
эмбеддингов. Выполняет hard-negative mining и сохраняет всё
в data/graphs/ для дальнейшего обучения.

Использование:
    # Один датасет
    python -m experiments.02_build_graphs --dataset beer

    # Все датасеты + объединённый граф
    python -m experiments.02_build_graphs --all

    # Возобновить прерванный запуск (пропустить готовые графы)
    python -m experiments.02_build_graphs --all --skip-existing

    # Построить только unified из уже готовых per-dataset графов
    python -m experiments.02_build_graphs --unified-only
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.dataset.download import DATASETS
from table_unifier.dataset.embedding_generation import TokenEmbedder
from table_unifier.dataset.graph_builder import build_graph
from table_unifier.dataset.pair_sampling import (
    build_triplet_indices,
    mine_hard_negatives,
    split_labeled_pairs,
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
#  Построение и сохранение графа одного датасета
# ------------------------------------------------------------------ #

def build_and_save_single(
    name: str,
    config: Config,
    token_embedder: TokenEmbedder,
    graphs_dir: Path,
) -> dict | None:
    """Построить граф одного датасета, сохранить в graphs_dir/<name>/."""
    logger.info("=" * 60)
    logger.info("Граф: %s", name)
    logger.info("=" * 60)

    data_dict = load_synth_dataset(
        name, config.data_dir / "synthetic", config.data_dir / "embeddings",
    )
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

    # Индексы ID → позиция в эмбеддинг-матрице
    id_to_idx_a = {str(row["id"]): i for i, (_, row) in enumerate(table_a.iterrows())}
    id_to_idx_b = {str(row["id"]): i for i, (_, row) in enumerate(table_b.iterrows())}

    # Hard-negative mining
    positives_train, negatives_train = split_labeled_pairs(splits["train"])
    hard_triplets_train = mine_hard_negatives(
        row_emb_a, row_emb_b, positives_train, id_to_idx_a, id_to_idx_b, top_k=5,
    )

    hard_triplets_val: list[tuple[str, str, str]] = []
    if "valid" in splits:
        positives_val, _ = split_labeled_pairs(splits["valid"])
        hard_triplets_val = mine_hard_negatives(
            row_emb_a, row_emb_b, positives_val, id_to_idx_a, id_to_idx_b, top_k=3,
        )

    # Построение графа
    graph, id_to_global_a, id_to_global_b = build_graph(
        table_a, table_b, column_embeddings, token_embedder,
        columns_a=columns_a, columns_b=columns_b,
        precomputed_row_embeddings_a=row_emb_a,
        precomputed_row_embeddings_b=row_emb_b,
    )

    # Триплеты → индексы в графе
    train_triplets = build_triplet_indices(hard_triplets_train, id_to_global_a, id_to_global_b)
    val_triplets = (
        build_triplet_indices(hard_triplets_val, id_to_global_a, id_to_global_b)
        if hard_triplets_val
        else torch.zeros((0, 3), dtype=torch.long)
    )

    # Сохранение
    out_dir = graphs_dir / name
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(graph, out_dir / "graph.pt")
    torch.save(train_triplets, out_dir / "train_triplets.pt")
    torch.save(val_triplets, out_dir / "val_triplets.pt")

    with open(out_dir / "id_to_global_a.json", "w") as f:
        json.dump(id_to_global_a, f)
    with open(out_dir / "id_to_global_b.json", "w") as f:
        json.dump(id_to_global_b, f)

    # Копируем test split (нужен для оценки)
    if "test" in splits:
        splits["test"].to_csv(out_dir / "test.csv", index=False)

    stats = {
        "name": name,
        "domain": DATASETS[name]["domain"],
        "n_rows": int(graph["row"].x.shape[0]),
        "n_rows_a": len(table_a),
        "n_rows_b": len(table_b),
        "n_tokens": int(graph["token"].x.shape[0]),
        "n_edges": int(graph["token", "in_row", "row"].edge_index.shape[1]),
        "row_dim": int(graph["row"].x.shape[1]),
        "token_dim": int(graph["token"].x.shape[1]),
        "edge_attr_dim": int(graph["token", "in_row", "row"].edge_attr.shape[1]),
        "n_train_triplets": int(len(train_triplets)),
        "n_val_triplets": int(len(val_triplets)),
        "n_train_positives": len(positives_train),
        "n_train_negatives": len(negatives_train),
        "has_test": "test" in splits,
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        "[%s] rows=%d, tokens=%d, edges=%d, train_triplets=%d, val_triplets=%d",
        name, stats["n_rows"], stats["n_tokens"], stats["n_edges"],
        stats["n_train_triplets"], stats["n_val_triplets"],
    )

    return stats


# ------------------------------------------------------------------ #
#  Объединение графов
# ------------------------------------------------------------------ #

def merge_graphs(graphs: list[HeteroData]) -> HeteroData:
    """Объединить несколько HeteroData-графов в один."""
    all_row_x: list[torch.Tensor] = []
    all_token_x: list[torch.Tensor] = []
    t2r_src_all: list[torch.Tensor] = []
    t2r_dst_all: list[torch.Tensor] = []
    t2r_attr_all: list[torch.Tensor] = []

    row_offset = 0
    token_offset = 0

    for g in graphs:
        n_rows = g["row"].x.shape[0]
        n_tokens = g["token"].x.shape[0]

        all_row_x.append(g["row"].x)
        all_token_x.append(g["token"].x)

        ei = g["token", "in_row", "row"].edge_index
        t2r_src_all.append(ei[0] + token_offset)
        t2r_dst_all.append(ei[1] + row_offset)
        t2r_attr_all.append(g["token", "in_row", "row"].edge_attr)

        row_offset += n_rows
        token_offset += n_tokens

    merged = HeteroData()
    merged["row"].x = torch.cat(all_row_x, dim=0)
    merged["token"].x = torch.cat(all_token_x, dim=0)

    t2r_edge_index = torch.stack([
        torch.cat(t2r_src_all), torch.cat(t2r_dst_all),
    ])
    t2r_edge_attr = torch.cat(t2r_attr_all, dim=0)

    merged["token", "in_row", "row"].edge_index = t2r_edge_index
    merged["token", "in_row", "row"].edge_attr = t2r_edge_attr
    merged["row", "has_token", "token"].edge_index = t2r_edge_index.flip(0)
    merged["row", "has_token", "token"].edge_attr = t2r_edge_attr.clone()

    return merged


def build_unified_graph(
    dataset_names: list[str],
    graphs_dir: Path,
) -> dict | None:
    """Загрузить per-dataset графы и объединить в unified."""
    logger.info("=" * 60)
    logger.info("Объединение графов в unified")
    logger.info("=" * 60)

    graphs: list[HeteroData] = []
    all_train: list[torch.Tensor] = []
    all_val: list[torch.Tensor] = []
    per_ds_meta: list[dict] = []
    row_offset = 0
    failed: list[str] = []

    for name in dataset_names:
        ds_dir = graphs_dir / name
        if not (ds_dir / "graph.pt").exists():
            failed.append(name)
            continue

        graph = torch.load(ds_dir / "graph.pt", weights_only=False)
        train_tri = torch.load(ds_dir / "train_triplets.pt", weights_only=False)
        val_tri = torch.load(ds_dir / "val_triplets.pt", weights_only=False)

        with open(ds_dir / "id_to_global_a.json") as f:
            id_to_global_a = json.load(f)
        with open(ds_dir / "id_to_global_b.json") as f:
            id_to_global_b = json.load(f)
        with open(ds_dir / "stats.json") as f:
            stats = json.load(f)

        n_rows = graph["row"].x.shape[0]

        # Смещённые триплеты и маппинги
        shifted_a = {k: v + row_offset for k, v in id_to_global_a.items()}
        shifted_b = {k: v + row_offset for k, v in id_to_global_b.items()}

        if len(train_tri) > 0:
            all_train.append(train_tri + row_offset)
        if len(val_tri) > 0:
            all_val.append(val_tri + row_offset)

        per_ds_meta.append({
            "name": name,
            "domain": stats.get("domain", ""),
            "row_offset": row_offset,
            "n_rows": n_rows,
            "id_to_global_a": shifted_a,
            "id_to_global_b": shifted_b,
        })

        graphs.append(graph)
        row_offset += n_rows

    if not graphs:
        logger.error("Ни один граф не найден — нечего объединять")
        return None

    # Merge
    unified = merge_graphs(graphs)

    unified_train = torch.cat(all_train, dim=0) if all_train else torch.zeros((0, 3), dtype=torch.long)
    unified_val = torch.cat(all_val, dim=0) if all_val else torch.zeros((0, 3), dtype=torch.long)

    # Сохранение
    out_dir = graphs_dir / "unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(unified, out_dir / "graph.pt")
    torch.save(unified_train, out_dir / "train_triplets.pt")
    torch.save(unified_val, out_dir / "val_triplets.pt")

    # Сохранить per-dataset маппинги
    # (id_to_global_{a,b} со смещениями — нужны для per-dataset оценки)
    for meta in per_ds_meta:
        ds_unified_dir = out_dir / meta["name"]
        ds_unified_dir.mkdir(parents=True, exist_ok=True)
        with open(ds_unified_dir / "id_to_global_a.json", "w") as f:
            json.dump(meta["id_to_global_a"], f)
        with open(ds_unified_dir / "id_to_global_b.json", "w") as f:
            json.dump(meta["id_to_global_b"], f)

        # Скопировать test.csv из per-dataset
        src_test = graphs_dir / meta["name"] / "test.csv"
        if src_test.exists():
            import shutil
            shutil.copy2(src_test, ds_unified_dir / "test.csv")

    stats = {
        "n_datasets": len(per_ds_meta),
        "failed": failed,
        "n_rows_total": int(unified["row"].x.shape[0]),
        "n_tokens_total": int(unified["token"].x.shape[0]),
        "n_edges_total": int(unified["token", "in_row", "row"].edge_index.shape[1]),
        "n_train_triplets": int(len(unified_train)),
        "n_val_triplets": int(len(unified_val)),
        "datasets": [
            {"name": m["name"], "domain": m["domain"],
             "row_offset": m["row_offset"], "n_rows": m["n_rows"]}
            for m in per_ds_meta
        ],
    }

    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info(
        "Unified: rows=%d, tokens=%d, edges=%d, train=%d, val=%d, datasets=%d",
        stats["n_rows_total"], stats["n_tokens_total"], stats["n_edges_total"],
        stats["n_train_triplets"], stats["n_val_triplets"], stats["n_datasets"],
    )

    return stats


# ------------------------------------------------------------------ #
#  Точка входа
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Построение графов для ER")
    parser.add_argument(
        "--dataset", default=None,
        help=f"Имя одного датасета. Доступные: {list(DATASETS.keys())}",
    )
    parser.add_argument("--all", action="store_true",
                        help="Построить графы для всех датасетов + unified")
    parser.add_argument("--unified-only", action="store_true",
                        help="Построить только unified граф из уже готовых per-dataset графов")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропустить датасеты, для которых граф уже построен")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if not args.all and not args.dataset and not args.unified_only:
        parser.error("Укажите --dataset <name>, --all или --unified-only")

    config = Config(data_dir=Path(args.data_dir))
    graphs_dir = config.data_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # --unified-only: построить только unified граф из готовых per-dataset графов
    if args.unified_only:
        available = [
            d.name for d in sorted(graphs_dir.iterdir())
            if d.is_dir() and d.name != "unified" and (d / "graph.pt").exists()
        ]
        if not available:
            logger.error("Нет готовых per-dataset графов в %s", graphs_dir)
            return
        logger.info("Найдено %d готовых графов: %s", len(available), available)
        unified_stats = build_unified_graph(available, graphs_dir)
        if unified_stats:
            logger.info("Unified граф построен успешно")
        return

    er_cfg = config.entity_resolution
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    token_embedder = TokenEmbedder(model_name=er_cfg.token_model_name, device=device)

    dataset_names = list(DATASETS.keys()) if args.all else [args.dataset]

    all_stats: list[dict] = []
    failed: list[str] = []

    for name in dataset_names:
        # --skip-existing: пропустить датасеты с готовым графом
        if args.skip_existing and (graphs_dir / name / "graph.pt").exists():
            logger.info("[%s] Граф уже существует — пропуск", name)
            stats_path = graphs_dir / name / "stats.json"
            if stats_path.exists():
                with open(stats_path) as f:
                    all_stats.append(json.load(f))
            continue
        try:
            stats = build_and_save_single(name, config, token_embedder, graphs_dir)
            if stats is not None:
                all_stats.append(stats)
            else:
                failed.append(name)
        except Exception:
            logger.exception("Ошибка при построении графа %s", name)
            failed.append(name)

    # Unified (только при --all)
    unified_stats = None
    if args.all and all_stats:
        successful = [s["name"] for s in all_stats]
        unified_stats = build_unified_graph(successful, graphs_dir)

    # Сводка
    summary = {
        "total": len(all_stats),
        "failed": failed,
        "datasets": all_stats,
        "unified": unified_stats,
    }
    summary_path = graphs_dir / "graph_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Сводка: %s", summary_path)

    logger.info("=" * 60)
    logger.info("ИТОГО: построено %d / %d графов", len(all_stats), len(dataset_names))
    if failed:
        logger.warning("Не удалось: %s", failed)
    for s in all_stats:
        logger.info(
            "  %s | rows=%d | tokens=%d | edges=%d | train_tri=%d",
            s["name"], s["n_rows"], s["n_tokens"], s["n_edges"],
            s["n_train_triplets"],
        )


if __name__ == "__main__":
    main()
