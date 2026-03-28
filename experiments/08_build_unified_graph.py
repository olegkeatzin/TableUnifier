"""Эксперимент 8 — Построение объединённого графа с глобальным IDF и split по строкам.

Объединяет 21 in-domain датасет в один граф, применяет глобальную IDF-фильтрацию,
и создаёт стратифицированный split по строкам (train 70% / val 15% / test 15%).

Cross-domain датасеты (electronics, anime, citations) строятся отдельно.

Сохраняет:
  - data/graphs/v3_unified/graph.pt — объединённый граф
  - data/graphs/v3_unified/train_pairs.pt — train пары [N, 3]
  - data/graphs/v3_unified/val_pairs.pt
  - data/graphs/v3_unified/test_pairs.pt
  - data/graphs/v3_unified/dataset_mappings.json
  - data/graphs/v3_unified/stats.json
  - data/graphs/v3_cross/{name}/graph.pt — cross-domain графы (отдельные)

Использование:
    python -m experiments.08_build_unified_graph
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch

from table_unifier.config import Config
from table_unifier.dataset.data_split import split_rows_stratified
from table_unifier.dataset.download import DATASETS
from table_unifier.dataset.embedding_generation import TokenEmbedder
from table_unifier.dataset.graph_builder import build_unified_graph_from_datasets, build_graph
from table_unifier.dataset.pair_sampling import split_labeled_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

CROSS_DOMAIN = {"electronics", "anime", "citations"}


def load_dataset_for_unified(name: str, synth_dir: Path, emb_dir: Path) -> dict | None:
    """Загрузить датасет со всеми labeled pairs."""
    ds_synth = synth_dir / name
    ds_emb = emb_dir / name

    required = [
        ds_synth / "tableA_synth.csv", ds_synth / "tableB_synth.csv",
        ds_emb / "column_embeddings_a.npz", ds_emb / "column_embeddings_b.npz",
        ds_emb / "row_embeddings_a.npy", ds_emb / "row_embeddings_b.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.warning("[%s] Отсутствуют файлы: %s", name, [p.name for p in missing])
        return None

    import pandas as pd
    table_a = pd.read_csv(ds_synth / "tableA_synth.csv")
    table_b = pd.read_csv(ds_synth / "tableB_synth.csv")

    col_emb_a = dict(np.load(ds_emb / "column_embeddings_a.npz"))
    col_emb_b = dict(np.load(ds_emb / "column_embeddings_b.npz"))
    row_emb_a = np.load(ds_emb / "row_embeddings_a.npy")
    row_emb_b = np.load(ds_emb / "row_embeddings_b.npy")

    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    # Собираем ВСЕ labeled pairs (train + valid + test вместе — split будет новый)
    labeled_pairs = []
    for split_name in ("train", "valid", "test"):
        p = ds_synth / f"{split_name}.csv"
        if p.exists():
            import pandas as pd
            df = pd.read_csv(p)
            pos, neg = split_labeled_pairs(df)
            for a_id, b_id in pos:
                labeled_pairs.append((a_id, b_id, 1))
            for a_id, b_id in neg:
                labeled_pairs.append((a_id, b_id, 0))

    return {
        "name": name,
        "table_a": table_a,
        "table_b": table_b,
        "columns_a": columns_a,
        "columns_b": columns_b,
        "column_embeddings": {**col_emb_a, **col_emb_b},
        "row_emb_a": row_emb_a,
        "row_emb_b": row_emb_b,
        "labeled_pairs": labeled_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение unified графа v3")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-token-df", type=float, default=0.05,
                        help="IDF порог (доля строк). Для unified графа 0.05 лучше чем 0.3")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    synth_dir = config.data_dir / "synthetic"
    emb_dir = config.data_dir / "embeddings"

    token_embedder = TokenEmbedder(model_name=config.entity_resolution.token_model_name, device=device)
    del token_embedder.model
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Загрузка in-domain датасетов
    in_domain_datasets = []
    for name in sorted(DATASETS.keys()):
        if name in CROSS_DOMAIN:
            continue
        ds = load_dataset_for_unified(name, synth_dir, emb_dir)
        if ds is not None and ds["labeled_pairs"]:
            in_domain_datasets.append(ds)

    logger.info("In-domain датасетов: %d", len(in_domain_datasets))

    # 2. Построение unified графа
    graph, dataset_mappings, all_labeled = build_unified_graph_from_datasets(
        in_domain_datasets, token_embedder,
        max_token_df=args.max_token_df,
    )

    # 3. Split по строкам
    train_pairs, val_pairs, test_pairs = split_rows_stratified(
        all_labeled, ratios=(0.7, 0.15, 0.15), seed=42,
    )

    # 4. Сохранение
    out_dir = config.data_dir / "graphs" / "v3_unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(graph, out_dir / "graph.pt")
    torch.save(train_pairs, out_dir / "train_pairs.pt")
    torch.save(val_pairs, out_dir / "val_pairs.pt")
    torch.save(test_pairs, out_dir / "test_pairs.pt")

    # Сериализуемые маппинги
    serializable_mappings = {
        name: {k: {str(kk): vv for kk, vv in v.items()} for k, v in maps.items()}
        for name, maps in dataset_mappings.items()
    }
    with open(out_dir / "dataset_mappings.json", "w") as f:
        json.dump(serializable_mappings, f)

    # Статистика фильтрации из графа
    fs = getattr(graph, "filter_stats", {})

    stats = {
        "n_rows": int(graph["row"].x.shape[0]),
        "n_tokens": int(graph["token"].x.shape[0]),
        "n_edges": int(graph["token", "in_row", "row"].edge_index.shape[1]),
        "n_labeled": int(len(all_labeled)),
        "n_train": int(len(train_pairs)),
        "n_val": int(len(val_pairs)),
        "n_test": int(len(test_pairs)),
        "n_datasets": len(in_domain_datasets),
        "datasets": [ds["name"] for ds in in_domain_datasets],
        "filter_stats": fs,
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Сохранено в %s", out_dir)
    logger.info("Stats: %s", stats)

    # 5. Cross-domain графы (отдельные)
    cross_dir = config.data_dir / "graphs" / "v3_cross"
    for name in sorted(CROSS_DOMAIN):
        ds = load_dataset_for_unified(name, synth_dir, emb_dir)
        if ds is None:
            continue

        logger.info("Cross-domain: %s", name)
        cg, id_a, id_b = build_graph(
            ds["table_a"], ds["table_b"], ds["column_embeddings"], token_embedder,
            columns_a=ds["columns_a"], columns_b=ds["columns_b"],
            precomputed_row_embeddings_a=ds["row_emb_a"],
            precomputed_row_embeddings_b=ds["row_emb_b"],
        )

        cd_out = cross_dir / name
        cd_out.mkdir(parents=True, exist_ok=True)
        torch.save(cg, cd_out / "graph.pt")
        with open(cd_out / "id_to_global_a.json", "w") as f:
            json.dump(id_a, f)
        with open(cd_out / "id_to_global_b.json", "w") as f:
            json.dump(id_b, f)

        # Labeled pairs для cross-domain оценки
        if ds["labeled_pairs"]:
            pairs = []
            for a_id, b_id, label in ds["labeled_pairs"]:
                ga = id_a.get(str(a_id))
                gb = id_b.get(str(b_id))
                if ga is not None and gb is not None:
                    pairs.append([ga, gb, label])
            if pairs:
                torch.save(torch.tensor(pairs, dtype=torch.long), cd_out / "labeled_pairs.pt")

    del token_embedder
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Готово!")


if __name__ == "__main__":
    main()
