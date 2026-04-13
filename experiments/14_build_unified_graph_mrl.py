"""Эксперимент 14 — Построение unified-графа с MRL-обрезкой qwen3-embedding.

Отличия от exp 08 (v3_unified):
  1. Column embeddings обрезаются 4096 → 312 через Matryoshka Representation Learning
     (qwen3-embedding официально поддерживает MRL: префикс + L2-renorm).
     После обрезки col_dim == row_dim (rubert-tiny2 hidden = 312), что позволит
     убрать learned-проекции в модели.
  2. Min-degree фильтрация токенов (min_token_count=2): убираем singleton-токены,
     которые бесполезны для message passing (одна строка → нечего сравнивать).
  3. Сниженный per-cell лимит (max_tokens_per_cell=8 вместо 16) — режет хвост
     по широким таблицам.

Сохраняет в data/graphs/v14_mrl/ — параллельная иерархия v3_unified.

Использование:
    python -m experiments.14_build_unified_graph_mrl
    python -m experiments.14_build_unified_graph_mrl --target-dim 312 --min-token-count 2
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
from table_unifier.dataset.graph_builder import build_graph, build_unified_graph_from_datasets
from table_unifier.dataset.pair_sampling import split_labeled_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

CROSS_DOMAIN = {"electronics", "anime", "citations"}


def mrl_truncate(emb: np.ndarray, target_dim: int) -> np.ndarray:
    """MRL truncation: префикс + L2-renorm.

    qwen3-embedding обучен с Matryoshka Representation Learning, поэтому
    префиксы фиксированной длины сохраняют семантическое качество.
    """
    if emb.ndim == 1:
        truncated = emb[:target_dim]
        norm = np.linalg.norm(truncated) + 1e-12
        return (truncated / norm).astype(np.float32)
    truncated = emb[:, :target_dim]
    norm = np.linalg.norm(truncated, axis=-1, keepdims=True) + 1e-12
    return (truncated / norm).astype(np.float32)


def load_dataset_for_unified(
    name: str, synth_dir: Path, emb_dir: Path, target_col_dim: int,
) -> dict | None:
    """Загрузить датасет с MRL-обрезанными column embeddings."""
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

    col_emb_a = {k: mrl_truncate(v, target_col_dim) for k, v in np.load(ds_emb / "column_embeddings_a.npz").items()}
    col_emb_b = {k: mrl_truncate(v, target_col_dim) for k, v in np.load(ds_emb / "column_embeddings_b.npz").items()}
    row_emb_a = np.load(ds_emb / "row_embeddings_a.npy")
    row_emb_b = np.load(ds_emb / "row_embeddings_b.npy")

    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    labeled_pairs = []
    for split_name in ("train", "valid", "test"):
        p = ds_synth / f"{split_name}.csv"
        if p.exists():
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
    parser = argparse.ArgumentParser(description="Построение MRL unified графа (exp 14)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-dim", type=int, default=312,
                        help="Размерность MRL-обрезки column embeddings (по умолч. 312 = rubert hidden)")
    parser.add_argument("--max-token-df", type=float, default=0.05,
                        help="IDF порог (доля строк). Дефолт 0.05 как в v3_unified")
    parser.add_argument("--min-token-count", type=int, default=2,
                        help="Min degree токенов (≥2 удаляет singleton-токены)")
    parser.add_argument("--max-tokens-per-cell", type=int, default=8,
                        help="Лимит токенов на ячейку (8 вместо v3-шных 16)")
    parser.add_argument("--out-subdir", default="v14_mrl",
                        help="Подкаталог в data/graphs/")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    synth_dir = config.data_dir / "synthetic"
    emb_dir = config.data_dir / "embeddings"

    token_embedder = TokenEmbedder(
        model_name=config.entity_resolution.token_model_name, device=device,
    )
    del token_embedder.model
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Загрузка in-domain датасетов
    in_domain_datasets = []
    for name in sorted(DATASETS.keys()):
        if name in CROSS_DOMAIN:
            continue
        ds = load_dataset_for_unified(name, synth_dir, emb_dir, args.target_dim)
        if ds is not None and ds["labeled_pairs"]:
            in_domain_datasets.append(ds)

    logger.info("In-domain датасетов: %d (col_dim после MRL = %d)",
                len(in_domain_datasets), args.target_dim)

    # 2. Построение unified графа
    graph, dataset_mappings, all_labeled = build_unified_graph_from_datasets(
        in_domain_datasets, token_embedder,
        max_token_df=args.max_token_df,
        max_tokens_per_cell=args.max_tokens_per_cell,
        min_token_count=args.min_token_count,
    )

    # 3. Split по строкам
    train_pairs, val_pairs, test_pairs = split_rows_stratified(
        all_labeled, ratios=(0.7, 0.15, 0.15), seed=42,
    )

    # 4. Сохранение
    out_dir = config.data_dir / "graphs" / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(graph, out_dir / "graph.pt")
    torch.save(train_pairs, out_dir / "train_pairs.pt")
    torch.save(val_pairs, out_dir / "val_pairs.pt")
    torch.save(test_pairs, out_dir / "test_pairs.pt")

    serializable_mappings = {
        name: {k: {str(kk): vv for kk, vv in v.items()} for k, v in maps.items()}
        for name, maps in dataset_mappings.items()
    }
    with open(out_dir / "dataset_mappings.json", "w") as f:
        json.dump(serializable_mappings, f)

    fs = getattr(graph, "filter_stats", {})
    stats = {
        "n_rows": int(graph["row"].x.shape[0]),
        "n_tokens": int(graph["token"].x.shape[0]),
        "n_edges": int(graph["token", "in_row", "row"].edge_index.shape[1]),
        "col_dim": int(graph.col_embeddings.shape[1]),
        "n_labeled": int(len(all_labeled)),
        "n_train": int(len(train_pairs)),
        "n_val": int(len(val_pairs)),
        "n_test": int(len(test_pairs)),
        "n_datasets": len(in_domain_datasets),
        "datasets": [ds["name"] for ds in in_domain_datasets],
        "filter_stats": fs,
        "mrl_target_dim": args.target_dim,
        "max_tokens_per_cell": args.max_tokens_per_cell,
        "min_token_count": args.min_token_count,
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Сохранено в %s", out_dir)
    logger.info("Stats: %s", stats)

    # 5. Cross-domain графы (отдельные)
    cross_dir = config.data_dir / "graphs" / f"{args.out_subdir}_cross"
    for name in sorted(CROSS_DOMAIN):
        ds = load_dataset_for_unified(name, synth_dir, emb_dir, args.target_dim)
        if ds is None:
            continue

        logger.info("Cross-domain: %s", name)
        cg, id_a, id_b = build_graph(
            ds["table_a"], ds["table_b"], ds["column_embeddings"], token_embedder,
            columns_a=ds["columns_a"], columns_b=ds["columns_b"],
            precomputed_row_embeddings_a=ds["row_emb_a"],
            precomputed_row_embeddings_b=ds["row_emb_b"],
            max_token_df=args.max_token_df,
            max_tokens_per_cell=args.max_tokens_per_cell,
            min_token_count=args.min_token_count,
        )

        cd_out = cross_dir / name
        cd_out.mkdir(parents=True, exist_ok=True)
        torch.save(cg, cd_out / "graph.pt")
        with open(cd_out / "id_to_global_a.json", "w") as f:
            json.dump(id_a, f)
        with open(cd_out / "id_to_global_b.json", "w") as f:
            json.dump(id_b, f)

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
