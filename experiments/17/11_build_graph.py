"""Эксперимент 17 — Шаг 11: unified-граф (Magellan in-domain + exp17 view-pairs).

Отличия от 14_build_unified_graph_mrl:
  1. Добавляет exp17 view-pair датасеты (lamoda/cars_ru/auto_ru/ozon — 20 датасетов)
  2. Devices view-pairs исключены по умолчанию (views_div=0.00, trivial)
  3. MRL-размерность по умолчанию = 1024 (bge-m3 hidden_dim)

Сохраняет:
  data/graphs/bge-m3/v17_views/        — unified граф + train/val/test pairs
  data/graphs/bge-m3/v17_views_cross/  — cross-domain графы (electronics, anime, citations)

Использование:
    uv run python -m experiments.17.11_build_graph
    uv run python -m experiments.17.11_build_graph --skip-magellan  # только exp17
    uv run python -m experiments.17.11_build_graph --include-devices
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from table_unifier.config import Config
from table_unifier.dataset.data_split import split_rows_stratified
from table_unifier.dataset.download import DATASETS
from table_unifier.dataset.embedding_generation import TokenEmbedder
from table_unifier.dataset.graph_builder import build_graph, build_unified_graph_from_datasets
from table_unifier.dataset.pair_sampling import split_labeled_pairs
from table_unifier.paths import columns_dir, rows_dir, unified_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

CROSS_DOMAIN = {"electronics", "anime", "citations"}
EXP17_SYNTH_SOURCES = ["lamoda", "cars_ru", "devices"]
EXP17_NATURAL_SOURCES = ["auto_ru", "ozon"]
EXP17_TRIVIAL = {"devices"}  # views_div=0.00

N_VIEWS = 4
DEFAULT_TAG = "bge-m3"
DEFAULT_ROW_MODEL = "BAAI/bge-m3"
DEFAULT_TARGET_DIM = 1024
DEFAULT_SUBDIR = "v17_views"


def mrl_truncate(emb: np.ndarray, target_dim: int) -> np.ndarray:
    if emb.ndim == 1:
        t = emb[:target_dim]
        return (t / (np.linalg.norm(t) + 1e-12)).astype(np.float32)
    t = emb[:, :target_dim]
    return (t / (np.linalg.norm(t, axis=-1, keepdims=True) + 1e-12)).astype(np.float32)


def exp17_dataset_names(include_devices: bool = False) -> list[str]:
    synth_srcs = EXP17_SYNTH_SOURCES if include_devices else [
        s for s in EXP17_SYNTH_SOURCES if s not in EXP17_TRIVIAL
    ]
    synth = [
        f"{src}_v{i}v{j}"
        for src in synth_srcs
        for i, j in combinations(range(N_VIEWS), 2)
    ]
    natural = [
        f"{src}_syn{k}"
        for src in EXP17_NATURAL_SOURCES
        for k in range(N_VIEWS)
    ]
    return synth + natural


def load_dataset_for_graph(
    name: str,
    synth_dir: Path,
    col_ds_dir: Path,
    row_ds_dir: Path,
    target_col_dim: int,
) -> dict | None:
    ta_path = synth_dir / "tableA_synth.csv"
    tb_path = synth_dir / "tableB_synth.csv"
    if not ta_path.exists():
        logger.warning("[%s] нет tableA_synth.csv — пропуск", name)
        return None

    required = [
        col_ds_dir / "column_embeddings_a.npz",
        col_ds_dir / "column_embeddings_b.npz",
        row_ds_dir / "row_embeddings_a.npy",
        row_ds_dir / "row_embeddings_b.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.warning("[%s] нет: %s — пропуск (запустите 10_gen_embeddings.py)",
                       name, [p.name for p in missing])
        return None

    table_a = pd.read_csv(ta_path)
    table_b = pd.read_csv(tb_path)

    col_emb_a = {k: mrl_truncate(v, target_col_dim)
                 for k, v in np.load(col_ds_dir / "column_embeddings_a.npz").items()}
    col_emb_b = {k: mrl_truncate(v, target_col_dim)
                 for k, v in np.load(col_ds_dir / "column_embeddings_b.npz").items()}
    row_emb_a = np.load(row_ds_dir / "row_embeddings_a.npy")
    row_emb_b = np.load(row_ds_dir / "row_embeddings_b.npy")

    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    labeled_pairs: list[tuple] = []
    for split_name in ("train", "valid", "test"):
        p = synth_dir / f"{split_name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            pos, neg = split_labeled_pairs(df)
            for a_id, b_id in pos:
                labeled_pairs.append((a_id, b_id, 1))
            for a_id, b_id in neg:
                labeled_pairs.append((a_id, b_id, 0))

    if not labeled_pairs:
        logger.warning("[%s] нет labeled pairs — пропуск", name)
        return None

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
    parser = argparse.ArgumentParser(description="Unified граф exp17 + Magellan")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-tag", default=DEFAULT_TAG)
    parser.add_argument("--row-model-name", default=DEFAULT_ROW_MODEL)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--target-dim", type=int, default=DEFAULT_TARGET_DIM,
                        help="MRL-обрезка column embeddings (должна == row_dim)")
    parser.add_argument("--out-subdir", default=DEFAULT_SUBDIR)
    parser.add_argument("--max-token-df", type=float, default=0.05)
    parser.add_argument("--min-token-count", type=int, default=2)
    parser.add_argument("--max-tokens-per-cell", type=int, default=8)
    parser.add_argument("--include-devices", action="store_true",
                        help="Включить devices view-pairs (trivial, views_div=0.00)")
    parser.add_argument("--skip-magellan", action="store_true",
                        help="Не включать Magellan датасеты")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    synth_base = config.data_dir / "synthetic"

    token_embedder = TokenEmbedder(
        model_name=args.row_model_name,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )
    del token_embedder.model
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Magellan in-domain датасеты
    magellan_datasets = []
    if not args.skip_magellan:
        for name in sorted(DATASETS.keys()):
            if name in CROSS_DOMAIN:
                continue
            ds = load_dataset_for_graph(
                name, synth_base / name,
                columns_dir(config.data_dir, name),
                rows_dir(config.data_dir, args.model_tag, name),
                args.target_dim,
            )
            if ds is not None:
                magellan_datasets.append(ds)
        logger.info("Magellan in-domain: %d датасетов", len(magellan_datasets))

    # 2. Exp17 view-pair датасеты
    exp17_names = exp17_dataset_names(include_devices=args.include_devices)
    exp17_datasets = []
    for name in exp17_names:
        ds = load_dataset_for_graph(
            name, synth_base / name,
            columns_dir(config.data_dir, name),
            rows_dir(config.data_dir, args.model_tag, name),
            args.target_dim,
        )
        if ds is not None:
            exp17_datasets.append(ds)
    logger.info("Exp17: %d/%d датасетов загружено", len(exp17_datasets), len(exp17_names))

    all_datasets = magellan_datasets + exp17_datasets
    if not all_datasets:
        raise RuntimeError("Нет данных. Сначала запустите 10_gen_embeddings.py")

    logger.info("Итого: %d датасетов, col_dim=%d", len(all_datasets), args.target_dim)

    # 3. Unified граф
    graph, dataset_mappings, all_labeled = build_unified_graph_from_datasets(
        all_datasets, token_embedder,
        max_token_df=args.max_token_df,
        max_tokens_per_cell=args.max_tokens_per_cell,
        min_token_count=args.min_token_count,
    )
    train_pairs, val_pairs, test_pairs = split_rows_stratified(
        all_labeled, ratios=(0.7, 0.15, 0.15), seed=42,
    )

    # 4. Сохранение
    out_dir = unified_dir(config.data_dir, args.model_tag, args.out_subdir)
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(graph, out_dir / "graph.pt")
    torch.save(train_pairs, out_dir / "train_pairs.pt")
    torch.save(val_pairs, out_dir / "val_pairs.pt")
    torch.save(test_pairs, out_dir / "test_pairs.pt")

    mappings_ser = {
        name: {k: {str(kk): vv for kk, vv in v.items()} for k, v in maps.items()}
        for name, maps in dataset_mappings.items()
    }
    with open(out_dir / "dataset_mappings.json", "w") as f:
        json.dump(mappings_ser, f)

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
        "n_datasets": len(all_datasets),
        "magellan_datasets": [ds["name"] for ds in magellan_datasets],
        "exp17_datasets": [ds["name"] for ds in exp17_datasets],
        "filter_stats": fs,
        "mrl_target_dim": args.target_dim,
        "model_tag": args.model_tag,
        "include_devices": args.include_devices,
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Unified граф → %s", out_dir)
    logger.info("rows=%d, tokens=%d, edges=%d | train=%d, val=%d, test=%d",
                stats["n_rows"], stats["n_tokens"], stats["n_edges"],
                stats["n_train"], stats["n_val"], stats["n_test"])

    # 5. Cross-domain графы (electronics, anime, citations)
    cross_dir = unified_dir(config.data_dir, args.model_tag, f"{args.out_subdir}_cross")
    for name in sorted(CROSS_DOMAIN):
        ds = load_dataset_for_graph(
            name, synth_base / name,
            columns_dir(config.data_dir, name),
            rows_dir(config.data_dir, args.model_tag, name),
            args.target_dim,
        )
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
            pairs = [
                [id_a[str(a)], id_b[str(b)], lbl]
                for a, b, lbl in ds["labeled_pairs"]
                if str(a) in id_a and str(b) in id_b
            ]
            if pairs:
                torch.save(torch.tensor(pairs, dtype=torch.long), cd_out / "labeled_pairs.pt")

        logger.info("  %s: %d rows, %d tokens", name,
                    cg["row"].x.shape[0], cg["token"].x.shape[0])

    del token_embedder
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Готово!")


if __name__ == "__main__":
    main()
