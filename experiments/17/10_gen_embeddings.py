"""Эксперимент 17 — Шаг 10: генерация эмбеддингов для view-pair датасетов.

Генерирует bge-m3 row embeddings + qwen3 column embeddings для:
  - Exp17 view-pair датасетов (lamoda/cars_ru/devices/auto_ru/ozon views)
  - Magellan датасетов (если нет bge-m3 row embeddings, или --no-skip-existing)

Column embeddings (qwen3) — shared: data/embeddings/columns/<dataset>/
Row embeddings (bge-m3)  — per-tag: data/embeddings/rows/bge-m3/<dataset>/

Использование:
    uv run python -m experiments.17.10_gen_embeddings
    uv run python -m experiments.17.10_gen_embeddings --skip-magellan
    uv run python -m experiments.17.10_gen_embeddings --only lamoda_v0v1 ozon_syn0
    uv run python -m experiments.17.10_gen_embeddings --skip-columns  # только row
"""

from __future__ import annotations

import argparse
import gc
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from table_unifier.config import Config
from table_unifier.dataset.download import DATASETS
from table_unifier.dataset.embedding_generation import (
    TokenEmbedder,
    generate_column_embeddings,
    serialize_row,
)
from table_unifier.ollama_client import OllamaClient
from table_unifier.paths import columns_dir, rows_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

EXP17_SYNTH_SOURCES = ["lamoda", "cars_ru", "devices"]
EXP17_NATURAL_SOURCES = ["auto_ru", "ozon"]
N_VIEWS = 4

DEFAULT_TAG = "bge-m3"
DEFAULT_MODEL = "BAAI/bge-m3"
DEFAULT_POOLING = "cls"


def exp17_dataset_names() -> list[str]:
    synth = [
        f"{src}_v{i}v{j}"
        for src in EXP17_SYNTH_SOURCES
        for i, j in combinations(range(N_VIEWS), 2)
    ]
    natural = [
        f"{src}_syn{k}"
        for src in EXP17_NATURAL_SOURCES
        for k in range(N_VIEWS)
    ]
    return synth + natural


def _embed_dataset(
    name: str,
    synth_base: Path,
    col_ds_dir: Path,
    row_ds_dir: Path,
    token_embedder: TokenEmbedder,
    ollama_client,
    skip_existing: bool,
) -> bool:
    ta_path = synth_base / "tableA_synth.csv"
    tb_path = synth_base / "tableB_synth.csv"
    if not ta_path.exists() or not tb_path.exists():
        logger.warning("[%s] нет tableA/B_synth.csv — пропуск", name)
        return False

    table_a = pd.read_csv(ta_path)
    table_b = pd.read_csv(tb_path)
    cols_a = [c for c in table_a.columns if c != "id"]
    cols_b = [c for c in table_b.columns if c != "id"]

    col_ds_dir.mkdir(parents=True, exist_ok=True)
    row_ds_dir.mkdir(parents=True, exist_ok=True)

    # column embeddings (qwen3, shared)
    ca_path = col_ds_dir / "column_embeddings_a.npz"
    cb_path = col_ds_dir / "column_embeddings_b.npz"
    if ollama_client is None:
        pass  # --skip-columns
    else:
        def _load_existing(path: Path) -> dict:
            return dict(np.load(path)) if path.exists() else {}

        existing_a = _load_existing(ca_path)
        existing_b = _load_existing(cb_path)

        missing_a = [c for c in cols_a if c not in existing_a]
        missing_b = [c for c in cols_b if c not in existing_b]

        if skip_existing and not missing_a and not missing_b:
            logger.info("[%s] column emb уже есть (все колонки) — пропуск", name)
        else:
            if missing_a or missing_b:
                logger.info("[%s] column embeddings: A=%d missing, B=%d missing …",
                            name, len(missing_a), len(missing_b))
            else:
                logger.info("[%s] column embeddings …", name)
            emb_a = generate_column_embeddings(ollama_client, table_a, cols_a, existing=existing_a)
            emb_b = generate_column_embeddings(ollama_client, table_b, cols_b, existing=existing_b)
            np.savez(ca_path, **emb_a)
            np.savez(cb_path, **emb_b)
            pd.DataFrame({"col": cols_a}).to_csv(col_ds_dir / "columns_a.csv", index=False)
            pd.DataFrame({"col": cols_b}).to_csv(col_ds_dir / "columns_b.csv", index=False)
            failed_a = [c for c in cols_a if c not in emb_a]
            failed_b = [c for c in cols_b if c not in emb_b]
            if failed_a or failed_b:
                logger.warning("[%s] не получены эмбеддинги: A=%s B=%s", name, failed_a, failed_b)

    # row embeddings (per model_tag)
    ra_path = row_ds_dir / "row_embeddings_a.npy"
    rb_path = row_ds_dir / "row_embeddings_b.npy"
    if skip_existing and ra_path.exists() and rb_path.exists():
        logger.info("[%s] row emb уже есть — пропуск", name)
    else:
        logger.info("[%s] row embeddings …", name)
        texts_a = [serialize_row(r, cols_a) for _, r in table_a.iterrows()]
        texts_b = [serialize_row(r, cols_b) for _, r in table_b.iterrows()]
        emb_a = token_embedder.embed_sentences(texts_a, desc=f"[{name}] A")
        emb_b = token_embedder.embed_sentences(texts_b, desc=f"[{name}] B")
        np.save(ra_path, emb_a)
        np.save(rb_path, emb_b)
        logger.info("[%s] A=%s B=%s → %s", name, emb_a.shape, emb_b.shape, row_ds_dir)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Генерация эмбеддингов для exp17 датасетов")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-tag", default=DEFAULT_TAG)
    parser.add_argument("--row-model-name", default=DEFAULT_MODEL)
    parser.add_argument("--pooling", default=DEFAULT_POOLING)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-columns", action="store_true",
                        help="Пропустить column embeddings (если Ollama недоступна)")
    parser.add_argument("--skip-existing", action="store_true", default=True,
                        help="Пропустить датасеты, где эмбеддинги уже есть (по умолч. включено)")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false")
    parser.add_argument("--skip-magellan", action="store_true",
                        help="Не обрабатывать Magellan датасеты")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Обработать только указанные датасеты")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    synth_base = config.data_dir / "synthetic"

    names_exp17 = exp17_dataset_names()
    names_magellan: list[str] = [] if args.skip_magellan else list(DATASETS.keys())
    all_names = names_exp17 + names_magellan

    if args.only:
        only_set = set(args.only)
        all_names = [n for n in all_names if n in only_set]

    logger.info("Обработка %d датасетов (exp17=%d, magellan=%d), tag=%s",
                len(all_names), len([n for n in names_exp17 if n in all_names]),
                len([n for n in names_magellan if n in all_names]),
                args.model_tag)

    token_embedder = TokenEmbedder(
        model_name=args.row_model_name,
        pooling=args.pooling,
        trust_remote_code=args.trust_remote_code,
        device=device,
    )
    ollama_client = None if args.skip_columns else OllamaClient(config.ollama)

    ok = skip = 0
    for name in tqdm(all_names, desc="datasets"):
        result = _embed_dataset(
            name,
            synth_base=synth_base / name,
            col_ds_dir=columns_dir(config.data_dir, name),
            row_ds_dir=rows_dir(config.data_dir, args.model_tag, name),
            token_embedder=token_embedder,
            ollama_client=ollama_client,
            skip_existing=args.skip_existing,
        )
        if result:
            ok += 1
        else:
            skip += 1

    del token_embedder
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Готово: обработано=%d, пропущено=%d", ok, skip)


if __name__ == "__main__":
    main()
