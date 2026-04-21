"""Эксперимент 1 — Исследование и генерация единого датасета.

Загрузка ВСЕХ датасетов из Magellan Data Repository, анализ, генерация
единого синтетического датасета (Schema Injection + Value Corruption)
по всем доменам (beer, movies, restaurants, books, и т.д.).

Использование:
    # Анализ одного датасета
    python -m experiments.01_data_exploration --dataset beer

    # Анализ всех датасетов
    python -m experiments.01_data_exploration --all

    # Генерация единого синтетического датасета по всем доменам
    python -m experiments.01_data_exploration --all --generate
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from table_unifier.config import Config
from table_unifier.dataset.download import DATASETS, download_dataset, load_dataset
from table_unifier.dataset.embedding_generation import (
    TokenEmbedder,
    generate_column_embeddings,
    serialize_row,
)
from table_unifier.dataset.schema_augmentation import augment_schema, apply_schema_injection
from table_unifier.dataset.value_corruption import corrupt_dataframe
from table_unifier.paths import columns_dir, rows_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def explore_dataset(name: str, tables: dict[str, pd.DataFrame]) -> None:
    """Вывести статистику по датасету."""
    print(f"\n{'#'*60}")
    print(f"  Датасет: {name}")
    print(f"{'#'*60}")
    for tbl_name, df in tables.items():
        print(f"\n{'='*60}")
        print(f"  {tbl_name}: {df.shape[0]} строк × {df.shape[1]} столбцов")
        print(f"  Столбцы: {list(df.columns)}")
        print(f"  Типы:\n{df.dtypes.to_string()}")
        print(f"  Пропуски:\n{df.isnull().sum().to_string()}")
        if tbl_name in ("train", "valid", "test"):
            if "label" in df.columns:
                counts = df["label"].value_counts()
                print(f"  Баланс label:\n{counts.to_string()}")
                ratio = counts.get(1, 0) / max(counts.get(0, 1), 1)
                print(f"  Ratio (pos/neg): {ratio:.3f}")
        else:
            print(df.head(3).to_string())


def generate_synthetic_for_one(
    name: str,
    tables: dict[str, pd.DataFrame],
    client,
    domain: str,
    out_dir: Path,
) -> dict:
    """Генерация синтетического датасета для одного набора данных.

    Возвращает метаданные: имена таблиц, маппинг столбцов, кол-во строк.
    """
    table_a = tables["tableA"]
    table_b = tables["tableB"]
    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    # ---- A. Schema Injection ---- #
    logger.info("[%s] Schema Injection …", name)
    synonym_map_a = augment_schema(client, columns_a, domain=domain, n_variants=3)
    synonym_map_b = augment_schema(client, columns_b, domain=domain, n_variants=3)

    rename_a = apply_schema_injection(columns_a, synonym_map_a, variant_idx=0)
    rename_b = apply_schema_injection(columns_b, synonym_map_b, variant_idx=1)

    synth_a = table_a.rename(columns=rename_a)
    synth_b = table_b.rename(columns=rename_b)

    logger.info("[%s] Table A: %s → %s", name, columns_a, list(rename_a.values()))
    logger.info("[%s] Table B: %s → %s", name, columns_b, list(rename_b.values()))

    # ---- B. Value Corruption ---- #
    logger.info("[%s] Value Corruption …", name)
    noisy_a = corrupt_dataframe(synth_a, row_prob=0.4, cell_prob=0.3)
    noisy_b = corrupt_dataframe(synth_b, row_prob=0.4, cell_prob=0.3)

    # Сохранение по отдельности
    ds_dir = out_dir / name
    ds_dir.mkdir(parents=True, exist_ok=True)
    noisy_a.to_csv(ds_dir / "tableA_synth.csv", index=False)
    noisy_b.to_csv(ds_dir / "tableB_synth.csv", index=False)

    # Ground truth: маппинг столбцов
    gt = {rename_a.get(c, c): rename_b.get(c, c) for c in columns_a if c in columns_b}
    pd.DataFrame(list(gt.items()), columns=["col_A", "col_B"]).to_csv(
        ds_dir / "schema_ground_truth.csv", index=False,
    )

    # Labeled data (train/valid/test)
    for split in ("train", "valid", "test"):
        if split in tables:
            tables[split].to_csv(ds_dir / f"{split}.csv", index=False)

    logger.info("[%s] Сохранён в %s", name, ds_dir)

    return {
        "name": name,
        "domain": domain,
        "rows_a": len(noisy_a),
        "rows_b": len(noisy_b),
        "columns_a_orig": columns_a,
        "columns_a_synth": list(rename_a.values()),
        "columns_b_orig": columns_b,
        "columns_b_synth": list(rename_b.values()),
        "schema_gt": gt,
        "n_train": len(tables.get("train", [])),
        "n_valid": len(tables.get("valid", [])),
        "n_test": len(tables.get("test", [])),
        "synth_a": noisy_a,
        "synth_b": noisy_b,
    }


def generate_unified_dataset(
    dataset_names: list[str],
    config: Config,
) -> None:
    """Генерация единого синтетического датасета по всем указанным наборам."""
    from table_unifier.ollama_client import OllamaClient

    client = OllamaClient(config.ollama)
    out_dir = config.data_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_meta: list[dict] = []
    failed: list[str] = []

    for name in dataset_names:
        ds_info = DATASETS[name]
        domain = ds_info["domain"]

        logger.info("=" * 60)
        logger.info("Обработка датасета: %s (домен: %s)", name, domain)
        logger.info("=" * 60)

        try:
            csv_path = download_dataset(name, config.data_dir)
            tables = load_dataset(csv_path, name=name)
        except Exception:
            logger.exception("Не удалось загрузить датасет %s — пропуск", name)
            failed.append(name)
            continue

        if "tableA" not in tables or "tableB" not in tables:
            logger.warning("Датасет %s: не найдены tableA/tableB — пропуск", name)
            failed.append(name)
            continue

        try:
            meta = generate_synthetic_for_one(name, tables, client, domain, out_dir)
            all_meta.append(meta)
        except Exception:
            logger.exception("Ошибка генерации для %s — пропуск", name)
            failed.append(name)

    # ---- Сводная статистика ---- #
    serializable_meta = [
        {k: v for k, v in m.items() if k not in ("synth_a", "synth_b")}
        for m in all_meta
    ]
    summary = {
        "total_datasets": len(all_meta),
        "failed_datasets": failed,
        "datasets": serializable_meta,
        "total_rows_a": sum(m["rows_a"] for m in all_meta),
        "total_rows_b": sum(m["rows_b"] for m in all_meta),
        "domains": sorted({m["domain"] for m in all_meta}),
    }

    summary_path = out_dir / "unified_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    logger.info("=" * 60)
    logger.info("Единый датасет готов: %s", out_dir)
    logger.info("Обработано: %d / %d датасетов", len(all_meta), len(all_meta) + len(failed))
    logger.info("Домены: %s", summary["domains"])
    logger.info("Всего строк: A=%d, B=%d", summary["total_rows_a"], summary["total_rows_b"])
    if failed:
        logger.warning("Не удалось: %s", failed)
    logger.info("Сводка: %s", summary_path)

    return all_meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Исследование и генерация датасетов")
    parser.add_argument(
        "--dataset", default=None,
        help=f"Имя одного датасета. Доступные: {list(DATASETS.keys())}",
    )
    parser.add_argument("--all", action="store_true",
                        help="Обработать ВСЕ доступные датасеты")
    parser.add_argument("--data-dir", default="data", help="Путь для данных")
    parser.add_argument("--generate", action="store_true",
                        help="Генерировать синтетический датасет (нужен Ollama)")
    parser.add_argument("--embeddings", action="store_true",
                        help="Генерировать эмбеддинги (столбцы, строки, токены)")
    parser.add_argument("--model-tag", default=None,
                        help="Namespace для data/embeddings/rows/<tag>/ "
                             "(по умолч. из config.entity_resolution.token_model_tag)")
    parser.add_argument("--row-model-name", default=None,
                        help="HF-имя модели-кодировщика строк. По умолч. "
                             "из config.entity_resolution.token_model_name")
    parser.add_argument("--pooling", default=None, choices=["cls", "mean"],
                        help="Как агрегировать last_hidden_state (cls/mean)")
    parser.add_argument("--row-prefix", default=None,
                        help="Префикс для текстов предложений (для e5: 'query: ')")
    parser.add_argument("--trust-remote-code", action="store_true", default=False)
    parser.add_argument("--skip-columns", action="store_true",
                        help="Не генерировать column embeddings заново (qwen3), "
                             "если они уже лежат в data/embeddings/columns/<ds>/")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))

    # Определяем список датасетов
    if args.all:
        dataset_names = list(DATASETS.keys())
    elif args.dataset:
        dataset_names = [args.dataset]
    else:
        parser.error("Укажите --dataset <name> или --all")
        return

    # Анализ — только при интерактивной эксплорации, не при --embeddings / --generate
    if not (args.embeddings or args.generate):
        for name in dataset_names:
            try:
                csv_path = download_dataset(name, config.data_dir)
                tables = load_dataset(csv_path, name=name)
                explore_dataset(name, tables)
            except Exception:
                logger.exception("Ошибка загрузки %s — пропуск", name)

    # Генерация (опционально)
    all_meta = None
    if args.generate:
        all_meta = generate_unified_dataset(dataset_names, config)

    # Генерация эмбеддингов
    if args.embeddings:
        er_cfg = config.entity_resolution
        generate_all_embeddings(
            dataset_names, config, all_meta,
            model_tag=args.model_tag or er_cfg.token_model_tag,
            row_model_name=args.row_model_name or er_cfg.token_model_name,
            pooling=args.pooling or er_cfg.row_pooling,
            row_prefix=args.row_prefix if args.row_prefix is not None else er_cfg.row_prefix,
            trust_remote_code=args.trust_remote_code or er_cfg.token_model_trust_remote_code,
            skip_columns=args.skip_columns,
        )


def generate_all_embeddings(
    dataset_names: list[str],
    config: Config,
    all_meta: list[dict] | None = None,
    *,
    model_tag: str = "rubert-tiny2",
    row_model_name: str = "cointegrated/rubert-tiny2",
    pooling: str = "cls",
    row_prefix: str = "",
    trust_remote_code: bool = False,
    skip_columns: bool = False,
) -> None:
    """Генерация эмбеддингов для всех датасетов.

    Column embeddings (qwen3) — shared: ``data/embeddings/columns/<dataset>/``.
    Row embeddings (HF-модель) — per-tag: ``data/embeddings/rows/<model_tag>/<dataset>/``.
    """
    from table_unifier.ollama_client import OllamaClient

    client = OllamaClient(config.ollama) if not skip_columns else None
    token_embedder = TokenEmbedder(
        model_name=row_model_name,
        pooling=pooling,
        row_prefix=row_prefix,
        trust_remote_code=trust_remote_code,
    )

    logger.info("Row-encoder: tag=%s, model=%s, pooling=%s, prefix=%r",
                model_tag, row_model_name, pooling, row_prefix)

    # Индекс синтетических таблиц по имени датасета
    synth_by_name: dict[str, dict] = {}
    if all_meta:
        synth_by_name = {m["name"]: m for m in all_meta}

    for name in dataset_names:
        logger.info("=" * 60)
        logger.info("Генерация эмбеддингов: %s", name)
        logger.info("=" * 60)

        col_ds_dir = columns_dir(config.data_dir, name)
        row_ds_dir = rows_dir(config.data_dir, model_tag, name)
        col_ds_dir.mkdir(parents=True, exist_ok=True)
        row_ds_dir.mkdir(parents=True, exist_ok=True)

        # Определяем таблицы (приоритет: in-memory → synthetic CSV → original)
        synth_dir = config.data_dir / "synthetic" / name
        if name in synth_by_name:
            table_a = synth_by_name[name]["synth_a"]
            table_b = synth_by_name[name]["synth_b"]
            logger.info("[%s] Используем синтетические таблицы (in-memory)", name)
        elif (synth_dir / "tableA_synth.csv").exists():
            table_a = pd.read_csv(synth_dir / "tableA_synth.csv")
            table_b = pd.read_csv(synth_dir / "tableB_synth.csv")
            logger.info("[%s] Загружены синтетические таблицы из %s", name, synth_dir)
        else:
            try:
                csv_path = download_dataset(name, config.data_dir)
                tables = load_dataset(csv_path, name=name)
                table_a, table_b = tables["tableA"], tables["tableB"]
                logger.info("[%s] Используем оригинальные таблицы", name)
            except Exception:
                logger.exception("Не удалось загрузить %s — пропуск", name)
                continue

        columns_a = [c for c in table_a.columns if c != "id"]
        columns_b = [c for c in table_b.columns if c != "id"]

        # ---- 1. Column embeddings (Ollama qwen3, shared) ---- #
        col_a_path = col_ds_dir / "column_embeddings_a.npz"
        col_b_path = col_ds_dir / "column_embeddings_b.npz"
        if skip_columns and col_a_path.exists() and col_b_path.exists():
            logger.info("[%s] Column embeddings уже есть в %s — пропуск", name, col_ds_dir)
        else:
            logger.info("[%s] Column embeddings …", name)
            col_emb_a = generate_column_embeddings(client, table_a, columns_a)
            col_emb_b = generate_column_embeddings(client, table_b, columns_b)
            np.savez(col_a_path, **{col: vec for col, vec in col_emb_a.items()})
            np.savez(col_b_path, **{col: vec for col, vec in col_emb_b.items()})
            pd.DataFrame({"col_a": columns_a}).to_csv(col_ds_dir / "columns_a.csv", index=False)
            pd.DataFrame({"col_b": columns_b}).to_csv(col_ds_dir / "columns_b.csv", index=False)
            logger.info("[%s] Column embeddings → %s (%d-dim)",
                        name, col_ds_dir,
                        next(iter(col_emb_a.values())).shape[0])

        # ---- 2. Row embeddings (per-tag) ---- #
        logger.info("[%s] Row embeddings (%s, %s) …", name, model_tag, pooling)
        row_texts_a = [serialize_row(row, columns_a) for _, row in table_a.iterrows()]
        row_texts_b = [serialize_row(row, columns_b) for _, row in table_b.iterrows()]
        row_emb_a = token_embedder.embed_sentences(
            row_texts_a, desc=f"[{name}] A ({len(row_texts_a):,} rows)")
        row_emb_b = token_embedder.embed_sentences(
            row_texts_b, desc=f"[{name}] B ({len(row_texts_b):,} rows)")

        np.save(row_ds_dir / "row_embeddings_a.npy", row_emb_a)
        np.save(row_ds_dir / "row_embeddings_b.npy", row_emb_b)

        logger.info("[%s] rows_a=%s, rows_b=%s → %s",
                    name, row_emb_a.shape, row_emb_b.shape, row_ds_dir)

    logger.info("Column → %s | Rows → %s",
                columns_dir(config.data_dir),
                rows_dir(config.data_dir, model_tag))


if __name__ == "__main__":
    main()
