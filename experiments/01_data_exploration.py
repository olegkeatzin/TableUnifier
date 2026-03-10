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
from table_unifier.dataset.schema_augmentation import augment_schema, apply_schema_injection
from table_unifier.dataset.value_corruption import corrupt_dataframe

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
    summary = {
        "total_datasets": len(all_meta),
        "failed_datasets": failed,
        "datasets": all_meta,
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

    # Анализ
    for name in dataset_names:
        try:
            csv_path = download_dataset(name, config.data_dir)
            tables = load_dataset(csv_path, name=name)
            explore_dataset(name, tables)
        except Exception:
            logger.exception("Ошибка загрузки %s — пропуск", name)

    # Генерация (опционально)
    if args.generate:
        generate_unified_dataset(dataset_names, config)


if __name__ == "__main__":
    main()
