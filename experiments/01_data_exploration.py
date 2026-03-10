"""Эксперимент 1 — Исследование и генерация датасета.

Загрузка данных из Magellan Data Repository, анализ, генерация
синтетического датасета (Schema Injection + Value Corruption).

Использование:
    python -m experiments.01_data_exploration --dataset Beer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from table_unifier.config import Config
from table_unifier.dataset.download import DATASETS, download_dataset, load_dataset
from table_unifier.dataset.schema_augmentation import augment_schema, apply_schema_injection
from table_unifier.dataset.value_corruption import corrupt_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def explore_dataset(tables: dict[str, pd.DataFrame]) -> None:
    """Вывести статистику по датасету."""
    for name, df in tables.items():
        print(f"\n{'='*60}")
        print(f"  {name}: {df.shape[0]} строк × {df.shape[1]} столбцов")
        print(f"  Столбцы: {list(df.columns)}")
        print(f"  Типы:\n{df.dtypes.to_string()}")
        print(f"  Пропуски:\n{df.isnull().sum().to_string()}")
        if name in ("train", "valid", "test"):
            if "label" in df.columns:
                counts = df["label"].value_counts()
                print(f"  Баланс label:\n{counts.to_string()}")
                ratio = counts.get(1, 0) / max(counts.get(0, 1), 1)
                print(f"  Ratio (pos/neg): {ratio:.3f}")
        else:
            print(df.head(3).to_string())


def generate_synthetic_dataset(
    tables: dict[str, pd.DataFrame],
    config: Config,
    domain: str = "general",
) -> None:
    """Генерация синтетического датасета (Schema + Value Noise)."""
    from table_unifier.ollama_client import OllamaClient

    client = OllamaClient(config.ollama)
    table_a = tables["tableA"]
    table_b = tables["tableB"]
    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    # ---- A. Schema Injection ---- #
    logger.info("=== Schema Injection ===")
    synonym_map_a = augment_schema(client, columns_a, domain=domain, n_variants=3)
    synonym_map_b = augment_schema(client, columns_b, domain=domain, n_variants=3)

    rename_a = apply_schema_injection(columns_a, synonym_map_a, variant_idx=0)
    rename_b = apply_schema_injection(columns_b, synonym_map_b, variant_idx=1)

    synth_a = table_a.rename(columns=rename_a)
    synth_b = table_b.rename(columns=rename_b)

    logger.info("Table A: %s → %s", columns_a, list(rename_a.values()))
    logger.info("Table B: %s → %s", columns_b, list(rename_b.values()))

    # ---- B. Value Corruption ---- #
    logger.info("=== Value Corruption ===")
    noisy_a = corrupt_dataframe(synth_a, row_prob=0.4, cell_prob=0.3)
    noisy_b = corrupt_dataframe(synth_b, row_prob=0.4, cell_prob=0.3)

    # Сохранение
    out_dir = config.data_dir / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    noisy_a.to_csv(out_dir / "tableA_synth.csv", index=False)
    noisy_b.to_csv(out_dir / "tableB_synth.csv", index=False)

    # Ground truth: маппинг столбцов
    gt = {rename_a.get(c, c): rename_b.get(c, c) for c in columns_a if c in columns_b}
    pd.DataFrame(list(gt.items()), columns=["col_A", "col_B"]).to_csv(
        out_dir / "schema_ground_truth.csv", index=False,
    )

    logger.info("Синтетический датасет сохранён в %s", out_dir)
    logger.info("Schema GT: %s", gt)


def main() -> None:
    parser = argparse.ArgumentParser(description="Исследование датасета")
    parser.add_argument(
        "--dataset", default="beer",
        help=f"Имя датасета. Доступные: {list(DATASETS.keys())}",
    )
    parser.add_argument("--data-dir", default="data", help="Путь для данных")
    parser.add_argument("--generate", action="store_true",
                        help="Генерировать синтетический датасет (нужен Ollama)")
    parser.add_argument("--domain", default="general",
                        help="Домен для генерации синонимов столбцов")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))

    # Загрузка
    path = download_dataset(args.dataset, config.data_dir)
    tables = load_dataset(path, name=args.dataset)

    # Анализ
    explore_dataset(tables)

    # Генерация (опционально)
    if args.generate:
        generate_synthetic_dataset(tables, config, domain=args.domain)


if __name__ == "__main__":
    main()
