"""Exp 17 — Пересплит уже построенного unified-графа по per-dataset стратификации.

Исходный split (11_build_graph.py) делает Union-Find компоненты глобально, что
приводит к одной гигантской компоненте на ozon (URL clustering 318K пар) и
аналогичной проблеме на других больших датасетах. Эти компоненты уходят
целиком в один split → train/val/test получают разные распределения.

Скрипт:
  1. Загружает существующие train/val/test_pairs.pt и dataset_mappings.json
  2. Восстанавливает all_labeled и для каждой пары определяет dataset_id
     (через id_to_global_a — глобальный индекс уникально принадлежит датасету)
  3. Пересплитывает per-dataset (каждый датасет 70/15/15)
  4. Бэкапит старые файлы в *.legacy.pt и сохраняет новые

Использование:
    uv run python -m experiments.17.11b_resplit --graph-subdir v17_views
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

import numpy as np
import torch

from table_unifier.config import Config
from table_unifier.dataset.data_split import split_rows_stratified
from table_unifier.paths import unified_dir

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-dataset пересплит unified графа")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--model-tag", default="bge-m3")
    parser.add_argument("--graph-subdir", default="v17_views")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = Config(data_dir=Path(args.data_dir))
    out_dir = unified_dir(cfg.data_dir, args.model_tag, args.graph_subdir)
    logger.info("Граф: %s", out_dir)

    # 1. Загрузка существующих сплитов + маппингов
    train = torch.load(out_dir / "train_pairs.pt", weights_only=False)
    val = torch.load(out_dir / "val_pairs.pt", weights_only=False)
    test = torch.load(out_dir / "test_pairs.pt", weights_only=False)
    all_labeled = torch.cat([train, val, test], dim=0)
    logger.info("Старые сплиты: train=%d, val=%d, test=%d → total=%d",
                len(train), len(val), len(test), len(all_labeled))

    with open(out_dir / "dataset_mappings.json") as f:
        mappings = json.load(f)

    # 2. global_idx → dataset_name
    #    Берём id_to_global_a (тот же row, что в pair[:, 0]) каждого датасета.
    #    Глобальные индексы уникальны → каждый row принадлежит ровно одному датасету.
    global_to_dataset: dict[int, str] = {}
    for name, m in mappings.items():
        for _local_id, gidx in m.get("id_to_global_a", {}).items():
            global_to_dataset[int(gidx)] = name
        for _local_id, gidx in m.get("id_to_global_b", {}).items():
            global_to_dataset[int(gidx)] = name

    # 3. dataset_id для каждой пары (через row 'a')
    name_to_id = {n: i for i, n in enumerate(sorted(mappings.keys()))}
    pair_dataset_ids = []
    missing = 0
    for ga, _gb, _label in all_labeled.tolist():
        name = global_to_dataset.get(int(ga))
        if name is None:
            missing += 1
            pair_dataset_ids.append(-1)
        else:
            pair_dataset_ids.append(name_to_id[name])
    if missing:
        logger.warning("Не удалось определить датасет для %d пар (будут пропущены)", missing)
    pair_dataset_ids = torch.tensor(pair_dataset_ids, dtype=torch.long)
    valid_mask = pair_dataset_ids >= 0
    all_labeled = all_labeled[valid_mask]
    pair_dataset_ids = pair_dataset_ids[valid_mask]

    # Распределение по датасетам (до сплита)
    logger.info("Распределение пар по датасетам:")
    for did, name in sorted(((v, k) for k, v in name_to_id.items())):
        n = int((pair_dataset_ids == did).sum())
        if n:
            logger.info("  %s: %d", name, n)

    # 4. Per-dataset сплит
    train_new, val_new, test_new = split_rows_stratified(
        all_labeled, ratios=(0.7, 0.15, 0.15), seed=args.seed,
        dataset_ids=pair_dataset_ids,
    )

    # 5. Бэкап + сохранение
    for name in ("train_pairs", "val_pairs", "test_pairs"):
        src = out_dir / f"{name}.pt"
        dst = out_dir / f"{name}.legacy.pt"
        if src.exists() and not dst.exists():
            shutil.copy(src, dst)
            logger.info("Бэкап: %s → %s", src.name, dst.name)

    torch.save(train_new, out_dir / "train_pairs.pt")
    torch.save(val_new, out_dir / "val_pairs.pt")
    torch.save(test_new, out_dir / "test_pairs.pt")

    # Обновить stats.json
    stats_path = out_dir / "stats.json"
    if stats_path.exists():
        with open(stats_path) as f:
            stats = json.load(f)
        stats["n_train"] = int(len(train_new))
        stats["n_val"] = int(len(val_new))
        stats["n_test"] = int(len(test_new))
        stats["split_strategy"] = "per_dataset"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        logger.info("stats.json обновлён")

    logger.info("Готово. Новые сплиты: train=%d, val=%d, test=%d",
                len(train_new), len(val_new), len(test_new))


if __name__ == "__main__":
    main()
