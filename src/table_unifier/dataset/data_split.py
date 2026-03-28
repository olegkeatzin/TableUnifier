# src/table_unifier/dataset/data_split.py
"""Стратифицированный split по строкам для GNN.

Группирует строки в связные компоненты (через labeled pairs),
затем распределяет компоненты по train/val/test с сохранением
пропорций positives/negatives.

Гарантия: строки из test-пар не появляются в train-графе.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_components(pairs: torch.Tensor) -> list[set[int]]:
    """Группировка строк в связные компоненты через Union-Find.

    Если строка A_3 участвует в паре с B_7, обе попадают в одну компоненту.
    Вся компонента пойдёт в один split.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in pairs:
        a, b = row[0].item(), row[1].item()
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    groups: dict[int, set[int]] = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)

    return list(groups.values())


def split_rows_stratified(
    labeled_pairs: torch.Tensor,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Разделить labeled pairs на train/val/test по строкам.

    Args:
        labeled_pairs: [N, 3] — (global_idx_a, global_idx_b, label)
        ratios: (train, val, test) доли
        seed: random seed

    Returns:
        (train_pairs, val_pairs, test_pairs) — каждый [M, 3]
        Гарантия: множества строк в split-ах не пересекаются.
    """
    rng = np.random.default_rng(seed)

    components = _build_components(labeled_pairs)

    # Для каждой компоненты считаем число positives
    comp_rows = []
    for comp in components:
        mask = torch.zeros(len(labeled_pairs), dtype=torch.bool)
        for i, row in enumerate(labeled_pairs):
            if row[0].item() in comp or row[1].item() in comp:
                mask[i] = True
        comp_pairs = labeled_pairs[mask]
        n_pos = (comp_pairs[:, 2] == 1).sum().item()
        comp_rows.append((comp, mask, n_pos, len(comp_pairs)))

    # Сортируем компоненты по размеру (большие первыми — лучше распределяются)
    comp_rows.sort(key=lambda x: -x[3])

    # Greedy: назначаем компоненту в split с наибольшим дефицитом
    target_pairs = [r * len(labeled_pairs) for r in ratios]
    current_pairs = [0.0, 0.0, 0.0]
    assignments: list[int] = []  # split index per component

    for _, _, _, n_pairs in comp_rows:
        # Выбираем split, у которого наибольший дефицит
        deficits = [target_pairs[i] - current_pairs[i] for i in range(3)]
        best = int(np.argmax(deficits))
        assignments.append(best)
        current_pairs[best] += n_pairs

    # Собираем пары по split
    split_masks = [torch.zeros(len(labeled_pairs), dtype=torch.bool) for _ in range(3)]
    for (_, mask, _, _), split_idx in zip(comp_rows, assignments):
        split_masks[split_idx] |= mask

    train_pairs = labeled_pairs[split_masks[0]]
    val_pairs = labeled_pairs[split_masks[1]]
    test_pairs = labeled_pairs[split_masks[2]]

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train_pairs), 100 * len(train_pairs) / len(labeled_pairs),
        len(val_pairs), 100 * len(val_pairs) / len(labeled_pairs),
        len(test_pairs), 100 * len(test_pairs) / len(labeled_pairs),
    )

    return train_pairs, val_pairs, test_pairs
