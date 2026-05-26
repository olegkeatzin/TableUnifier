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

    pairs_np = pairs.numpy() if isinstance(pairs, torch.Tensor) else np.asarray(pairs)
    for a, b in pairs_np[:, :2].tolist():
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
    components = _build_components(labeled_pairs)

    # node → comp_id (по построению Union-Find row[0] и row[1] каждой пары
    # лежат в одной компоненте, поэтому для маркировки пары достаточно одного из них).
    node_to_comp: dict[int, int] = {}
    for cid, comp in enumerate(components):
        for node in comp:
            node_to_comp[node] = cid

    pairs_np = labeled_pairs.numpy()
    n_comps = len(components)

    # comp_id для каждой пары (по row[0])
    comp_id_per_pair = np.fromiter(
        (node_to_comp[int(a)] for a in pairs_np[:, 0]),
        dtype=np.int64, count=len(pairs_np),
    )

    n_pairs_per_comp = np.bincount(comp_id_per_pair, minlength=n_comps)

    # Greedy в порядке убывания размера компоненты
    order = np.argsort(-n_pairs_per_comp)
    n_total = len(labeled_pairs)
    target_pairs = np.array(ratios, dtype=np.float64) * n_total
    current_pairs = np.zeros(3, dtype=np.float64)
    split_per_comp = np.empty(n_comps, dtype=np.int64)

    for cid in order:
        deficits = target_pairs - current_pairs
        best = int(np.argmax(deficits))
        split_per_comp[cid] = best
        current_pairs[best] += n_pairs_per_comp[cid]

    # Векторизованно: split каждой пары
    split_per_pair = split_per_comp[comp_id_per_pair]

    train_pairs = labeled_pairs[torch.from_numpy(split_per_pair == 0)]
    val_pairs = labeled_pairs[torch.from_numpy(split_per_pair == 1)]
    test_pairs = labeled_pairs[torch.from_numpy(split_per_pair == 2)]

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train_pairs), 100 * len(train_pairs) / n_total,
        len(val_pairs), 100 * len(val_pairs) / n_total,
        len(test_pairs), 100 * len(test_pairs) / n_total,
    )

    return train_pairs, val_pairs, test_pairs
