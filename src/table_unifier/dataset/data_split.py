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
    dataset_ids: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Разделить labeled pairs на train/val/test по строкам.

    Args:
        labeled_pairs: [N, 3] — (global_idx_a, global_idx_b, label)
        ratios: (train, val, test) доли
        seed: random seed
        dataset_ids: [N] — id датасета каждой пары. Если задано,
            стратификация выполняется per-dataset (каждый датасет
            делится 70/15/15 отдельно). Это важно когда один датасет —
            одна гигантская компонента (например, ozon URL clustering).

    Returns:
        (train_pairs, val_pairs, test_pairs) — каждый [M, 3]
        Гарантия: множества строк в split-ах не пересекаются.
    """
    if dataset_ids is not None:
        return _split_per_dataset(labeled_pairs, dataset_ids, ratios, seed)

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


def _split_one_dataset(
    pairs: torch.Tensor,
    ratios: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Сплит одного датасета: компоненты → greedy в 3 ведра."""
    if len(pairs) == 0:
        empty = torch.zeros((0, 3), dtype=pairs.dtype)
        return empty, empty.clone(), empty.clone()

    components = _build_components(pairs)
    node_to_comp = {n: cid for cid, comp in enumerate(components) for n in comp}
    pairs_np = pairs.numpy()
    comp_id_per_pair = np.fromiter(
        (node_to_comp[int(a)] for a in pairs_np[:, 0]),
        dtype=np.int64, count=len(pairs_np),
    )
    n_comps = len(components)
    n_pairs_per_comp = np.bincount(comp_id_per_pair, minlength=n_comps)
    order = np.argsort(-n_pairs_per_comp)

    n_total = len(pairs)
    target = np.array(ratios, dtype=np.float64) * n_total
    current = np.zeros(3, dtype=np.float64)
    split_per_comp = np.empty(n_comps, dtype=np.int64)

    for cid in order:
        best = int(np.argmax(target - current))
        split_per_comp[cid] = best
        current[best] += n_pairs_per_comp[cid]

    split_per_pair = split_per_comp[comp_id_per_pair]
    return (
        pairs[torch.from_numpy(split_per_pair == 0)],
        pairs[torch.from_numpy(split_per_pair == 1)],
        pairs[torch.from_numpy(split_per_pair == 2)],
    )


def _split_per_dataset(
    labeled_pairs: torch.Tensor,
    dataset_ids: torch.Tensor,
    ratios: tuple[float, float, float],
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-dataset стратификация: каждый датасет делится 70/15/15 независимо."""
    assert len(dataset_ids) == len(labeled_pairs), (
        f"dataset_ids ({len(dataset_ids)}) != labeled_pairs ({len(labeled_pairs)})"
    )
    ids_np = dataset_ids.numpy() if isinstance(dataset_ids, torch.Tensor) else np.asarray(dataset_ids)
    unique_ids = np.unique(ids_np)

    train_parts, val_parts, test_parts = [], [], []
    for did in unique_ids:
        mask = torch.from_numpy(ids_np == did)
        ds_pairs = labeled_pairs[mask]
        tr, va, te = _split_one_dataset(ds_pairs, ratios)
        train_parts.append(tr)
        val_parts.append(va)
        test_parts.append(te)
        logger.info(
            "  dataset=%s: %d pairs → train=%d, val=%d, test=%d",
            did, len(ds_pairs), len(tr), len(va), len(te),
        )

    train_pairs = torch.cat(train_parts, dim=0) if train_parts else labeled_pairs[:0]
    val_pairs = torch.cat(val_parts, dim=0) if val_parts else labeled_pairs[:0]
    test_pairs = torch.cat(test_parts, dim=0) if test_parts else labeled_pairs[:0]

    n_total = len(labeled_pairs)
    logger.info(
        "Per-dataset split: train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        len(train_pairs), 100 * len(train_pairs) / max(n_total, 1),
        len(val_pairs), 100 * len(val_pairs) / max(n_total, 1),
        len(test_pairs), 100 * len(test_pairs) / max(n_total, 1),
    )

    # Перемешать внутри каждого split (важно для NeighborLoader iter)
    gen = torch.Generator().manual_seed(seed)
    for t in (train_pairs, val_pairs, test_pairs):
        if len(t) > 0:
            idx = torch.randperm(len(t), generator=gen)
            t[:] = t[idx]

    return train_pairs, val_pairs, test_pairs
