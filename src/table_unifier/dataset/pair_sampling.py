"""Формирование пар (positive / hard-negative) для обучения.

Этап 2b из схемы Синтетический датасет.canvas.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import torch
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)


def split_labeled_pairs(
    labels_df: pd.DataFrame,
    ltable_id_col: str = "ltable_id",
    rtable_id_col: str = "rtable_id",
    label_col: str = "label",
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Разделить размеченные пары на positives (label=1) и negatives (label=0)."""
    positives: list[tuple[str, str]] = []
    negatives: list[tuple[str, str]] = []
    for _, row in labels_df.iterrows():
        label_val = row[label_col]
        if pd.isna(label_val):
            continue
        pair = (str(row[ltable_id_col]), str(row[rtable_id_col]))
        if int(label_val) == 1:
            positives.append(pair)
        else:
            negatives.append(pair)
    logger.info("Positives: %d, Negatives: %d", len(positives), len(negatives))
    return positives, negatives


def mine_hard_negatives(
    row_embeddings_a: np.ndarray,
    row_embeddings_b: np.ndarray,
    positives: list[tuple[str, str]],
    id_to_idx_a: dict[str, int],
    id_to_idx_b: dict[str, int],
    top_k: int = 5,
) -> list[tuple[str, str, str]]:
    """Найти hard-negative тройки на основе косинусной близости начальных эмбеддингов.

    Для каждой положительной пары (a, p) ищем top-k ближайших строк из B,
    которые **не** являются positive-партнёром.

    Returns:
        Список (anchor_id, positive_id, negative_id).
    """
    positive_set = set(positives)
    sim_matrix = cosine_similarity(row_embeddings_a, row_embeddings_b)

    triplets: list[tuple[str, str, str]] = []
    idx_to_id_b = {v: k for k, v in id_to_idx_b.items()}

    for a_id, p_id in positives:
        if a_id not in id_to_idx_a or p_id not in id_to_idx_b:
            continue
        a_idx = id_to_idx_a[a_id]
        sims = sim_matrix[a_idx]
        # Сортировка по убыванию сходства
        ranked = np.argsort(-sims)
        count = 0
        for b_local_idx in ranked:
            n_id = idx_to_id_b.get(int(b_local_idx))
            if n_id is None:
                continue
            if n_id == p_id or (a_id, n_id) in positive_set:
                continue
            triplets.append((a_id, p_id, n_id))
            count += 1
            if count >= top_k:
                break

    logger.info("Hard-negative triplets: %d", len(triplets))
    return triplets


def build_triplet_indices(
    triplets: list[tuple[str, str, str]],
    id_to_global_a: dict[str, int],
    id_to_global_b: dict[str, int],
) -> torch.Tensor:
    """Преобразовать тройки ID → глобальные индексы строк в графе.

    Returns:
        Tensor [N_triplets, 3] — (anchor_idx, positive_idx, negative_idx).
    """
    result = []
    for a_id, p_id, n_id in triplets:
        ga = id_to_global_a.get(a_id)
        gp = id_to_global_b.get(p_id)
        gn = id_to_global_b.get(n_id)
        if ga is not None and gp is not None and gn is not None:
            result.append([ga, gp, gn])
    if not result:
        return torch.zeros((0, 3), dtype=torch.long)
    return torch.tensor(result, dtype=torch.long)
