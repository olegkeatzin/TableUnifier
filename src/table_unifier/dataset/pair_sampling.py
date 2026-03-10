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
        pair = (str(row[ltable_id_col]), str(row[rtable_id_col]))
        if int(row[label_col]) == 1:
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
    ids_b = sorted(id_to_idx_b.keys())

    for a_id, p_id in positives:
        a_idx = id_to_idx_a[a_id]
        sims = sim_matrix[a_idx]
        # Сортировка по убыванию сходства
        ranked = np.argsort(-sims)
        count = 0
        for b_local_idx in ranked:
            n_id = ids_b[b_local_idx]
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
        result.append([id_to_global_a[a_id], id_to_global_b[p_id], id_to_global_b[n_id]])
    return torch.tensor(result, dtype=torch.long)
