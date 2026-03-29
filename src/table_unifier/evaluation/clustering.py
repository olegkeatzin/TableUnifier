# src/table_unifier/evaluation/clustering.py
"""Threshold-based evaluation for Entity Resolution.

Production pipeline:
  1. GNN produces L2-normalised row embeddings
  2. Cosine similarity between candidate pairs
  3. Threshold θ → match / no-match

Evaluation:
  - find_best_threshold(): sweep on val pairs → θ that maximises F1
  - evaluate_pairs_at_threshold(): apply θ to test pairs → P / R / F1
  - evaluate_pairs_auc(): ROC-AUC + Average Precision (threshold-free)
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: torch.Tensor | np.ndarray,
    min_cluster_size: int = 15,
    metric: str = "euclidean",
) -> np.ndarray:
    """Cluster row embeddings with HDBSCAN.

    Returns:
        Integer label array (length = n_rows). Label -1 means noise.
    """
    import hdbscan

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    clusterer = hdbscan.HDBSCAN(min_cluster_size=min_cluster_size, metric=metric)
    return clusterer.fit_predict(embeddings)


def _cosine_scores(embeddings: torch.Tensor, pairs: torch.Tensor) -> np.ndarray:
    """Cosine similarity for each pair (embeddings already L2-normalised)."""
    idx_a = pairs[:, 0]
    idx_b = pairs[:, 1]
    return (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1).numpy()


def find_best_threshold(
    embeddings: torch.Tensor,
    val_pairs: torch.Tensor,
    n_thresholds: int = 200,
) -> tuple[float, float]:
    """Find cosine similarity threshold that maximises F1 on validation pairs.

    Returns:
        (best_threshold, best_f1)
    """
    scores = _cosine_scores(embeddings, val_pairs)
    labels = val_pairs[:, 2].numpy()

    if len(np.unique(labels)) < 2:
        logger.warning("Val pairs have single class — returning default threshold 0.5")
        return 0.5, 0.0

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    # precision_recall_curve returns len(thresholds) = len(precision) - 1
    f1_scores = np.where(
        (precision[:-1] + recall[:-1]) > 0,
        2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1]),
        0.0,
    )
    best_idx = np.argmax(f1_scores)
    return float(thresholds[best_idx]), float(f1_scores[best_idx])


def evaluate_pairs_at_threshold(
    embeddings: torch.Tensor,
    pairs: torch.Tensor,
    threshold: float,
) -> dict:
    """Apply threshold to pairs → Precision, Recall, F1.

    This mirrors the production decision: sim >= θ → duplicate.
    """
    scores = _cosine_scores(embeddings, pairs)
    labels = pairs[:, 2].numpy()

    if len(np.unique(labels)) < 2:
        return {}

    predictions = (scores >= threshold).astype(int)
    return {
        "threshold": threshold,
        "f1": float(f1_score(labels, predictions)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "n_pairs": int(len(labels)),
        "n_pos": int(labels.sum()),
        "n_predicted_pos": int(predictions.sum()),
    }


def evaluate_pairs_auc(
    embeddings: torch.Tensor,
    pairs: torch.Tensor,
) -> dict:
    """Threshold-free metrics: ROC-AUC and Average Precision."""
    scores = _cosine_scores(embeddings, pairs)
    labels = pairs[:, 2].numpy()

    if len(np.unique(labels)) < 2:
        return {}

    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "avg_precision": float(average_precision_score(labels, scores)),
    }
