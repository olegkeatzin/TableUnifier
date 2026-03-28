# src/table_unifier/evaluation/clustering.py
"""HDBSCAN кластеризация эмбеддингов и метрики оценки."""

from __future__ import annotations

import logging

import hdbscan
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: torch.Tensor,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Кластеризация эмбеддингов через HDBSCAN.

    Args:
        embeddings: [N, D] tensor (L2-normalized рекомендуется)
        min_cluster_size: минимальный размер кластера
        min_samples: минимальное число точек в окрестности (default=min_cluster_size)
        metric: метрика расстояния

    Returns:
        labels: [N] numpy array, -1 = шум
    """
    X = embeddings.detach().cpu().numpy().astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info("HDBSCAN: %d кластеров, %d шум (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(labels))

    return labels


def evaluate_clusters(
    labels: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> dict:
    """Оценить качество кластеризации.

    Args:
        labels: [N] — метки кластеров (-1 = шум)
        ground_truth: [N] — истинные метки (опционально)

    Returns:
        dict с метриками:
        - coverage: доля точек не в шуме
        - n_clusters: число кластеров
        - noise_ratio: доля шума
        - ari: Adjusted Rand Index (если есть ground_truth)
        - nmi: Normalized Mutual Info (если есть ground_truth)
    """
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    metrics: dict = {
        "coverage": (n_total - n_noise) / n_total if n_total > 0 else 0.0,
        "n_clusters": n_clusters,
        "noise_ratio": n_noise / n_total if n_total > 0 else 0.0,
        "n_total": n_total,
        "n_noise": n_noise,
    }

    if ground_truth is not None:
        # Оцениваем только по не-шумовым точкам
        non_noise = labels != -1
        if non_noise.sum() > 0:
            metrics["ari"] = float(adjusted_rand_score(ground_truth[non_noise], labels[non_noise]))
            metrics["nmi"] = float(normalized_mutual_info_score(ground_truth[non_noise], labels[non_noise]))

    return metrics
