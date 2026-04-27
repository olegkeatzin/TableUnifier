"""Общая фитнес-функция и pair-метрики для ГА-кластеризации (HDBSCAN / CC).

Правило на парах одинаковое для обоих кластеризаторов::

    match(a, b)  ⇔  cluster[a] == cluster[b]   и   cluster[a] != noise_label

Для HDBSCAN ``noise_label = -1``. Для connected-components шума нет
(каждая строка обязательно в каком-то компоненте), достаточно
``noise_label = None`` — тогда условие "≠ noise" тождественно true.

Фитнес — F-beta на парах с penalty за гигантский кластер:

* β конфигурируем (1.0 по умолчанию). β < 1 наказывает FP сильнее (актуально
  для ER, где ложные совпадения дороже пропущенных).
* Если максимальный non-noise кластер занимает > ``giant_cluster_threshold``
  всех строк, F-beta домножается на ``giant_cluster_penalty`` (< 1). Защита
  от «слить всё в один компонент» при несбалансированных val_pairs.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import torch
from sklearn.metrics import f1_score, precision_score, recall_score


def _to_np_pairs(pairs: torch.Tensor | np.ndarray) -> np.ndarray:
    return pairs.cpu().numpy() if isinstance(pairs, torch.Tensor) else np.asarray(pairs)


def cluster_labels_to_pair_preds(
    cluster_labels: np.ndarray,
    pairs: torch.Tensor | np.ndarray,
    *,
    noise_label: int | None = -1,
) -> np.ndarray:
    """match(a,b) = same cluster (и не noise, если noise_label указан)."""
    pairs_np = _to_np_pairs(pairs)
    la = cluster_labels[pairs_np[:, 0]]
    lb = cluster_labels[pairs_np[:, 1]]
    same = la == lb
    if noise_label is None:
        return same.astype(int)
    return (same & (la != noise_label)).astype(int)


def pair_metrics_from_labels(
    cluster_labels: np.ndarray,
    pairs: torch.Tensor | np.ndarray,
    *,
    noise_label: int | None = -1,
) -> dict[str, float]:
    """F1/Precision/Recall на парах + статистика кластеров."""
    pairs_np = _to_np_pairs(pairs)
    y_true = pairs_np[:, 2].astype(int)
    y_pred = cluster_labels_to_pair_preds(cluster_labels, pairs_np, noise_label=noise_label)

    if noise_label is not None:
        non_noise = cluster_labels[cluster_labels != noise_label]
        n_noise = int((cluster_labels == noise_label).sum())
    else:
        non_noise = cluster_labels
        n_noise = 0
    n_clusters = int(len(np.unique(non_noise))) if len(non_noise) > 0 else 0

    if len(np.unique(y_true)) < 2:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0,
                "n_clusters": n_clusters, "n_noise": n_noise,
                "n_pairs": int(len(y_true)), "n_pos": int(y_true.sum()),
                "n_predicted_pos": int(y_pred.sum())}

    return {
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "n_clusters": n_clusters,
        "n_noise": n_noise,
        "n_pairs": int(len(y_true)),
        "n_pos": int(y_true.sum()),
        "n_predicted_pos": int(y_pred.sum()),
    }


def _fbeta(precision: float, recall: float, beta: float) -> float:
    if precision + recall == 0.0:
        return 0.0
    b2 = beta * beta
    return (1.0 + b2) * precision * recall / (b2 * precision + recall)


def _max_cluster_fraction(cluster_labels: np.ndarray, noise_label: int | None) -> float:
    if noise_label is not None:
        valid = cluster_labels[cluster_labels != noise_label]
    else:
        valid = cluster_labels
    if len(valid) == 0:
        return 0.0
    counts = Counter(valid.tolist())
    return max(counts.values()) / len(cluster_labels)


def pair_fitness_from_labels(
    cluster_labels: np.ndarray,
    pairs: torch.Tensor | np.ndarray,
    *,
    fbeta: float = 1.0,
    giant_cluster_threshold: float = 0.5,
    giant_cluster_penalty: float = 0.5,
    noise_label: int | None = -1,
) -> float:
    """F-beta на парах с penalty за «гигантский кластер».

    Возвращает 0.0, если предсказаний нет или y_true вырожден.
    """
    pairs_np = _to_np_pairs(pairs)
    y_true = pairs_np[:, 2].astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.0
    y_pred = cluster_labels_to_pair_preds(cluster_labels, pairs_np, noise_label=noise_label)

    p = float(precision_score(y_true, y_pred, zero_division=0))
    r = float(recall_score(y_true, y_pred, zero_division=0))
    score = _fbeta(p, r, fbeta)

    if giant_cluster_threshold < 1.0:
        frac = _max_cluster_fraction(cluster_labels, noise_label=noise_label)
        if frac > giant_cluster_threshold:
            score *= giant_cluster_penalty

    return float(score)
