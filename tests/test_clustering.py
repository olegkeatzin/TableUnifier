# tests/test_clustering.py
"""Тесты для evaluation/clustering.py."""

import numpy as np
import pytest
import torch

from table_unifier.evaluation.clustering import cluster_embeddings, evaluate_clusters


class TestClusterEmbeddings:
    def test_returns_labels(self):
        emb = torch.randn(50, 16)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        labels = cluster_embeddings(emb, min_cluster_size=5)
        assert labels.shape == (50,)
        assert labels.dtype == np.int64 or labels.dtype == np.intp

    def test_noise_label_is_minus_one(self):
        """HDBSCAN помечает шум как -1."""
        emb = torch.randn(100, 16)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        labels = cluster_embeddings(emb, min_cluster_size=5)
        assert -1 in labels or len(np.unique(labels)) > 1

    def test_similar_points_same_cluster(self):
        """Два кластера явно разделённых точек."""
        cluster_a = torch.randn(20, 8) + torch.tensor([5.0, 0, 0, 0, 0, 0, 0, 0])
        cluster_b = torch.randn(20, 8) + torch.tensor([-5.0, 0, 0, 0, 0, 0, 0, 0])
        emb = torch.cat([cluster_a, cluster_b])
        labels = cluster_embeddings(emb, min_cluster_size=5)
        # Точки 0-19 должны быть в одном кластере, 20-39 — в другом
        assert labels[0] == labels[10]
        assert labels[20] == labels[30]
        assert labels[0] != labels[20]


class TestEvaluateClusters:
    def test_with_ground_truth(self):
        labels = np.array([0, 0, 1, 1, -1])
        ground_truth = np.array([0, 0, 1, 1, 0])
        metrics = evaluate_clusters(labels, ground_truth)
        assert "ari" in metrics
        assert "nmi" in metrics
        assert "coverage" in metrics

    def test_coverage_excludes_noise(self):
        labels = np.array([0, 0, -1, -1, -1])
        metrics = evaluate_clusters(labels)
        assert metrics["coverage"] == pytest.approx(0.4)

    def test_no_ground_truth(self):
        labels = np.array([0, 0, 1, 1, -1])
        metrics = evaluate_clusters(labels)
        assert "coverage" in metrics
        assert "n_clusters" in metrics
        assert "ari" not in metrics
