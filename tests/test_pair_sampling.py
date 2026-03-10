"""Тесты для dataset/pair_sampling.py."""

import numpy as np
import pandas as pd
import pytest
import torch

from table_unifier.dataset.pair_sampling import (
    build_triplet_indices,
    mine_hard_negatives,
    split_labeled_pairs,
)


class TestSplitLabeledPairs:
    def test_basic_split(self, labels_df):
        pos, neg = split_labeled_pairs(labels_df)
        assert len(pos) == 2
        assert len(neg) == 2

    def test_all_positive(self):
        df = pd.DataFrame({
            "ltable_id": [0, 1],
            "rtable_id": [0, 1],
            "label": [1, 1],
        })
        pos, neg = split_labeled_pairs(df)
        assert len(pos) == 2
        assert len(neg) == 0

    def test_all_negative(self):
        df = pd.DataFrame({
            "ltable_id": [0, 1],
            "rtable_id": [2, 3],
            "label": [0, 0],
        })
        pos, neg = split_labeled_pairs(df)
        assert len(pos) == 0
        assert len(neg) == 2

    def test_returns_correct_ids(self, labels_df):
        pos, neg = split_labeled_pairs(labels_df)
        assert (0, 0) in pos
        assert (1, 1) in pos
        assert (0, 2) in neg
        assert (2, 0) in neg


class TestMineHardNegatives:
    def test_returns_triplets(self):
        rng = np.random.default_rng(42)
        emb_a = rng.standard_normal((3, 16)).astype(np.float32)
        emb_b = rng.standard_normal((3, 16)).astype(np.float32)
        positives = [(0, 0)]
        id_to_idx_a = {0: 0, 1: 1, 2: 2}
        id_to_idx_b = {0: 0, 1: 1, 2: 2}

        triplets = mine_hard_negatives(
            emb_a, emb_b, positives, id_to_idx_a, id_to_idx_b, top_k=2,
        )
        assert len(triplets) == 2
        for a, p, n in triplets:
            assert a == 0
            assert p == 0
            assert n != 0  # negative не может быть positive-партнёром

    def test_top_k_limits(self):
        rng = np.random.default_rng(0)
        emb_a = rng.standard_normal((5, 8)).astype(np.float32)
        emb_b = rng.standard_normal((5, 8)).astype(np.float32)
        positives = [(0, 0), (1, 1)]
        id_to_idx_a = {i: i for i in range(5)}
        id_to_idx_b = {i: i for i in range(5)}

        triplets = mine_hard_negatives(
            emb_a, emb_b, positives, id_to_idx_a, id_to_idx_b, top_k=1,
        )
        # 2 positives × top_k=1
        assert len(triplets) == 2


class TestBuildTripletIndices:
    def test_shape(self):
        triplets = [(0, 0, 1), (1, 1, 2)]
        id_to_global_a = {0: 0, 1: 1}
        id_to_global_b = {0: 3, 1: 4, 2: 5}
        result = build_triplet_indices(triplets, id_to_global_a, id_to_global_b)
        assert result.shape == (2, 3)
        assert result.dtype == torch.long

    def test_values(self):
        triplets = [(10, 20, 30)]
        id_to_global_a = {10: 0}
        id_to_global_b = {20: 5, 30: 7}
        result = build_triplet_indices(triplets, id_to_global_a, id_to_global_b)
        assert result[0, 0].item() == 0
        assert result[0, 1].item() == 5
        assert result[0, 2].item() == 7
