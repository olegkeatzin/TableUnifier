# tests/test_data_split.py
"""Тесты для dataset/data_split.py."""

import pytest
import torch

from table_unifier.dataset.data_split import split_rows_stratified


class TestSplitRowsStratified:
    @pytest.fixture()
    def labeled_pairs(self):
        """Пары, образующие несколько связных компонент с pos/neg внутри каждой.

        comp A: rows {0, 10, 11} — 1 pos + 1 neg = 2 pairs
        comp B: rows {1, 12, 13} — 1 pos + 1 neg = 2 pairs
        comp C: rows {2, 14, 15} — 1 pos + 1 neg = 2 pairs
        comp D: rows {3, 16, 17} — 1 pos + 1 neg = 2 pairs
        comp E: rows {4, 18, 19} — 1 pos + 1 neg = 2 pairs
        """
        return torch.tensor([
            [0, 10, 1],  # comp A pos
            [0, 11, 0],  # comp A neg
            [1, 12, 1],  # comp B pos
            [1, 13, 0],  # comp B neg
            [2, 14, 1],  # comp C pos
            [2, 15, 0],  # comp C neg
            [3, 16, 1],  # comp D pos
            [3, 17, 0],  # comp D neg
            [4, 18, 1],  # comp E pos
            [4, 19, 0],  # comp E neg
        ], dtype=torch.long)

    def test_returns_three_splits(self, labeled_pairs):
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_row_overlap(self, labeled_pairs):
        """Строки из test не должны быть в train."""
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        train_rows = set(train[:, 0].tolist() + train[:, 1].tolist())
        val_rows = set(val[:, 0].tolist() + val[:, 1].tolist())
        test_rows = set(test[:, 0].tolist() + test[:, 1].tolist())
        assert train_rows.isdisjoint(test_rows), "Train and test share rows!"
        assert train_rows.isdisjoint(val_rows), "Train and val share rows!"
        assert val_rows.isdisjoint(test_rows), "Val and test share rows!"

    def test_all_pairs_assigned(self, labeled_pairs):
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(labeled_pairs)

    def test_positive_ratio_preserved(self, labeled_pairs):
        """Доля positives примерно одинакова во всех split."""
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        overall_pos = (labeled_pairs[:, 2] == 1).float().mean().item()
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            if len(split) < 2:
                continue
            pos_ratio = (split[:, 2] == 1).float().mean().item()
            assert abs(pos_ratio - overall_pos) < 0.4, (
                f"{name} positive ratio {pos_ratio:.2f} too far from overall {overall_pos:.2f}"
            )

    def test_deterministic(self, labeled_pairs):
        t1, v1, te1 = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        t2, v2, te2 = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        assert torch.equal(t1, t2)
        assert torch.equal(v1, v2)
        assert torch.equal(te1, te2)
