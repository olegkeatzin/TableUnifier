"""Тесты для training/schema_trainer.py."""

import numpy as np
import pytest
import torch

from table_unifier.config import SchemaMatchingConfig
from table_unifier.training.schema_trainer import _build_sm_triplets, train_schema_matching


class TestBuildSmTriplets:
    def test_basic_triplets(self):
        rng = np.random.default_rng(42)
        col_emb = {
            "title_A": rng.standard_normal(16).astype(np.float32),
            "title_B": rng.standard_normal(16).astype(np.float32),
            "brand_A": rng.standard_normal(16).astype(np.float32),
            "brand_B": rng.standard_normal(16).astype(np.float32),
        }
        gt = [("title_A", "title_B"), ("brand_A", "brand_B")]
        anc, pos, neg = _build_sm_triplets(col_emb, gt)
        assert len(anc) > 0
        assert len(anc) == len(pos) == len(neg)
        assert anc.shape[1] == 16

    def test_skips_missing_columns(self):
        col_emb = {"a": np.zeros(4, dtype=np.float32)}
        gt = [("a", "missing")]
        anc, pos, neg = _build_sm_triplets(col_emb, gt)
        assert len(anc) == 0

    def test_no_self_negatives(self):
        rng = np.random.default_rng(0)
        col_emb = {
            "a": rng.standard_normal(8).astype(np.float32),
            "b": rng.standard_normal(8).astype(np.float32),
            "c": rng.standard_normal(8).astype(np.float32),
        }
        gt = [("a", "b")]
        anc, pos, neg = _build_sm_triplets(col_emb, gt)
        # neg should only be from "c" (not "a" or "b")
        assert len(anc) == 1  # only c as negative


class TestTrainSchemaMatching:
    def test_trains_and_returns_model(self):
        rng = np.random.default_rng(42)
        dim = 32
        col_emb = {
            f"col_{i}": rng.standard_normal(dim).astype(np.float32)
            for i in range(6)
        }
        gt = [("col_0", "col_1"), ("col_2", "col_3")]

        config = SchemaMatchingConfig(
            embedding_dim=dim, hidden_dim=16, projection_dim=8,
            epochs=3, batch_size=4, lr=1e-2,
        )
        model = train_schema_matching(col_emb, gt, config=config, device="cpu")

        # Проверить что модель работает
        x = torch.randn(2, dim)
        model.eval()
        with torch.no_grad():
            out = model(x)
        assert out.shape == (2, 8)

    def test_raises_on_empty_triplets(self):
        col_emb = {"a": np.zeros(8, dtype=np.float32)}
        gt = [("a", "missing")]
        with pytest.raises(ValueError, match="триплеты"):
            train_schema_matching(col_emb, gt, device="cpu")

    def test_saves_model(self, tmp_path):
        rng = np.random.default_rng(0)
        dim = 16
        col_emb = {f"c{i}": rng.standard_normal(dim).astype(np.float32) for i in range(4)}
        gt = [("c0", "c1")]
        config = SchemaMatchingConfig(
            embedding_dim=dim, hidden_dim=8, projection_dim=4,
            epochs=2, batch_size=4,
        )
        save_path = tmp_path / "sm.pt"
        train_schema_matching(col_emb, gt, config=config, device="cpu", save_path=save_path)
        assert save_path.exists()
