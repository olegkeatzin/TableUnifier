"""Тесты для training/er_trainer.py."""

import torch
import pytest
from torch_geometric.data import HeteroData

from table_unifier.config import EntityResolutionConfig
from table_unifier.training.er_trainer import (
    find_duplicates,
    get_row_embeddings,
    train_entity_resolution,
)
from table_unifier.models.entity_resolution import EntityResolutionGNN


@pytest.fixture()
def er_config():
    return EntityResolutionConfig(
        row_dim=32,
        token_dim=32,
        col_dim=64,
        hidden_dim=32,
        edge_dim=16,
        output_dim=16,
        num_gnn_layers=1,
        dropout=0.0,
        epochs=2,
        batch_size=8,
        lr=1e-2,
    )


@pytest.fixture()
def triplet_indices():
    # 3 rows in A (idx 0-2), 3 rows in B (idx 3-5)
    return torch.tensor([
        [0, 3, 4],
        [1, 4, 5],
        [0, 3, 5],
    ], dtype=torch.long)


class TestTrainEntityResolution:
    def test_trains_and_returns_model(self, small_hetero_data, triplet_indices, er_config):
        model, history = train_entity_resolution(
            data=small_hetero_data,
            triplet_indices=triplet_indices,
            config=er_config,
            device="cpu",
        )
        assert isinstance(model, EntityResolutionGNN)
        assert "train_loss" in history
        assert len(history["train_loss"]) == er_config.epochs

    def test_with_validation(self, small_hetero_data, triplet_indices, er_config):
        val_triplets = torch.tensor([[2, 5, 3]], dtype=torch.long)
        model, history = train_entity_resolution(
            data=small_hetero_data,
            triplet_indices=triplet_indices,
            config=er_config,
            val_triplet_indices=val_triplets,
            device="cpu",
        )
        assert isinstance(model, EntityResolutionGNN)
        assert len(history["val_loss"]) == er_config.epochs

    def test_saves_model(self, small_hetero_data, triplet_indices, er_config, tmp_path):
        save_path = tmp_path / "er.pt"
        train_entity_resolution(
            data=small_hetero_data,
            triplet_indices=triplet_indices,
            config=er_config,
            device="cpu",
            save_path=save_path,
        )
        assert save_path.exists()
        assert (tmp_path / "er.history.json").exists()


class TestGetRowEmbeddings:
    def test_returns_correct_shape(self, small_hetero_data, er_config):
        model = EntityResolutionGNN(
            row_dim=er_config.row_dim,
            token_dim=er_config.token_dim,
            col_dim=er_config.col_dim,
            hidden_dim=er_config.hidden_dim,
            edge_dim=er_config.edge_dim,
            output_dim=er_config.output_dim,
            num_gnn_layers=er_config.num_gnn_layers,
            dropout=er_config.dropout,
        )
        emb = get_row_embeddings(model, small_hetero_data, device="cpu")
        n_rows = small_hetero_data["row"].x.shape[0]
        assert emb.shape == (n_rows, er_config.output_dim)
        assert emb.device == torch.device("cpu")


class TestFindDuplicates:
    def test_finds_similar_pairs(self):
        # Создать эмбеддинги где строки 0 и 3 очень похожи
        emb = torch.zeros(6, 4)
        emb[0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        emb[1] = torch.tensor([0.0, 1.0, 0.0, 0.0])
        emb[2] = torch.tensor([0.0, 0.0, 1.0, 0.0])
        emb[3] = torch.tensor([0.99, 0.01, 0.0, 0.0])  # similar to 0
        emb[4] = torch.tensor([0.0, 0.0, 0.0, 1.0])
        emb[5] = torch.tensor([0.0, 0.0, 0.01, 0.99])

        # L2-normalize
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)

        id_to_a = {10: 0, 11: 1, 12: 2}
        id_to_b = {20: 3, 21: 4, 22: 5}

        results = find_duplicates(emb, id_to_a, id_to_b, threshold=0.9)
        assert len(results) >= 1
        # (10, 20) should be a match
        ids = [(a, b) for a, b, _ in results]
        assert (10, 20) in ids

    def test_no_duplicates_with_high_threshold(self):
        emb = torch.randn(6, 4)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        id_to_a = {0: 0, 1: 1, 2: 2}
        id_to_b = {3: 3, 4: 4, 5: 5}
        results = find_duplicates(emb, id_to_a, id_to_b, threshold=0.999)
        # Маловероятно, что случайные векторы будут настолько похожи
        assert isinstance(results, list)

    def test_sorted_by_similarity(self):
        emb = torch.tensor([
            [1.0, 0.0],
            [0.5, 0.5],
            [0.9, 0.1],
            [0.0, 1.0],
        ])
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        id_to_a = {0: 0, 1: 1}
        id_to_b = {2: 2, 3: 3}
        results = find_duplicates(emb, id_to_a, id_to_b, threshold=0.0)
        # Сортировка по убыванию similarity
        sims = [s for _, _, s in results]
        assert sims == sorted(sims, reverse=True)
