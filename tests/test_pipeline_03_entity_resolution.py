"""
Тесты для Pipeline 3: Entity Resolution — GNN модель, датасет, графы, обучение.

Покрывает:
- ERConfig создание и параметры
- EntityResolutionGNN: forward pass, output shape, L2 norm
- GNNLayer: bidirectional message passing
- ERGraphDataset и er_collate_fn
- Построение графов из HeteroData
- ERTrainer: полный цикл обучения на синтетических графах
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from table_unifier.entity_resolution.config import ERConfig
from table_unifier.entity_resolution.gnn_model import (
    EntityResolutionGNN,
    GNNLayer,
)

try:
    from torch_geometric.data import HeteroData, Batch
except ImportError:
    pytest.skip("torch_geometric not installed", allow_module_level=True)

FASTTEXT_DIM = 300
SM_OUTPUT_DIM = 256
ER_OUTPUT_DIM = 128
ER_HIDDEN_DIM = 256
ER_EDGE_DIM = 128


# ─────────────────── Утилиты для создания синтетических графов ───────────────────

def make_synthetic_graph(
    n_rows_a: int = 5,
    n_rows_b: int = 4,
    n_tokens: int = 20,
    n_edges: int = 40,
    row_dim: int = FASTTEXT_DIM,
    col_dim: int = SM_OUTPUT_DIM,
    n_duplicates: int = 2,
) -> HeteroData:
    """Создать синтетический HeteroData-граф (как в ER pipeline)."""
    n_rows = n_rows_a + n_rows_b
    data = HeteroData()

    # Row nodes
    data["row"].x = torch.randn(n_rows, row_dim)
    # Table ID: 0 для A, 1 для B
    data["row"].table_id = torch.cat([
        torch.zeros(n_rows_a, dtype=torch.long),
        torch.ones(n_rows_b, dtype=torch.long),
    ])
    # Entity labels: первые n_duplicates из A совпадают с первыми n_duplicates из B
    labels_a = list(range(n_rows_a))
    labels_b = list(range(n_duplicates)) + list(range(n_rows_a, n_rows_a + n_rows_b - n_duplicates))
    data["row"].entity_label = torch.tensor(labels_a + labels_b, dtype=torch.long)
    data["row"].num_rows_a = torch.tensor(n_rows_a)

    # Token nodes
    data["token"].x = torch.randn(n_tokens, row_dim)

    # Edges: row -> token
    src_rows = torch.randint(0, n_rows, (n_edges,))
    dst_tokens = torch.randint(0, n_tokens, (n_edges,))
    data["row", "has_token", "token"].edge_index = torch.stack([src_rows, dst_tokens])
    data["row", "has_token", "token"].edge_attr = torch.randn(n_edges, col_dim)

    # Reverse edges: token -> row
    data["token", "in_row", "row"].edge_index = torch.stack([dst_tokens, src_rows])
    data["token", "in_row", "row"].edge_attr = torch.randn(n_edges, col_dim)

    return data


def make_er_config(**overrides) -> ERConfig:
    """Минимальный ERConfig для тестов."""
    defaults = dict(
        token_embed_dim=FASTTEXT_DIM,
        hidden_dim=ER_HIDDEN_DIM,
        edge_dim=ER_EDGE_DIM,
        num_gnn_layers=2,
        num_heads=4,
        dropout=0.1,
        use_jumping_knowledge=True,
        output_dim=ER_OUTPUT_DIM,
        batch_size=2,
        learning_rate=1e-3,
        num_epochs=2,
        margin=0.3,
        mining_strategy="all",
        scheduler_patience=1,
        early_stopping_patience=3,
    )
    defaults.update(overrides)
    return ERConfig(**defaults)


# ═══════════════════════ ERConfig ═══════════════════════

class TestERConfig:
    """Тесты конфигурации ER."""

    def test_default_config(self):
        config = ERConfig()
        assert config.hidden_dim == 256
        assert config.output_dim == 128
        assert config.num_gnn_layers == 2
        assert config.num_heads == 4
        assert config.use_jumping_knowledge is True

    def test_hidden_dim_divisible_by_heads(self):
        """hidden_dim должен делиться на num_heads."""
        config = make_er_config()
        assert config.hidden_dim % config.num_heads == 0

    def test_save_load(self, tmp_path):
        config = make_er_config()
        path = str(tmp_path / "er_config.json")
        config.save(path)
        loaded = ERConfig.load(path)
        assert loaded.hidden_dim == config.hidden_dim
        assert loaded.output_dim == config.output_dim


# ═══════════════════════ GNNLayer ═══════════════════════

class TestGNNLayer:
    """Тесты одного слоя GNN."""

    def test_forward_shape(self):
        layer = GNNLayer(
            hidden_dim=ER_HIDDEN_DIM,
            edge_dim=ER_EDGE_DIM,
            num_heads=4,
        )
        n_rows, n_tokens, n_edges = 10, 20, 40

        row_x = torch.randn(n_rows, ER_HIDDEN_DIM)
        token_x = torch.randn(n_tokens, ER_HIDDEN_DIM)

        src_rows = torch.randint(0, n_tokens, (n_edges,))
        dst_rows = torch.randint(0, n_rows, (n_edges,))
        edge_index_t2r = torch.stack([src_rows, dst_rows])

        src_r = torch.randint(0, n_rows, (n_edges,))
        dst_t = torch.randint(0, n_tokens, (n_edges,))
        edge_index_r2t = torch.stack([src_r, dst_t])

        edge_attr_t2r = torch.randn(n_edges, ER_EDGE_DIM)
        edge_attr_r2t = torch.randn(n_edges, ER_EDGE_DIM)

        new_row, new_token = layer(
            row_x, token_x,
            edge_index_t2r, edge_index_r2t,
            edge_attr_t2r, edge_attr_r2t,
        )

        assert new_row.shape == (n_rows, ER_HIDDEN_DIM)
        assert new_token.shape == (n_tokens, ER_HIDDEN_DIM)

    def test_residual_connection(self):
        """Residual: выход не идентичен входу, но размерности совпадают."""
        layer = GNNLayer(hidden_dim=ER_HIDDEN_DIM, edge_dim=ER_EDGE_DIM)
        n_rows, n_tokens, n_edges = 5, 10, 20

        row_x = torch.randn(n_rows, ER_HIDDEN_DIM)
        token_x = torch.randn(n_tokens, ER_HIDDEN_DIM)
        edge_index_t2r = torch.stack([
            torch.randint(0, n_tokens, (n_edges,)),
            torch.randint(0, n_rows, (n_edges,)),
        ])
        edge_index_r2t = torch.stack([
            torch.randint(0, n_rows, (n_edges,)),
            torch.randint(0, n_tokens, (n_edges,)),
        ])
        edge_attr = torch.randn(n_edges, ER_EDGE_DIM)

        new_row, new_token = layer(
            row_x, token_x,
            edge_index_t2r, edge_index_r2t,
            edge_attr, edge_attr,
        )

        # Размерности сохраняются (residual)
        assert new_row.shape == row_x.shape
        assert new_token.shape == token_x.shape


# ═══════════════════════ EntityResolutionGNN ═══════════════════════

class TestEntityResolutionGNN:
    """Тесты полной GNN модели для ER."""

    def test_forward_shape(self):
        config = make_er_config()
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        graph = make_synthetic_graph()
        out = model(graph)
        n_rows = graph["row"].x.shape[0]
        assert out.shape == (n_rows, ER_OUTPUT_DIM)

    def test_l2_normalized(self):
        """Выход GNN L2-нормализован."""
        config = make_er_config()
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        graph = make_synthetic_graph()
        out = model(graph)
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_without_jumping_knowledge(self):
        """Модель без JumpingKnowledge."""
        config = make_er_config(use_jumping_knowledge=False)
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        graph = make_synthetic_graph()
        out = model(graph)
        n_rows = graph["row"].x.shape[0]
        assert out.shape == (n_rows, ER_OUTPUT_DIM)

    def test_different_gnn_layers(self):
        """1, 2, 3 слоя GNN."""
        for n_layers in [1, 2, 3]:
            config = make_er_config(num_gnn_layers=n_layers)
            model = EntityResolutionGNN(
                row_input_dim=FASTTEXT_DIM,
                col_embed_dim=SM_OUTPUT_DIM,
                config=config,
            )
            graph = make_synthetic_graph()
            out = model(graph)
            assert out.shape[1] == ER_OUTPUT_DIM

    def test_get_row_embeddings(self):
        """get_row_embeddings фильтрует по table_id."""
        config = make_er_config()
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        model.eval()
        n_a, n_b = 5, 4
        graph = make_synthetic_graph(n_rows_a=n_a, n_rows_b=n_b)

        with torch.no_grad():
            emb_a = model.get_row_embeddings(graph, table_id=0)
            emb_b = model.get_row_embeddings(graph, table_id=1)
            emb_all = model.get_row_embeddings(graph, table_id=None)

        assert emb_a.shape == (n_a, ER_OUTPUT_DIM)
        assert emb_b.shape == (n_b, ER_OUTPUT_DIM)
        assert emb_all.shape == (n_a + n_b, ER_OUTPUT_DIM)

    def test_eval_mode_no_crash(self):
        """Модель работает в eval mode (dropout отключён)."""
        config = make_er_config(dropout=0.5)
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        model.eval()
        graph = make_synthetic_graph()
        with torch.no_grad():
            out = model(graph)
        assert out.shape[1] == ER_OUTPUT_DIM


# ═══════════════════════ ERGraphDataset и collate ═══════════════════════

class TestERGraphDataset:
    """Тесты Dataset для ER (загрузка .pt файлов)."""

    def test_load_graphs(self, tmp_path):
        """Dataset загружает сохранённые .pt графы."""
        graphs_dir = tmp_path / "graphs"
        graphs_dir.mkdir()

        for i in range(3):
            g = make_synthetic_graph()
            torch.save(g, graphs_dir / f"graph_{i:04d}.pt")

        from table_unifier.entity_resolution.er_dataset import ERGraphDataset
        ds = ERGraphDataset(str(graphs_dir))
        assert len(ds) == 3

        g = ds[0]
        assert "row" in g.node_types
        assert "token" in g.node_types

    def test_empty_dir_raises(self, tmp_path):
        """Пустая директория вызывает ошибку."""
        from table_unifier.entity_resolution.er_dataset import ERGraphDataset
        empty = tmp_path / "empty"
        empty.mkdir()

        with pytest.raises(FileNotFoundError):
            ERGraphDataset(str(empty))

    def test_collate_fn(self):
        """er_collate_fn корректно батчирует графы."""
        from table_unifier.entity_resolution.er_dataset import er_collate_fn

        graphs = [make_synthetic_graph() for _ in range(3)]
        batch = er_collate_fn(graphs)

        # Все entity_labels уникальны между графами
        labels = batch["row"].entity_label
        # Проверяем что label offset работает
        total_rows = sum(g["row"].x.shape[0] for g in graphs)
        assert labels.shape[0] == total_rows

    def test_collate_label_offset(self):
        """entity_labels из разных графов не пересекаются после collate."""
        from table_unifier.entity_resolution.er_dataset import er_collate_fn

        g1 = make_synthetic_graph(n_rows_a=3, n_rows_b=2, n_duplicates=1)
        g2 = make_synthetic_graph(n_rows_a=4, n_rows_b=3, n_duplicates=2)

        labels_g1_before = g1["row"].entity_label.clone()
        labels_g2_before = g2["row"].entity_label.clone()

        batch = er_collate_fn([g1, g2])
        labels = batch["row"].entity_label

        # Метки из g2 должны быть сдвинуты
        n1 = labels_g1_before.shape[0]
        labels_from_g2 = labels[n1:]
        max_label_g1 = labels_g1_before.max().item() + 1
        assert labels_from_g2.min().item() >= max_label_g1


# ═══════════════════════ ERTrainer (мини-обучение) ═══════════════════════

class TestERTrainer:
    """Интеграционный тест тренера ER на синтетических графах."""

    def _prepare_graph_dirs(self, base_dir, n_train=5, n_val=2):
        """Создать директории с синтетическими графами."""
        train_dir = os.path.join(base_dir, "train")
        val_dir = os.path.join(base_dir, "val")
        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(val_dir, exist_ok=True)

        for i in range(n_train):
            g = make_synthetic_graph(
                n_rows_a=5, n_rows_b=4, n_tokens=15, n_edges=30, n_duplicates=2,
            )
            torch.save(g, os.path.join(train_dir, f"graph_{i:04d}.pt"))

        for i in range(n_val):
            g = make_synthetic_graph(
                n_rows_a=4, n_rows_b=3, n_tokens=12, n_edges=24, n_duplicates=1,
            )
            torch.save(g, os.path.join(val_dir, f"graph_{i:04d}.pt"))

        return train_dir, val_dir

    def test_training_loop(self, tmp_path):
        """Полный цикл обучения ER на синтетических графах."""
        from table_unifier.entity_resolution.trainer import ERTrainer

        graph_base = str(tmp_path / "er_graphs")
        train_dir, val_dir = self._prepare_graph_dirs(graph_base)

        config = make_er_config(num_epochs=2, mining_strategy="all")
        trainer = ERTrainer(
            config=config,
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            device="cpu",
        )

        save_dir = str(tmp_path / "er_model_output")
        history = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            save_dir=save_dir,
        )

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0

        # Проверяем сохранённые файлы
        assert os.path.exists(os.path.join(save_dir, "best_model.pt"))

    def test_loss_is_finite(self, tmp_path):
        """Loss не NaN и не inf."""
        from table_unifier.entity_resolution.trainer import ERTrainer

        graph_base = str(tmp_path / "er_graphs")
        train_dir, val_dir = self._prepare_graph_dirs(graph_base)

        config = make_er_config(num_epochs=2, mining_strategy="all")
        trainer = ERTrainer(
            config=config,
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            device="cpu",
        )

        history = trainer.train(
            train_dir=train_dir,
            val_dir=val_dir,
            save_dir=str(tmp_path / "er_finite"),
        )

        for loss in history["train_loss"]:
            assert np.isfinite(loss), f"Train loss is not finite: {loss}"
        for loss in history["val_loss"]:
            assert np.isfinite(loss), f"Val loss is not finite: {loss}"

    def test_model_output_after_training(self, tmp_path):
        """После обучения модель даёт корректные эмбеддинги."""
        from table_unifier.entity_resolution.trainer import ERTrainer

        graph_base = str(tmp_path / "er_graphs")
        train_dir, val_dir = self._prepare_graph_dirs(graph_base)

        config = make_er_config(num_epochs=1, mining_strategy="all")
        trainer = ERTrainer(
            config=config,
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            device="cpu",
        )

        save_dir = str(tmp_path / "er_output_test")
        trainer.train(train_dir=train_dir, val_dir=val_dir, save_dir=save_dir)

        # Загружаем модель и проверяем инференс
        checkpoint = torch.load(
            os.path.join(save_dir, "best_model.pt"),
            map_location="cpu",
            weights_only=False,
        )
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        graph = make_synthetic_graph()
        with torch.no_grad():
            out = model(graph)

        assert out.shape[1] == ER_OUTPUT_DIM
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
