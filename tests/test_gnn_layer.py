"""Тесты для models/gnn_layer.py."""

import torch
import pytest

from table_unifier.models.gnn_layer import GNNLayer


class TestGNNLayer:
    @pytest.fixture()
    def layer(self):
        return GNNLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0)

    @pytest.fixture()
    def graph_tensors(self):
        """Минимальный набор тензоров для прогона GNN слоя."""
        n_rows = 4
        n_tokens = 6
        n_edges = 10
        hidden = 32
        edge_dim = 16

        row_x = torch.randn(n_rows, hidden)
        token_x = torch.randn(n_tokens, hidden)

        t2r_src = torch.randint(0, n_tokens, (n_edges,))
        t2r_dst = torch.randint(0, n_rows, (n_edges,))
        t2r_edge_index = torch.stack([t2r_src, t2r_dst])

        r2t_edge_index = torch.stack([t2r_dst, t2r_src])

        edge_attr_t2r = torch.randn(n_edges, edge_dim)
        edge_attr_r2t = torch.randn(n_edges, edge_dim)

        return row_x, token_x, t2r_edge_index, edge_attr_t2r, r2t_edge_index, edge_attr_r2t

    def test_output_shapes(self, layer, graph_tensors):
        row_x, token_x, *rest = graph_tensors
        row_out, token_out = layer(row_x, token_x, *rest)
        assert row_out.shape == row_x.shape
        assert token_out.shape == token_x.shape

    def test_residual_connection(self, layer, graph_tensors):
        row_x, token_x, *rest = graph_tensors
        layer.eval()
        with torch.no_grad():
            row_out, token_out = layer(row_x, token_x, *rest)
        # Выходы не должны быть идентичны входам (конволюция меняет)
        assert not torch.allclose(row_out, row_x, atol=1e-6)

    def test_gradient_flows(self, layer, graph_tensors):
        row_x, token_x, *rest = graph_tensors
        row_x = row_x.requires_grad_(True)
        token_x = token_x.requires_grad_(True)
        row_out, token_out = layer(row_x, token_x, *rest)
        loss = row_out.sum() + token_out.sum()
        loss.backward()
        assert row_x.grad is not None
        assert token_x.grad is not None

    def test_odd_hidden_dim(self):
        # Простая архитектура не требует деления на heads
        layer = GNNLayer(hidden_dim=33, edge_dim=16, num_heads=1)
        assert layer is not None
