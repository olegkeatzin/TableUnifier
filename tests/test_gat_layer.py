# tests/test_gat_layer.py
"""Тесты для models/gat_layer.py."""

import torch
import pytest
from table_unifier.models.gat_layer import GATLayer


class TestGATLayer:
    @pytest.fixture()
    def layer(self):
        return GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0)

    @pytest.fixture()
    def graph_tensors(self):
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

    def test_gradient_flows(self, layer, graph_tensors):
        row_x, token_x, *rest = graph_tensors
        row_x = row_x.requires_grad_(True)
        token_x = token_x.requires_grad_(True)
        row_out, token_out = layer(row_x, token_x, *rest)
        loss = row_out.sum() + token_out.sum()
        loss.backward()
        assert row_x.grad is not None
        assert token_x.grad is not None

    def test_bidirectional(self):
        layer = GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0, bidirectional=True)
        assert hasattr(layer, "conv_r2t")

    def test_not_bidirectional(self):
        layer = GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0, bidirectional=False)
        assert not hasattr(layer, "conv_r2t")

    def test_attention_weights_exist(self, layer, graph_tensors):
        """GATv2Conv возвращает attention weights при return_attention_weights=True."""
        row_x, token_x, t2r_ei, ea_t2r, *_ = graph_tensors
        out, (edge_index_out, attn_weights) = layer.conv_t2r(
            (token_x, row_x), t2r_ei, ea_t2r, return_attention_weights=True,
        )
        # GATv2Conv may add self-loops, so edge count can be >= original
        assert attn_weights.shape[0] >= t2r_ei.shape[1]
        assert attn_weights.shape[1] == layer.conv_t2r.heads

    def test_deterministic_in_eval(self, graph_tensors):
        layer = GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.1)
        layer.eval()
        row_x, token_x, *rest = graph_tensors
        with torch.no_grad():
            out1, _ = layer(row_x, token_x, *rest)
            out2, _ = layer(row_x, token_x, *rest)
        torch.testing.assert_close(out1, out2)
