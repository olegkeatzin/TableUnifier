"""Тесты для EntityResolutionGAT."""

import torch
import pytest
from table_unifier.models.entity_resolution import EntityResolutionGAT


class TestEntityResolutionGAT:
    @pytest.fixture()
    def model(self):
        return EntityResolutionGAT(
            row_dim=32, token_dim=32, col_dim=64,
            hidden_dim=32, edge_dim=16, output_dim=16,
            num_gnn_layers=2, num_heads=4, dropout=0.0,
        )

    def test_output_shape(self, model, small_hetero_data):
        out = model(small_hetero_data)
        n_rows = small_hetero_data["row"].x.shape[0]
        assert out.shape == (n_rows, 16)

    def test_l2_normalized(self, model, small_hetero_data):
        model.eval()
        with torch.no_grad():
            out = model(small_hetero_data)
        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(norms.shape[0]), atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self, model, small_hetero_data):
        out = model(small_hetero_data)
        loss = out.sum()
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert any(grads)

    def test_num_gat_layers(self, model):
        assert len(model.gnn_layers) == 2

    def test_forward_signature_matches_gnn(self, model, small_hetero_data):
        """Тот же интерфейс что у EntityResolutionGNN — принимает HeteroData, возвращает Tensor."""
        out = model(small_hetero_data)
        assert isinstance(out, torch.Tensor)
        assert out.dim() == 2
