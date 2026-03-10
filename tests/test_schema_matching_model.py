"""Тесты для models/schema_matching.py."""

import torch
import pytest

from table_unifier.models.schema_matching import SchemaProjector


class TestSchemaProjector:
    @pytest.fixture()
    def model(self):
        return SchemaProjector(input_dim=64, hidden_dim=32, output_dim=16, dropout=0.0)

    def test_output_shape(self, model):
        x = torch.randn(8, 64)
        out = model(x)
        assert out.shape == (8, 16)

    def test_l2_normalized(self, model):
        x = torch.randn(4, 64)
        out = model(x)
        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(4), atol=1e-5, rtol=1e-5)

    def test_single_sample(self, model):
        model.eval()  # BatchNorm requires batch_size > 1 in train mode
        x = torch.randn(1, 64)
        out = model(x)
        assert out.shape == (1, 16)
        norm = torch.norm(out, p=2).item()
        assert abs(norm - 1.0) < 1e-5

    def test_default_dims(self):
        model = SchemaProjector()
        x = torch.randn(2, 4096)
        out = model(x)
        assert out.shape == (2, 256)

    def test_gradient_flows(self, model):
        x = torch.randn(4, 64, requires_grad=True)
        out = model(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (4, 64)

    def test_eval_mode(self, model):
        model.eval()
        x = torch.randn(4, 64)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (4, 16)
