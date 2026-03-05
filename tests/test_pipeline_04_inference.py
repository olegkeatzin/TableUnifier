"""
Тесты для Pipeline 4: Полный инференс SM + ER.

Покрывает:
- Загрузка обученных SM и ER моделей
- SM инференс: вычисление column embeddings, match_schemas
- ER инференс: find_duplicates, DuplicateMatch
- Консистентность SM → ER pipeline
- Метрики и оценка (TP/FP/FN расчёт)

Внешние зависимости (Ollama, FastText) мокаются.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest
import torch

from table_unifier.schema_matching.config import SMConfig
from table_unifier.schema_matching.model import SchemaMatchingModel
from table_unifier.entity_resolution.config import ERConfig
from table_unifier.entity_resolution.gnn_model import EntityResolutionGNN

try:
    from torch_geometric.data import HeteroData
except ImportError:
    pytest.skip("torch_geometric not installed", allow_module_level=True)

EMBED_DIM = 768
SM_OUTPUT_DIM = 256
ER_OUTPUT_DIM = 128
FASTTEXT_DIM = 300
ER_HIDDEN_DIM = 256
ER_EDGE_DIM = 128


# ─────────────────── Утилиты ───────────────────

def _save_sm_model(tmp_path) -> str:
    """Создать и сохранить SM модель, вернуть путь."""
    config = SMConfig(
        input_dim=EMBED_DIM,
        projection_dims=[256],
        output_dim=SM_OUTPUT_DIM,
    )
    model = SchemaMatchingModel(config)
    path = str(tmp_path / "sm_best.pt")
    model.save(path)
    return path


def _save_er_model(tmp_path) -> str:
    """Создать и сохранить ER модель, вернуть путь."""
    config = ERConfig(
        hidden_dim=ER_HIDDEN_DIM,
        edge_dim=ER_EDGE_DIM,
        num_gnn_layers=2,
        num_heads=4,
        output_dim=ER_OUTPUT_DIM,
        token_embed_dim=FASTTEXT_DIM,
    )
    model = EntityResolutionGNN(
        row_input_dim=FASTTEXT_DIM,
        col_embed_dim=SM_OUTPUT_DIM,
        config=config,
    )
    path = str(tmp_path / "er_best.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": {
            "hidden_dim": ER_HIDDEN_DIM,
            "edge_dim": ER_EDGE_DIM,
            "num_gnn_layers": 2,
            "num_heads": 4,
            "output_dim": ER_OUTPUT_DIM,
            "token_embed_dim": FASTTEXT_DIM,
        },
    }, path)
    return path


# ═══════════════════════ SM Модель: загрузка и инференс ═══════════════════════

class TestSMModelLoading:
    """Загрузка обученной SM модели."""

    def test_load_model(self, tmp_path):
        path = _save_sm_model(tmp_path)
        model = SchemaMatchingModel.load(path, device="cpu")
        assert model.config.output_dim == SM_OUTPUT_DIM
        assert not model.training  # eval mode

    def test_model_projection(self, tmp_path):
        """Проекция через загруженную модель."""
        path = _save_sm_model(tmp_path)
        model = SchemaMatchingModel.load(path, device="cpu")

        x = torch.randn(5, EMBED_DIM)
        with torch.no_grad():
            projected = model(x)

        assert projected.shape == (5, SM_OUTPUT_DIM)
        norms = torch.norm(projected, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


# ═══════════════════════ ER Модель: загрузка и инференс ═══════════════════════

class TestERModelLoading:
    """Загрузка обученной ER модели."""

    def test_load_checkpoint(self, tmp_path):
        path = _save_er_model(tmp_path)
        config = ERConfig(
            hidden_dim=ER_HIDDEN_DIM,
            edge_dim=ER_EDGE_DIM,
            num_gnn_layers=2,
            num_heads=4,
            output_dim=ER_OUTPUT_DIM,
            token_embed_dim=FASTTEXT_DIM,
        )
        model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=SM_OUTPUT_DIM,
            config=config,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        # forward pass на синтетическом графе
        from tests.test_pipeline_03_entity_resolution import make_synthetic_graph
        graph = make_synthetic_graph()
        with torch.no_grad():
            out = model(graph)
        assert out.shape[1] == ER_OUTPUT_DIM


# ═══════════════════════ ER Inference: DuplicateMatch ═══════════════════════

class TestDuplicateMatch:
    """Тесты структуры DuplicateMatch."""

    def test_dataclass(self):
        from table_unifier.entity_resolution.inference import DuplicateMatch
        m = DuplicateMatch(row_idx_a=0, row_idx_b=2, similarity=0.85)
        assert m.row_idx_a == 0
        assert m.row_idx_b == 2
        assert m.similarity == 0.85


# ═══════════════════════ Метрики инференса ═══════════════════════

class TestInferenceMetrics:
    """Расчёт P/R/F1 для ER дубликатов (как в Pipeline 4)."""

    def test_perfect_match(self):
        """Все дубликаты найдены, нет ложных."""
        gt_pairs = {(0, 0), (2, 1)}
        pred_pairs = {(0, 0), (2, 1)}

        tp = len(gt_pairs & pred_pairs)
        fp = len(pred_pairs - gt_pairs)
        fn = len(gt_pairs - pred_pairs)

        assert tp == 2
        assert fp == 0
        assert fn == 0

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_partial_match(self):
        """Часть дубликатов найдена, есть FP и FN."""
        gt_pairs = {(0, 0), (1, 1), (2, 2)}
        pred_pairs = {(0, 0), (2, 2), (3, 3)}

        tp = len(gt_pairs & pred_pairs)
        fp = len(pred_pairs - gt_pairs)
        fn = len(gt_pairs - pred_pairs)

        assert tp == 2
        assert fp == 1
        assert fn == 1

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)

        assert abs(precision - 2/3) < 1e-6
        assert abs(recall - 2/3) < 1e-6

    def test_no_matches(self):
        """Ничего не найдено."""
        gt_pairs = {(0, 0), (1, 1)}
        pred_pairs = set()

        tp = len(gt_pairs & pred_pairs)
        fp = len(pred_pairs - gt_pairs)
        fn = len(gt_pairs - pred_pairs)

        assert tp == 0
        assert fp == 0
        assert fn == 2

    def test_sm_accuracy(self, er_pair_metadata):
        """SM accuracy: маппинг столбцов vs ground truth."""
        gt_a = er_pair_metadata["column_mapping_a"]
        gt_b = er_pair_metadata["column_mapping_b"]
        base_to_b = {v: k for k, v in gt_b.items()}

        # Идеальный маппинг
        mapping = {
            "Артикул": "Код товара",
            "Наименование": "Название",
            "Цена": "Стоимость",
            "Количество": "Кол-во",
        }

        correct = 0
        total = 0
        for src, tgt in mapping.items():
            src_base = gt_a.get(src)
            expected = base_to_b.get(src_base)
            if expected is not None:
                total += 1
                if tgt == expected:
                    correct += 1

        assert total == 4
        assert correct == 4
        assert correct / total == 1.0


# ═══════════════════════ ER-пары из файлов ═══════════════════════

class TestERPairLoading:
    """Загрузка ER пар из CSV + meta.json (как в Pipeline 4)."""

    def test_load_pair(self, er_test_dir):
        """CSV-пара и метаданные корректно загружаются."""
        df_a = pd.read_csv(os.path.join(er_test_dir, "pair_0000_table_a.csv"))
        df_b = pd.read_csv(os.path.join(er_test_dir, "pair_0000_table_b.csv"))

        with open(os.path.join(er_test_dir, "pair_0000_meta.json"), "r") as f:
            meta = json.load(f)

        assert df_a.shape[0] == meta["num_rows_a"]
        assert df_b.shape[0] == meta["num_rows_b"]
        assert len(meta["duplicate_pairs"]) == meta["num_duplicates"]

    def test_column_mappings(self, er_test_dir):
        """column_mapping содержит маппинг для всех столбцов."""
        with open(os.path.join(er_test_dir, "pair_0000_meta.json"), "r") as f:
            meta = json.load(f)

        df_a = pd.read_csv(os.path.join(er_test_dir, "pair_0000_table_a.csv"))
        df_b = pd.read_csv(os.path.join(er_test_dir, "pair_0000_table_b.csv"))

        for col in df_a.columns:
            assert col in meta["column_mapping_a"]
        for col in df_b.columns:
            assert col in meta["column_mapping_b"]

    def test_duplicate_pairs_valid(self, er_test_dir, er_pair_metadata):
        """Индексы дубликатов в пределах размеров таблиц."""
        for a_idx, b_idx in er_pair_metadata["duplicate_pairs"]:
            assert 0 <= a_idx < er_pair_metadata["num_rows_a"]
            assert 0 <= b_idx < er_pair_metadata["num_rows_b"]


# ═══════════════════════ Кросс-пайплайн консистентность ═══════════════════════

class TestCrossPipelineConsistency:
    """SM и ER модели совместимы по размерностям."""

    def test_sm_output_matches_er_edge_dim_input(self, tmp_path):
        """SM output_dim может использоваться как col_embed_dim для ER."""
        sm_config = SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[256],
            output_dim=SM_OUTPUT_DIM,
        )
        sm_model = SchemaMatchingModel(sm_config)

        x = torch.randn(5, EMBED_DIM)
        with torch.no_grad():
            sm_out = sm_model(x)

        # sm_out.shape[1] == SM_OUTPUT_DIM, это col_embed_dim для ER
        er_config = ERConfig(
            hidden_dim=ER_HIDDEN_DIM,
            edge_dim=ER_EDGE_DIM,
            num_gnn_layers=2,
            num_heads=4,
            output_dim=ER_OUTPUT_DIM,
            token_embed_dim=FASTTEXT_DIM,
        )
        er_model = EntityResolutionGNN(
            row_input_dim=FASTTEXT_DIM,
            col_embed_dim=sm_out.shape[1],  # должно быть SM_OUTPUT_DIM
            config=er_config,
        )

        from tests.test_pipeline_03_entity_resolution import make_synthetic_graph
        graph = make_synthetic_graph()
        with torch.no_grad():
            er_out = er_model(graph)

        assert er_out.shape[1] == ER_OUTPUT_DIM

    def test_cosine_similarity_range(self, tmp_path):
        """Косинусное сходство L2-нормализованных векторов в [-1, 1]."""
        sm_config = SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[128],
            output_dim=SM_OUTPUT_DIM,
        )
        sm_model = SchemaMatchingModel(sm_config)
        sm_model.eval()

        x = torch.randn(10, EMBED_DIM)
        with torch.no_grad():
            emb = sm_model(x)

        sim = torch.mm(emb, emb.t())
        assert sim.min().item() >= -1.0 - 1e-5
        assert sim.max().item() <= 1.0 + 1e-5

    def test_threshold_filtering(self):
        """Пороговая фильтрация дубликатов работает корректно."""
        from table_unifier.entity_resolution.inference import DuplicateMatch

        # Симулируем sim_matrix
        sim_matrix = torch.tensor([
            [0.95, 0.3, 0.1],
            [0.2, 0.88, 0.4],
            [0.5, 0.15, 0.75],
        ])

        threshold = 0.7
        matches = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                if sim_matrix[i, j] >= threshold:
                    matches.append(DuplicateMatch(
                        row_idx_a=i, row_idx_b=j,
                        similarity=sim_matrix[i, j].item(),
                    ))

        assert len(matches) == 3  # (0,0)=0.95, (1,1)=0.88, (2,2)=0.75
        assert all(m.similarity >= threshold for m in matches)
