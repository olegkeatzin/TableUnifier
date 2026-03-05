"""
Тесты для Pipeline 2: Schema Matching — датасет, модель, обучение, save/load.

Покрывает:
- SMConfig создание и параметры
- SMDataset: загрузка, __getitem__, embed_dim, num_classes
- load_and_split_dataset: стратификация, пропорции
- ProjectionHead / SchemaMatchingModel: forward pass, output shape, L2 norm
- SchemaMatchingModel save/load roundtrip
- SMTrainer: полный цикл обучения на синтетических данных
"""

import json
import os
import tempfile

import numpy as np
import pytest
import torch

from table_unifier.schema_matching.config import SMConfig
from table_unifier.schema_matching.dataset import SMDataset, load_and_split_dataset, create_dataloaders
from table_unifier.schema_matching.model import ProjectionHead, SchemaMatchingModel

EMBED_DIM = 768
SM_OUTPUT_DIM = 256
NUM_CLASSES = 5
SAMPLES_PER_CLASS = 10


# ═══════════════════════ SMConfig ═══════════════════════

class TestSMConfig:
    """Тесты конфигурации SM."""

    def test_default_config(self):
        config = SMConfig()
        assert config.output_dim == 256
        assert config.dropout == 0.1
        assert config.mining_strategy in ("hard", "semihard", "all")

    def test_from_dict(self, full_config):
        params = full_config["schema_matching"]
        config = SMConfig(**{k: v for k, v in params.items()
                            if k in SMConfig.__dataclass_fields__})
        assert config.output_dim == params["output_dim"]
        assert config.num_epochs == params["num_epochs"]
        assert config.margin == params["margin"]

    def test_ratios_sum(self, full_config):
        params = full_config["schema_matching"]
        total = params["train_ratio"] + params["val_ratio"] + params["test_ratio"]
        assert abs(total - 1.0) < 1e-6


# ═══════════════════════ SMDataset ═══════════════════════

class TestSMDataset:
    """Тесты PyTorch Dataset для SM."""

    def test_dataset_creation(self, sm_dataset_entries):
        ds = SMDataset(sm_dataset_entries)
        assert len(ds) == NUM_CLASSES * SAMPLES_PER_CLASS
        assert ds.num_classes == NUM_CLASSES
        assert ds.embed_dim == EMBED_DIM

    def test_getitem(self, sm_dataset_entries):
        ds = SMDataset(sm_dataset_entries)
        emb, label = ds[0]
        assert isinstance(emb, torch.Tensor)
        assert isinstance(label, torch.Tensor)
        assert emb.shape == (EMBED_DIM,)
        assert label.dim() == 0  # scalar

    def test_label_mapping(self, sm_dataset_entries):
        ds = SMDataset(sm_dataset_entries)
        # label2idx и idx2label — взаимно обратные
        for label, idx in ds.label2idx.items():
            assert ds.idx2label[idx] == label

    def test_class_distribution(self, sm_dataset_entries):
        ds = SMDataset(sm_dataset_entries)
        dist = ds.get_class_distribution()
        for cls_name, count in dist.items():
            assert count == SAMPLES_PER_CLASS

    def test_custom_label2idx(self, sm_dataset_entries):
        custom = {f"class_{i}": i * 10 for i in range(NUM_CLASSES)}
        ds = SMDataset(sm_dataset_entries, label2idx=custom)
        for label, idx in ds.label2idx.items():
            assert idx == custom[label]

    def test_embeddings_tensor_dtype(self, sm_dataset_entries):
        ds = SMDataset(sm_dataset_entries)
        assert ds.embeddings.dtype == torch.float32
        assert ds.labels.dtype == torch.long


# ═══════════════════════ Загрузка и стратификация ═══════════════════════

class TestLoadAndSplit:
    """load_and_split_dataset — стратификация по классам."""

    def test_split_sizes(self, sm_dataset_path):
        train, val, test = load_and_split_dataset(sm_dataset_path)
        total = len(train) + len(val) + len(test)
        assert total == NUM_CLASSES * SAMPLES_PER_CLASS

    def test_split_proportions(self, sm_dataset_path):
        train, val, test = load_and_split_dataset(
            sm_dataset_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        total = len(train) + len(val) + len(test)
        assert len(train) >= total * 0.5  # не меньше половины

    def test_all_classes_present(self, sm_dataset_path):
        """Каждый сплит содержит все классы (стратификация)."""
        train, val, test = load_and_split_dataset(sm_dataset_path)
        for ds in [train, val, test]:
            class_labels = set(ds.idx2label[l.item()] for l in ds.labels)
            assert len(class_labels) == NUM_CLASSES

    def test_shared_label2idx(self, sm_dataset_path):
        """Все сплиты используют единый label2idx."""
        train, val, test = load_and_split_dataset(sm_dataset_path)
        assert train.label2idx == val.label2idx == test.label2idx

    def test_reproducibility(self, sm_dataset_path):
        """С одинаковым seed — одинаковые сплиты."""
        t1, v1, te1 = load_and_split_dataset(sm_dataset_path, seed=42)
        t2, v2, te2 = load_and_split_dataset(sm_dataset_path, seed=42)
        assert torch.equal(t1.embeddings, t2.embeddings)
        assert torch.equal(t1.labels, t2.labels)

    def test_dataloaders(self, sm_dataset_path):
        """create_dataloaders возвращает рабочие DataLoader-ы."""
        train, val, test = load_and_split_dataset(sm_dataset_path)
        tl, vl, tel = create_dataloaders(train, val, test, batch_size=4)
        batch = next(iter(tl))
        assert len(batch) == 2  # (embeddings, labels)
        assert batch[0].shape[1] == EMBED_DIM


# ═══════════════════════ Модель ═══════════════════════

class TestProjectionHead:
    """Тесты ProjectionHead MLP."""

    def test_output_shape(self):
        head = ProjectionHead(
            input_dim=EMBED_DIM,
            projection_dims=[512, 384],
            output_dim=SM_OUTPUT_DIM,
        )
        x = torch.randn(8, EMBED_DIM)
        out = head(x)
        assert out.shape == (8, SM_OUTPUT_DIM)

    def test_l2_normalized(self):
        """Выход L2-нормализован (unit vectors)."""
        head = ProjectionHead(
            input_dim=EMBED_DIM,
            projection_dims=[512, 384],
            output_dim=SM_OUTPUT_DIM,
        )
        x = torch.randn(16, EMBED_DIM)
        out = head(x)
        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    def test_no_batch_norm(self):
        """Модель без BatchNorm работает."""
        head = ProjectionHead(
            input_dim=EMBED_DIM,
            projection_dims=[256],
            output_dim=SM_OUTPUT_DIM,
            use_batch_norm=False,
        )
        x = torch.randn(4, EMBED_DIM)
        out = head(x)
        assert out.shape == (4, SM_OUTPUT_DIM)

    def test_single_sample_eval(self):
        """Один сэмпл в eval mode (BatchNorm не ломается)."""
        head = ProjectionHead(
            input_dim=EMBED_DIM,
            projection_dims=[256],
            output_dim=SM_OUTPUT_DIM,
        )
        head.eval()
        x = torch.randn(1, EMBED_DIM)
        out = head(x)
        assert out.shape == (1, SM_OUTPUT_DIM)


class TestSchemaMatchingModel:
    """Тесты полной SM модели."""

    def _make_config(self):
        return SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[512, 384],
            output_dim=SM_OUTPUT_DIM,
        )

    def test_forward(self):
        model = SchemaMatchingModel(self._make_config())
        x = torch.randn(8, EMBED_DIM)
        out = model(x)
        assert out.shape == (8, SM_OUTPUT_DIM)

    def test_get_embeddings(self):
        model = SchemaMatchingModel(self._make_config())
        x = torch.randn(8, EMBED_DIM)
        emb = model.get_embeddings(x)
        assert emb.shape == (8, SM_OUTPUT_DIM)
        # модель должна быть в eval mode после get_embeddings
        assert not model.training

    def test_save_load_roundtrip(self, tmp_path):
        """Модель сохраняется и загружается с идентичными весами."""
        config = self._make_config()
        model = SchemaMatchingModel(config)
        model.eval()
        x = torch.randn(4, EMBED_DIM)

        with torch.no_grad():
            out_before = model(x)

        path = str(tmp_path / "sm_model.pt")
        model.save(path)

        loaded = SchemaMatchingModel.load(path, device="cpu")

        with torch.no_grad():
            out_after = loaded(x)

        assert torch.allclose(out_before, out_after, atol=1e-6)

    def test_load_preserves_config(self, tmp_path):
        """Загруженная модель имеет правильную конфигурацию."""
        config = self._make_config()
        model = SchemaMatchingModel(config)

        path = str(tmp_path / "sm_model.pt")
        model.save(path)

        loaded = SchemaMatchingModel.load(path)
        assert loaded.config.input_dim == EMBED_DIM
        assert loaded.config.output_dim == SM_OUTPUT_DIM
        assert loaded.config.projection_dims == [512, 384]


# ═══════════════════════ SMTrainer (мини-обучение) ═══════════════════════

class TestSMTrainer:
    """Интеграционный тест тренера SM на синтетических данных."""

    def test_training_loop(self, sm_dataset_path, tmp_path):
        """Полный цикл обучения на минимальных данных."""
        from table_unifier.schema_matching.trainer import SMTrainer

        config = SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[256],
            output_dim=SM_OUTPUT_DIM,
            batch_size=8,
            num_epochs=2,
            learning_rate=1e-3,
            margin=0.2,
            mining_strategy="all",
            scheduler_patience=1,
            early_stopping_patience=3,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )

        trainer = SMTrainer(config, device="cpu")
        save_dir = str(tmp_path / "sm_train_output")

        history = trainer.train(
            dataset_path=sm_dataset_path,
            save_dir=save_dir,
        )

        # Проверяем историю
        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) > 0
        assert len(history["val_loss"]) > 0

        # Проверяем сохранённые файлы
        assert os.path.exists(os.path.join(save_dir, "best_model.pt"))
        assert os.path.exists(os.path.join(save_dir, "training_results.json"))
        assert os.path.exists(os.path.join(save_dir, "label2idx.json"))

    def test_model_improves_or_stable(self, sm_dataset_path, tmp_path):
        """Loss не взрывается за несколько эпох."""
        from table_unifier.schema_matching.trainer import SMTrainer

        config = SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[128],
            output_dim=SM_OUTPUT_DIM,
            batch_size=16,
            num_epochs=3,
            mining_strategy="all",
            early_stopping_patience=5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        trainer = SMTrainer(config, device="cpu")

        history = trainer.train(
            dataset_path=sm_dataset_path,
            save_dir=str(tmp_path / "sm_stable"),
        )

        losses = history["train_loss"]
        # Loss не должен быть NaN или inf
        for loss in losses:
            assert np.isfinite(loss), f"Loss is not finite: {loss}"

    def test_training_results_json(self, sm_dataset_path, tmp_path):
        """training_results.json содержит все необходимые ключи."""
        from table_unifier.schema_matching.trainer import SMTrainer

        config = SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[128],
            output_dim=SM_OUTPUT_DIM,
            num_epochs=1,
            mining_strategy="all",
            early_stopping_patience=5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        trainer = SMTrainer(config, device="cpu")
        save_dir = str(tmp_path / "sm_results")
        trainer.train(dataset_path=sm_dataset_path, save_dir=save_dir)

        with open(os.path.join(save_dir, "training_results.json"), "r") as f:
            results = json.load(f)

        assert "best_epoch" in results
        assert "best_val_loss" in results
        assert "test_metrics" in results
        assert "dataset" in results
        assert results["dataset"]["num_classes"] == NUM_CLASSES

    def test_loaded_model_inference(self, sm_dataset_path, tmp_path):
        """После обучения загруженная модель даёт корректный output."""
        from table_unifier.schema_matching.trainer import SMTrainer

        config = SMConfig(
            input_dim=EMBED_DIM,
            projection_dims=[128],
            output_dim=SM_OUTPUT_DIM,
            num_epochs=1,
            mining_strategy="all",
            early_stopping_patience=5,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
        )
        trainer = SMTrainer(config, device="cpu")
        save_dir = str(tmp_path / "sm_inference")
        trainer.train(dataset_path=sm_dataset_path, save_dir=save_dir)

        model = SchemaMatchingModel.load(
            os.path.join(save_dir, "best_model.pt"), device="cpu"
        )
        x = torch.randn(3, EMBED_DIM)
        out = model.get_embeddings(x)
        assert out.shape == (3, SM_OUTPUT_DIM)

        norms = torch.norm(out, p=2, dim=1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)
