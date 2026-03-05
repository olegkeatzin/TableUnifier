"""
Тесты для Pipeline 1: загрузка конфигурации и создание DataGenConfig.

Покрывает:
- Загрузка config.json и доступ к параметрам data_generation
- Создание DataGenConfig из словаря
- Валидация параметров DataGenConfig
- Свойства total_er_pairs, total_tables
"""

import json
import pytest

from table_unifier.data_generation import DataGenConfig


class TestConfigLoading:
    """Загрузка конфигурации из JSON."""

    def test_load_config_json(self, config_json_path):
        """config.json корректно читается и содержит все секции."""
        with open(config_json_path, "r", encoding="utf-8") as f:
            config = json.load(f)

        assert "data_generation" in config
        assert "schema_matching" in config
        assert "entity_resolution" in config
        assert "inference" in config

    def test_data_gen_params_fields(self, full_config):
        """Секция data_generation содержит все необходимые ключи."""
        params = full_config["data_generation"]
        required = [
            "ollama_host", "llm_model", "embedding_model",
            "num_train_pairs", "num_val_pairs", "num_test_pairs",
            "num_extra_sm_tables", "output_dir",
            "sm_dataset_file", "sm_metadata_file", "er_raw_dir",
        ]
        for key in required:
            assert key in params, f"Ключ '{key}' отсутствует в data_generation"


class TestDataGenConfig:
    """Создание и валидация DataGenConfig."""

    def test_create_from_dict(self, full_config):
        """DataGenConfig создаётся из словаря без ошибок."""
        params = full_config["data_generation"]
        config = DataGenConfig(**params)

        assert config.ollama_host == params["ollama_host"]
        assert config.llm_model == params["llm_model"]
        assert config.num_train_pairs == params["num_train_pairs"]
        assert config.output_dir == params["output_dir"]

    def test_total_er_pairs(self, full_config):
        """total_er_pairs = train + val + test."""
        params = full_config["data_generation"]
        config = DataGenConfig(**params)

        expected = params["num_train_pairs"] + params["num_val_pairs"] + params["num_test_pairs"]
        assert config.total_er_pairs == expected

    def test_total_tables(self, full_config):
        """total_tables = er_pairs * 2 + extra SM."""
        params = full_config["data_generation"]
        config = DataGenConfig(**params)

        expected = config.total_er_pairs * 2 + params["num_extra_sm_tables"]
        assert config.total_tables == expected

    def test_default_values(self):
        """DataGenConfig с дефолтами создаётся без ошибок."""
        config = DataGenConfig()
        assert config.min_rows_per_table <= config.max_rows_per_table
        assert config.min_common_entities <= config.max_common_entities
        assert config.min_unique_entities <= config.max_unique_entities
        assert 0.0 <= config.perturbation_prob <= 1.0
        assert 0.0 <= config.missing_value_prob <= 1.0

    def test_save_load_roundtrip(self, full_config, tmp_path):
        """DataGenConfig сохраняется и загружается без потерь."""
        params = full_config["data_generation"]
        config = DataGenConfig(**params)

        path = str(tmp_path / "config_test.json")
        config.save(path)

        with open(path, "r", encoding="utf-8") as f:
            loaded = json.load(f)

        assert loaded["ollama_host"] == config.ollama_host
        assert loaded["num_train_pairs"] == config.num_train_pairs
        assert loaded["output_dir"] == config.output_dir

    def test_er_raw_dir_path(self, full_config):
        """Путь к ER данным строится корректно."""
        import os
        params = full_config["data_generation"]
        config = DataGenConfig(**params)

        er_dir = os.path.join(config.output_dir, config.er_raw_dir)
        assert er_dir == os.path.join("unified_dataset", "raw")
