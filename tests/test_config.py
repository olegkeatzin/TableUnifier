"""Тесты для config.py."""

from pathlib import Path

from table_unifier.config import (
    Config,
    EntityResolutionConfig,
    OllamaConfig,
    SchemaMatchingConfig,
)


class TestOllamaConfig:
    def test_defaults(self):
        cfg = OllamaConfig()
        assert cfg.host == "http://localhost:11434"
        assert cfg.llm_model == "qwen3.5:9b"
        assert cfg.embedding_model == "qwen3-embedding:8b"

    def test_custom_host(self):
        cfg = OllamaConfig(host="http://remote:11434")
        assert cfg.host == "http://remote:11434"


class TestSchemaMatchingConfig:
    def test_defaults(self):
        cfg = SchemaMatchingConfig()
        assert cfg.embedding_dim == 4096
        assert cfg.hidden_dim == 1024
        assert cfg.projection_dim == 256
        assert 0 < cfg.dropout < 1
        assert cfg.margin > 0

    def test_custom(self):
        cfg = SchemaMatchingConfig(embedding_dim=768, projection_dim=128)
        assert cfg.embedding_dim == 768
        assert cfg.projection_dim == 128


class TestEntityResolutionConfig:
    def test_defaults(self):
        cfg = EntityResolutionConfig()
        assert cfg.row_dim == 312
        assert cfg.token_dim == 312
        assert cfg.col_dim == 4096
        assert cfg.hidden_dim == 128
        assert cfg.num_gnn_layers == 2


class TestConfig:
    def test_defaults(self):
        cfg = Config()
        assert isinstance(cfg.ollama, OllamaConfig)
        assert isinstance(cfg.schema_matching, SchemaMatchingConfig)
        assert isinstance(cfg.entity_resolution, EntityResolutionConfig)
        assert cfg.data_dir == Path("data")
        assert cfg.output_dir == Path("output")

    def test_nested_override(self):
        cfg = Config(ollama=OllamaConfig(host="http://gpu-server:11434"))
        assert cfg.ollama.host == "http://gpu-server:11434"
