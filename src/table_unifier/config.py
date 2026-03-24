from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class OllamaConfig:
    """Настройки Ollama (поддержка удалённого хоста)."""

    host: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embedding_model: str = "qwen3-embedding:8b"


@dataclass
class SchemaMatchingConfig:
    """Настройки модели Schema Matching (проекция эмбеддингов)."""

    embedding_dim: int = 4096  # выход qwen3-embedding:8b
    hidden_dim: int = 1024
    projection_dim: int = 256
    dropout: float = 0.1
    margin: float = 0.3
    lr: float = 1e-3
    epochs: int = 50
    batch_size: int = 64


@dataclass
class EntityResolutionConfig:
    """Настройки модели Entity Resolution (GNN)."""

    token_model_name: str = "cointegrated/rubert-tiny2"
    row_dim: int = 312  # hidden size rubert-tiny2
    token_dim: int = 312  # vocabulary embedding dim
    col_dim: int = 4096  # qwen3-embedding:8b
    hidden_dim: int = 128
    edge_dim: int = 64
    output_dim: int = 128
    num_gnn_layers: int = 2
    dropout: float = 0.1
    margin: float = 0.3
    lr: float = 1e-3
    weight_decay: float = 0.0
    epochs: int = 50
    batch_size: int = 32
    bidirectional: bool = False


@dataclass
class Config:
    """Главная конфигурация проекта."""

    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    schema_matching: SchemaMatchingConfig = field(default_factory=SchemaMatchingConfig)
    entity_resolution: EntityResolutionConfig = field(default_factory=EntityResolutionConfig)
    data_dir: Path = Path("data")
    output_dir: Path = Path("output")
