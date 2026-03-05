"""
TableUnifier — система унификации табличных схем.

Три основных подмодуля:
- data_generation:    генерация синтетических данных (SM + ER)
- schema_matching:    обучение и инференс Schema Matching (Triplet Loss)
- entity_resolution:  обучение и инференс Entity Resolution (GNN)

Общие утилиты:
- models:  Ollama API обёртки (OllamaLLM, OllamaEmbedding)
- core:    TableUnifier (column embedding pipeline)
- config:  AppConfig (базовая конфигурация Ollama + обработки столбцов)
"""

__version__ = "3.0.0"
__author__ = "Oleg Keatzin"
__license__ = "MIT"

from .config import AppConfig, OllamaConfig, EmbeddingConfig, ClassifierConfig
from .models import OllamaLLM, OllamaEmbedding
from .core import TableUnifier, EmbSerialV2

__all__ = [
    # Config
    'AppConfig',
    'OllamaConfig',
    'EmbeddingConfig',
    'ClassifierConfig',
    # Ollama wrappers
    'OllamaLLM',
    'OllamaEmbedding',
    # Column embedding pipeline
    'TableUnifier',
    'EmbSerialV2',
]
