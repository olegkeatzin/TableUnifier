"""
TableUnifier - Intelligent Table Schema Unification System

A powerful system for automatic table unification with different schemas
based on semantic analysis of columns using vector embeddings.
"""

__version__ = "2.0.0"
__author__ = "Oleg Keatzin"
__license__ = "MIT"

from .core import TableUnifier, EmbSerialV2
from .config import AppConfig, OllamaConfig, EmbeddingConfig, ClassifierConfig
from .models import OllamaLLM, OllamaEmbedding
from .visualizer import (
    plot_similarity_matrix,
    plot_matching_quality,
    plot_embedding_space,
    generate_mapping_report,
    print_summary
)

__all__ = [
    'TableUnifier',
    'EmbSerialV2',
    'AppConfig',
    'OllamaConfig',
    'EmbeddingConfig',
    'ClassifierConfig',
    'OllamaLLM',
    'OllamaEmbedding',
    'plot_similarity_matrix',
    'plot_matching_quality',
    'plot_embedding_space',
    'generate_mapping_report',
    'print_summary',
]
