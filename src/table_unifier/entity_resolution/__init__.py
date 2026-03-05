"""
Entity Resolution на основе Graph Neural Networks.

Модуль расширяет курсовой проект по Schema Matching до дипломной работы
по поиску дубликатов строк в табличных данных с разными схемами.

Архитектура:
    ┌───────────────────────────────────────────────────────────┐
    │  Таблица A        Таблица B                              │
    │  ┌──────────┐    ┌──────────┐                            │
    │  │ Row Nodes │    │ Row Nodes │  ← Ollama embeddings     │
    │  └─────┬────┘    └────┬─────┘                            │
    │        │              │                                   │
    │        └──── edges ───┘  ← edge_attr = column embeddings │
    │              │                (из курсового проекта)       │
    │        ┌─────┴────┐                                      │
    │        │Token Nodes│  ← CharNgram embeddings (learnable)  │
    │        └─────┬────┘                                      │
    │              │                                            │
    │     GNN (TransformerConv + edge_attr)                     │
    │              │                                            │
    │     Row Embeddings (L2-normalized)                        │
    │              │                                            │
    │     Metric Learning (Triplet Loss)                        │
    │              │                                            │
    │     Duplicate Detection (cosine similarity)               │
    └───────────────────────────────────────────────────────────┘

Ключевые компоненты:
    - ERConfig: конфигурация всех параметров
    - FastTextEmbedder: предобученные эмбеддинги токенов (устойчивы к опечаткам)
    - TablePairGraphBuilder: построение двудольных графов
    - EntityResolutionGNN: GNN с TransformerConv + edge attributes
    - ERTablePairGenerator: генерация синтетических данных для обучения
    - ERTrainer: обучение с Triplet Loss + mining
    - DuplicateDetector: инференс для поиска дубликатов
"""

from .config import ERConfig
from .token_embedder import FastTextEmbedder
from .graph_builder import TablePairGraphBuilder, compute_column_embeddings
from .gnn_model import EntityResolutionGNN, GNNLayer
from .er_data_generator import ERTablePairGenerator, EntityPool, TablePairData
from .er_dataset import ERGraphDataset, er_collate_fn, build_and_save_graphs, get_embedding_dims, precompute_column_embeddings
from .trainer import ERTrainer
from .inference import DuplicateDetector, DuplicateMatch, evaluate_on_test

__all__ = [
    # Config
    'ERConfig',
    # Token embedding
    'FastTextEmbedder',
    # Graph building
    'TablePairGraphBuilder',
    'compute_column_embeddings',
    # GNN model
    'EntityResolutionGNN',
    'GNNLayer',
    # Data generation
    'ERTablePairGenerator',
    'EntityPool',
    'TablePairData',
    # Dataset
    'ERGraphDataset',
    'er_collate_fn',
    'build_and_save_graphs',
    'precompute_column_embeddings',
    'get_embedding_dims',
    # Training
    'ERTrainer',
    # Inference
    'DuplicateDetector',
    'DuplicateMatch',
    'evaluate_on_test',
]
