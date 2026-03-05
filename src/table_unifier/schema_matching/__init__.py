"""
Schema Matching на основе Triplet Loss.

Модуль обучает проекцию column embeddings для точного сопоставления
столбцов таблиц с разными схемами.

Архитектура:
    ┌─────────────────────────────────────────────────────────────┐
    │  Столбец: "Цена за ед."                                    │
    │     ↓                                                       │
    │  LLM (qwen3.5:9b): генерация описания                     │
    │     ↓                                                       │
    │  Embedding (qwen3-embedding:8b): сырой вектор [4096]       │
    │     ↓                                                       │
    │  ProjectionHead (MLP, trained): [4096] → [256]             │
    │     ↓                                                       │
    │  L2-нормализация                                            │
    │     ↓                                                       │
    │  Triplet Loss обучение                                      │
    │     ↓                                                       │
    │  Schema Matching (cosine sim + Hungarian algorithm)          │
    │     ↓                                                       │
    │  Column Embeddings → GNN edge_attr (Entity Resolution)      │
    └─────────────────────────────────────────────────────────────┘

Компоненты:
    - SMConfig:              конфигурация всех параметров
    - SMDatasetGenerator:    генерация данных (таблицы + описания + эмбеддинги)
    - SMDataset:             PyTorch Dataset для triplet mining
    - SchemaMatchingModel:   MLP projection head
    - SMTrainer:             обучение с Triplet Loss
    - SchemaMatcherInference: инференс (матчинг схем, column embeddings для GNN)
"""

from .config import SMConfig
from .data_generator import SMDatasetGenerator, SMTableGenerator
from .dataset import SMDataset, load_and_split_dataset, create_dataloaders
from .model import SchemaMatchingModel, ProjectionHead
from .trainer import SMTrainer
from .inference import SchemaMatcherInference

__all__ = [
    # Config
    'SMConfig',
    
    # Data Generation
    'SMDatasetGenerator',
    'SMTableGenerator',
    
    # Dataset
    'SMDataset',
    'load_and_split_dataset',
    'create_dataloaders',
    
    # Model
    'SchemaMatchingModel',
    'ProjectionHead',
    
    # Training
    'SMTrainer',
    
    # Inference
    'SchemaMatcherInference',
]
