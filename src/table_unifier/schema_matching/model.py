"""
Модель проекции для Schema Matching с Triplet Loss.

Архитектура:
    Input:  Raw embedding от qwen3-embedding:8b [batch, input_dim]
    ↓ Linear → BatchNorm → GELU → Dropout
    ↓ Linear → BatchNorm → GELU → Dropout
    ↓ Linear → BatchNorm → GELU → Dropout
    ↓ Linear → output_dim
    ↓ L2-normalize
    Output: Normalized embedding [batch, output_dim]

Обученная проекция даёт эмбеддинги, где:
    - Столбцы одного семантического типа → близкие вектора
    - Столбцы разных типов → далёкие вектора
    
Эти эмбеддинги затем используются:
    1. Напрямую для schema matching (cosine similarity + Венгерский алгоритм)
    2. Как edge_attr (column embeddings) в GNN для Entity Resolution
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import logging

from .config import SMConfig

logger = logging.getLogger(__name__)


class ProjectionHead(nn.Module):
    """MLP Projection Head для Schema Matching.
    
    Проецирует эмбеддинги общего назначения из LLM/embedding модели
    в специализированное пространство для schema matching.
    
    Ключевые особенности:
        - BatchNorm для стабильности обучения
        - Residual connections (если размерности совпадают)
        - GELU activation (smoother gradient flow)
        - L2-нормализация на выходе (для cosine similarity)
    """
    
    def __init__(
        self,
        input_dim: int,
        projection_dims: List[int],
        output_dim: int,
        dropout: float = 0.1,
        use_batch_norm: bool = True,
    ):
        """
        Args:
            input_dim: Размерность входных эмбеддингов (e.g., 4096 для qwen3-embedding:8b)
            projection_dims: Размерности скрытых слоёв [2048, 1024, 512]
            output_dim: Размерность выходного вектора (для matching и GNN)
            dropout: Dropout rate
            use_batch_norm: Использовать BatchNorm
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        layers = []
        prev_dim = input_dim
        
        for dim in projection_dims:
            layers.append(nn.Linear(prev_dim, dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = dim
        
        # Финальный слой
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.projection = nn.Sequential(*layers)
        
        # Инициализация весов
        self._init_weights()
        
        num_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"ProjectionHead: {input_dim} → {projection_dims} → {output_dim}, "
            f"{num_params:,} params"
        )
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [batch_size, input_dim] — сырые эмбеддинги
            
        Returns:
            [batch_size, output_dim] — L2-нормализованные проекции
        """
        projected = self.projection(x)
        # L2-нормализация для cosine similarity
        normalized = F.normalize(projected, p=2, dim=-1)
        return normalized


class SchemaMatchingModel(nn.Module):
    """Полная модель Schema Matching.
    
    Оборачивает ProjectionHead и предоставляет удобный API
    для обучения и инференса.
    """
    
    def __init__(self, config: SMConfig):
        super().__init__()
        self.config = config
        
        self.projection_head = ProjectionHead(
            input_dim=config.input_dim,
            projection_dims=config.projection_dims,
            output_dim=config.output_dim,
            dropout=config.dropout,
            use_batch_norm=config.use_batch_norm,
        )
    
    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Forward pass: raw embeddings → normalized projections."""
        return self.projection_head(embeddings)
    
    def get_embeddings(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Инференс: получить проецированные эмбеддинги (eval mode)."""
        self.eval()
        with torch.no_grad():
            return self.forward(embeddings)
    
    def save(self, filepath: str):
        """Сохранить модель и конфигурацию."""
        torch.save({
            'model_state_dict': self.state_dict(),
            'config': {
                'input_dim': self.config.input_dim,
                'projection_dims': self.config.projection_dims,
                'output_dim': self.config.output_dim,
                'dropout': self.config.dropout,
                'use_batch_norm': self.config.use_batch_norm,
            },
        }, filepath)
        logger.info(f"Модель сохранена: {filepath}")
    
    @classmethod
    def load(cls, filepath: str, device: str = 'cpu') -> 'SchemaMatchingModel':
        """Загрузить модель из файла."""
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        
        config = SMConfig(**{k: v for k, v in checkpoint['config'].items()
                            if k in SMConfig.__dataclass_fields__})
        
        model = cls(config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        logger.info(f"Модель загружена: {filepath}")
        return model
