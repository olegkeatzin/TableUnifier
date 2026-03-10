"""Модель проекции для Schema Matching.

Архитектура (из Модель проекции.canvas):
  input (4096 dim) → Linear → BatchNorm → GELU → Dropout
                    → Linear → L2-normalize → output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SchemaProjector(nn.Module):
    """Проектор сырого эмбеддинга столбца в метрическое пространство."""

    def __init__(
        self,
        input_dim: int = 4096,
        hidden_dim: int = 1024,
        output_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Вход: [B, input_dim]. Выход: L2-нормализованный [B, output_dim]."""
        out = self.net(x)
        return F.normalize(out, p=2, dim=-1)
