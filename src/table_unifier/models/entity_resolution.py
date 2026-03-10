"""Полная модель Entity Resolution на основе GNN.

Архитектура (из Entity Resolution.canvas):
  1. Проекционные слои (row_proj, token_proj, edge_proj)
  2. L слоёв GNNLayer
  3. Jumping Knowledge (concat + linear)
  4. Output Head (MLP + L2-нормализация)

Выход: [N_rows, D_output] — эмбеддинги строк для Triplet Loss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from table_unifier.models.gnn_layer import GNNLayer


class EntityResolutionGNN(nn.Module):
    def __init__(
        self,
        row_dim: int = 312,
        token_dim: int = 312,
        col_dim: int = 4096,
        hidden_dim: int = 256,
        edge_dim: int = 128,
        output_dim: int = 128,
        num_gnn_layers: int = 3,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers

        # ---- 1. Проекционные слои ---- #

        # row_proj: Linear → LayerNorm → GELU → Dropout
        self.row_proj = nn.Sequential(
            nn.Linear(row_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # token_proj: Linear → LayerNorm → GELU → Dropout
        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # edge_proj: Linear → GELU
        self.edge_proj = nn.Sequential(
            nn.Linear(col_dim, edge_dim),
            nn.GELU(),
        )

        # ---- 2. GNN слои ---- #
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        # ---- 3. Jumping Knowledge ---- #
        self.jk_linear = nn.Linear(hidden_dim * (num_gnn_layers + 1), hidden_dim)

        # ---- 4. Output Head ---- #
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Args:
            data: HeteroData с узлами 'row', 'token' и рёбрами
                  ('token', 'in_row', 'row'), ('row', 'has_token', 'token').

        Returns:
            L2-нормализованные эмбеддинги строк [N_rows, D_output].
        """
        # Проекция входных признаков
        row_x = self.row_proj(data["row"].x)
        token_x = self.token_proj(data["token"].x)

        edge_attr_t2r = self.edge_proj(
            data["token", "in_row", "row"].edge_attr
        )
        edge_attr_r2t = self.edge_proj(
            data["row", "has_token", "token"].edge_attr
        )

        t2r_edge_index = data["token", "in_row", "row"].edge_index
        r2t_edge_index = data["row", "has_token", "token"].edge_index

        # Jumping Knowledge: собираем row_x с каждого уровня
        row_representations = [row_x]

        for layer in self.gnn_layers:
            row_x, token_x = layer(
                row_x, token_x,
                t2r_edge_index, edge_attr_t2r,
                r2t_edge_index, edge_attr_r2t,
            )
            row_representations.append(row_x)

        # Concat всех слоёв: [N_rows, D_hidden * (L+1)]
        jk_input = torch.cat(row_representations, dim=-1)
        output = self.jk_linear(jk_input)

        # Output head → L2-нормализация
        output = self.output_head(output)
        output = F.normalize(output, p=2, dim=-1)

        return output
