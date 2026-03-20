"""Модель Entity Resolution на основе GNN.

Упрощённая архитектура:
  1. Проекционные слои (row_proj, token_proj, edge_proj)
  2. L слоёв GNNLayer (token→row, mean aggregation, edge features)
  3. Output Head (Linear + L2-нормализация)

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
        hidden_dim: int = 128,
        edge_dim: int = 64,
        output_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 1,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_gnn_layers = num_gnn_layers

        # ---- 1. Проекционные слои ---- #
        self.row_proj = nn.Sequential(
            nn.Linear(row_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.token_proj = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # edge_proj: col_embeddings [n_cols, 4096] → [n_cols, edge_dim]
        self.edge_proj = nn.Sequential(
            nn.Linear(col_dim, edge_dim),
            nn.GELU(),
        )

        # ---- 2. GNN слои ---- #
        self.gnn_layers = nn.ModuleList([
            GNNLayer(hidden_dim, edge_dim, num_heads, dropout)
            for _ in range(num_gnn_layers)
        ])

        # ---- 3. Output Head ---- #
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """
        Returns:
            L2-нормализованные эмбеддинги строк [N_rows, D_output].
        """
        row_x = self.row_proj(data["row"].x)
        token_x = self.token_proj(data["token"].x)

        # Проецируем col_embeddings ДО индексации
        col_emb_proj = self.edge_proj(data.col_embeddings)  # [n_cols, edge_dim]
        t2r_col_idx = data["token", "in_row", "row"].edge_col_idx
        r2t_col_idx = data["row", "has_token", "token"].edge_col_idx

        edge_attr_t2r = col_emb_proj[t2r_col_idx]  # [E, edge_dim]
        edge_attr_r2t = col_emb_proj[r2t_col_idx]  # [E, edge_dim]

        t2r_edge_index = data["token", "in_row", "row"].edge_index
        r2t_edge_index = data["row", "has_token", "token"].edge_index

        for layer in self.gnn_layers:
            row_x, token_x = layer(
                row_x, token_x,
                t2r_edge_index, edge_attr_t2r,
                r2t_edge_index, edge_attr_r2t,
            )

        output = self.output_head(row_x)
        output = F.normalize(output, p=2, dim=-1)

        return output
