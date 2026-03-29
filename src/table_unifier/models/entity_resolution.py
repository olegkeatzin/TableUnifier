"""Модели Entity Resolution на основе GNN.

Две архитектуры:
  - EntityResolutionGNN: EdgeMeanConv (оригинальная)
  - EntityResolutionGAT: GATv2Conv с edge features (v3)

Общая схема:
  1. Проекционные слои (row_proj, token_proj, edge_proj)
  2. L слоёв GNN/GAT
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
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

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
            GNNLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                dropout=dropout,
                bidirectional=bidirectional,
            )
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
        t2r_col_idx = data["token", "in_row", "row"].edge_col_idx.long()
        r2t_col_idx = data["row", "has_token", "token"].edge_col_idx.long()

        edge_attr_t2r = col_emb_proj[t2r_col_idx]  # [E, edge_dim]
        edge_attr_r2t = col_emb_proj[r2t_col_idx]  # [E, edge_dim]

        t2r_edge_index = data["token", "in_row", "row"].edge_index.long()
        r2t_edge_index = data["row", "has_token", "token"].edge_index.long()

        for layer in self.gnn_layers:
            row_x, token_x = layer(
                row_x, token_x,
                t2r_edge_index, edge_attr_t2r,
                r2t_edge_index, edge_attr_r2t,
            )

        output = self.output_head(row_x)
        output = F.normalize(output, p=2, dim=-1)

        return output


# ------------------------------------------------------------------ #
#  GATv2 вариант
# ------------------------------------------------------------------ #

from table_unifier.models.gat_layer import GATLayer


class EntityResolutionGAT(nn.Module):
    """Entity Resolution модель на основе GATv2 с edge features.

    Тот же интерфейс что у EntityResolutionGNN:
      forward(data: HeteroData) → [N_rows, output_dim] L2-normalized
    """

    def __init__(
        self,
        row_dim: int = 312,
        token_dim: int = 312,
        col_dim: int = 4096,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        output_dim: int = 128,
        num_gnn_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()

        # Проекционные слои (идентичны EntityResolutionGNN)
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
        self.edge_proj = nn.Sequential(
            nn.Linear(col_dim, edge_dim),
            nn.GELU(),
        )

        # GATv2 слои
        self.gnn_layers = nn.ModuleList([
            GATLayer(
                hidden_dim=hidden_dim,
                edge_dim=edge_dim,
                num_heads=num_heads,
                dropout=dropout,
                attention_dropout=attention_dropout,
                bidirectional=bidirectional,
            )
            for _ in range(num_gnn_layers)
        ])

        # Output Head
        self.output_head = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, data: HeteroData) -> torch.Tensor:
        """Returns: L2-нормализованные эмбеддинги строк [N_rows, output_dim]."""
        row_x = self.row_proj(data["row"].x)
        token_x = self.token_proj(data["token"].x)

        col_emb_proj = self.edge_proj(data.col_embeddings)
        t2r_col_idx = data["token", "in_row", "row"].edge_col_idx.long()
        r2t_col_idx = data["row", "has_token", "token"].edge_col_idx.long()

        edge_attr_t2r = col_emb_proj[t2r_col_idx]
        edge_attr_r2t = col_emb_proj[r2t_col_idx]

        t2r_edge_index = data["token", "in_row", "row"].edge_index.long()
        r2t_edge_index = data["row", "has_token", "token"].edge_index.long()

        for layer in self.gnn_layers:
            row_x, token_x = layer(
                row_x, token_x,
                t2r_edge_index, edge_attr_t2r,
                r2t_edge_index, edge_attr_r2t,
            )

        output = self.output_head(row_x)
        output = F.normalize(output, p=2, dim=-1)
        return output


# ------------------------------------------------------------------ #
#  Classification head для BCE-обучения
# ------------------------------------------------------------------ #


class PairClassifier(nn.Module):
    """Классификатор пар строк поверх GNN/GAT backbone.

    Берёт |emb_a - emb_b| и пропускает через MLP.
    Возвращает raw logits — использовать с BCEWithLogitsLoss
    (безопасен для mixed precision / AMP).
    """

    def __init__(self, backbone: nn.Module, embedding_dim: int = 128):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(embedding_dim, 1),
        )

    def get_embeddings(self, data: HeteroData) -> torch.Tensor:
        """L2-нормализованные эмбеддинги строк (для eval/кластеризации)."""
        return self.backbone(data)

    def forward(
        self, data: HeteroData, pairs: torch.Tensor,
    ) -> torch.Tensor:
        """Raw logits для пар строк.

        Args:
            data:  HeteroData граф.
            pairs: [P, 2] — индексы строк (idx_a, idx_b).

        Returns:
            [P] — logits (до sigmoid). Для P(match): torch.sigmoid(logits).
        """
        embeddings = self.backbone(data)
        emb_a = embeddings[pairs[:, 0]]
        emb_b = embeddings[pairs[:, 1]]
        diff = torch.abs(emb_a - emb_b)
        return self.classifier(diff).squeeze(-1)
