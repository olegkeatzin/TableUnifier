# src/table_unifier/models/gat_layer.py
"""GATv2 Layer с edge features для гетерогенного графа.

GATv2Conv: a^T · LeakyReLU(W · [h_i || h_j || e_ij])
— dynamic attention, учитывающий и узлы, и контекст столбца (edge_attr).

Направления:
  - unidirectional: только token → row
  - bidirectional: token → row + row → token
"""

import torch
import torch.nn as nn
from torch_geometric.nn import GATv2Conv


class GATLayer(nn.Module):
    """Слой GAT: Token → Row (+ опционально Row → Token) с edge features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_heads: int = 4,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
        )
        head_dim = hidden_dim // num_heads

        # Token → Row
        self.conv_t2r = GATv2Conv(
            in_channels=(hidden_dim, hidden_dim),
            out_channels=head_dim,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=attention_dropout,
            concat=True,  # output = num_heads * head_dim = hidden_dim
            add_self_loops=False,  # bipartite: разные типы узлов
        )
        self.norm_row = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Row → Token (опционально)
        if bidirectional:
            self.conv_r2t = GATv2Conv(
                in_channels=(hidden_dim, hidden_dim),
                out_channels=head_dim,
                heads=num_heads,
                edge_dim=edge_dim,
                dropout=attention_dropout,
                concat=True,
                add_self_loops=False,  # bipartite: разные типы узлов
            )
            self.norm_token = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        row_x: torch.Tensor,
        token_x: torch.Tensor,
        t2r_edge_index: torch.Tensor,
        edge_attr_t2r: torch.Tensor,
        r2t_edge_index: torch.Tensor,
        edge_attr_r2t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Token → Row
        row_msg = self.conv_t2r((token_x, row_x), t2r_edge_index, edge_attr_t2r)
        row_x = self.norm_row(row_x + self.dropout(row_msg))

        # Row → Token (если bidirectional)
        if self.bidirectional:
            token_msg = self.conv_r2t((row_x, token_x), r2t_edge_index, edge_attr_r2t)
            token_x = self.norm_token(token_x + self.dropout(token_msg))

        return row_x, token_x
