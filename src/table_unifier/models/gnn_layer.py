"""Один слой GNN с двунаправленным TransformerConv.

Пинг-понг Message Passing (из Entity Resolution.canvas):
  Шаг 1: Token → Row  (t2r_conv)  + Residual + LayerNorm
  Шаг 2: Row → Token   (r2t_conv) + Residual + LayerNorm
"""

import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class GNNLayer(nn.Module):
    """Двудольный слой GNN с TransformerConv."""

    def __init__(
        self,
        hidden_dim: int = 256,
        edge_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0

        # Шаг 1: Token → Row
        self.t2r_conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.row_norm = nn.LayerNorm(hidden_dim)
        self.row_dropout = nn.Dropout(dropout)

        # Шаг 2: Row → Token
        self.r2t_conv = TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
        )
        self.token_norm = nn.LayerNorm(hidden_dim)
        self.token_dropout = nn.Dropout(dropout)

    def forward(
        self,
        row_x: torch.Tensor,
        token_x: torch.Tensor,
        t2r_edge_index: torch.Tensor,
        edge_attr_t2r: torch.Tensor,
        r2t_edge_index: torch.Tensor,
        edge_attr_r2t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            row_x:          [N_rows, D_hidden]
            token_x:        [N_tokens, D_hidden]
            t2r_edge_index: [2, E] — token(src) → row(dst)
            edge_attr_t2r:  [E, D_edge]
            r2t_edge_index: [2, E] — row(src) → token(dst)
            edge_attr_r2t:  [E, D_edge]

        Returns:
            (row_x_updated, token_x_updated)
        """
        # Шаг 1: Tokens → Rows
        row_msg = self.t2r_conv(
            (token_x, row_x), t2r_edge_index, edge_attr_t2r,
        )
        row_x = self.row_norm(row_x + self.row_dropout(row_msg))

        # Шаг 2: Rows → Tokens
        token_msg = self.r2t_conv(
            (row_x, token_x), r2t_edge_index, edge_attr_r2t,
        )
        token_x = self.token_norm(token_x + self.token_dropout(token_msg))

        return row_x, token_x
