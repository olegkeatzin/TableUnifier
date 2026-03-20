"""Один слой GNN с edge-aware message passing.

Простой однонаправленный слой (token → row):
  message = Linear(concat(token_h, edge_attr))
  aggregation = mean
  update = residual + LayerNorm
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing


class EdgeMeanConv(MessagePassing):
    """Mean-aggregation с edge features.

    message_j = Linear(concat(h_j, edge_attr))
    row_h_new = mean(messages) → residual + LayerNorm
    """

    def __init__(self, in_dim: int, edge_dim: int, out_dim: int):
        super().__init__(aggr="mean")
        self.lin = nn.Linear(in_dim + edge_dim, out_dim)

    def forward(
        self,
        x_src: torch.Tensor,
        x_dst: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        size = (x_src.size(0), x_dst.size(0))
        return self.propagate(edge_index, x=x_src, edge_attr=edge_attr, size=size)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))


class GNNLayer(nn.Module):
    """Однонаправленный слой: Token → Row с edge features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_heads: int = 1,  # не используется, оставлен для совместимости
        dropout: float = 0.1,
    ):
        super().__init__()
        self.conv = EdgeMeanConv(hidden_dim, edge_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        row_x: torch.Tensor,
        token_x: torch.Tensor,
        t2r_edge_index: torch.Tensor,
        edge_attr_t2r: torch.Tensor,
        r2t_edge_index: torch.Tensor,
        edge_attr_r2t: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Только Token → Row (без обратного направления)
        row_msg = self.conv(token_x, row_x, t2r_edge_index, edge_attr_t2r)
        row_x = self.norm(row_x + self.dropout(row_msg))

        # token_x не обновляется — возвращаем как есть
        return row_x, token_x
