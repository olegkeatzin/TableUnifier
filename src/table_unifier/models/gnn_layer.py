"""Один слой GNN с edge-aware message passing.

Поддерживает два типа свёрток:
  - "mean": EdgeMeanConv — Linear(concat(h_j, edge_attr)) + mean aggregation
  - "transformer": TransformerConv — multi-head attention с edge features

Направление:
  - unidirectional: только token → row
  - bidirectional: token → row + row → token
"""

import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing, TransformerConv


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
        return self.propagate(edge_index.long(), x=x_src, edge_attr=edge_attr, size=size)

    def message(self, x_j: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.lin(torch.cat([x_j, edge_attr], dim=-1))


def _make_conv(
    conv_type: str, hidden_dim: int, edge_dim: int, num_heads: int,
) -> nn.Module:
    """Фабрика свёрточных слоёв."""
    if conv_type == "mean":
        return EdgeMeanConv(hidden_dim, edge_dim, hidden_dim)
    if conv_type == "transformer":
        return TransformerConv(
            in_channels=hidden_dim,
            out_channels=hidden_dim // num_heads,
            heads=num_heads,
            edge_dim=edge_dim,
            dropout=0.0,
            concat=True,  # out = num_heads * (hidden_dim // num_heads) = hidden_dim
        )
    raise ValueError(f"Unknown conv_type: {conv_type!r}")


class GNNLayer(nn.Module):
    """Слой GNN: Token → Row (+ опционально Row → Token) с edge features."""

    def __init__(
        self,
        hidden_dim: int = 128,
        edge_dim: int = 64,
        num_heads: int = 1,
        dropout: float = 0.1,
        conv_type: str = "mean",
        bidirectional: bool = False,
    ):
        super().__init__()
        self.bidirectional = bidirectional

        # Token → Row
        self.conv_t2r = _make_conv(conv_type, hidden_dim, edge_dim, num_heads)
        self.norm_row = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Row → Token (опционально)
        if bidirectional:
            self.conv_r2t = _make_conv(conv_type, hidden_dim, edge_dim, num_heads)
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
        # Гарантируем int dtype (checkpoint может конвертировать в float)
        t2r_edge_index = t2r_edge_index.long()
        r2t_edge_index = r2t_edge_index.long()

        # Token → Row
        row_msg = self.conv_t2r(token_x, row_x, t2r_edge_index, edge_attr_t2r)
        row_x = self.norm_row(row_x + self.dropout(row_msg))

        # Row → Token (если bidirectional)
        if self.bidirectional:
            token_msg = self.conv_r2t(row_x, token_x, r2t_edge_index, edge_attr_r2t)
            token_x = self.norm_token(token_x + self.dropout(token_msg))

        return row_x, token_x
