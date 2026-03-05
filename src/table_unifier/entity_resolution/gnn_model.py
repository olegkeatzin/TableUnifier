"""
GNN-модель для Entity Resolution с поддержкой атрибутов рёбер.

Архитектура:
    1. Проекционные слои:
       - row_proj:   D_row → D_hidden (проекция Bag-of-Words row embeddings)
       - token_proj: D_token → D_hidden (проекция предобученных FastText эмбеддингов)
       - edge_proj:  D_col → D_edge (проекция column embeddings для TransformerConv)
    
    2. GNN слои (TransformerConv с edge_attr):
       - Token → Row message passing: строки агрегируют информацию от своих токенов
       - Row → Token message passing: токены агрегируют информацию от строк
       - Residual connections + LayerNorm (борьба с over-smoothing)
    
    3. Выходная голова:
       - JumpingKnowledge (опционально): конкатенация представлений со всех слоёв
       - Linear → L2-normalize → финальный вектор строки для метрического обучения

Ключевая инновация:
    Column Embeddings как Edge Features позволяют модели понимать КОНТЕКСТ связи,
    даже для новых, невиданных ранее типов столбцов (Zero-Shot generalization).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple

try:
    from torch_geometric.nn import TransformerConv
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError(
        "PyTorch Geometric не установлен! "
        "pip install torch-geometric"
    )

from .config import ERConfig


class GNNLayer(nn.Module):
    """Один слой двунаправленного Message Passing.
    
    Содержит два TransformerConv:
        - t2r_conv: сообщения от токенов к строкам (Token → Row)
        - r2t_conv: сообщения от строк к токенам (Row → Token)
    
    Каждый conv использует edge_attr (column embeddings) для attention.
    Residual connections + LayerNorm для стабильности.
    """
    
    def __init__(
        self,
        hidden_dim: int,
        edge_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        assert hidden_dim % num_heads == 0, (
            f"hidden_dim ({hidden_dim}) должен делиться на num_heads ({num_heads})"
        )
        head_dim = hidden_dim // num_heads
        
        # Token → Row: строки собирают информацию от токенов
        # Bipartite: in_channels = (src_dim, dst_dim) = (hidden_dim, hidden_dim)
        self.t2r_conv = TransformerConv(
            in_channels=(hidden_dim, hidden_dim),
            out_channels=head_dim,
            heads=num_heads,
            concat=True,       # concat heads → output = num_heads * head_dim = hidden_dim
            edge_dim=edge_dim,
            dropout=dropout,
            root_weight=True,  # self-loop (residual внутри conv)
        )
        
        # Row → Token: токены собирают информацию от строк
        self.r2t_conv = TransformerConv(
            in_channels=(hidden_dim, hidden_dim),
            out_channels=head_dim,
            heads=num_heads,
            concat=True,
            edge_dim=edge_dim,
            dropout=dropout,
            root_weight=True,
        )
        
        # Layer normalization после каждого direction
        self.row_norm = nn.LayerNorm(hidden_dim)
        self.token_norm = nn.LayerNorm(hidden_dim)
        
        # Dropout для регуляризации
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        row_x: torch.Tensor,      # [N_rows, D_hidden]
        token_x: torch.Tensor,    # [N_tokens, D_hidden]
        edge_index_t2r: torch.Tensor,   # [2, E] token(src) → row(dst)
        edge_index_r2t: torch.Tensor,   # [2, E] row(src) → token(dst)
        edge_attr_t2r: torch.Tensor,    # [E, D_edge]
        edge_attr_r2t: torch.Tensor,    # [E, D_edge]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            (new_row_x, new_token_x) — обновлённые представления
        """
        # Token → Row: строки агрегируют информацию от токенов
        row_msg = self.t2r_conv(
            x=(token_x, row_x),
            edge_index=edge_index_t2r,
            edge_attr=edge_attr_t2r,
        )
        row_x = self.row_norm(row_x + self.dropout(row_msg))    # Residual + Norm
        
        # Row → Token: токены агрегируют информацию от строк
        token_msg = self.r2t_conv(
            x=(row_x, token_x),
            edge_index=edge_index_r2t,
            edge_attr=edge_attr_r2t,
        )
        token_x = self.token_norm(token_x + self.dropout(token_msg))  # Residual + Norm
        
        return row_x, token_x


class EntityResolutionGNN(nn.Module):
    """GNN для Entity Resolution.
    
    Принимает HeteroData граф, возвращает L2-нормированные эмбеддинги строк
    для метрического обучения (Triplet Loss).
    
    Архитектура:
        Input → Projection → L×GNNLayer → (JumpingKnowledge) → Output Head → L2 Norm
    """
    
    def __init__(
        self,
        row_input_dim: int,
        col_embed_dim: int,
        config: ERConfig,
    ):
        """
        Args:
            row_input_dim: Размерность row embeddings (BoW из FastText, обычно 300)
            col_embed_dim: Размерность column embeddings (атрибуты рёбер)
            config: Конфигурация ER
        """
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        
        # ═══════════════ Проекции входных признаков ═══════════════
        
        # Проекция row embeddings (BoW из FastText) в скрытое пространство
        self.row_proj = nn.Sequential(
            nn.Linear(row_input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Проекция предобученных FastText token embeddings в скрытое пространство
        self.token_proj = nn.Sequential(
            nn.Linear(config.token_embed_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
        )
        
        # Проекция column embeddings (edge features) в edge dimension
        self.edge_proj = nn.Sequential(
            nn.Linear(col_embed_dim, config.edge_dim),
            nn.GELU(),
        )
        
        # ═══════════════ GNN слои ═══════════════
        
        self.gnn_layers = nn.ModuleList([
            GNNLayer(
                hidden_dim=hidden,
                edge_dim=config.edge_dim,
                num_heads=config.num_heads,
                dropout=config.dropout,
            )
            for _ in range(config.num_gnn_layers)
        ])
        
        # ═══════════════ Выходная голова ═══════════════
        
        if config.use_jumping_knowledge:
            # JK: конкатенируем row представления со всех слоёв + начальное
            jk_input_dim = hidden * (config.num_gnn_layers + 1)
            self.jk_linear = nn.Linear(jk_input_dim, hidden)
        
        # Финальная проекция в пространство метрического обучения
        self.output_head = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(hidden, config.output_dim),
        )
    
    def forward(self, data: HeteroData) -> torch.Tensor:
        """Forward pass: HeteroData → L2-нормированные row embeddings.
        
        Args:
            data: HeteroData граф из TablePairGraphBuilder
            
        Returns:
            [N_rows, output_dim] L2-нормированные эмбеддинги строк
        """
        # ═══════════════ 1. Проекция входных признаков ═══════════════
        
        # Row embeddings (BoW из FastText) → hidden
        row_x = self.row_proj(data['row'].x)  # [N_rows, D_hidden]
        
        # Token embeddings (предобученные FastText) → hidden
        token_x = self.token_proj(data['token'].x)  # [N_tokens, D_hidden]
        
        # Edge features → projected
        edge_attr_r2t = self.edge_proj(
            data['row', 'has_token', 'token'].edge_attr
        )  # [E, D_edge]
        edge_attr_t2r = self.edge_proj(
            data['token', 'in_row', 'row'].edge_attr
        )  # [E, D_edge]
        
        # Edge indices
        edge_index_r2t = data['row', 'has_token', 'token'].edge_index
        edge_index_t2r = data['token', 'in_row', 'row'].edge_index
        
        # ═══════════════ 2. GNN Message Passing ═══════════════
        
        # Для JumpingKnowledge: собираем представления со всех слоёв
        row_representations = [row_x] if self.config.use_jumping_knowledge else []
        
        for gnn_layer in self.gnn_layers:
            row_x, token_x = gnn_layer(
                row_x=row_x,
                token_x=token_x,
                edge_index_t2r=edge_index_t2r,
                edge_index_r2t=edge_index_r2t,
                edge_attr_t2r=edge_attr_t2r,
                edge_attr_r2t=edge_attr_r2t,
            )
            
            if self.config.use_jumping_knowledge:
                row_representations.append(row_x)
        
        # ═══════════════ 3. Выходная голова ═══════════════
        
        if self.config.use_jumping_knowledge:
            # Конкатенируем представления со всех слоёв
            row_x = torch.cat(row_representations, dim=-1)  # [N_rows, D_hidden * (L+1)]
            row_x = self.jk_linear(row_x)  # [N_rows, D_hidden]
        
        # Финальная проекция
        output = self.output_head(row_x)  # [N_rows, D_output]
        
        # L2-нормализация для метрического обучения
        output = F.normalize(output, p=2, dim=-1)
        
        return output
    
    def get_row_embeddings(
        self,
        data: HeteroData,
        table_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Получить эмбеддинги строк, опционально отфильтрованные по таблице.
        
        Args:
            data: HeteroData граф
            table_id: 0 для таблицы A, 1 для таблицы B, None для всех
            
        Returns:
            [N, output_dim] эмбеддинги строк
        """
        all_embeddings = self.forward(data)
        
        if table_id is not None:
            mask = data['row'].table_id == table_id
            return all_embeddings[mask]
        
        return all_embeddings
