# V3: GAT + Mini-batch Training + HDBSCAN Evaluation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Переработать pipeline обучения и оценки ER: GAT v2 архитектура с edge features, честный split по строкам, mini-batch обучение через NeighborLoader на объединённом графе, HDBSCAN кластеризация для оценки.

**Architecture:** Объединённый граф из 21 in-domain датасета с глобальной IDF-фильтрацией. Стратифицированный split по строкам (train 70% / val 15% / test 15%) через группировку связных компонент. GATv2Conv с edge features для message passing. NeighborLoader для mini-batch обучения (граф в CPU RAM, батчи на GPU). HDBSCAN для кластеризации эмбеддингов при оценке. 3 cross-domain датасета (electronics, anime, citations) для transfer learning оценки.

**Tech Stack:** PyTorch, PyG (GATv2Conv, NeighborLoader, HeteroData), hdbscan, scikit-learn, numpy, pandas

---

## File Structure

### New Files
- `src/table_unifier/models/gat_layer.py` — GATv2 layer with edge features (replaces gnn_layer.py usage)
- `src/table_unifier/dataset/data_split.py` — стратифицированный split по строкам с группировкой компонент
- `src/table_unifier/evaluation/clustering.py` — HDBSCAN кластеризация и метрики
- `tests/test_gat_layer.py` — тесты GAT layer
- `tests/test_data_split.py` — тесты split логики
- `tests/test_clustering.py` — тесты HDBSCAN оценки
- `experiments/08_build_unified_graph.py` — построение объединённого графа с новым split
- `experiments/09_train_gat.py` — обучение GAT модели mini-batch
- `experiments/10_evaluate.py` — оценка: in-domain test, cross-domain, real data (HDBSCAN)

### Modified Files
- `src/table_unifier/models/entity_resolution.py` — новый `EntityResolutionGAT` (параллельно существующему `EntityResolutionGNN`)
- `src/table_unifier/config.py` — добавить `num_heads`, `attention_dropout` в `EntityResolutionConfig`
- `src/table_unifier/training/er_trainer.py` — добавить `train_entity_resolution_minibatch()`
- `src/table_unifier/dataset/graph_builder.py` — `build_unified_graph_from_datasets()` с глобальным IDF
- `pyproject.toml` — добавить `hdbscan` в зависимости

---

## Task 1: Добавить hdbscan в зависимости

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Добавить hdbscan**

В секции `[project] dependencies` добавить:

```
"hdbscan>=0.8.33",
```

- [ ] **Step 2: Установить**

Run: `uv sync`
Expected: hdbscan установлен без ошибок

- [ ] **Step 3: Проверить импорт**

Run: `uv run python -c "import hdbscan; print(hdbscan.__version__)"`
Expected: версия hdbscan

- [ ] **Step 4: Commit**

```bash
git add pyproject.toml uv.lock
git commit -m "deps: add hdbscan for clustering evaluation"
```

---

## Task 2: GATv2 Layer с edge features

**Files:**
- Create: `src/table_unifier/models/gat_layer.py`
- Create: `tests/test_gat_layer.py`

- [ ] **Step 1: Написать failing тесты**

```python
# tests/test_gat_layer.py
"""Тесты для models/gat_layer.py."""

import torch
import pytest
from table_unifier.models.gat_layer import GATLayer


class TestGATLayer:
    @pytest.fixture()
    def layer(self):
        return GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0)

    @pytest.fixture()
    def graph_tensors(self):
        n_rows = 4
        n_tokens = 6
        n_edges = 10
        hidden = 32
        edge_dim = 16

        row_x = torch.randn(n_rows, hidden)
        token_x = torch.randn(n_tokens, hidden)
        t2r_src = torch.randint(0, n_tokens, (n_edges,))
        t2r_dst = torch.randint(0, n_rows, (n_edges,))
        t2r_edge_index = torch.stack([t2r_src, t2r_dst])
        r2t_edge_index = torch.stack([t2r_dst, t2r_src])
        edge_attr_t2r = torch.randn(n_edges, edge_dim)
        edge_attr_r2t = torch.randn(n_edges, edge_dim)

        return row_x, token_x, t2r_edge_index, edge_attr_t2r, r2t_edge_index, edge_attr_r2t

    def test_output_shapes(self, layer, graph_tensors):
        row_x, token_x, *rest = graph_tensors
        row_out, token_out = layer(row_x, token_x, *rest)
        assert row_out.shape == row_x.shape
        assert token_out.shape == token_x.shape

    def test_gradient_flows(self, layer, graph_tensors):
        row_x, token_x, *rest = graph_tensors
        row_x = row_x.requires_grad_(True)
        token_x = token_x.requires_grad_(True)
        row_out, token_out = layer(row_x, token_x, *rest)
        loss = row_out.sum() + token_out.sum()
        loss.backward()
        assert row_x.grad is not None
        assert token_x.grad is not None

    def test_bidirectional(self):
        layer = GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0, bidirectional=True)
        assert hasattr(layer, "conv_r2t")

    def test_not_bidirectional(self):
        layer = GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.0, bidirectional=False)
        assert not hasattr(layer, "conv_r2t")

    def test_attention_weights_exist(self, layer, graph_tensors):
        """GATv2Conv возвращает attention weights при return_attention_weights=True."""
        row_x, token_x, t2r_ei, ea_t2r, *_ = graph_tensors
        # GATv2Conv поддерживает return_attention_weights
        out, (edge_index_out, attn_weights) = layer.conv_t2r(
            (token_x, row_x), t2r_ei, ea_t2r, return_attention_weights=True,
        )
        assert attn_weights.shape[0] == t2r_ei.shape[1]
        assert attn_weights.shape[1] == layer.conv_t2r.heads

    def test_deterministic_in_eval(self, graph_tensors):
        layer = GATLayer(hidden_dim=32, edge_dim=16, num_heads=4, dropout=0.1)
        layer.eval()
        row_x, token_x, *rest = graph_tensors
        with torch.no_grad():
            out1, _ = layer(row_x, token_x, *rest)
            out2, _ = layer(row_x, token_x, *rest)
        torch.testing.assert_close(out1, out2)
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_gat_layer.py -v`
Expected: ImportError — `gat_layer` не существует

- [ ] **Step 3: Реализовать GATLayer**

```python
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
```

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_gat_layer.py -v`
Expected: все тесты проходят

- [ ] **Step 5: Commit**

```bash
git add src/table_unifier/models/gat_layer.py tests/test_gat_layer.py
git commit -m "feat: add GATv2 layer with edge features"
```

---

## Task 3: EntityResolutionGAT модель

**Files:**
- Modify: `src/table_unifier/models/entity_resolution.py`
- Modify: `src/table_unifier/config.py`
- Create: `tests/test_entity_resolution_gat.py`

- [ ] **Step 1: Добавить поля в config**

В `EntityResolutionConfig` в `src/table_unifier/config.py` добавить после `bidirectional`:

```python
    num_heads: int = 4
    attention_dropout: float = 0.1
```

- [ ] **Step 2: Написать failing тесты**

```python
# tests/test_entity_resolution_gat.py
"""Тесты для EntityResolutionGAT."""

import torch
import pytest
from table_unifier.models.entity_resolution import EntityResolutionGAT


class TestEntityResolutionGAT:
    @pytest.fixture()
    def model(self):
        return EntityResolutionGAT(
            row_dim=32, token_dim=32, col_dim=64,
            hidden_dim=32, edge_dim=16, output_dim=16,
            num_gnn_layers=2, num_heads=4, dropout=0.0,
        )

    def test_output_shape(self, model, small_hetero_data):
        out = model(small_hetero_data)
        n_rows = small_hetero_data["row"].x.shape[0]
        assert out.shape == (n_rows, 16)

    def test_l2_normalized(self, model, small_hetero_data):
        model.eval()
        with torch.no_grad():
            out = model(small_hetero_data)
        norms = torch.norm(out, p=2, dim=-1)
        torch.testing.assert_close(norms, torch.ones(norms.shape[0]), atol=1e-5, rtol=1e-5)

    def test_gradient_flows(self, model, small_hetero_data):
        out = model(small_hetero_data)
        loss = out.sum()
        loss.backward()
        grads = [p.grad is not None for p in model.parameters() if p.requires_grad]
        assert any(grads)

    def test_num_gat_layers(self, model):
        assert len(model.gnn_layers) == 2

    def test_forward_signature_matches_gnn(self, model, small_hetero_data):
        """Тот же интерфейс что у EntityResolutionGNN — принимает HeteroData, возвращает Tensor."""
        out = model(small_hetero_data)
        assert isinstance(out, torch.Tensor)
        assert out.dim() == 2
```

- [ ] **Step 3: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_entity_resolution_gat.py -v`
Expected: ImportError — `EntityResolutionGAT` не существует

- [ ] **Step 4: Реализовать EntityResolutionGAT**

Добавить в `src/table_unifier/models/entity_resolution.py` **после** существующего `EntityResolutionGNN`:

```python
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
```

- [ ] **Step 5: Запустить тесты**

Run: `uv run pytest tests/test_entity_resolution_gat.py tests/test_entity_resolution_model.py -v`
Expected: все тесты проходят (новые GAT + старые GNN)

- [ ] **Step 6: Commit**

```bash
git add src/table_unifier/models/entity_resolution.py src/table_unifier/config.py tests/test_entity_resolution_gat.py
git commit -m "feat: add EntityResolutionGAT model with GATv2 + edge features"
```

---

## Task 4: Стратифицированный split по строкам

**Files:**
- Create: `src/table_unifier/dataset/data_split.py`
- Create: `tests/test_data_split.py`

- [ ] **Step 1: Написать failing тесты**

```python
# tests/test_data_split.py
"""Тесты для dataset/data_split.py."""

import pytest
import torch

from table_unifier.dataset.data_split import split_rows_stratified


class TestSplitRowsStratified:
    @pytest.fixture()
    def labeled_pairs(self):
        """Простые пары: (global_idx_a, global_idx_b, label)."""
        return torch.tensor([
            [0, 10, 1],  # positive
            [1, 11, 1],  # positive
            [2, 12, 1],  # positive
            [3, 13, 1],  # positive
            [4, 14, 1],  # positive
            [0, 11, 0],  # negative
            [1, 12, 0],  # negative
            [2, 10, 0],  # negative
            [3, 14, 0],  # negative
            [4, 13, 0],  # negative
        ], dtype=torch.long)

    def test_returns_three_splits(self, labeled_pairs):
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        assert len(train) > 0
        assert len(val) > 0
        assert len(test) > 0

    def test_no_row_overlap(self, labeled_pairs):
        """Строки из test не должны быть в train."""
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        train_rows = set(train[:, 0].tolist() + train[:, 1].tolist())
        val_rows = set(val[:, 0].tolist() + val[:, 1].tolist())
        test_rows = set(test[:, 0].tolist() + test[:, 1].tolist())
        assert train_rows.isdisjoint(test_rows), "Train and test share rows!"
        assert train_rows.isdisjoint(val_rows), "Train and val share rows!"
        assert val_rows.isdisjoint(test_rows), "Val and test share rows!"

    def test_all_pairs_assigned(self, labeled_pairs):
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        total = len(train) + len(val) + len(test)
        assert total == len(labeled_pairs)

    def test_positive_ratio_preserved(self, labeled_pairs):
        """Доля positives примерно одинакова во всех split."""
        train, val, test = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        overall_pos = (labeled_pairs[:, 2] == 1).float().mean().item()
        for split, name in [(train, "train"), (val, "val"), (test, "test")]:
            if len(split) < 2:
                continue
            pos_ratio = (split[:, 2] == 1).float().mean().item()
            assert abs(pos_ratio - overall_pos) < 0.3, (
                f"{name} positive ratio {pos_ratio:.2f} too far from overall {overall_pos:.2f}"
            )

    def test_deterministic(self, labeled_pairs):
        t1, v1, te1 = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        t2, v2, te2 = split_rows_stratified(labeled_pairs, ratios=(0.6, 0.2, 0.2), seed=42)
        assert torch.equal(t1, t2)
        assert torch.equal(v1, v2)
        assert torch.equal(te1, te2)
```

- [ ] **Step 2: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_data_split.py -v`
Expected: ImportError

- [ ] **Step 3: Реализовать split_rows_stratified**

```python
# src/table_unifier/dataset/data_split.py
"""Стратифицированный split по строкам для GNN.

Группирует строки в связные компоненты (через labeled pairs),
затем распределяет компоненты по train/val/test с сохранением
пропорций positives/negatives.

Гарантия: строки из test-пар не появляются в train-графе.
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _build_components(pairs: torch.Tensor) -> list[set[int]]:
    """Группировка строк в связные компоненты через Union-Find.

    Если строка A_3 участвует в паре с B_7, обе попадают в одну компоненту.
    Вся компонента пойдёт в один split.
    """
    parent: dict[int, int] = {}

    def find(x: int) -> int:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for row in pairs:
        a, b = row[0].item(), row[1].item()
        parent.setdefault(a, a)
        parent.setdefault(b, b)
        union(a, b)

    groups: dict[int, set[int]] = defaultdict(set)
    for node in parent:
        groups[find(node)].add(node)

    return list(groups.values())


def split_rows_stratified(
    labeled_pairs: torch.Tensor,
    ratios: tuple[float, float, float] = (0.7, 0.15, 0.15),
    seed: int = 42,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Разделить labeled pairs на train/val/test по строкам.

    Args:
        labeled_pairs: [N, 3] — (global_idx_a, global_idx_b, label)
        ratios: (train, val, test) доли
        seed: random seed

    Returns:
        (train_pairs, val_pairs, test_pairs) — каждый [M, 3]
        Гарантия: множества строк в split-ах не пересекаются.
    """
    rng = np.random.default_rng(seed)

    components = _build_components(labeled_pairs)

    # Для каждой компоненты считаем число positives
    comp_rows = []
    for comp in components:
        mask = torch.zeros(len(labeled_pairs), dtype=torch.bool)
        for i, row in enumerate(labeled_pairs):
            if row[0].item() in comp or row[1].item() in comp:
                mask[i] = True
        comp_pairs = labeled_pairs[mask]
        n_pos = (comp_pairs[:, 2] == 1).sum().item()
        comp_rows.append((comp, mask, n_pos, len(comp_pairs)))

    # Сортируем компоненты по размеру (большие первыми — лучше распределяются)
    comp_rows.sort(key=lambda x: -x[3])

    # Greedy: назначаем компоненту в split с наибольшим дефицитом
    target_pairs = [r * len(labeled_pairs) for r in ratios]
    current_pairs = [0.0, 0.0, 0.0]
    assignments: list[int] = []  # split index per component

    for _, _, _, n_pairs in comp_rows:
        # Выбираем split, у которого наибольший дефицит
        deficits = [target_pairs[i] - current_pairs[i] for i in range(3)]
        best = int(np.argmax(deficits))
        assignments.append(best)
        current_pairs[best] += n_pairs

    # Собираем пары по split
    split_masks = [torch.zeros(len(labeled_pairs), dtype=torch.bool) for _ in range(3)]
    for (_, mask, _, _), split_idx in zip(comp_rows, assignments):
        split_masks[split_idx] |= mask

    train_pairs = labeled_pairs[split_masks[0]]
    val_pairs = labeled_pairs[split_masks[1]]
    test_pairs = labeled_pairs[split_masks[2]]

    logger.info(
        "Split: train=%d (%.0f%%), val=%d (%.0f%%), test=%d (%.0f%%)",
        len(train_pairs), 100 * len(train_pairs) / len(labeled_pairs),
        len(val_pairs), 100 * len(val_pairs) / len(labeled_pairs),
        len(test_pairs), 100 * len(test_pairs) / len(labeled_pairs),
    )

    return train_pairs, val_pairs, test_pairs
```

- [ ] **Step 4: Запустить тесты**

Run: `uv run pytest tests/test_data_split.py -v`
Expected: все тесты проходят

- [ ] **Step 5: Commit**

```bash
git add src/table_unifier/dataset/data_split.py tests/test_data_split.py
git commit -m "feat: add stratified row-level split with connected components"
```

---

## Task 5: Построение объединённого графа с глобальным IDF

**Files:**
- Modify: `src/table_unifier/dataset/graph_builder.py`
- Create: `experiments/08_build_unified_graph.py`

- [ ] **Step 1: Добавить build_unified_graph_from_datasets() в graph_builder.py**

Добавить в конец `src/table_unifier/dataset/graph_builder.py`:

```python
def build_unified_graph_from_datasets(
    datasets: list[dict],
    token_embedder: TokenEmbedder,
    max_token_df: float = 0.3,
    max_tokens_per_cell: int = 16,
) -> tuple[HeteroData, dict[str, dict], torch.Tensor]:
    """Построить один граф из нескольких датасетов с глобальным IDF.

    Args:
        datasets: список словарей, каждый содержит:
            - "name": str
            - "table_a": pd.DataFrame
            - "table_b": pd.DataFrame
            - "columns_a": list[str]
            - "columns_b": list[str]
            - "column_embeddings": dict[str, np.ndarray]
            - "row_emb_a": np.ndarray
            - "row_emb_b": np.ndarray
            - "labeled_pairs": list[tuple[str, str, int]]  # (id_a, id_b, label)
        token_embedder: TokenEmbedder для get_token_ids и get_vocab_embedding
        max_token_df: порог IDF (глобальный — по всем строкам всех датасетов)
        max_tokens_per_cell: лимит токенов на ячейку

    Returns:
        (graph, dataset_mappings, all_labeled_pairs)
        - graph: HeteroData с единым пространством индексов
        - dataset_mappings: {name: {"id_to_global_a": {...}, "id_to_global_b": {...}}}
        - all_labeled_pairs: [N, 3] tensor (global_idx_a, global_idx_b, label)
    """
    # Объединяем все строки в единые таблицы
    all_rows: list[pd.Series] = []
    all_row_columns: list[list[str]] = []
    all_row_embeddings: list[np.ndarray] = []
    dataset_mappings: dict[str, dict] = {}
    all_labeled: list[list[int]] = []

    global_idx = 0

    for ds in datasets:
        name = ds["name"]
        table_a, table_b = ds["table_a"], ds["table_b"]
        cols_a, cols_b = ds["columns_a"], ds["columns_b"]

        id_to_global_a: dict[str, int] = {}
        id_to_global_b: dict[str, int] = {}

        for _, row in table_a.iterrows():
            id_to_global_a[str(row["id"])] = global_idx
            all_rows.append(row)
            all_row_columns.append(cols_a)
            global_idx += 1

        for _, row in table_b.iterrows():
            id_to_global_b[str(row["id"])] = global_idx
            all_rows.append(row)
            all_row_columns.append(cols_b)
            global_idx += 1

        all_row_embeddings.append(ds["row_emb_a"])
        all_row_embeddings.append(ds["row_emb_b"])

        # labeled pairs → global indices
        for a_id, b_id, label in ds["labeled_pairs"]:
            ga = id_to_global_a.get(str(a_id))
            gb = id_to_global_b.get(str(b_id))
            if ga is not None and gb is not None:
                all_labeled.append([ga, gb, label])

        dataset_mappings[name] = {
            "id_to_global_a": id_to_global_a,
            "id_to_global_b": id_to_global_b,
        }

    n_rows = len(all_rows)
    row_embeddings = np.concatenate(all_row_embeddings, axis=0)

    # Собираем все column_embeddings в единый dict
    merged_col_emb: dict[str, np.ndarray] = {}
    for ds in datasets:
        merged_col_emb.update(ds["column_embeddings"])

    logger.info("Unified: %d строк, %d датасетов", n_rows, len(datasets))

    # Дальше — тот же алгоритм что в build_graph(), но с глобальным IDF
    # (код аналогичен build_graph начиная с шага 3, но all_rows уже объединены)
    token_vocab: dict[int, int] = {}
    token_ids_list: list[int] = []
    raw_edges: list[tuple[int, int, int]] = []
    token_row_sets: dict[int, set] = defaultdict(set)
    col_emb_list: list[np.ndarray] = []
    col_to_idx: dict[str, int] = {}

    for row_idx, (row, cols) in enumerate(zip(all_rows, all_row_columns)):
        for col in cols:
            val = row.get(col, "")
            if pd.isna(val) or not str(val).strip():
                continue
            cell_text = str(val)
            tids = token_embedder.get_token_ids(cell_text)

            col_emb = merged_col_emb.get(col)
            if col_emb is None:
                continue

            if col not in col_to_idx:
                col_to_idx[col] = len(col_emb_list)
                col_emb_list.append(col_emb)
            cidx = col_to_idx[col]

            seen_in_cell: set[int] = set()
            for tid in tids:
                if tid in seen_in_cell:
                    continue
                seen_in_cell.add(tid)

                if tid not in token_vocab:
                    token_vocab[tid] = len(token_ids_list)
                    token_ids_list.append(tid)
                token_node = token_vocab[tid]

                raw_edges.append((token_node, row_idx, cidx))
                token_row_sets[token_node].add(row_idx)

    # Глобальная IDF-фильтрация
    idf_threshold = int(max_token_df * n_rows)
    stopword_nodes = {
        node for node, rows in token_row_sets.items()
        if len(rows) > idf_threshold
    }
    logger.info("Глобальная IDF: удалено %d/%d токенов (df > %.0f%% из %d строк)",
                len(stopword_nodes), len(token_ids_list), max_token_df * 100, n_rows)

    # Финальные рёбра с лимитом per-cell
    cell_tokens: dict[tuple[int, int], list[int]] = defaultdict(list)
    for token_node, row_idx, cidx in raw_edges:
        if token_node not in stopword_nodes:
            cell_tokens[(row_idx, cidx)].append(token_node)

    t2r_src: list[int] = []
    t2r_dst: list[int] = []
    t2r_col_idx: list[int] = []

    for (row_idx, cidx), nodes in cell_tokens.items():
        if len(nodes) > max_tokens_per_cell:
            nodes = random.sample(nodes, max_tokens_per_cell)
        for token_node in nodes:
            t2r_src.append(token_node)
            t2r_dst.append(row_idx)
            t2r_col_idx.append(cidx)

    # Ремаппинг токенов
    used_nodes = set(t2r_src)
    old_to_new: dict[int, int] = {}
    new_token_ids_list: list[int] = []
    for old_node, tid in enumerate(token_ids_list):
        if old_node in used_nodes:
            old_to_new[old_node] = len(new_token_ids_list)
            new_token_ids_list.append(tid)

    t2r_src = [old_to_new[n] for n in t2r_src]
    token_ids_list = new_token_ids_list

    # Token embeddings
    token_embeddings = np.stack([
        token_embedder.get_vocab_embedding(tid) for tid in token_ids_list
    ])

    # Собрать HeteroData
    data = HeteroData()
    data["row"].x = torch.tensor(row_embeddings, dtype=torch.float32)
    data["token"].x = torch.tensor(token_embeddings, dtype=torch.float32)

    col_emb_matrix = torch.tensor(np.stack(col_emb_list), dtype=torch.float32)
    data.col_embeddings = col_emb_matrix

    edge_col_idx = torch.tensor(t2r_col_idx, dtype=torch.long)
    t2r_edge_index = torch.tensor([t2r_src, t2r_dst], dtype=torch.long)
    data["token", "in_row", "row"].edge_index = t2r_edge_index
    data["token", "in_row", "row"].edge_col_idx = edge_col_idx
    data["row", "has_token", "token"].edge_index = torch.tensor([t2r_dst, t2r_src], dtype=torch.long)
    data["row", "has_token", "token"].edge_col_idx = edge_col_idx

    labeled_tensor = torch.tensor(all_labeled, dtype=torch.long) if all_labeled else torch.zeros((0, 3), dtype=torch.long)

    logger.info("Unified graph: rows=%d, tokens=%d, edges=%d, labeled_pairs=%d",
                data["row"].x.shape[0], data["token"].x.shape[0],
                t2r_edge_index.shape[1], len(labeled_tensor))

    return data, dataset_mappings, labeled_tensor
```

- [ ] **Step 2: Создать эксперимент 08**

```python
# experiments/08_build_unified_graph.py
"""Эксперимент 8 — Построение объединённого графа с глобальным IDF и split по строкам.

Объединяет 21 in-domain датасет в один граф, применяет глобальную IDF-фильтрацию,
и создаёт стратифицированный split по строкам (train 70% / val 15% / test 15%).

Cross-domain датасеты (electronics, anime, citations) строятся отдельно.

Сохраняет:
  - data/graphs/v3_unified/graph.pt — объединённый граф
  - data/graphs/v3_unified/train_pairs.pt — train пары [N, 3]
  - data/graphs/v3_unified/val_pairs.pt
  - data/graphs/v3_unified/test_pairs.pt
  - data/graphs/v3_unified/dataset_mappings.json
  - data/graphs/v3_unified/stats.json
  - data/graphs/v3_cross/{name}/graph.pt — cross-domain графы (отдельные)

Использование:
    python -m experiments.08_build_unified_graph
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch

from table_unifier.config import Config
from table_unifier.dataset.data_split import split_rows_stratified
from table_unifier.dataset.download import DATASETS
from table_unifier.dataset.embedding_generation import TokenEmbedder
from table_unifier.dataset.graph_builder import build_unified_graph_from_datasets, build_graph
from table_unifier.dataset.pair_sampling import split_labeled_pairs

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

CROSS_DOMAIN = {"electronics", "anime", "citations"}


def load_dataset_for_unified(name: str, synth_dir: Path, emb_dir: Path) -> dict | None:
    """Загрузить датасет со всеми labeled pairs."""
    ds_synth = synth_dir / name
    ds_emb = emb_dir / name

    required = [
        ds_synth / "tableA_synth.csv", ds_synth / "tableB_synth.csv",
        ds_emb / "column_embeddings_a.npz", ds_emb / "column_embeddings_b.npz",
        ds_emb / "row_embeddings_a.npy", ds_emb / "row_embeddings_b.npy",
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        logger.warning("[%s] Отсутствуют файлы: %s", name, [p.name for p in missing])
        return None

    import pandas as pd
    table_a = pd.read_csv(ds_synth / "tableA_synth.csv")
    table_b = pd.read_csv(ds_synth / "tableB_synth.csv")

    col_emb_a = dict(np.load(ds_emb / "column_embeddings_a.npz"))
    col_emb_b = dict(np.load(ds_emb / "column_embeddings_b.npz"))
    row_emb_a = np.load(ds_emb / "row_embeddings_a.npy")
    row_emb_b = np.load(ds_emb / "row_embeddings_b.npy")

    columns_a = [c for c in table_a.columns if c != "id"]
    columns_b = [c for c in table_b.columns if c != "id"]

    # Собираем ВСЕ labeled pairs (train + valid + test вместе — split будет новый)
    labeled_pairs = []
    for split_name in ("train", "valid", "test"):
        p = ds_synth / f"{split_name}.csv"
        if p.exists():
            df = pd.read_csv(p)
            pos, neg = split_labeled_pairs(df)
            for a_id, b_id in pos:
                labeled_pairs.append((a_id, b_id, 1))
            for a_id, b_id in neg:
                labeled_pairs.append((a_id, b_id, 0))

    return {
        "name": name,
        "table_a": table_a,
        "table_b": table_b,
        "columns_a": columns_a,
        "columns_b": columns_b,
        "column_embeddings": {**col_emb_a, **col_emb_b},
        "row_emb_a": row_emb_a,
        "row_emb_b": row_emb_b,
        "labeled_pairs": labeled_pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Построение unified графа v3")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    synth_dir = config.data_dir / "synthetic"
    emb_dir = config.data_dir / "embeddings"

    token_embedder = TokenEmbedder(model_name=config.entity_resolution.token_model_name, device=device)
    del token_embedder.model
    torch.cuda.empty_cache()
    gc.collect()

    # 1. Загрузка in-domain датасетов
    in_domain_datasets = []
    for name in sorted(DATASETS.keys()):
        if name in CROSS_DOMAIN:
            continue
        ds = load_dataset_for_unified(name, synth_dir, emb_dir)
        if ds is not None and ds["labeled_pairs"]:
            in_domain_datasets.append(ds)

    logger.info("In-domain датасетов: %d", len(in_domain_datasets))

    # 2. Построение unified графа
    graph, dataset_mappings, all_labeled = build_unified_graph_from_datasets(
        in_domain_datasets, token_embedder,
    )

    # 3. Split по строкам
    train_pairs, val_pairs, test_pairs = split_rows_stratified(
        all_labeled, ratios=(0.7, 0.15, 0.15), seed=42,
    )

    # 4. Сохранение
    out_dir = config.data_dir / "graphs" / "v3_unified"
    out_dir.mkdir(parents=True, exist_ok=True)

    torch.save(graph, out_dir / "graph.pt")
    torch.save(train_pairs, out_dir / "train_pairs.pt")
    torch.save(val_pairs, out_dir / "val_pairs.pt")
    torch.save(test_pairs, out_dir / "test_pairs.pt")

    # Сериализуемые маппинги
    serializable_mappings = {
        name: {k: {str(kk): vv for kk, vv in v.items()} for k, v in maps.items()}
        for name, maps in dataset_mappings.items()
    }
    with open(out_dir / "dataset_mappings.json", "w") as f:
        json.dump(serializable_mappings, f)

    stats = {
        "n_rows": int(graph["row"].x.shape[0]),
        "n_tokens": int(graph["token"].x.shape[0]),
        "n_edges": int(graph["token", "in_row", "row"].edge_index.shape[1]),
        "n_labeled": int(len(all_labeled)),
        "n_train": int(len(train_pairs)),
        "n_val": int(len(val_pairs)),
        "n_test": int(len(test_pairs)),
        "n_datasets": len(in_domain_datasets),
        "datasets": [ds["name"] for ds in in_domain_datasets],
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Сохранено в %s", out_dir)
    logger.info("Stats: %s", stats)

    # 5. Cross-domain графы (отдельные)
    cross_dir = config.data_dir / "graphs" / "v3_cross"
    for name in sorted(CROSS_DOMAIN):
        ds = load_dataset_for_unified(name, synth_dir, emb_dir)
        if ds is None:
            continue

        logger.info("Cross-domain: %s", name)
        cg, id_a, id_b = build_graph(
            ds["table_a"], ds["table_b"], ds["column_embeddings"], token_embedder,
            columns_a=ds["columns_a"], columns_b=ds["columns_b"],
            precomputed_row_embeddings_a=ds["row_emb_a"],
            precomputed_row_embeddings_b=ds["row_emb_b"],
        )

        cd_out = cross_dir / name
        cd_out.mkdir(parents=True, exist_ok=True)
        torch.save(cg, cd_out / "graph.pt")
        with open(cd_out / "id_to_global_a.json", "w") as f:
            json.dump(id_a, f)
        with open(cd_out / "id_to_global_b.json", "w") as f:
            json.dump(id_b, f)

        # Labeled pairs для cross-domain оценки
        if ds["labeled_pairs"]:
            pairs = []
            for a_id, b_id, label in ds["labeled_pairs"]:
                ga = id_a.get(str(a_id))
                gb = id_b.get(str(b_id))
                if ga is not None and gb is not None:
                    pairs.append([ga, gb, label])
            if pairs:
                torch.save(torch.tensor(pairs, dtype=torch.long), cd_out / "labeled_pairs.pt")

    del token_embedder
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Готово!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Запустить**

Run: `uv run python -m experiments.08_build_unified_graph`
Expected: unified граф и cross-domain графы сохранены в `data/graphs/v3_unified/` и `data/graphs/v3_cross/`

- [ ] **Step 4: Commit**

```bash
git add src/table_unifier/dataset/graph_builder.py experiments/08_build_unified_graph.py
git commit -m "feat: build unified graph with global IDF and row-level split"
```

---

## Task 6: Mini-batch training с NeighborLoader

**Files:**
- Modify: `src/table_unifier/training/er_trainer.py`
- Create: `experiments/09_train_gat.py`

- [ ] **Step 1: Добавить train_entity_resolution_minibatch() в er_trainer.py**

Добавить в конец файла `src/table_unifier/training/er_trainer.py` (перед утилитами):

```python
# ------------------------------------------------------------------ #
#  Mini-batch обучение на большом графе (NeighborLoader)
# ------------------------------------------------------------------ #

def train_entity_resolution_minibatch(
    graph: HeteroData,
    train_pairs: torch.Tensor,
    val_pairs: torch.Tensor | None = None,
    config: EntityResolutionConfig | None = None,
    device: str | None = None,
    save_path: Path | None = None,
    epoch_callback: "Callable[[int, float | None], None] | None" = None,
    model_class: str = "gat",
) -> tuple[nn.Module, dict]:
    """Mini-batch обучение ER на объединённом графе.

    Граф хранится в CPU RAM. На каждом шаге NeighborLoader
    сэмплирует подграф для батча seed-узлов (строки из триплетов),
    подграф загружается на GPU для forward + backward.

    Args:
        graph:        HeteroData — объединённый граф (в CPU)
        train_pairs:  [N, 3] — (idx_a, idx_b, label), label=1 positive
        val_pairs:    [M, 3] — аналогично, для валидации
        config:       EntityResolutionConfig
        device:       cuda / cpu
        save_path:    путь сохранения лучшей модели
        epoch_callback: для early stopping / pruning
        model_class:  "gat" или "gnn"

    Returns:
        (model, history)
    """
    from torch_geometric.loader import NeighborLoader

    config = config or EntityResolutionConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Размерности из графа
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    # Модель
    if model_class == "gat":
        from table_unifier.models.entity_resolution import EntityResolutionGAT
        model = EntityResolutionGAT(
            row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
            hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
            num_gnn_layers=config.num_gnn_layers, num_heads=config.num_heads,
            dropout=config.dropout, attention_dropout=config.attention_dropout,
            bidirectional=config.bidirectional,
        ).to(device)
    else:
        model = _build_model(config, device)

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    # Формируем триплеты из labeled pairs
    # Positives: (a, b) с label=1; для каждого positive берём random negative
    pos_mask = train_pairs[:, 2] == 1
    neg_mask = train_pairs[:, 2] == 0
    pos_pairs = train_pairs[pos_mask]  # [P, 3]
    neg_pairs = train_pairs[neg_mask]  # [N_neg, 3]

    # Все row indices из train_pairs — seed nodes для NeighborLoader
    all_train_rows = torch.unique(torch.cat([train_pairs[:, 0], train_pairs[:, 1]]))

    # NeighborLoader: сэмплирует подграф для заданных seed nodes
    # num_neighbors: сколько соседей сэмплировать на каждом слое
    num_neighbors = {
        ("token", "in_row", "row"): [32] * config.num_gnn_layers,
        ("row", "has_token", "token"): [32] * config.num_gnn_layers,
    }

    best_val_loss = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        model.train()

        # Генерируем триплеты для этой эпохи
        rng = torch.Generator().manual_seed(epoch)
        if len(neg_pairs) > 0:
            neg_idx = torch.randint(0, len(neg_pairs), (len(pos_pairs),), generator=rng)
            triplet_a = pos_pairs[:, 0]     # anchor (from table A)
            triplet_p = pos_pairs[:, 1]     # positive (from table B)
            triplet_n = neg_pairs[neg_idx, 1]  # negative (from table B of a neg pair)
        else:
            # Fallback: random negatives
            all_b = torch.unique(train_pairs[:, 1])
            triplet_a = pos_pairs[:, 0]
            triplet_p = pos_pairs[:, 1]
            triplet_n = all_b[torch.randint(0, len(all_b), (len(pos_pairs),), generator=rng)]

        # Seed nodes = все строки из триплетов
        seed_rows = torch.unique(torch.cat([triplet_a, triplet_p, triplet_n]))

        # NeighborLoader для подграфа
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            input_nodes=("row", seed_rows),
            batch_size=min(config.batch_size, len(seed_rows)),
            shuffle=False,
        )

        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            embeddings = model(batch)

            # Маппинг: global row idx → local idx в батче
            # batch["row"].n_id содержит оригинальные индексы
            global_to_local = {gid.item(): local for local, gid in enumerate(batch["row"].n_id)}

            # Фильтруем триплеты где все 3 строки в батче
            batch_triplets = []
            for a, p, n in zip(triplet_a.tolist(), triplet_p.tolist(), triplet_n.tolist()):
                if a in global_to_local and p in global_to_local and n in global_to_local:
                    batch_triplets.append([global_to_local[a], global_to_local[p], global_to_local[n]])

            if not batch_triplets:
                batch.to("cpu")
                continue

            bt = torch.tensor(batch_triplets, device=device)
            a_emb = embeddings[bt[:, 0]]
            p_emb = embeddings[bt[:, 1]]
            n_emb = embeddings[bt[:, 2]]

            # Semi-hard mining
            a_mined, p_mined, n_mined = mine_semi_hard(
                embeddings, bt, margin=config.margin,
            )
            if a_mined.shape[0] == 0:
                a_mined, p_mined, n_mined = a_emb, p_emb, n_emb

            loss = criterion(a_mined, p_mined, n_mined)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            batch.to("cpu")
            torch.cuda.empty_cache()

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # Validation
        val_loss = None
        if val_pairs is not None and len(val_pairs) > 0:
            model.eval()
            val_pos = val_pairs[val_pairs[:, 2] == 1]
            val_neg = val_pairs[val_pairs[:, 2] == 0]

            if len(val_pos) > 0 and len(val_neg) > 0:
                val_seed = torch.unique(torch.cat([val_pairs[:, 0], val_pairs[:, 1]]))
                val_loader = NeighborLoader(
                    graph, num_neighbors=num_neighbors,
                    input_nodes=("row", val_seed),
                    batch_size=min(512, len(val_seed)),
                    shuffle=False,
                )

                val_loss_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        val_emb = model(batch)
                        global_to_local = {gid.item(): local for local, gid in enumerate(batch["row"].n_id)}

                        # Score all val_pairs in this batch
                        for row in val_pairs:
                            a, b, label = row[0].item(), row[1].item(), row[2].item()
                            if a in global_to_local and b in global_to_local:
                                ea = val_emb[global_to_local[a]]
                                eb = val_emb[global_to_local[b]]
                                sim = (ea * eb).sum().item()
                                # Для triplet-style val: positive пары должны быть ближе
                                # Используем 1-sim как proxy для loss
                                if label == 1:
                                    val_loss_sum += max(0, 1 - sim)
                                val_count += 1

                        batch.to("cpu")
                        torch.cuda.empty_cache()

                val_loss = val_loss_sum / max(val_count, 1) if val_count > 0 else None

        if val_loss is not None:
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        if epoch % 5 == 0 or epoch == 1:
            logger.info("ER [minibatch] Epoch %d/%d — train_loss=%.4f%s, lr=%.2e",
                        epoch, config.epochs, train_loss, val_info, current_lr)

        if epoch_callback is not None:
            try:
                epoch_callback(epoch, val_loss)
            except StopIteration:
                logger.info("Обучение остановлено callback на эпохе %d", epoch)
                break

    # Загрузить лучший checkpoint
    if save_path and save_path.exists() and history["val_loss"]:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info("Загружен лучший checkpoint (val_loss=%.4f)", best_val_loss)

    # Сохранить историю
    if save_path:
        import json as _json
        history_path = save_path.with_suffix(".history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            _json.dump(history, f)

    return model, history
```

- [ ] **Step 2: Создать эксперимент 09**

```python
# experiments/09_train_gat.py
"""Эксперимент 9 — Обучение GAT модели mini-batch на объединённом графе.

Использование:
    python -m experiments.09_train_gat
    python -m experiments.09_train_gat --max-epochs 500 --patience 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.training.er_trainer import train_entity_resolution_minibatch

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


class EarlyStopping:
    def __init__(self, patience: int = 30):
        self.patience = patience
        self.best_val = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def __call__(self, epoch: int, val_loss: float | None) -> None:
        if val_loss is None:
            return
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_epoch = epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
            logger.info("Early stopping: best=%.4f @ ep %d", self.best_val, self.best_epoch)
            raise StopIteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение GAT mini-batch")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    unified_dir = config.data_dir / "graphs" / "v3_unified"
    save_path = config.output_dir / "v3_gat_model.pt"

    # Загрузка
    logger.info("Загрузка unified графа...")
    graph = torch.load(unified_dir / "graph.pt", weights_only=False)
    train_pairs = torch.load(unified_dir / "train_pairs.pt", weights_only=False)
    val_pairs = torch.load(unified_dir / "val_pairs.pt", weights_only=False)

    logger.info("Graph: %d rows, %d tokens, %d edges",
                graph["row"].x.shape[0], graph["token"].x.shape[0],
                graph["token", "in_row", "row"].edge_index.shape[1])
    logger.info("Train: %d pairs, Val: %d pairs", len(train_pairs), len(val_pairs))

    # Загрузка лучших гиперпараметров из HPO (если есть)
    hpo_arch = config.output_dir / "hpo_architecture.json"
    hpo_train = config.output_dir / "hpo_training.json"

    if hpo_arch.exists() and hpo_train.exists():
        with open(hpo_arch) as f:
            best_arch = json.load(f)["best_params"]
        with open(hpo_train) as f:
            best_training = json.load(f)["best_training"]
        er_config = EntityResolutionConfig(
            hidden_dim=best_arch["hidden_dim"],
            edge_dim=best_arch["edge_dim"],
            num_gnn_layers=best_arch["num_gnn_layers"],
            dropout=best_arch["dropout"],
            bidirectional=best_arch["bidirectional"],
            lr=best_training["lr"],
            margin=best_training["margin"],
            weight_decay=best_training["weight_decay"],
            num_heads=4,
            attention_dropout=0.1,
        )
        logger.info("Конфигурация из HPO + GAT defaults")
    else:
        er_config = EntityResolutionConfig(
            hidden_dim=128, edge_dim=128, num_gnn_layers=3,
            dropout=0.0, bidirectional=True,
            lr=7.5e-4, margin=0.1, weight_decay=0.0,
            num_heads=4, attention_dropout=0.1,
        )
        logger.info("Конфигурация по умолчанию")

    er_config.epochs = args.max_epochs
    er_config.batch_size = args.batch_size

    early_stop = EarlyStopping(patience=args.patience)

    model, history = train_entity_resolution_minibatch(
        graph=graph,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        config=er_config,
        device=device,
        save_path=save_path,
        epoch_callback=early_stop,
        model_class="gat",
    )

    logger.info("Обучение завершено. Модель: %s", save_path)


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Commit**

```bash
git add src/table_unifier/training/er_trainer.py experiments/09_train_gat.py
git commit -m "feat: add mini-batch GAT training with NeighborLoader"
```

---

## Task 7: HDBSCAN кластеризация и оценка

**Files:**
- Create: `src/table_unifier/evaluation/clustering.py`
- Create: `tests/test_clustering.py`

- [ ] **Step 1: Создать __init__.py для evaluation модуля**

```bash
mkdir -p src/table_unifier/evaluation
touch src/table_unifier/evaluation/__init__.py
```

- [ ] **Step 2: Написать failing тесты**

```python
# tests/test_clustering.py
"""Тесты для evaluation/clustering.py."""

import numpy as np
import pytest
import torch

from table_unifier.evaluation.clustering import cluster_embeddings, evaluate_clusters


class TestClusterEmbeddings:
    def test_returns_labels(self):
        emb = torch.randn(50, 16)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        labels = cluster_embeddings(emb, min_cluster_size=5)
        assert labels.shape == (50,)
        assert labels.dtype == np.int64 or labels.dtype == np.intp

    def test_noise_label_is_minus_one(self):
        """HDBSCAN помечает шум как -1."""
        emb = torch.randn(100, 16)
        emb = torch.nn.functional.normalize(emb, p=2, dim=-1)
        labels = cluster_embeddings(emb, min_cluster_size=5)
        assert -1 in labels or len(np.unique(labels)) > 1

    def test_similar_points_same_cluster(self):
        """Два кластера явно разделённых точек."""
        cluster_a = torch.randn(20, 8) + torch.tensor([5.0, 0, 0, 0, 0, 0, 0, 0])
        cluster_b = torch.randn(20, 8) + torch.tensor([-5.0, 0, 0, 0, 0, 0, 0, 0])
        emb = torch.cat([cluster_a, cluster_b])
        labels = cluster_embeddings(emb, min_cluster_size=5)
        # Точки 0-19 должны быть в одном кластере, 20-39 — в другом
        assert labels[0] == labels[10]
        assert labels[20] == labels[30]
        assert labels[0] != labels[20]


class TestEvaluateClusters:
    def test_with_ground_truth(self):
        labels = np.array([0, 0, 1, 1, -1])
        ground_truth = np.array([0, 0, 1, 1, 0])
        metrics = evaluate_clusters(labels, ground_truth)
        assert "ari" in metrics
        assert "nmi" in metrics
        assert "coverage" in metrics

    def test_coverage_excludes_noise(self):
        labels = np.array([0, 0, -1, -1, -1])
        metrics = evaluate_clusters(labels)
        assert metrics["coverage"] == pytest.approx(0.4)

    def test_no_ground_truth(self):
        labels = np.array([0, 0, 1, 1, -1])
        metrics = evaluate_clusters(labels)
        assert "coverage" in metrics
        assert "n_clusters" in metrics
        assert "ari" not in metrics
```

- [ ] **Step 3: Запустить тесты, убедиться что падают**

Run: `uv run pytest tests/test_clustering.py -v`
Expected: ImportError

- [ ] **Step 4: Реализовать clustering.py**

```python
# src/table_unifier/evaluation/clustering.py
"""HDBSCAN кластеризация эмбеддингов и метрики оценки."""

from __future__ import annotations

import logging

import hdbscan
import numpy as np
import torch
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

logger = logging.getLogger(__name__)


def cluster_embeddings(
    embeddings: torch.Tensor,
    min_cluster_size: int = 10,
    min_samples: int | None = None,
    metric: str = "euclidean",
) -> np.ndarray:
    """Кластеризация эмбеддингов через HDBSCAN.

    Args:
        embeddings: [N, D] tensor (L2-normalized рекомендуется)
        min_cluster_size: минимальный размер кластера
        min_samples: минимальное число точек в окрестности (default=min_cluster_size)
        metric: метрика расстояния

    Returns:
        labels: [N] numpy array, -1 = шум
    """
    X = embeddings.detach().cpu().numpy().astype(np.float64)

    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric=metric,
        core_dist_n_jobs=-1,
    )
    labels = clusterer.fit_predict(X)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = (labels == -1).sum()
    logger.info("HDBSCAN: %d кластеров, %d шум (%.1f%%)",
                n_clusters, n_noise, 100 * n_noise / len(labels))

    return labels


def evaluate_clusters(
    labels: np.ndarray,
    ground_truth: np.ndarray | None = None,
) -> dict:
    """Оценить качество кластеризации.

    Args:
        labels: [N] — метки кластеров (-1 = шум)
        ground_truth: [N] — истинные метки (опционально)

    Returns:
        dict с метриками:
        - coverage: доля точек не в шуме
        - n_clusters: число кластеров
        - noise_ratio: доля шума
        - ari: Adjusted Rand Index (если есть ground_truth)
        - nmi: Normalized Mutual Info (если есть ground_truth)
    """
    n_total = len(labels)
    n_noise = int((labels == -1).sum())
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)

    metrics: dict = {
        "coverage": (n_total - n_noise) / n_total if n_total > 0 else 0.0,
        "n_clusters": n_clusters,
        "noise_ratio": n_noise / n_total if n_total > 0 else 0.0,
        "n_total": n_total,
        "n_noise": n_noise,
    }

    if ground_truth is not None:
        # Оцениваем только по не-шумовым точкам
        non_noise = labels != -1
        if non_noise.sum() > 0:
            metrics["ari"] = float(adjusted_rand_score(ground_truth[non_noise], labels[non_noise]))
            metrics["nmi"] = float(normalized_mutual_info_score(ground_truth[non_noise], labels[non_noise]))

    return metrics
```

- [ ] **Step 5: Запустить тесты**

Run: `uv run pytest tests/test_clustering.py -v`
Expected: все тесты проходят

- [ ] **Step 6: Commit**

```bash
git add src/table_unifier/evaluation/__init__.py src/table_unifier/evaluation/clustering.py tests/test_clustering.py
git commit -m "feat: add HDBSCAN clustering evaluation module"
```

---

## Task 8: Эксперимент 10 — Unified evaluation pipeline

**Files:**
- Create: `experiments/10_evaluate.py`

- [ ] **Step 1: Создать эксперимент оценки**

```python
# experiments/10_evaluate.py
"""Эксперимент 10 — Оценка GAT модели.

Три уровня оценки:
  1. In-domain test set — ROC-AUC, AP на held-out строках
  2. Cross-domain — ROC-AUC, AP + HDBSCAN ARI/NMI на unseen датасетах
  3. Real data — HDBSCAN кластеризация (unsupervised)

Использование:
    python -m experiments.10_evaluate
    python -m experiments.10_evaluate --model-path output/v3_gat_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.evaluation.clustering import cluster_embeddings, evaluate_clusters
from table_unifier.models.entity_resolution import EntityResolutionGAT
from table_unifier.training.er_trainer import get_row_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_pairs(embeddings: torch.Tensor, pairs: torch.Tensor) -> dict:
    """ROC-AUC и AP из labeled pairs."""
    if len(pairs) == 0:
        return {}
    idx_a = pairs[:, 0]
    idx_b = pairs[:, 1]
    labels = pairs[:, 2].numpy()
    scores = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1).numpy()
    if len(np.unique(labels)) < 2:
        return {}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "avg_precision": float(average_precision_score(labels, scores)),
    }


def load_gat_model(graph, config: EntityResolutionConfig, model_path: Path, device: str):
    """Загрузить обученную GAT модель."""
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    model = EntityResolutionGAT(
        row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
        hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
        num_gnn_layers=config.num_gnn_layers, num_heads=config.num_heads,
        dropout=config.dropout, attention_dropout=config.attention_dropout,
        bidirectional=config.bidirectional,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка GAT модели")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path or (config.output_dir / "v3_gat_model.pt")

    er_config = EntityResolutionConfig(
        hidden_dim=128, edge_dim=128, num_gnn_layers=3,
        dropout=0.0, bidirectional=True,
        num_heads=4, attention_dropout=0.1,
    )

    results = {}

    # ===== 1. In-domain test =====
    logger.info("=" * 60)
    logger.info("1. In-domain test evaluation")
    logger.info("=" * 60)

    unified_dir = config.data_dir / "graphs" / "v3_unified"
    graph = torch.load(unified_dir / "graph.pt", weights_only=False)
    test_pairs = torch.load(unified_dir / "test_pairs.pt", weights_only=False)

    model = load_gat_model(graph, er_config, model_path, device)

    # Для большого графа: получаем эмбеддинги батчами
    # (get_row_embeddings загружает весь граф — может не влезть)
    # Для оценки можно загрузить на CPU
    embeddings = get_row_embeddings(model, graph, device="cpu")

    in_domain_metrics = evaluate_pairs(embeddings, test_pairs)
    logger.info("In-domain: %s", in_domain_metrics)
    results["in_domain"] = in_domain_metrics

    # HDBSCAN на test строках
    test_rows = torch.unique(torch.cat([test_pairs[:, 0], test_pairs[:, 1]]))
    test_emb = embeddings[test_rows]
    test_labels = cluster_embeddings(test_emb, min_cluster_size=args.min_cluster_size)

    # Ground truth для кластеризации: строки из одной positive пары → один кластер
    gt = np.full(len(test_rows), -1)
    row_to_local = {r.item(): i for i, r in enumerate(test_rows)}
    cluster_id = 0
    for pair in test_pairs:
        if pair[2].item() == 1:
            a_local = row_to_local.get(pair[0].item())
            b_local = row_to_local.get(pair[1].item())
            if a_local is not None and b_local is not None:
                gt[a_local] = cluster_id
                gt[b_local] = cluster_id
                cluster_id += 1

    cluster_metrics = evaluate_clusters(test_labels, gt)
    results["in_domain_clustering"] = cluster_metrics
    logger.info("In-domain clustering: %s", cluster_metrics)

    del graph, embeddings
    torch.cuda.empty_cache()

    # ===== 2. Cross-domain =====
    logger.info("=" * 60)
    logger.info("2. Cross-domain evaluation")
    logger.info("=" * 60)

    cross_dir = config.data_dir / "graphs" / "v3_cross"
    cross_results = []

    for ds_dir in sorted(cross_dir.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "graph.pt").exists():
            continue

        name = ds_dir.name
        logger.info("Cross-domain: %s", name)

        cg = torch.load(ds_dir / "graph.pt", weights_only=False)
        model_cd = load_gat_model(cg, er_config, model_path, device)
        cd_emb = get_row_embeddings(model_cd, cg, device="cpu")

        cd_metrics = {"name": name}

        # Pair-level metrics
        lp_path = ds_dir / "labeled_pairs.pt"
        if lp_path.exists():
            lp = torch.load(lp_path, weights_only=False)
            pair_metrics = evaluate_pairs(cd_emb, lp)
            cd_metrics.update(pair_metrics)

        # HDBSCAN
        cd_labels = cluster_embeddings(cd_emb, min_cluster_size=args.min_cluster_size)
        cd_cluster = evaluate_clusters(cd_labels)
        cd_metrics["clustering"] = cd_cluster

        cross_results.append(cd_metrics)
        logger.info("  %s: %s", name, cd_metrics)

        del cg, cd_emb
        torch.cuda.empty_cache()

    results["cross_domain"] = cross_results

    # ===== 3. Сохранение =====
    out_path = config.output_dir / "v3_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Результаты сохранены: %s", out_path)

    # Сводка
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ v3")
    print("=" * 60)
    if in_domain_metrics:
        print(f"\nIn-domain test:")
        print(f"  ROC-AUC:       {in_domain_metrics.get('roc_auc', 0):.4f}")
        print(f"  Avg Precision: {in_domain_metrics.get('avg_precision', 0):.4f}")
    if cluster_metrics:
        print(f"  HDBSCAN ARI:   {cluster_metrics.get('ari', 'N/A')}")
        print(f"  HDBSCAN NMI:   {cluster_metrics.get('nmi', 'N/A')}")
        print(f"  Coverage:      {cluster_metrics.get('coverage', 0):.1%}")

    if cross_results:
        print(f"\nCross-domain:")
        for r in cross_results:
            auc = r.get("roc_auc", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
            print(f"  {r['name']:20s} AUC={auc_str}")

    print("=" * 60)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Commit**

```bash
git add experiments/10_evaluate.py
git commit -m "feat: add unified evaluation pipeline (in-domain, cross-domain, HDBSCAN)"
```

---

## Task 9: Запуск полного pipeline

- [ ] **Step 1: Построить unified граф (эксп. 8)**

Run: `uv run python -m experiments.08_build_unified_graph`
Expected: `data/graphs/v3_unified/` и `data/graphs/v3_cross/` созданы

- [ ] **Step 2: Обучить GAT модель (эксп. 9)**

Run: `uv run python -m experiments.09_train_gat --max-epochs 500 --patience 30`
Expected: `output/v3_gat_model.pt` создан

- [ ] **Step 3: Оценить модель (эксп. 10)**

Run: `uv run python -m experiments.10_evaluate`
Expected: `output/v3_evaluation_results.json` создан, результаты в консоли

- [ ] **Step 4: Commit результаты**

```bash
git add experiments/ src/
git commit -m "results: v3 GAT model training and evaluation"
```

---

## Порядок зависимостей

```
Task 1 (hdbscan dep)
  ↓
Task 2 (GATLayer)  →  Task 3 (EntityResolutionGAT)  →  Task 6 (mini-batch trainer)
  ↓                                                       ↓
Task 4 (data_split)  →  Task 5 (unified graph)  →  Task 9 (run pipeline)
  ↓                                                       ↑
Task 7 (HDBSCAN eval)  →  Task 8 (evaluate.py)  ────────┘
```

Параллельно можно делать: Task 2 + Task 4 + Task 7 (независимы).
