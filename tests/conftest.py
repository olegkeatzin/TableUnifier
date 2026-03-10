"""Общие фикстуры для тестов."""

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import HeteroData


# ------------------------------------------------------------------ #
#  Данные
# ------------------------------------------------------------------ #

@pytest.fixture()
def table_a() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [0, 1, 2],
        "title": ["iPhone 13", "Galaxy S21", "Pixel 6"],
        "brand": ["Apple", "Samsung", "Google"],
        "price": [999, 849, 599],
    })


@pytest.fixture()
def table_b() -> pd.DataFrame:
    return pd.DataFrame({
        "id": [0, 1, 2],
        "title": ["Apple iPhone 13", "Samsung Galaxy S21", "Huawei P50"],
        "brand": ["Apple", "Samsung", "Huawei"],
        "price": [999, 850, 699],
    })


@pytest.fixture()
def labels_df() -> pd.DataFrame:
    return pd.DataFrame({
        "ltable_id": [0, 1, 0, 2],
        "rtable_id": [0, 1, 2, 0],
        "label":     [1, 1, 0, 0],
    })


@pytest.fixture()
def column_embeddings() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    return {
        "title": rng.standard_normal(64).astype(np.float32),
        "brand": rng.standard_normal(64).astype(np.float32),
        "price": rng.standard_normal(64).astype(np.float32),
    }


# ------------------------------------------------------------------ #
#  Граф (маленький, для юнит-тестов — без реального TokenEmbedder)
# ------------------------------------------------------------------ #

@pytest.fixture()
def small_hetero_data() -> HeteroData:
    """Минимальный HeteroData для тестов моделей."""
    n_rows = 6
    n_tokens = 10
    n_edges = 20
    row_dim = 32
    token_dim = 32
    col_dim = 64

    rng = np.random.default_rng(42)
    data = HeteroData()

    data["row"].x = torch.tensor(
        rng.standard_normal((n_rows, row_dim)), dtype=torch.float32,
    )
    data["token"].x = torch.tensor(
        rng.standard_normal((n_tokens, token_dim)), dtype=torch.float32,
    )

    t2r_src = torch.randint(0, n_tokens, (n_edges,))
    t2r_dst = torch.randint(0, n_rows, (n_edges,))
    edge_attr = torch.tensor(
        rng.standard_normal((n_edges, col_dim)), dtype=torch.float32,
    )

    data["token", "in_row", "row"].edge_index = torch.stack([t2r_src, t2r_dst])
    data["token", "in_row", "row"].edge_attr = edge_attr
    data["row", "has_token", "token"].edge_index = torch.stack([t2r_dst, t2r_src])
    data["row", "has_token", "token"].edge_attr = edge_attr.clone()

    return data
