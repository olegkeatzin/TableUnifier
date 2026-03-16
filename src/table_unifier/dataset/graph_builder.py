"""Построение гетерогенного графа для Entity Resolution.

Двудольный граф HeteroData (PyG):
  — Узлы «row»  : строки обеих таблиц
  — Узлы «token»: уникальные субтокены
  — Рёбра «token → row» (in_row)  с атрибутом = эмбеддинг столбца
  — Рёбра «row → token» (has_token) с атрибутом = эмбеддинг столбца
"""

from __future__ import annotations

import logging
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData

from table_unifier.dataset.embedding_generation import TokenEmbedder, serialize_row

logger = logging.getLogger(__name__)


def build_graph(
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    column_embeddings: dict[str, np.ndarray],
    token_embedder: TokenEmbedder,
    columns_a: list[str] | None = None,
    columns_b: list[str] | None = None,
    precomputed_row_embeddings_a: np.ndarray | None = None,
    precomputed_row_embeddings_b: np.ndarray | None = None,
) -> tuple[HeteroData, dict[str, int], dict[str, int]]:
    """Построить HeteroData-граф из двух таблиц.

    Args:
        precomputed_row_embeddings_a: готовые row-эмбеддинги Table A [N_a, D].
            Если переданы, вызов token_embedder.embed_sentences пропускается.
        precomputed_row_embeddings_b: аналогично для Table B.

    Returns:
        (data, id_to_global_a, id_to_global_b)
        где id_to_global_* — отображение оригинального ID строки
        во внутренний индекс узла «row».
    """
    columns_a = columns_a or [c for c in table_a.columns if c != "id"]
    columns_b = columns_b or [c for c in table_b.columns if c != "id"]

    # ---- 1. Сбор всех строк ---- #
    all_rows: list[pd.Series] = []
    all_row_columns: list[list[str]] = []  # столбцы каждой строки
    id_to_global_a: dict[str, int] = {}
    id_to_global_b: dict[str, int] = {}

    for idx, (_, row) in enumerate(table_a.iterrows()):
        id_to_global_a[str(row["id"])] = idx
        all_rows.append(row)
        all_row_columns.append(columns_a)
    offset = len(table_a)
    for idx, (_, row) in enumerate(table_b.iterrows()):
        id_to_global_b[str(row["id"])] = offset + idx
        all_rows.append(row)
        all_row_columns.append(columns_b)

    n_rows = len(all_rows)
    logger.info("Всего строк: %d (A=%d, B=%d)", n_rows, len(table_a), len(table_b))

    # ---- 2. CLS-эмбеддинги строк ---- #
    if precomputed_row_embeddings_a is not None and precomputed_row_embeddings_b is not None:
        row_embeddings = np.concatenate(
            [precomputed_row_embeddings_a, precomputed_row_embeddings_b], axis=0
        )
        logger.info("Используются готовые row-эмбеддинги (shape=%s)", row_embeddings.shape)
    else:
        row_texts = [
            serialize_row(row, cols) for row, cols in zip(all_rows, all_row_columns)
        ]
        row_embeddings = token_embedder.embed_sentences(row_texts)  # [N_rows, D_row]

    # ---- 3. Построение связей и токенов ---- #
    token_vocab: dict[int, int] = {}  # token_id → node_index
    token_ids_list: list[int] = []    # node_index → original token_id

    t2r_src: list[int] = []
    t2r_dst: list[int] = []
    t2r_col_idx: list[int] = []  # индекс столбца → col_emb_list

    col_emb_list: list[np.ndarray] = []
    col_to_idx: dict[str, int] = {}

    for row_idx, (row, cols) in enumerate(zip(all_rows, all_row_columns)):
        for col in cols:
            val = row.get(col, "")
            if pd.isna(val) or not str(val).strip():
                continue
            cell_text = str(val)
            tids = token_embedder.get_token_ids(cell_text)

            col_emb = column_embeddings.get(col)
            if col_emb is None:
                continue

            if col not in col_to_idx:
                col_to_idx[col] = len(col_emb_list)
                col_emb_list.append(col_emb)
            cidx = col_to_idx[col]

            for tid in tids:
                if tid not in token_vocab:
                    token_vocab[tid] = len(token_ids_list)
                    token_ids_list.append(tid)
                token_node = token_vocab[tid]

                t2r_src.append(token_node)
                t2r_dst.append(row_idx)
                t2r_col_idx.append(cidx)

    n_tokens = len(token_ids_list)
    logger.info("Уникальных токенов: %d, рёбер: %d", n_tokens, len(t2r_src))

    # ---- 4. Vocabulary embeddings токенов ---- #
    token_embeddings = np.stack([
        token_embedder.get_vocab_embedding(tid) for tid in token_ids_list
    ])  # [N_tokens, D_token]

    # ---- 5. Собрать HeteroData ---- #
    data = HeteroData()

    data["row"].x = torch.tensor(row_embeddings, dtype=torch.float32)
    data["token"].x = torch.tensor(token_embeddings, dtype=torch.float32)

    # Token → Row
    t2r_edge_index = torch.tensor([t2r_src, t2r_dst], dtype=torch.long)
    # Собираем edge_attr через индекс столбца (вместо хранения N копий вектора)
    col_emb_matrix = np.stack(col_emb_list)  # [n_cols, 4096]
    col_idx_arr = np.array(t2r_col_idx, dtype=np.intp)
    t2r_edge_attr = torch.tensor(col_emb_matrix[col_idx_arr], dtype=torch.float32)
    del col_idx_arr
    data["token", "in_row", "row"].edge_index = t2r_edge_index
    data["token", "in_row", "row"].edge_attr = t2r_edge_attr

    # Row → Token (обратные рёбра с тем же атрибутом)
    r2t_edge_index = torch.tensor([t2r_dst, t2r_src], dtype=torch.long)
    data["row", "has_token", "token"].edge_index = r2t_edge_index
    data["row", "has_token", "token"].edge_attr = t2r_edge_attr.clone()

    logger.info(
        "Граф: row=%d, token=%d, edges=%d",
        data["row"].x.shape[0],
        data["token"].x.shape[0],
        t2r_edge_index.shape[1],
    )

    return data, id_to_global_a, id_to_global_b
