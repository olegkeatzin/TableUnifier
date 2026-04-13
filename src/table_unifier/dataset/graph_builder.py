"""Построение гетерогенного графа для Entity Resolution.

Двудольный граф HeteroData (PyG):
  — Узлы «row»  : строки обеих таблиц
  — Узлы «token»: уникальные субтокены
  — Рёбра «token → row» (in_row)  с атрибутом = эмбеддинг столбца
  — Рёбра «row → token» (has_token) с атрибутом = эмбеддинг столбца
"""

from __future__ import annotations

import logging
import random
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
    max_token_df: float = 0.3,
    max_tokens_per_cell: int = 16,
    min_token_count: int = 1,
) -> tuple[HeteroData, dict[str, int], dict[str, int]]:
    """Построить HeteroData-граф из двух таблиц.

    Токены фильтруются в три этапа:
    1. IDF-фильтрация (верхний хвост): удаляются токены, встречающиеся в
       >max_token_df доле строк (стоп-слова, пунктуация) — они не помогают
       различать сущности.
    2. Min-degree фильтрация (нижний хвост): удаляются токены, встречающиеся
       в <min_token_count строках. Singleton-токены (degree=1) бесполезны для
       message passing — их сообщение идёт в одну строку и не помогает
       сравнивать её с другими.
    3. Если после фильтрации ячейка даёт >max_tokens_per_cell токенов,
       берётся случайная выборка (не первые N, чтобы охватить всю ячейку).

    Args:
        precomputed_row_embeddings_a: готовые row-эмбеддинги Table A [N_a, D].
            Если переданы, вызов token_embedder.embed_sentences пропускается.
        precomputed_row_embeddings_b: аналогично для Table B.
        max_token_df: порог document frequency (доля строк). Токены, встречающиеся
            чаще, считаются стоп-словами и удаляются. По умолч. 0.3 (30% строк).
        max_tokens_per_cell: лимит токенов на ячейку после IDF-фильтрации.
            При превышении — случайная выборка. По умолч. 16.
        min_token_count: минимальное число строк, в которых должен встречаться
            токен. По умолч. 1 (без фильтрации). Установить ≥2 чтобы убрать
            singleton-токены (уникальные ID, опечатки).

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

    # ---- 3. Построение связей и токенов (проход 1: сбор всех рёбер) ---- #
    token_vocab: dict[int, int] = {}  # token_id → node_index
    token_ids_list: list[int] = []    # node_index → original token_id

    # Сырые рёбра до IDF-фильтрации: (token_node, row_idx, col_idx)
    raw_edges: list[tuple[int, int, int]] = []

    # Для вычисления document frequency: token_node → множество строк
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

            col_emb = column_embeddings.get(col)
            if col_emb is None:
                continue

            if col not in col_to_idx:
                col_to_idx[col] = len(col_emb_list)
                col_emb_list.append(col_emb)
            cidx = col_to_idx[col]

            # Дедупликация токенов внутри ячейки: одно ребро token→row на токен
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

    # ---- 3b. IDF-фильтрация: удаляем стоп-токены ---- #
    # Токены, встречающиеся в >max_token_df доле строк — стоп-слова/пунктуация.
    # Для ER они бесполезны: одинаково присутствуют в matching и non-matching парах.
    idf_threshold = int(max_token_df * n_rows)
    stopword_nodes = {
        node for node, rows in token_row_sets.items()
        if len(rows) > idf_threshold
    }
    if stopword_nodes:
        logger.info(
            "IDF-фильтрация: удалено %d/%d токенов (df > %.0f%% строк)",
            len(stopword_nodes), len(token_ids_list), max_token_df * 100,
        )

    # ---- 3b'. Min-degree фильтрация: удаляем singleton-токены ---- #
    if min_token_count > 1:
        singleton_nodes = {
            node for node, rows in token_row_sets.items()
            if len(rows) < min_token_count and node not in stopword_nodes
        }
        if singleton_nodes:
            logger.info(
                "Min-degree фильтрация: удалено %d/%d токенов (df < %d строк)",
                len(singleton_nodes), len(token_ids_list), min_token_count,
            )
        stopword_nodes = stopword_nodes | singleton_nodes

    # ---- 3c. Построение финальных рёбер с лимитом per-cell ---- #
    # Группируем по (row_idx, col_idx), фильтруем стоп-токены,
    # при превышении лимита берём случайную выборку (не первые N).
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

    # Ремаппинг: оставляем только токены, задействованные в рёбрах
    used_nodes = set(t2r_src)
    old_to_new: dict[int, int] = {}
    new_token_ids_list: list[int] = []
    for old_node, tid in enumerate(token_ids_list):
        if old_node in used_nodes:
            old_to_new[old_node] = len(new_token_ids_list)
            new_token_ids_list.append(tid)

    t2r_src = [old_to_new[n] for n in t2r_src]
    token_ids_list = new_token_ids_list

    n_tokens = len(token_ids_list)
    logger.info(
        "Уникальных токенов: %d (после фильтрации), рёбер: %d",
        n_tokens, len(t2r_src),
    )

    # ---- 4. Vocabulary embeddings токенов ---- #
    token_embeddings = np.stack([
        token_embedder.get_vocab_embedding(tid) for tid in token_ids_list
    ])  # [N_tokens, D_token]

    # ---- 5. Собрать HeteroData ---- #
    data = HeteroData()

    data["row"].x = torch.tensor(row_embeddings, dtype=torch.float32)
    data["token"].x = torch.tensor(token_embeddings, dtype=torch.float32)

    # Матрица эмбеддингов столбцов (компактная: [n_cols, 4096])
    col_emb_matrix = torch.tensor(np.stack(col_emb_list), dtype=torch.float32)
    data.col_embeddings = col_emb_matrix

    # Индекс столбца для каждого ребра (int, [n_edges])
    edge_col_idx = torch.tensor(t2r_col_idx, dtype=torch.long)

    # Token → Row
    t2r_edge_index = torch.tensor([t2r_src, t2r_dst], dtype=torch.long)
    data["token", "in_row", "row"].edge_index = t2r_edge_index
    data["token", "in_row", "row"].edge_col_idx = edge_col_idx

    # Row → Token (обратные рёбра с тем же индексом столбца)
    r2t_edge_index = torch.tensor([t2r_dst, t2r_src], dtype=torch.long)
    data["row", "has_token", "token"].edge_index = r2t_edge_index
    data["row", "has_token", "token"].edge_col_idx = edge_col_idx

    logger.info(
        "Граф: row=%d, token=%d, edges=%d, cols=%d",
        data["row"].x.shape[0],
        data["token"].x.shape[0],
        t2r_edge_index.shape[1],
        col_emb_matrix.shape[0],
    )

    return data, id_to_global_a, id_to_global_b


# ------------------------------------------------------------------ #
#  Объединённый граф из нескольких датасетов
# ------------------------------------------------------------------ #


def build_unified_graph_from_datasets(
    datasets: list[dict],
    token_embedder: "TokenEmbedder",
    max_token_df: float = 0.3,
    max_tokens_per_cell: int = 16,
    min_token_count: int = 1,
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

    # --- Тот же алгоритм что в build_graph(), но с глобальным IDF ---
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
    n_raw_edges = len(raw_edges)
    n_raw_tokens = len(token_ids_list)
    idf_threshold = int(max_token_df * n_rows)
    stopword_nodes = {
        node for node, rows in token_row_sets.items()
        if len(rows) > idf_threshold
    }
    n_tokens_removed_idf = len(stopword_nodes)
    n_edges_removed_idf = sum(1 for tn, _, _ in raw_edges if tn in stopword_nodes)
    logger.info("Глобальная IDF: удалено %d/%d токенов (df > %.1f%% из %d строк), "
                "рёбер до=%d, удалено=%d (%.1f%%)",
                n_tokens_removed_idf, n_raw_tokens, max_token_df * 100, n_rows,
                n_raw_edges, n_edges_removed_idf,
                100 * n_edges_removed_idf / n_raw_edges if n_raw_edges else 0)

    # Min-degree фильтрация: режем нижний хвост (singleton-токены)
    n_tokens_removed_mindeg = 0
    n_edges_removed_mindeg = 0
    if min_token_count > 1:
        singleton_nodes = {
            node for node, rows in token_row_sets.items()
            if len(rows) < min_token_count and node not in stopword_nodes
        }
        n_tokens_removed_mindeg = len(singleton_nodes)
        n_edges_removed_mindeg = sum(1 for tn, _, _ in raw_edges if tn in singleton_nodes)
        logger.info("Min-degree: удалено %d/%d токенов (df < %d), рёбер удалено=%d (%.1f%%)",
                    n_tokens_removed_mindeg, n_raw_tokens, min_token_count,
                    n_edges_removed_mindeg,
                    100 * n_edges_removed_mindeg / n_raw_edges if n_raw_edges else 0)
        stopword_nodes = stopword_nodes | singleton_nodes

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

    n_final_edges = t2r_edge_index.shape[1]
    n_edges_after_idf = n_raw_edges - n_edges_removed_idf
    n_edges_after_filters = n_edges_after_idf - n_edges_removed_mindeg
    n_edges_removed_cell_limit = n_edges_after_filters - n_final_edges

    logger.info("Unified graph: rows=%d, tokens=%d, edges=%d "
                "(raw=%d → after IDF=%d → after min-deg=%d → final=%d), labeled_pairs=%d",
                data["row"].x.shape[0], data["token"].x.shape[0],
                n_final_edges, n_raw_edges, n_edges_after_idf, n_edges_after_filters,
                n_final_edges, len(labeled_tensor))

    # Сохраняем статистику фильтрации как атрибуты графа
    data.filter_stats = {
        "raw_tokens": n_raw_tokens,
        "raw_edges": n_raw_edges,
        "idf_threshold": max_token_df,
        "min_token_count": min_token_count,
        "tokens_removed_idf": n_tokens_removed_idf,
        "tokens_removed_mindeg": n_tokens_removed_mindeg,
        "edges_removed_idf": n_edges_removed_idf,
        "edges_removed_mindeg": n_edges_removed_mindeg,
        "edges_after_idf": n_edges_after_idf,
        "edges_removed_cell_limit": n_edges_removed_cell_limit,
        "final_tokens": len(token_ids_list),
        "final_edges": n_final_edges,
    }

    return data, dataset_mappings, labeled_tensor
