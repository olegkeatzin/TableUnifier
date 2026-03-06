"""
Построение двудольного графа (Bipartite Graph) для Entity Resolution.

Архитектура графа:
    Узлы:
        - Row Nodes: по одному на каждую строку обеих таблиц
          Признак: Bag-of-Words из FastText-эмбеддингов токенов строки
        - Token Nodes: по одному на каждый уникальный токен
          Признак: предобученный FastText-эмбеддинг (замороженный)
    
    Рёбра:
        - (row, has_token, token): строка содержит токен в какой-то ячейке
        - (token, in_row, row): обратное ребро (для message passing)
        - edge_attr: эмбеддинг столбца, из которого взят токен
          (ключевая интеграция с курсовым проектом → Zero-Shot!)
    
    Борьба с hub nodes:
        Токены, встречающиеся в > max_token_doc_freq доле строк, отбрасываются.
        Это убирает слишком частые слова ("шт.", "руб." и т.д.).
"""

import re
import numpy as np
import torch
import logging
from typing import List, Dict, Tuple, Optional

try:
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError(
        "PyTorch Geometric не установлен! Установите: "
        "pip install torch-geometric"
    )

from .config import ERConfig
from .token_embedder import FastTextEmbedder

logger = logging.getLogger(__name__)

# Паттерн для токенизации: разбиваем по пробелам и знакам препинания
_TOKENIZE_PATTERN = re.compile(r'[\s\-_/\\,.;:!?()[\]{}\"\'+*=<>@#$%^&~`|]+')


def tokenize_cell(value: str, min_len: int = 2, max_len: int = 50) -> List[str]:
    """Токенизация значения ячейки.
    
    Простой, но эффективный токенизатор:
    - Переводит в нижний регистр
    - Разбивает по пробелам и пунктуации
    - Фильтрует по длине
    
    Args:
        value: Значение ячейки (приводится к строке)
        min_len: Минимальная длина токена
        max_len: Максимальная длина токена
        
    Returns:
        Список уникальных токенов из ячейки
    """
    text = str(value).lower().strip()
    if not text or text == 'nan' or text == 'none':
        return []
    
    tokens = _TOKENIZE_PATTERN.split(text)
    tokens = [t for t in tokens if min_len <= len(t) <= max_len]
    
    # Дедупликация внутри ячейки (сохраняем порядок)
    seen = set()
    unique = []
    for t in tokens:
        if t not in seen:
            seen.add(t)
            unique.append(t)
    
    return unique


class TablePairGraphBuilder:
    """Строитель графов для пар таблиц.
    
    Основной API:
        ft = FastTextEmbedder("cc.ru.300.bin")
        builder = TablePairGraphBuilder(config, ft)
        graph = builder.build_graph(
            df_a, df_b,
            col_embeddings_a, col_embeddings_b,
            duplicate_pairs=[(0, 2), (3, 1)]  # ground truth
        )
    """
    
    def __init__(self, config: ERConfig, fasttext_embedder: FastTextEmbedder):
        self.config = config
        self._ft_embedder = fasttext_embedder
    
    def build_graph(
        self,
        df_a,                                       # pd.DataFrame таблицы A
        df_b,                                       # pd.DataFrame таблицы B
        col_embeddings_a: Dict[str, np.ndarray],    # col_name → embedding
        col_embeddings_b: Dict[str, np.ndarray],    # col_name → embedding
        duplicate_pairs: Optional[List[Tuple[int, int]]] = None,  # [(idx_a, idx_b), ...]
    ) -> HeteroData:
        """Построить HeteroData граф из пары таблиц.
        
        Токены эмбеддятся через предобученный FastText.
        Строки эмбеддятся как Bag-of-Words (среднее FastText-векторов токенов).
        
        Args:
            df_a, df_b: DataFrames двух таблиц
            col_embeddings_a: Словарь {column_name: column_embedding} для таблицы A
            col_embeddings_b: Словарь {column_name: column_embedding} для таблицы B
            duplicate_pairs: Список пар (row_idx_A, row_idx_B) — ground truth дубликатов
                             Индексы относительны к df_a / df_b
        
        Returns:
            HeteroData граф с узлами 'row' и 'token', рёбрами и атрибутами
        """
        n_a = len(df_a)
        n_b = len(df_b)
        n_rows = n_a + n_b
        
        # ═══════════════ 1. Токенизация всех ячеек ═══════════════
        # Для каждой строки и каждого столбца извлекаем токены
        # Храним: (row_global_idx, token_str, col_embedding)
        edge_records = []  # [(row_idx, token_str, col_emb_vector), ...]
        
        # Таблица A
        for row_idx in range(n_a):
            for col_name in df_a.columns:
                cell_val = df_a.iloc[row_idx][col_name]
                tokens = tokenize_cell(
                    cell_val,
                    min_len=self.config.min_token_length,
                    max_len=self.config.max_token_length,
                )
                col_emb = col_embeddings_a.get(col_name)
                if col_emb is None:
                    continue
                for tok in tokens:
                    edge_records.append((row_idx, tok, col_emb))
        
        # Таблица B (row indices сдвинуты на n_a)
        for row_idx in range(n_b):
            global_row_idx = n_a + row_idx
            for col_name in df_b.columns:
                cell_val = df_b.iloc[row_idx][col_name]
                tokens = tokenize_cell(
                    cell_val,
                    min_len=self.config.min_token_length,
                    max_len=self.config.max_token_length,
                )
                col_emb = col_embeddings_b.get(col_name)
                if col_emb is None:
                    continue
                for tok in tokens:
                    edge_records.append((global_row_idx, tok, col_emb))
        
        if not edge_records:
            logger.warning("Нет рёбер! Проверьте данные и column embeddings.")
            return self._empty_graph(df_a, df_b, duplicate_pairs)
        
        # ═══════════════ 2. Фильтрация hub nodes ═══════════════
        # Считаем в скольких строках встречается каждый токен
        token_row_sets: Dict[str, set] = {}
        for row_idx, tok, _ in edge_records:
            if tok not in token_row_sets:
                token_row_sets[tok] = set()
            token_row_sets[tok].add(row_idx)
        
        max_freq = int(n_rows * self.config.max_token_doc_freq)
        hub_tokens = {tok for tok, rows in token_row_sets.items() if len(rows) > max_freq}
        
        if hub_tokens:
            logger.debug(f"Удалено {len(hub_tokens)} hub-токенов (freq > {self.config.max_token_doc_freq})")
        
        # Фильтруем рёбра
        edge_records = [(r, t, e) for r, t, e in edge_records if t not in hub_tokens]
        
        if not edge_records:
            logger.warning("После фильтрации hub-нод не осталось рёбер!")
            return self._empty_graph(df_a, df_b, duplicate_pairs)
        
        # ═══════════════ 3. Индексация токенов ═══════════════
        unique_tokens = sorted(set(t for _, t, _ in edge_records))
        token_to_idx = {tok: idx for idx, tok in enumerate(unique_tokens)}
        n_tokens = len(unique_tokens)
        
        # ═══════════════ 4. Построение рёбер ═══════════════
        row_indices = []
        token_indices = []
        edge_attrs = []
        
        for row_idx, tok, col_emb in edge_records:
            row_indices.append(row_idx)
            token_indices.append(token_to_idx[tok])
            edge_attrs.append(col_emb)
        
        edge_col_dim = edge_attrs[0].shape[0] if edge_attrs else 0
        
        # row → token
        edge_index_r2t = torch.tensor(
            [row_indices, token_indices], dtype=torch.long
        )
        edge_attr = torch.tensor(
            np.array(edge_attrs), dtype=torch.float32
        )
        
        # token → row (reverse)
        edge_index_t2r = torch.tensor(
            [token_indices, row_indices], dtype=torch.long
        )
        
        # ═══════════════ 5. Признаки узлов ═══════════════
        # Token nodes: предобученные FastText-эмбеддинги
        token_x = torch.tensor(
            self._ft_embedder.get_batch_vectors(unique_tokens),
            dtype=torch.float32,
        )
        
        # Row nodes: Bag-of-Words (среднее FastText-векторов всех токенов строки)
        row_embeddings = []
        for row_global_idx in range(n_rows):
            # Собираем все токены данной строки (включая hub-токены, они полезны для BoW)
            if row_global_idx < n_a:
                df = df_a
                local_idx = row_global_idx
            else:
                df = df_b
                local_idx = row_global_idx - n_a
            
            row_tokens = []
            for col_name in df.columns:
                cell_val = df.iloc[local_idx][col_name]
                row_tokens.extend(tokenize_cell(
                    cell_val,
                    min_len=self.config.min_token_length,
                    max_len=self.config.max_token_length,
                ))
            
            row_embeddings.append(self._ft_embedder.get_bow_embedding(row_tokens))
        
        row_x = torch.tensor(
            np.stack(row_embeddings),
            dtype=torch.float32,
        )
        
        # Table ID: 0 для таблицы A, 1 для таблицы B
        table_ids = torch.tensor(
            [0] * n_a + [1] * n_b, dtype=torch.long
        )
        
        # ═══════════════ 6. Метки сущностей (для Metric Learning) ═══════════════
        entity_labels = self._build_entity_labels(n_a, n_b, duplicate_pairs)
        
        # ═══════════════ 7. Сборка HeteroData ═══════════════
        data = HeteroData()
        
        # Row nodes
        data['row'].x = row_x
        data['row'].table_id = table_ids
        data['row'].entity_label = entity_labels
        data['row'].num_rows_a = torch.tensor([n_a])
        data['row'].num_rows_b = torch.tensor([n_b])
        
        # Token nodes (предобученные FastText-эмбеддинги)
        data['token'].x = token_x
        data['token'].num_nodes = n_tokens
        
        # Edges: row → token
        data['row', 'has_token', 'token'].edge_index = edge_index_r2t
        data['row', 'has_token', 'token'].edge_attr = edge_attr
        
        # Edges: token → row (reverse, те же атрибуты)
        data['token', 'in_row', 'row'].edge_index = edge_index_t2r
        data['token', 'in_row', 'row'].edge_attr = edge_attr.clone()
        
        # Метаданные
        data.num_duplicate_pairs = len(duplicate_pairs) if duplicate_pairs else 0
        
        return data
    
    def _build_entity_labels(
        self,
        n_a: int,
        n_b: int,
        duplicate_pairs: Optional[List[Tuple[int, int]]],
    ) -> torch.Tensor:
        """Назначить метки сущностей для метрического обучения.
        
        Дубликаты получают ОДИНАКОВУЮ метку.
        Уникальные строки получают УНИКАЛЬНЫЕ метки.
        
        Returns:
            [n_a + n_b] LongTensor меток
        """
        n_total = n_a + n_b
        # Начальное назначение: каждая строка — уникальная сущность
        labels = list(range(n_total))
        
        if duplicate_pairs:
            # Строим Union-Find для связывания дубликатов
            for idx_a, idx_b in duplicate_pairs:
                global_b = n_a + idx_b
                # Назначаем строке B ту же метку, что и строке A
                labels[global_b] = labels[idx_a]
        
        return torch.tensor(labels, dtype=torch.long)
    
    def _empty_graph(
        self,
        df_a,
        df_b,
        duplicate_pairs: Optional[List[Tuple[int, int]]],
    ) -> HeteroData:
        """Создать пустой граф (fallback при отсутствии рёбер)."""
        n_a = len(df_a)
        n_b = len(df_b)
        ft_dim = self._ft_embedder.dim
        
        data = HeteroData()
        data['row'].x = torch.zeros((n_a + n_b, ft_dim), dtype=torch.float32)
        data['row'].table_id = torch.tensor([0]*n_a + [1]*n_b, dtype=torch.long)
        data['row'].entity_label = self._build_entity_labels(n_a, n_b, duplicate_pairs)
        data['row'].num_rows_a = torch.tensor([n_a])
        data['row'].num_rows_b = torch.tensor([n_b])
        
        data['token'].x = torch.zeros((1, ft_dim), dtype=torch.float32)
        data['token'].num_nodes = 1
        
        data['row', 'has_token', 'token'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['row', 'has_token', 'token'].edge_attr = torch.zeros((0, ft_dim), dtype=torch.float)
        data['token', 'in_row', 'row'].edge_index = torch.zeros((2, 0), dtype=torch.long)
        data['token', 'in_row', 'row'].edge_attr = torch.zeros((0, ft_dim), dtype=torch.float)
        
        data.num_duplicate_pairs = 0
        return data



def compute_column_embeddings(
    df,
    sm_inference,
) -> Dict[str, np.ndarray]:
    """Вычислить эмбеддинги столбцов через SchemaMatcherInference (из курсового проекта).
    
    Использует LLM для генерации описания столбца и embedding model для вектора.
    
    Args:
        df: DataFrame
        sm_inference: SchemaMatcherInference instance
        
    Returns:
        Dict {column_name: np.ndarray embedding}
    """
    embeddings, column_names = sm_inference.compute_column_embeddings(df)
    return {name: emb for name, emb in zip(column_names, embeddings)}
