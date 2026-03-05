"""
PyTorch Dataset для Entity Resolution.

Обеспечивает:
1. Загрузку предпостроенных графов (.pt файлов) для быстрого обучения
2. Построение графов из пар таблиц + FastText эмбеддинги + column embeddings
3. Правильный collate для батчирования HeteroData графов
4. Сдвиг entity_labels между графами в батче (чтобы метки не пересекались)
"""

import os
import torch
import numpy as np
import logging
from typing import List, Dict, Optional, Callable
from pathlib import Path

try:
    from torch_geometric.data import HeteroData, Batch
except ImportError:
    raise ImportError("PyTorch Geometric не установлен! pip install torch-geometric")

from .config import ERConfig
from .graph_builder import (
    TablePairGraphBuilder,
    compute_column_embeddings,
)
from .token_embedder import FastTextEmbedder
from .er_data_generator import TablePairData

logger = logging.getLogger(__name__)


class ERGraphDataset(torch.utils.data.Dataset):
    """Dataset из предпостроенных графов (.pt).
    
    Используется при обучении: графы строятся один раз (медленно, Ollama),
    сохраняются на диск, а dataloader лишь читает .pt файлы (быстро).
    
    Usage:
        # Построить и сохранить (один раз):
        build_and_save_graphs(pairs, config, output_dir)
        
        # Загружать при обучении:
        dataset = ERGraphDataset('er_dataset/graphs/train')
        loader = DataLoader(dataset, batch_size=16, collate_fn=er_collate_fn)
    """
    
    def __init__(self, graphs_dir: str):
        """
        Args:
            graphs_dir: Путь к директории с .pt файлами
        """
        self.graphs_dir = Path(graphs_dir)
        self.file_list = sorted(self.graphs_dir.glob("*.pt"))
        
        if not self.file_list:
            raise FileNotFoundError(
                f"Нет .pt файлов в {graphs_dir}. "
                "Сначала запустите build_and_save_graphs()."
            )
        
        logger.info(f"ERGraphDataset: {len(self.file_list)} графов из {graphs_dir}")
    
    def __len__(self):
        return len(self.file_list)
    
    def __getitem__(self, idx) -> HeteroData:
        return torch.load(self.file_list[idx], weights_only=False)


def er_collate_fn(data_list: List[HeteroData]) -> HeteroData:
    """Collate function для батчирования ER графов.
    
    Ключевое отличие от стандартного Batch.from_data_list():
    - Сдвигает entity_labels между графами, чтобы метки из разных
      графов не совпадали (важно для triplet mining).
    
    Пример:
        Граф 1: entity_labels = [0, 1, 2, 0, 3]  (0 — дубликат)
        Граф 2: entity_labels = [0, 1, 0, 2]      (0 — дубликат)
        
        После collate:
        Батч: entity_labels = [0, 1, 2, 0, 3,  |  4, 5, 4, 6]
                               ── граф 1 ──     ── граф 2 ──
                               
        Метки 0 из графа 1 != метка 4 из графа 2, хотя в исходных
        графах обе были "0". Это предотвращает ложные позитивы.
    """
    if not data_list:
        return Batch.from_data_list([])
    
    # Сдвигаем entity_labels
    offset = 0
    for data in data_list:
        labels = data['row'].entity_label
        max_label = labels.max().item() + 1
        data['row'].entity_label = labels + offset
        offset += max_label
    
    # Стандартный PyG батч (автоматически сдвигает edge_index, конкатенирует фичи)
    batch = Batch.from_data_list(data_list)
    
    return batch


def _column_cache_key(col_name: str, content_sample: list) -> str:
    """Создать ключ кэша для столбца на основе имени и содержимого."""
    sample_str = "|".join(str(v) for v in content_sample[:5])
    return f"{col_name}::{sample_str}"


def precompute_column_embeddings(
    pairs: List[TablePairData],
    unifier,
    config: ERConfig,
) -> Dict[str, np.ndarray]:
    """Предвычислить column embeddings для всех уникальных столбцов во всех парах.
    
    Колонки с одинаковым именем и первыми 5 значениями считаются идентичными.
    Вызов LLM + embedding делается ОДИН раз для каждого уникального столбца,
    вместо повторных вызовов для каждой пары таблиц.
    
    Рекомендуется вызывать эту функцию один раз для всех сплитов (train/val/test)
    и передавать результат в build_and_save_graphs через параметр col_cache,
    чтобы избежать повторного вычисления эмбеддингов для одинаковых столбцов.
    
    Args:
        pairs: Список пар таблиц (или объединённый список из всех сплитов)
        unifier: TableUnifier instance
        config: Конфигурация ER
    
    Returns:
        Dict {cache_key: np.ndarray embedding}
    """
    import pandas as pd
    
    # 1. Собираем уникальные столбцы по (имя, сэмпл содержимого)
    unique_columns: Dict[str, tuple] = {}  # cache_key → (col_name, content_sample, data_type)
    
    for pair in pairs:
        for df in [pair.df_a, pair.df_b]:
            for col in df.columns:
                content = df[col].dropna().tolist()[:5]
                key = _column_cache_key(col, content)
                if key not in unique_columns:
                    unique_columns[key] = (col, content, str(df[col].dtype))
    
    logger.info(
        f"Column embedding cache: {len(unique_columns)} уникальных столбцов "
        f"(из ~{sum(len(p.df_a.columns) + len(p.df_b.columns) for p in pairs)} всего)"
    )
    
    # 2. Вычисляем embeddings для уникальных столбцов через unifier
    cache: Dict[str, np.ndarray] = {}
    keys = list(unique_columns.keys())
    
    # Batch обработка: собираем все столбцы в один DataFrame-like вызов
    columns_data = []
    for key in keys:
        col_name, content, data_type = unique_columns[key]
        columns_data.append((col_name, content, data_type))
    
    # Обрабатываем через batch_processor напрямую
    results = unifier.batch_processor.process_columns_batch(columns_data)
    
    for key, result in zip(keys, results):
        cache[key] = result['embedding']
    
    return cache


def _get_cached_column_embeddings(
    df,
    col_cache: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Получить column embeddings из кэша для DataFrame."""
    result = {}
    for col in df.columns:
        content = df[col].dropna().tolist()[:5]
        key = _column_cache_key(col, content)
        if key in col_cache:
            result[col] = col_cache[key]
        else:
            logger.warning(f"Column '{col}' не найден в кэше (key={key[:60]}...)")
    return result


def build_and_save_graphs(
    pairs: List[TablePairData],
    config: ERConfig,
    output_dir: str,
    unifier=None,
    fasttext_embedder: Optional[FastTextEmbedder] = None,
    progress_callback: Optional[Callable] = None,
    skip_existing: bool = True,
    col_cache: Optional[Dict[str, np.ndarray]] = None,
) -> int:
    """Построить графы из пар таблиц и сохранить на диск.
    
    Этот метод выполняет:
    1. Для каждой пары таблиц вычисляет column embeddings через TableUnifier
    2. Вычисляет token embeddings (предобученный FastText) и row embeddings (BoW)
    3. Строит HeteroData граф
    4. Сохраняет как .pt файл
    
    Оптимизации:
    - Column embeddings кэшируются: LLM вызывается только для уникальных
      столбцов (по имени + содержимому), а не для каждой пары
    - Уже построенные графы пропускаются (skip_existing=True)
    - Row embeddings вычисляются локально через BoW из FastText (без Ollama)
    - col_cache позволяет передать предвычисленный кэш столбцов
    
    Args:
        pairs: Список пар таблиц (из ERTablePairGenerator)
        config: Конфигурация ER
        output_dir: Директория для сохранения .pt файлов
        unifier: TableUnifier instance (если None — создаётся из config)
        fasttext_embedder: FastTextEmbedder instance (если None — создаётся из config)
        progress_callback: Callable(i, total) для отслеживания прогресса
        skip_existing: Пропускать ли уже существующие .pt файлы
        col_cache: Предвычисленный кэш column embeddings (из precompute_column_embeddings).
                   Если передан, column embeddings не вычисляются повторно.
        
    Returns:
        Число успешно созданных графов
    """
    import time as _time
    from ..core import TableUnifier
    from ..config import AppConfig
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Инициализация моделей (если не переданы)
    if fasttext_embedder is None:
        fasttext_embedder = FastTextEmbedder(config.fasttext_model_path)
    
    if unifier is None:
        app_config = AppConfig()
        app_config.ollama.host = config.ollama_host
        app_config.ollama.embedding_model = config.embedding_model
        app_config.ollama.llm_model = config.llm_model
        app_config.embedding.batch_size = config.embedding_batch_size
        unifier = TableUnifier(app_config)
    
    graph_builder = TablePairGraphBuilder(config, fasttext_embedder)
    
    # ═══════════════ Определяем, какие графы нужно строить ═══════════════
    pairs_to_build = []
    for i, pair in enumerate(pairs):
        filepath = os.path.join(output_dir, f"graph_{i:06d}.pt")
        if skip_existing and os.path.exists(filepath):
            continue
        pairs_to_build.append((i, pair))
    
    skipped = len(pairs) - len(pairs_to_build)
    if skipped > 0:
        logger.info(f"Пропущено {skipped} уже построенных графов")
    
    if not pairs_to_build:
        logger.info("Все графы уже построены!")
        # Обновляем прогресс для всех
        if progress_callback:
            for i in range(len(pairs)):
                progress_callback(i + 1, len(pairs))
        return len(pairs)
    
    # ═══════════════ Предвычисление column embeddings (кэш) ═══════════════
    if col_cache is None:
        t0 = _time.time()
        pairs_for_cache = [pair for _, pair in pairs_to_build]
        col_cache = precompute_column_embeddings(pairs_for_cache, unifier, config)
        t_col = _time.time() - t0
        logger.info(f"Column embeddings предвычислены за {t_col:.1f}с")
    else:
        logger.info(f"Используется переданный col_cache ({len(col_cache)} столбцов)")
    
    # ═══════════════ Построение графов ═══════════════
    # Row embeddings теперь вычисляются локально внутри build_graph
    # через BoW из FastText (без вызовов Ollama)
    successful = skipped  # Уже существующие считаем успешными
    
    # Обновляем прогресс для пропущенных
    if progress_callback and skipped > 0:
        for i in range(skipped):
            progress_callback(i + 1, len(pairs))
    
    for local_idx, (i, pair) in enumerate(pairs_to_build):
        try:
            col_emb_a = _get_cached_column_embeddings(pair.df_a, col_cache)
            col_emb_b = _get_cached_column_embeddings(pair.df_b, col_cache)
            
            # Построение графа (row embeddings вычисляются внутри как BoW)
            graph = graph_builder.build_graph(
                df_a=pair.df_a,
                df_b=pair.df_b,
                col_embeddings_a=col_emb_a,
                col_embeddings_b=col_emb_b,
                duplicate_pairs=pair.duplicate_pairs,
            )
            
            # Сохранение
            filepath = os.path.join(output_dir, f"graph_{i:06d}.pt")
            torch.save(graph, filepath)
            successful += 1
            
        except Exception as e:
            logger.error(f"Ошибка построения графа {i}: {e}")
        
        if progress_callback:
            progress_callback(i + 1, len(pairs))
    
    logger.info(f"Построено {successful}/{len(pairs)} графов в {output_dir}")
    return successful


def get_embedding_dims(sample_graph_path: str) -> Dict[str, int]:
    """Определить размерности из сэмпла графа.
    
    Полезно для автоматической инициализации модели.
    
    Returns:
        {'row_dim': D_row, 'col_dim': D_col, 'token_dim': max_ngrams}
    """
    data = torch.load(sample_graph_path, weights_only=False)
    
    dims = {
        'row_dim': data['row'].x.shape[1],
        'col_dim': data['row', 'has_token', 'token'].edge_attr.shape[1]
                   if data['row', 'has_token', 'token'].edge_attr.shape[0] > 0
                   else 0,
    }
    
    return dims
