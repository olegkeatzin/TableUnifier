"""
Модуль инференса для Entity Resolution.

Предоставляет DuplicateDetector — класс для поиска дубликатов строк
между двумя таблицами с произвольными схемами.

Pipeline:
    1. Получить column embeddings (SchemaMatcherInference (SM) / курсовой проект)
    2. Построить граф из пары таблиц (FastText для токенов, BoW для строк)
    3. Forward pass через обученную GNN
    4. Сравнить row embeddings из таблицы A и B
    5. Вернуть пары дубликатов с confidence score

Также включает evaluate() для оценки качества на тестовых данных.
"""

import torch
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from torch_geometric.data import HeteroData
except ImportError:
    raise ImportError("PyTorch Geometric не установлен!")

from .config import ERConfig
from .gnn_model import EntityResolutionGNN
from .graph_builder import (
    TablePairGraphBuilder,
    compute_column_embeddings,
)
from .token_embedder import FastTextEmbedder
from .er_dataset import ERGraphDataset, er_collate_fn

logger = logging.getLogger(__name__)


@dataclass
class DuplicateMatch:
    """Результат сопоставления дубликатов."""
    row_idx_a: int          # Индекс строки в таблице A
    row_idx_b: int          # Индекс строки в таблице B
    similarity: float       # Косинусное сходство (0..1)
    

class DuplicateDetector:
    """Детектор дубликатов строк между таблицами.
    
    Usage:
        detector = DuplicateDetector.from_checkpoint(
            'er_model.pt',
            config=ERConfig(),
            row_input_dim=300,
            col_embed_dim=256,
        )
        
        matches = detector.find_duplicates(df_a, df_b, threshold=0.7)
        for m in matches:
            print(f"Row A[{m.row_idx_a}] ≈ Row B[{m.row_idx_b}] (sim={m.similarity:.3f})")
    """
    
    def __init__(
        self,
        model: EntityResolutionGNN,
        config: ERConfig,
        fasttext_embedder: FastTextEmbedder = None,
        device: torch.device = None,
    ):
        self.model = model
        self.config = config
        self.device = device or torch.device('cpu')
        self.model.to(self.device)
        self.model.eval()
        
        # Загрузка FastText u043cодели (для построения графов при инференсе)
        if fasttext_embedder is None:
            fasttext_embedder = FastTextEmbedder(config.fasttext_model_path)
        self._ft_embedder = fasttext_embedder
        
        self.graph_builder = TablePairGraphBuilder(config, fasttext_embedder)
    
    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str,
        config: ERConfig,
        row_input_dim: int,
        col_embed_dim: int,
        fasttext_embedder: FastTextEmbedder = None,
        device: str = None,
    ) -> 'DuplicateDetector':
        """Создать детектор из сохранённого чекпоинта.
        
        Args:
            checkpoint_path: Путь к .pt файлу модели
            config: Конфигурация ER
            row_input_dim: Размерность row embeddings (FastText dim, обычно 300)
            col_embed_dim: Размерность column embeddings
            fasttext_embedder: FastTextEmbedder instance (если None — создаётся из config)
            device: 'cuda', 'cpu', или None (авто)
        """
        dev = torch.device(device) if device else \
              torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model = EntityResolutionGNN(
            row_input_dim=row_input_dim,
            col_embed_dim=col_embed_dim,
            config=config,
        )
        
        checkpoint = torch.load(checkpoint_path, map_location=dev, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        logger.info(f"Модель загружена из {checkpoint_path}")
        
        return cls(
            model=model,
            config=config,
            fasttext_embedder=fasttext_embedder,
            device=dev,
        )
    
    @torch.no_grad()
    def find_duplicates(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        threshold: float = 0.7,
        top_k: int = None,
        sm_inference=None,
    ) -> List[DuplicateMatch]:
        """Найти дубликаты строк между двумя таблицами.
        
        Args:
            df_a: Первая таблица
            df_b: Вторая таблица
            threshold: Минимальное косинусное сходство для считания дубликатом
            top_k: Вернуть только top-K лучших совпадений (None = все выше threshold)
            sm_inference: SchemaMatcherInference (если None — создаётся из config)
            
        Returns:
            Список DuplicateMatch, отсортированный по убыванию similarity
        """
        # Ленивый импорт для инженерии зависимостей
        from ..schema_matching.config import SMConfig
        from ..schema_matching.inference import SchemaMatcherInference
        
        # Инициализация SM inference (для column embeddings)
        if sm_inference is None:
            sm_config = SMConfig(
                ollama_host=self.config.ollama_host,
                embedding_model=self.config.embedding_model,
                llm_model=self.config.llm_model,
            )
            sm_inference = SchemaMatcherInference(
                model_path=self.config.sm_model_path,
                config=sm_config,
            )
        
        # 1. Column embeddings (через LLM + Ollama embed)
        logger.info("Вычисление column embeddings...")
        col_emb_a = compute_column_embeddings(df_a, sm_inference)
        col_emb_b = compute_column_embeddings(df_b, sm_inference)
        
        # 2. Построение графа (row embeddings = BoW из FastText, вычисляется внутри)
        logger.info("Построение графа (FastText BoW)...")
        graph = self.graph_builder.build_graph(
            df_a=df_a, df_b=df_b,
            col_embeddings_a=col_emb_a,
            col_embeddings_b=col_emb_b,
        )
        graph = graph.to(self.device)
        
        # 3. Инференс GNN
        logger.info("Инференс GNN...")
        all_embeddings = self.model(graph)
        
        n_a = graph['row'].num_rows_a.item()
        emb_a = all_embeddings[:n_a]   # [N_a, D]
        emb_b = all_embeddings[n_a:]   # [N_b, D]
        
        # 5. Cosine similarity matrix
        sim_matrix = torch.mm(emb_a, emb_b.t())  # [N_a, N_b]
        
        # 6. Извлечение совпадений
        matches = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                sim = sim_matrix[i, j].item()
                if sim >= threshold:
                    matches.append(DuplicateMatch(
                        row_idx_a=i,
                        row_idx_b=j,
                        similarity=sim,
                    ))
        
        # Сортировка по similarity (убывание)
        matches.sort(key=lambda m: m.similarity, reverse=True)
        
        if top_k is not None:
            matches = matches[:top_k]
        
        logger.info(f"Найдено {len(matches)} дубликатов (threshold={threshold})")
        
        return matches
    
    @torch.no_grad()
    def find_duplicates_from_graph(
        self,
        graph: HeteroData,
        threshold: float = 0.7,
    ) -> List[DuplicateMatch]:
        """Найти дубликаты из предпостроенного графа (без Ollama).
        
        Args:
            graph: HeteroData граф (из build_graph или .pt файл)
            threshold: Минимальное сходство
            
        Returns:
            Список DuplicateMatch
        """
        graph = graph.to(self.device)
        all_embeddings = self.model(graph)
        
        n_a = graph['row'].num_rows_a.item()
        emb_a = all_embeddings[:n_a]
        emb_b = all_embeddings[n_a:]
        
        sim_matrix = torch.mm(emb_a, emb_b.t())
        
        matches = []
        for i in range(sim_matrix.shape[0]):
            for j in range(sim_matrix.shape[1]):
                sim = sim_matrix[i, j].item()
                if sim >= threshold:
                    matches.append(DuplicateMatch(
                        row_idx_a=i,
                        row_idx_b=j,
                        similarity=sim,
                    ))
        
        matches.sort(key=lambda m: m.similarity, reverse=True)
        return matches
    
    def matches_to_dataframe(
        self,
        matches: List[DuplicateMatch],
        df_a: pd.DataFrame = None,
        df_b: pd.DataFrame = None,
    ) -> pd.DataFrame:
        """Конвертировать результаты в удобный DataFrame.
        
        Args:
            matches: Список DuplicateMatch
            df_a: Исходная таблица A (для отображения значений)
            df_b: Исходная таблица B
            
        Returns:
            DataFrame с колонками: row_a, row_b, similarity, [+данные строк]
        """
        records = []
        for m in matches:
            record = {
                'row_idx_a': m.row_idx_a,
                'row_idx_b': m.row_idx_b,
                'similarity': round(m.similarity, 4),
            }
            
            if df_a is not None:
                for col in df_a.columns[:3]:  # первые 3 столбца для контекста
                    record[f'A_{col}'] = df_a.iloc[m.row_idx_a].get(col, '')
            
            if df_b is not None:
                for col in df_b.columns[:3]:
                    record[f'B_{col}'] = df_b.iloc[m.row_idx_b].get(col, '')
            
            records.append(record)
        
        return pd.DataFrame(records)


def evaluate_on_test(
    model: EntityResolutionGNN,
    test_dir: str,
    config: ERConfig,
    device: torch.device = None,
    thresholds: List[float] = None,
) -> Dict[str, float]:
    """Оценка модели на тестовом наборе данных.
    
    Вычисляет:
    - Precision, Recall, F1 при разных threshold
    - Mean Average Precision (MAP)
    - Average cosine similarity для дубликатов vs не-дубликатов (separability)
    
    Args:
        model: Обученная модель
        test_dir: Директория с тестовыми .pt графами
        config: Конфигурация
        device: Устройство
        thresholds: Список пороговых значений для evaluation
        
    Returns:
        Словарь метрик
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    
    from torch_geometric.loader import DataLoader as PyGDataLoader
    
    model.to(device)
    model.eval()
    
    test_dataset = ERGraphDataset(test_dir)
    test_loader = PyGDataLoader(
        test_dataset,
        batch_size=1,  # По одному графу для точных метрик
        shuffle=False,
    )
    
    all_dup_sims = []   # сходства для истинных дубликатов
    all_non_sims = []   # сходства для не-дубликатов
    
    per_threshold_metrics = {t: {'tp': 0, 'fp': 0, 'fn': 0} for t in thresholds}
    
    with torch.no_grad():
        for graph in test_loader:
            graph = graph.to(device)
            embeddings = model(graph)
            
            n_a = graph['row'].num_rows_a.item()
            emb_a = embeddings[:n_a]
            emb_b = embeddings[n_a:]
            labels = graph['row'].entity_label
            labels_a = labels[:n_a]
            labels_b = labels[n_a:]
            
            if emb_a.shape[0] == 0 or emb_b.shape[0] == 0:
                continue
            
            sim_matrix = torch.mm(emb_a, emb_b.t()).cpu()
            
            # Ground truth пары
            gt_pairs = set()
            for i, la in enumerate(labels_a):
                for j, lb in enumerate(labels_b):
                    if la == lb:
                        gt_pairs.add((i, j.item() if isinstance(j, torch.Tensor) else j))
            
            # Собираем сходства
            for i in range(sim_matrix.shape[0]):
                for j in range(sim_matrix.shape[1]):
                    sim = sim_matrix[i, j].item()
                    if (i, j) in gt_pairs:
                        all_dup_sims.append(sim)
                    else:
                        all_non_sims.append(sim)
            
            # Метрики при каждом threshold
            for t in thresholds:
                for i in range(sim_matrix.shape[0]):
                    best_sim, best_j = sim_matrix[i].max(dim=0)
                    best_j = best_j.item()
                    best_sim = best_sim.item()
                    
                    has_gt = any(pair[0] == i for pair in gt_pairs)
                    
                    if has_gt:
                        gt_j = [pair[1] for pair in gt_pairs if pair[0] == i]
                        if best_sim >= t and best_j in gt_j:
                            per_threshold_metrics[t]['tp'] += 1
                        else:
                            per_threshold_metrics[t]['fn'] += 1
                    else:
                        if best_sim >= t:
                            per_threshold_metrics[t]['fp'] += 1
    
    # Агрегированные метрики
    results = {}
    
    if all_dup_sims:
        results['avg_duplicate_similarity'] = float(np.mean(all_dup_sims))
        results['std_duplicate_similarity'] = float(np.std(all_dup_sims))
    if all_non_sims:
        results['avg_non_duplicate_similarity'] = float(np.mean(all_non_sims))
        results['std_non_duplicate_similarity'] = float(np.std(all_non_sims))
    
    if all_dup_sims and all_non_sims:
        results['separability'] = results['avg_duplicate_similarity'] - results['avg_non_duplicate_similarity']
    
    for t in thresholds:
        m = per_threshold_metrics[t]
        tp, fp, fn = m['tp'], m['fp'], m['fn']
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-8)
        
        results[f'precision@{t}'] = precision
        results[f'recall@{t}'] = recall
        results[f'f1@{t}'] = f1
    
    return results
