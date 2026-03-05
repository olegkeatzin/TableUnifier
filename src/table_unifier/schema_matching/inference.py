"""
Инференс для Schema Matching.

Использует обученную ProjectionHead для сопоставления столбцов
между двумя таблицами с разными схемами.

Пайплайн инференса:
    1. Для каждого столбца: генерация описания через LLM → эмбеддинг
    2. Проекция через обученную ProjectionHead
    3. Косинусная матрица сходства
    4. Оптимальное сопоставление (Венгерский алгоритм)
    5. Фильтрация по порогу сходства

Также предоставляет функции для:
    - Вычисления column embeddings для GNN (edge_attr)
    - Batch-инференса для множества таблиц
"""

import numpy as np
import torch
import logging
from typing import Dict, List, Tuple, Optional
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity

from .config import SMConfig
from .model import SchemaMatchingModel

logger = logging.getLogger(__name__)


class SchemaMatcherInference:
    """Инференс для Schema Matching с обученной моделью.
    
    Usage:
        matcher = SchemaMatcherInference(
            model_path='sm_models/best_model.pt',
            config=SMConfig(ollama_host='http://localhost:11434'),
        )
        
        mapping, metrics = matcher.match_schemas(
            source_df, target_df,
            similarity_threshold=0.6,
        )
    """
    
    def __init__(
        self,
        model_path: str,
        config: Optional[SMConfig] = None,
        device: Optional[str] = None,
    ):
        """
        Args:
            model_path: Путь к обученной модели (.pt)
            config: Конфигурация (для Ollama моделей)
            device: 'cuda' или 'cpu'
        """
        self.config = config or SMConfig()
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Загрузка модели
        self.model = SchemaMatchingModel.load(model_path, device=str(self.device))
        
        # Инициализация Ollama
        import ollama
        self.ollama_client = ollama.Client(host=self.config.ollama_host)
        
        logger.info(f"SchemaMatcherInference: model={model_path}, device={self.device}")
    
    def _generate_description(self, col_name: str, content: List, data_type: str) -> str:
        """Генерация описания столбца через LLM."""
        sample = content[:self.config.sample_size]
        type_info = f" Тип данных: {data_type}." if data_type else ""
        
        prompt = (
            f"Дай краткое описание для столбца таблицы с названием '{col_name}'.{type_info}\n"
            f"Если по названию не понятно, что это за столбец, попробуй угадать на основе "
            f"содержимого: {sample}.\n"
            f"Описание должно быть универсальным.\n"
            f"Выведи только описание и ничего больше. /no_think"
        )
        
        try:
            response = self.ollama_client.generate(model=self.config.llm_model, prompt=prompt)
            return f"{col_name}: {response['response']}"
        except Exception as e:
            logger.warning(f"LLM error for '{col_name}': {e}")
            return f"{col_name}: {', '.join(map(str, sample[:3]))}"
    
    def _embed_texts(self, texts: List[str]) -> np.ndarray:
        """Эмбеддинг текстов через Ollama."""
        all_embeddings = []
        batch_size = self.config.embedding_batch_size
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.ollama_client.embed(
                    model=self.config.embedding_model,
                    input=batch
                )
                all_embeddings.extend(response['embeddings'])
            except Exception as e:
                logger.error(f"Embedding error: {e}")
                for text in batch:
                    try:
                        resp = self.ollama_client.embed(
                            model=self.config.embedding_model,
                            input=[text]
                        )
                        all_embeddings.extend(resp['embeddings'])
                    except:
                        all_embeddings.append([0.0] * self.config.input_dim)
        
        return np.array(all_embeddings)
    
    def _project_embeddings(self, raw_embeddings: np.ndarray) -> np.ndarray:
        """Проекция через обученную модель."""
        self.model.eval()
        with torch.no_grad():
            tensor = torch.tensor(raw_embeddings, dtype=torch.float32).to(self.device)
            projected = self.model(tensor)
            return projected.cpu().numpy()
    
    def compute_column_embeddings(
        self,
        df,
        column_names: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        """Вычислить проецированные column embeddings для DataFrame.
        
        Используется для:
        - Schema matching (cosine similarity)
        - GNN edge features (column embeddings)
        
        Args:
            df: pandas DataFrame
            column_names: Список столбцов (если None - все)
            
        Returns:
            (embeddings [N, output_dim], column_names)
        """
        import pandas as pd
        
        if column_names is None:
            column_names = list(df.columns)
        
        descriptions = []
        for col in column_names:
            content = df[col].dropna().tolist()[:self.config.sample_size]
            data_type = str(df[col].dtype)
            desc = self._generate_description(col, content, data_type)
            descriptions.append(desc)
        
        # Raw embeddings
        raw_emb = self._embed_texts(descriptions)
        
        # Projection
        projected = self._project_embeddings(raw_emb)
        
        return projected, column_names
    
    def match_schemas(
        self,
        source_df,
        target_df,
        similarity_threshold: Optional[float] = None,
    ) -> Tuple[Dict[str, Optional[str]], Dict]:
        """Сопоставить схемы двух таблиц.
        
        Args:
            source_df: Исходная (эталонная) таблица
            target_df: Целевая таблица для сопоставления
            similarity_threshold: Порог сходства (из конфига по умолчанию)
            
        Returns:
            - mapping: {source_col: target_col или None}
            - metrics: метрики сопоставления
        """
        if similarity_threshold is None:
            similarity_threshold = self.config.similarity_threshold
        
        # Вычисляем embeddings
        source_emb, source_cols = self.compute_column_embeddings(source_df)
        target_emb, target_cols = self.compute_column_embeddings(target_df)
        
        # Матрица сходства
        sim_matrix = cosine_similarity(source_emb, target_emb)
        
        # Венгерский алгоритм
        cost_matrix = 1 - sim_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Формируем mapping
        mapping = {}
        similarities = []
        detailed = []
        
        for src_idx, tgt_idx in zip(row_ind, col_ind):
            sim = sim_matrix[src_idx, tgt_idx]
            src_col = source_cols[src_idx]
            
            if sim >= similarity_threshold:
                mapping[src_col] = target_cols[tgt_idx]
                similarities.append(sim)
                detailed.append({
                    'source': src_col,
                    'target': target_cols[tgt_idx],
                    'similarity': float(sim),
                })
            else:
                mapping[src_col] = None
                detailed.append({
                    'source': src_col,
                    'target': None,
                    'similarity': float(sim),
                })
        
        # Немаппированные столбцы
        for col in source_cols:
            if col not in mapping:
                mapping[col] = None
        
        metrics = {
            'matched_columns': len([v for v in mapping.values() if v is not None]),
            'total_source_columns': len(source_cols),
            'total_target_columns': len(target_cols),
            'avg_similarity': float(np.mean(similarities)) if similarities else 0.0,
            'min_similarity': float(np.min(similarities)) if similarities else 0.0,
            'max_similarity': float(np.max(similarities)) if similarities else 0.0,
            'threshold': similarity_threshold,
            'detailed_mapping': detailed,
        }
        
        return mapping, metrics
    
    def compute_column_embeddings_raw(
        self,
        descriptions: List[str],
    ) -> np.ndarray:
        """Вычислить проецированные embeddings из готовых описаний.
        
        Удобно для batch-инференса без повторной генерации описаний.
        
        Args:
            descriptions: Список описаний столбцов
            
        Returns:
            [N, output_dim] — проецированные нормализованные эмбеддинги
        """
        raw_emb = self._embed_texts(descriptions)
        return self._project_embeddings(raw_emb)
