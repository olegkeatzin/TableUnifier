"""
Система унификации таблиц с батчингом и метриками (БЕЗ кэширования)
"""
from .models import OllamaLLM, OllamaEmbedding
import numpy as np
from numpy.linalg import norm
from typing import List, Optional, Dict, Tuple
import pandas as pd
from .config import AppConfig
import logging

logger = logging.getLogger(__name__)


class EmbSerialV2:
    """Представление столбца с эмбеддингом"""
    
    def __init__(self, name: str, content: list, embedding: np.ndarray, 
                 description: str, data_type: Optional[str] = None):
        self.name = name
        self.content = content
        self.embedding = embedding
        self.description = description
        self.data_type = data_type
    
    def __sub__(self, other):
        """Вычисление косинусного расстояния"""
        if not isinstance(other, EmbSerialV2):
            return NotImplemented
        
        vec1 = self.embedding
        vec2 = other.embedding
        
        cosine_similarity = np.dot(vec1, vec2) / (norm(vec1) * norm(vec2))
        cosine_distance = 1 - cosine_similarity
        
        return cosine_distance
    
    def similarity_score(self, other) -> float:
        """Вычисление косинусного сходства (0-1, где 1 - идентичные)"""
        if not isinstance(other, EmbSerialV2):
            return 0.0
        
        vec1 = self.embedding
        vec2 = other.embedding
        
        return float(np.dot(vec1, vec2) / (norm(vec1) * norm(vec2)))
    
    def to_dict(self) -> Dict:
        """Экспорт в словарь"""
        return {
            'name': self.name,
            'description': self.description,
            'data_type': self.data_type,
            'sample_content': self.content[:5],
            'embedding_shape': self.embedding.shape
        }
    
    def __repr__(self):
        type_str = f" ({self.data_type})" if self.data_type else ""
        return f"EmbSerialV2('{self.name}'{type_str})"


class SimpleBatchProcessor:
    """Простой батч-процессор БЕЗ кэширования для максимального разнообразия датасета"""
    
    def __init__(self, embedder, llm, batch_size: int = 10):
        self.embedder = embedder
        self.llm = llm
        self.batch_size = batch_size
    
    def process_columns_batch(self, columns_data: List[Tuple[str, List, Optional[str]]]) -> List[Dict]:
        """Обработать несколько столбцов за раз БЕЗ кэширования"""
        results = []
        descriptions_to_embed = []
        
        # Генерируем описания для всех столбцов
        for idx, (name, content, data_type) in enumerate(columns_data):
            description = self._generate_description(name, content, data_type)
            descriptions_to_embed.append((idx, name, description, content, data_type))
        
        # Батчинг эмбеддингов
        all_descriptions = [desc[2] for desc in descriptions_to_embed]
        embeddings = self._batch_embed(all_descriptions)
        
        # Формируем результаты
        for (idx, name, description, content, data_type), embedding in zip(descriptions_to_embed, embeddings):
            results.append({
                'index': idx,
                'name': name,
                'description': description,
                'embedding': embedding,
                'data_type': data_type
            })
        
        return results
    
    def _generate_description(self, name: str, content: List, data_type: Optional[str]) -> str:
        """Генерация описания столбца через LLM"""
        type_info = f" Тип данных: {data_type}." if data_type else ""
        
        # Выбор 5 случайных элементов или меньше, если меньше доступно
        sample_size = min(5, len(content))
        if sample_size > 0:
            random_elements = list(np.random.choice(content, size=sample_size, replace=False))
        else:
            random_elements = []
        
        prompt = f'''
        Дай краткое описание для столбца таблицы с названием '{name}'.{type_info}
        Если по названию не понятно, что это за столбец, попробуй угадать на основе содержимого: {random_elements}.
        Описание должно быть универсальным, чтобы подходить для любых значений в этом столбце.
        Если столбец описывает что-то конкретное, думай шире - в столбце могут быть более разнообразные данные.
        Выведи только описание и ничего больше.
        '''
        
        try:
            desc = self.llm.generate(prompt)
            return f'{name}: {desc}'
        except Exception as e:
            print(f"Ошибка генерации описания для '{name}': {e}")
            return f'{name}: {", ".join(map(str, content[:3]))}'
    
    def _batch_embed(self, texts: List[str]) -> List[np.ndarray]:
        """Генерация эмбеддингов батчами"""
        all_embeddings = []
        
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            try:
                embeddings = self.embedder.embed(batch)
                all_embeddings.extend([np.array(emb) for emb in embeddings])
            except Exception as e:
                logger.error(f"Ошибка при генерации эмбеддингов для батча {i}: {e}")
                # Fallback: по одному
                for text in batch:
                    try:
                        emb = self.embedder.embed([text])[0]
                        all_embeddings.append(np.array(emb))
                    except Exception as e2:
                        logger.error(f"Критическая ошибка эмбеддинга: {e2}")
                        all_embeddings.append(np.zeros(768))
        
        return all_embeddings


class TableUnifier:
    """Главный класс для унификации таблиц"""
    
    def __init__(self, config: Optional[AppConfig] = None):
        self.config = config or AppConfig()
        
        # Инициализация моделей
        self.llm = OllamaLLM(
            host=self.config.ollama.host,
            model=self.config.ollama.llm_model
        )
        self.embedder = OllamaEmbedding(
            host=self.config.ollama.host,
            model=self.config.ollama.embedding_model
        )
        
        # Батч-процессор
        self.batch_processor = SimpleBatchProcessor(
            self.embedder, 
            self.llm, 
            self.config.embedding.batch_size
        )
        
        self.reference_columns: Optional[List[EmbSerialV2]] = None
    
    def process_dataframe(self, df: pd.DataFrame, 
                         include_types: bool = None) -> List[EmbSerialV2]:
        """
        Обработать DataFrame и создать EmbSerial для каждого столбца
        
        Args:
            df: pandas DataFrame
            include_types: учитывать ли типы данных (из конфига по умолчанию)
        
        Returns:
            Список объектов EmbSerialV2
        """
        if include_types is None:
            include_types = self.config.embedding.include_data_types
        
        # Подготовка данных для батч-обработки
        columns_data = []
        for col in df.columns:
            content = df[col].dropna().tolist()[:self.config.embedding.sample_size]
            data_type = str(df[col].dtype) if include_types else None
            columns_data.append((col, content, data_type))
        
        logger.debug(f"Обработка {len(columns_data)} столбцов...")
        
        # Батч-обработка
        results = self.batch_processor.process_columns_batch(columns_data)
        
        # Создание объектов EmbSerialV2
        emb_serials = []
        for result in results:
            emb_serial = EmbSerialV2(
                name=result['name'],
                content=columns_data[result['index']][1],
                embedding=result['embedding'],
                description=result['description'],
                data_type=result['data_type']
            )
            emb_serials.append(emb_serial)
        
        logger.debug(f"Обработка {len(emb_serials)} столбцов завершена")
        return emb_serials
    
    def set_reference_schema(self, reference_df: pd.DataFrame):
        """Установить эталонную схему таблицы"""
        logger.debug("Установка эталонной схемы...")
        self.reference_columns = self.process_dataframe(reference_df)
        logger.debug(f"Эталонная схема: {[col.name for col in self.reference_columns]}")
    
    def unify_table(self, target_df: pd.DataFrame, 
                    similarity_threshold: float = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Унифицировать таблицу к эталонной схеме
        
        Returns:
            - Унифицированный DataFrame
            - Словарь с метриками и информацией о сопоставлении
        """
        if self.reference_columns is None:
            raise ValueError("Сначала установите эталонную схему через set_reference_schema()")
        
        if similarity_threshold is None:
            similarity_threshold = self.config.classifier.min_similarity
        
        # Обработка целевой таблицы
        target_columns = self.process_dataframe(target_df)
        
        # Сопоставление столбцов
        mapping, metrics = self._match_columns(
            self.reference_columns, 
            target_columns, 
            similarity_threshold
        )
        
        # Создание унифицированной таблицы
        unified_df = pd.DataFrame()
        for ref_col, target_col in mapping.items():
            if target_col is not None:
                unified_df[ref_col.name] = target_df[target_col.name]
            else:
                # Столбец отсутствует в целевой таблице
                unified_df[ref_col.name] = None
        
        # Метрики
        metrics['unified_columns'] = list(unified_df.columns)
        metrics['missing_columns'] = [col.name for col, val in mapping.items() if val is None]
        
        return unified_df, metrics
    
    def _match_columns(self, reference_cols: List[EmbSerialV2], 
                      target_cols: List[EmbSerialV2],
                      threshold: float) -> Tuple[Dict[EmbSerialV2, Optional[EmbSerialV2]], Dict]:
        """
        Сопоставить столбцы целевой таблицы с эталонными
        
        Returns:
            - Словарь {эталонный_столбец: соответствующий_столбец_или_None}
            - Метрики сопоставления
        """
        from scipy.optimize import linear_sum_assignment
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Матрица сходства
        ref_embeddings = np.array([col.embedding for col in reference_cols])
        target_embeddings = np.array([col.embedding for col in target_cols])
        
        similarity_matrix = cosine_similarity(ref_embeddings, target_embeddings)
        
        # Венгерский алгоритм для оптимального сопоставления
        cost_matrix = 1 - similarity_matrix
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        mapping = {}
        similarities = []
        
        for ref_idx, target_idx in zip(row_ind, col_ind):
            sim = similarity_matrix[ref_idx, target_idx]
            ref_col = reference_cols[ref_idx]
            
            if sim >= threshold:
                mapping[ref_col] = target_cols[target_idx]
                similarities.append(sim)
            else:
                mapping[ref_col] = None
        
        # Несопоставленные эталонные столбцы
        for ref_col in reference_cols:
            if ref_col not in mapping:
                mapping[ref_col] = None
        
        # Метрики
        metrics = {
            'total_reference_columns': len(reference_cols),
            'total_target_columns': len(target_cols),
            'matched_columns': len([v for v in mapping.values() if v is not None]),
            'average_similarity': float(np.mean(similarities)) if similarities else 0.0,
            'min_similarity': float(np.min(similarities)) if similarities else 0.0,
            'max_similarity': float(np.max(similarities)) if similarities else 0.0,
            'threshold_used': threshold,
            'detailed_mapping': [
                {
                    'reference': ref_col.name,
                    'target': target_col.name if target_col else None,
                    'similarity': float(similarity_matrix[reference_cols.index(ref_col), 
                                       target_cols.index(target_col)]) if target_col else 0.0
                }
                for ref_col, target_col in mapping.items()
            ]
        }
        
        return mapping, metrics
