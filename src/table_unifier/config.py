"""
Конфигурационный файл для системы унификации таблиц
"""
from dataclasses import dataclass
from typing import Optional, List
import json
import logging

logger = logging.getLogger(__name__)

# ═══════════════ Реестр размерностей embedding-моделей ═══════════════

EMBEDDING_MODEL_DIMS: dict[str, int] = {
    "qwen3-embedding:8b": 4096,
    "embeddinggemma": 768,
    "nomic-embed-text": 768,
    "mxbai-embed-large": 1024,
    "snowflake-arctic-embed": 1024,
    "all-minilm": 384,
}
"""Известные embedding-модели и размерности их выходных векторов."""


def get_embedding_dim(model_name: str) -> int:
    """Получить размерность эмбеддинга по имени модели.
    
    Args:
        model_name: Имя Ollama embedding модели
        
    Returns:
        Размерность вектора эмбеддинга
        
    Raises:
        ValueError: Если модель неизвестна
    """
    if model_name in EMBEDDING_MODEL_DIMS:
        return EMBEDDING_MODEL_DIMS[model_name]
    
    # Проверяем по базовому имени (без тега, e.g. "qwen3-embedding" для "qwen3-embedding:8b")
    base_name = model_name.split(":")[0]
    for known, dim in EMBEDDING_MODEL_DIMS.items():
        if known.split(":")[0] == base_name:
            logger.warning(
                f"Точное совпадение для '{model_name}' не найдено, "
                f"используется '{known}' → {dim}"
            )
            return dim
    
    raise ValueError(
        f"Неизвестная embedding модель: '{model_name}'. "
        f"Известные модели: {list(EMBEDDING_MODEL_DIMS.keys())}. "
        f"Добавьте модель в EMBEDDING_MODEL_DIMS или задайте input_dim вручную."
    )


def get_default_projection_dims(input_dim: int, output_dim: int = 256) -> List[int]:
    """Автоматический расчёт размерностей projection head.
    
    Поэтапное уменьшение в 2 раза от input_dim до output_dim.
    
    Примеры:
        4096, 256 → [2048, 1024, 512]
        768, 256  → [384]
        1024, 256 → [512]
    """
    dims = []
    d = input_dim
    while d // 2 > output_dim:
        d = d // 2
        dims.append(d)
    return dims if dims else [(input_dim + output_dim) // 2]


@dataclass
class OllamaConfig:
    """Конфигурация подключения к Ollama"""
    host: str = "http://localhost:11434"
    llm_model: str = "qwen3.5:9b"
    embedding_model: str = "qwen3-embedding:8b"
    timeout: int = 120  # секунды
    

@dataclass
class EmbeddingConfig:
    """Конфигурация для работы с эмбеддингами"""
    batch_size: int = 10
    similarity_threshold: float = 0.5
    include_data_types: bool = True  # Учитывать типы данных при описании
    sample_size: int = 5  # Сколько примеров данных использовать
    

@dataclass
class ClassifierConfig:
    """Конфигурация классификатора"""
    update_alpha: float = 0.3  # Вес новых данных при обновлении
    min_similarity: float = 0.6  # Минимальное сходство для сопоставления
    auto_update: bool = False  # Автоматически обновлять эталонные эмбеддинги


@dataclass
class AppConfig:
    """Конфигурация приложения для TableUnifier (column embedding pipeline).
    
    Используется модулем core.py (TableUnifier) для подключения к Ollama
    и настройки обработки столбцов. Специализированные конфиги для SM и ER
    находятся в соответствующих подмодулях.
    """
    ollama: OllamaConfig = None
    embedding: EmbeddingConfig = None
    classifier: ClassifierConfig = None
    
    def __post_init__(self):
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.classifier is None:
            self.classifier = ClassifierConfig()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AppConfig':
        """Загрузить конфигурацию из JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            ollama=OllamaConfig(**data.get('ollama', {})),
            embedding=EmbeddingConfig(**data.get('embedding', {})),
            classifier=ClassifierConfig(**data.get('classifier', {})),
        )
    
    def to_file(self, filepath: str):
        """Сохранить конфигурацию в JSON файл"""
        data = {
            'ollama': self.ollama.__dict__,
            'embedding': self.embedding.__dict__,
            'classifier': self.classifier.__dict__,
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
