"""
Конфигурационный файл для системы унификации таблиц
"""
from dataclasses import dataclass
from typing import Optional
import json


@dataclass
class OllamaConfig:
    """Конфигурация подключения к Ollama"""
    host: str = "http://localhost:11434"
    llm_model: str = "gemma3:1b"
    embedding_model: str = "embeddinggemma"
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
class DatasetGenerationConfig:
    """Конфигурация генерации датасета"""
    num_tables: int = 500
    min_rows_per_table: int = 5
    max_rows_per_table: int = 20
    min_optional_columns: int = 2  # Минимум дополнительных столбцов
    max_optional_columns: int = 8  # Максимум дополнительных столбцов (всего 8 опциональных)
    num_workers: int = 16  # Количество параллельных потоков
    column_name_variation_level: float = 0.7
    include_typos: bool = True
    include_abbreviations: bool = True
    include_translations: bool = True
    output_dir: str = "generated_dataset_v3"
    tables_dir: str = "tables"
    dataset_file: str = "dataset_v3.json"
    metadata_file: str = "metadata_v3.json"
    locales: list = None
    
    def __post_init__(self):
        if self.locales is None:
            self.locales = ["ru_RU", "en_US"]
    

@dataclass
class AppConfig:
    """Главная конфигурация приложения"""
    ollama: OllamaConfig = None
    embedding: EmbeddingConfig = None
    classifier: ClassifierConfig = None
    dataset_generation: DatasetGenerationConfig = None
    
    def __post_init__(self):
        if self.ollama is None:
            self.ollama = OllamaConfig()
        if self.embedding is None:
            self.embedding = EmbeddingConfig()
        if self.classifier is None:
            self.classifier = ClassifierConfig()
        if self.dataset_generation is None:
            self.dataset_generation = DatasetGenerationConfig()
    
    @classmethod
    def from_file(cls, filepath: str) -> 'AppConfig':
        """Загрузить конфигурацию из JSON файла"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return cls(
            ollama=OllamaConfig(**data.get('ollama', {})),
            embedding=EmbeddingConfig(**data.get('embedding', {})),
            classifier=ClassifierConfig(**data.get('classifier', {})),
            dataset_generation=DatasetGenerationConfig(**data.get('dataset_generation', {}))
        )
    
    def to_file(self, filepath: str):
        """Сохранить конфигурацию в JSON файл"""
        data = {
            'ollama': self.ollama.__dict__,
            'embedding': self.embedding.__dict__,
            'classifier': self.classifier.__dict__,
            'dataset_generation': self.dataset_generation.__dict__
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# Конфигурация по умолчанию
DEFAULT_CONFIG = AppConfig()
