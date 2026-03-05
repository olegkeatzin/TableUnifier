"""
Конфигурация модуля Schema Matching с Triplet Loss.

Содержит все параметры для:
- Генерации данных (столбцы, описания, эмбеддинги)
- Модели проекции (MLP projection head)
- Обучения (triplet loss, mining, scheduler)
- Инференса (матчинг схем)
"""

from dataclasses import dataclass, field, asdict
from typing import Tuple, Optional, List, Dict, Any
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class SMConfig:
    """Конфигурация Schema Matching на базе Triplet Loss.
    
    Архитектура:
        1. Генерация описания столбца через LLM
        2. Эмбеддинг описания через embedding model
        3. Projection Head (MLP) обученный через Triplet Loss
        4. L2-нормализованные выходные вектора для косинусного сходства
        5. Венгерский алгоритм для оптимального сопоставления столбцов
    
    Связь с GNN:
        Выходные эмбеддинги используются как edge_attr (column embeddings)
        в Entity Resolution GNN модели.
    
    Автоматический подбор размерностей:
        При изменении embedding_model автоматически пересчитываются
        input_dim и projection_dims. Для ручного переопределения
        задайте их явно (не None).
    """
    
    # ─────────────────── Ollama модели ───────────────────
    ollama_host: str = "http://127.0.0.1:11434"
    """Адрес Ollama сервера"""
    
    llm_model: str = "qwen3.5:9b"
    """LLM модель для генерации описаний столбцов"""
    
    embedding_model: str = "qwen3-embedding:8b"
    """Embedding модель для векторизации описаний.
    Изменение модели автоматически пересчитает input_dim и projection_dims."""
    
    embedding_batch_size: int = 50
    """Размер батча для Ollama embed API"""
    
    # ─────────────────── Генерация данных ───────────────────
    num_tables: int = 1000
    """Количество таблиц для генерации"""
    
    min_rows_per_table: int = 5
    """Минимальное число строк в таблице"""
    
    max_rows_per_table: int = 15
    """Максимальное число строк в таблице"""
    
    min_optional_columns: int = 2
    """Минимум опциональных столбцов"""
    
    max_optional_columns: int = 8
    """Максимум опциональных столбцов"""
    
    num_workers: int = 8
    """Количество потоков для параллельной генерации"""
    
    column_name_variation_level: float = 0.7
    """Уровень вариативности названий столбцов (0-1)"""
    
    include_typos: bool = True
    """Включать опечатки в названия столбцов"""
    
    include_abbreviations: bool = True
    """Включать сокращения"""
    
    include_translations: bool = True
    """Включать англо-русские вариации"""
    
    sample_size: int = 5
    """Количество примеров значений для генерации описания"""
    
    locales: List[str] = field(default_factory=lambda: ["ru_RU", "en_US"])
    """Локали Faker для генерации данных"""
    
    # ─────────────── Расширение разнообразия датасета ───────────────
    extra_column_variants: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    """Дополнительные варианты названий для существующих столбцов.
    
    Формат: { "base_name": { "ru": ["вариант1", ...], "en": ["variant1", ...] } }
    Примеры:
        {
            "quantity": {"ru": ["Штук", "Кол-во единиц"], "en": ["No. of items"]},
            "base_price": {"ru": ["Тариф", "Расценка"], "en": ["Rate", "Tariff"]}
        }
    """
    
    extra_product_groups: List[str] = field(default_factory=list)
    """Дополнительные категории/группы товаров.
    Примеры: ["Медикаменты", "Продукты питания", "Химреактивы"]
    """
    
    extra_units: List[str] = field(default_factory=list)
    """Дополнительные единицы измерения.
    Примеры: ["т", "м²", "м³", "рулон", "бухта"]
    """
    
    extra_manufacturers: List[str] = field(default_factory=list)
    """Дополнительные производители/бренды.
    Примеры: ["КАМАЗ", "Газпром", "Северсталь"]
    """
    
    extra_vat_rates: List[int] = field(default_factory=list)
    """Дополнительные ставки НДС (в процентах).
    Примеры: [5, 7, 12]
    """
    
    extra_article_prefixes: List[str] = field(default_factory=list)
    """Дополнительные префиксы для артикулов.
    Примеры: ["МАТ", "ОБР", "ЗАП", "КОМ"]
    """
    
    extra_product_name_parts: List[str] = field(default_factory=list)
    """Дополнительные модификаторы для наименований товаров.
    Примеры: ["Turbo", "Industrial", "Heavy-Duty", "Compact"]
    """
    
    extra_countries: List[str] = field(default_factory=list)
    """Дополнительные страны (вместо/вместе с Faker).
    Примеры: ["Китай", "Россия", "Германия", "Турция"]
    """
    
    extra_notes_templates: List[str] = field(default_factory=list)
    """Дополнительные шаблоны для примечаний.
    Примеры: ["Срочная поставка", "Под заказ", "Со склада"]
    """
    
    custom_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Пользовательские типы столбцов (добавляются к встроенным).
    
    Формат:
        {
            "base_name": {
                "description": "Описание столбца",
                "variants": {"ru": [...], "en": [...]},
                "data_type": "str" | "int" | "float",
                "values": [...],           # список возможных значений
                "mandatory": false         # обязательный или опциональный
            }
        }
    Примеры:
        {
            "delivery_date": {
                "description": "Дата поставки товара",
                "variants": {
                    "ru": ["Дата поставки", "Дата доставки", "Срок поставки"],
                    "en": ["Delivery Date", "Ship Date", "ETA"]
                },
                "data_type": "str",
                "values": [],
                "mandatory": false
            },
            "warehouse": {
                "description": "Склад хранения",
                "variants": {
                    "ru": ["Склад", "Хранилище", "Место хранения"],
                    "en": ["Warehouse", "Storage", "Location"]
                },
                "data_type": "str",
                "values": ["Склад-1", "Склад-2", "Центральный", "Филиал"],
                "mandatory": false
            }
        }
    """
    
    # ─────────────────── Модель проекции ───────────────────
    input_dim: Optional[int] = None
    """Размерность входных эмбеддингов.
    None → автоматически определяется из embedding_model.
    При обучении перезаписывается фактической размерностью из данных."""
    
    projection_dims: Optional[List[int]] = None
    """Размерности скрытых слоёв проекции.
    None → автоматически рассчитываются из input_dim (поэтапное деление на 2)."""
    
    output_dim: int = 256
    """Размерность выходного эмбеддинга (для schema matching и GNN edge_attr)"""
    
    dropout: float = 0.1
    """Dropout для регуляризации"""
    
    use_batch_norm: bool = True
    """Использовать BatchNorm в projection head"""
    
    # ─────────────────── Обучение ───────────────────
    batch_size: int = 64
    """Размер батча"""
    
    learning_rate: float = 1e-3
    """Начальная скорость обучения"""
    
    weight_decay: float = 1e-5
    """L2-регуляризация"""
    
    num_epochs: int = 30
    """Максимальное число эпох"""
    
    margin: float = 0.2
    """Маржа для Triplet Loss"""
    
    mining_strategy: str = "semihard"
    """Стратегия майнинга триплетов: 'hard', 'semihard', 'all'"""
    
    scheduler_patience: int = 5
    """Patience для ReduceLROnPlateau"""
    
    early_stopping_patience: int = 10
    """Patience для early stopping"""
    
    train_ratio: float = 0.7
    """Доля данных для обучения"""
    
    val_ratio: float = 0.15
    """Доля данных для валидации"""
    
    test_ratio: float = 0.15
    """Доля данных для тестирования"""
    
    # ─────────────────── Инференс ───────────────────
    similarity_threshold: float = 0.6
    """Минимальное косинусное сходство для матчинга"""
    
    # ─────────────────── Пути ───────────────────
    output_dir: str = "sm_dataset"
    """Корневая директория для данных Schema Matching"""
    
    dataset_file: str = "sm_dataset.json"
    """Файл датасета (эмбеддинги + метки)"""
    
    metadata_file: str = "sm_metadata.json"
    """Файл метаданных генерации"""
    
    model_dir: str = "sm_models"
    """Директория для сохранения моделей"""
    
    def __post_init__(self):
        """Автоматический расчёт размерностей из embedding_model."""
        from ..config import get_embedding_dim, get_default_projection_dims
        
        if self.input_dim is None:
            self.input_dim = get_embedding_dim(self.embedding_model)
            logger.info(
                f"input_dim автоматически определён из '{self.embedding_model}': "
                f"{self.input_dim}"
            )
        
        if self.projection_dims is None:
            self.projection_dims = get_default_projection_dims(
                self.input_dim, self.output_dim
            )
            logger.info(
                f"projection_dims автоматически рассчитаны: "
                f"{self.input_dim} → {self.projection_dims} → {self.output_dim}"
            )
    
    def recalculate_dims(self):
        """Принудительно пересчитать input_dim и projection_dims из embedding_model."""
        from ..config import get_embedding_dim, get_default_projection_dims
        
        self.input_dim = get_embedding_dim(self.embedding_model)
        self.projection_dims = get_default_projection_dims(
            self.input_dim, self.output_dim
        )
        logger.info(
            f"Размерности пересчитаны для '{self.embedding_model}': "
            f"{self.input_dim} → {self.projection_dims} → {self.output_dim}"
        )
    
    def save(self, filepath: str):
        """Сохранить конфигурацию в JSON"""
        data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'SMConfig':
        """Загрузить конфигурацию из JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    def __repr__(self):
        lines = ["SMConfig("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
