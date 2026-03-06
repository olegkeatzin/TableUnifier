"""
Конфигурация унифицированной генерации данных для Schema Matching и Entity Resolution.

Один конфиг контролирует:
- Общие шаблоны столбцов и расширения разнообразия
- Параметры генерации таблиц (EntityPool, пары, пертурбации)
- Параметры Ollama с автоматическим подбором batch_size
- Пути для сохранения SM + ER данных
"""

from dataclasses import dataclass, field, asdict
from typing import Dict, List, Any, Optional, Tuple
import json


@dataclass
class DataGenConfig:
    """Конфигурация генерации объединённого датасета SM + ER.
    
    Единый генератор создаёт:
    1. Пулл канонических сущностей (EntityPool)
    2. Пары таблиц с контролируемым пересечением (для ER)
    3. Описания + эмбеддинги столбцов из каждой таблицы (для SM)
    
    Ollama-ускорение:
    - auto_batch_size: автоматический подбор оптимального batch_size
    - keep_alive: модель остаётся загруженной в GPU между запросами
    - num_predict: ограничение длины генерации LLM
    - num_ctx: размер контекстного окна
    - llm_concurrency: параллельные запросы к LLM
    """
    
    # ─────────────────── Ollama модели ───────────────────
    ollama_host: str = "http://127.0.0.1:11434"
    """Адрес Ollama сервера"""
    
    llm_model: str = "qwen3.5:9b"
    """LLM модель для генерации описаний столбцов"""
    
    embedding_model: str = "qwen3-embedding:8b"
    """Эмбеддинг модель для векторизации описаний.
    Изменение модели автоматически перестроит размерности в SM конфиге."""
    
    # ─────────────────── Ollama ускорение ───────────────────
    auto_batch_size: bool = True
    """Автоматический подбор оптимального batch_size для embed API"""
    
    initial_batch_size: int = 8
    """Начальный batch_size (если auto_batch_size=False — используемый)"""
    
    max_batch_size: int = 512
    """Верхний предел batch_size при автоподборе"""
    
    min_batch_size: int = 1
    """Нижний предел batch_size"""
    
    keep_alive: str = "30m"
    """Время удержания модели в GPU (Ollama keep_alive). 
    '-1' = навсегда, '0' = выгрузить сразу, '30m' = 30 минут"""
    
    num_predict: int = 150
    """Максимум токенов для LLM генерации описания столбца"""
    
    num_ctx: int = 2048
    """Размер контекстного окна LLM (меньше = быстрее для коротких промптов)"""
    
    warmup: bool = True
    """Прогреть модели перед генерацией (загрузить в GPU)"""
    
    auto_llm_parallel: bool = True
    """Автоматический подбор оптимального числа параллельных LLM запросов"""
    
    max_llm_parallel: int = 16
    """Верхний предел num_parallel для LLM при автоподборе"""
    
    min_llm_parallel: int = 1
    """Нижний предел num_parallel для LLM"""
    
    initial_llm_parallel: int = 1
    """Начальный num_parallel (если auto_llm_parallel=False — используемый)"""
    
    # ─────────────────── Генерация таблиц ───────────────────
    locales: List[str] = field(default_factory=lambda: ["ru_RU", "en_US"])
    """Локали Faker для генерации данных"""
    
    min_rows_per_table: int = 5
    """Минимальное число строк в таблице"""
    
    max_rows_per_table: int = 15
    """Максимальное число строк в таблице"""
    
    min_optional_columns: int = 2
    """Минимум опциональных столбцов"""
    
    max_optional_columns: int = 8
    """Максимум опциональных столбцов"""
    
    column_name_variation_level: float = 0.7
    """Уровень вариативности названий столбцов (0-1)"""
    
    include_typos: bool = True
    """Включать опечатки в названия столбцов"""
    
    include_abbreviations: bool = True
    """Включать сокращения"""
    
    sample_size: int = 5
    """Количество примеров значений для генерации описания"""
    
    # ─────────────────── Entity Pool (для ER) ───────────────────
    num_entity_pool: int = 500
    """Размер пула канонических сущностей"""
    
    # ─────────────────── ER пары таблиц ───────────────────
    num_train_pairs: int = 300
    """Число пар таблиц для обучения ER"""
    
    num_val_pairs: int = 50
    """Число пар таблиц для валидации ER"""
    
    num_test_pairs: int = 50
    """Число пар таблиц для тестирования ER"""
    
    min_common_entities: int = 3
    """Минимальное число общих сущностей (дубликатов) в паре"""
    
    max_common_entities: int = 8
    """Максимальное число общих сущностей"""
    
    min_unique_entities: int = 1
    """Минимальное число уникальных сущностей в каждой таблице"""
    
    max_unique_entities: int = 5
    """Максимальное число уникальных сущностей"""
    
    perturbation_prob: float = 0.3
    """Вероятность пертурбации значения ячейки"""
    
    missing_value_prob: float = 0.1
    """Вероятность пропуска значения (NaN)"""
    
    # ─────────────────── SM дополнительно ───────────────────
    num_extra_sm_tables: int = 200
    """Число дополнительных standalone-таблиц для SM 
    (сверх тех, что генерируются для ER пар)"""
    
    # ─────────────── Расширение разнообразия датасета ───────────────
    extra_column_variants: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    """Дополнительные варианты названий для существующих столбцов.
    Формат: { "base_name": { "ru": ["вариант1", ...], "en": ["variant1", ...] } }
    """
    
    extra_product_groups: List[str] = field(default_factory=list)
    """Дополнительные категории/группы товаров"""
    
    extra_units: List[str] = field(default_factory=list)
    """Дополнительные единицы измерения"""
    
    extra_manufacturers: List[str] = field(default_factory=list)
    """Дополнительные производители/бренды"""
    
    extra_vat_rates: List[int] = field(default_factory=list)
    """Дополнительные ставки НДС"""
    
    extra_article_prefixes: List[str] = field(default_factory=list)
    """Дополнительные префиксы артикулов"""
    
    extra_product_name_parts: List[str] = field(default_factory=list)
    """Дополнительные модификаторы наименований"""
    
    extra_countries: List[str] = field(default_factory=list)
    """Дополнительные страны"""
    
    extra_notes_templates: List[str] = field(default_factory=list)
    """Дополнительные шаблоны примечаний"""
    
    custom_columns: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    """Пользовательские типы столбцов.
    Формат:
        {
            "base_name": {
                "description": "...",
                "variants": {"ru": [...], "en": [...]},
                "data_type": "str"|"int"|"float",
                "values": [...],
                "mandatory": false,
                "entity_column": true  # включить в EntityPool для ER
            }
        }
    """
    
    # ─────────────────── Выходные пути ───────────────────
    output_dir: str = "unified_dataset"
    """Корневая директория для всех данных"""
    
    sm_dataset_file: str = "sm_dataset.json"
    """Файл SM датасета (column embeddings + метки)"""
    
    sm_metadata_file: str = "sm_metadata.json"
    """Файл SM метаданных"""
    
    er_raw_dir: str = "raw"
    """Поддиректория для ER сырых данных (CSV пары + meta.json)"""
    
    # ─────────────────── Методы ───────────────────
    
    def save(self, filepath: str):
        """Сохранить конфигурацию в JSON"""
        data = asdict(self)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DataGenConfig':
        """Загрузить конфигурацию из JSON"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Поддержка двух форматов:
        # 1) Плоский DataGenConfig JSON (результат save)
        # 2) Экспериментальный JSON c обёрткой {"data_gen_config": {...}}
        if isinstance(data, dict) and isinstance(data.get("data_gen_config"), dict):
            data = data["data_gen_config"]

        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)
    
    @property
    def total_er_pairs(self) -> int:
        """Общее число ER пар."""
        return self.num_train_pairs + self.num_val_pairs + self.num_test_pairs
    
    @property
    def total_tables(self) -> int:
        """Прогноз общего числа таблиц (ER пары × 2 + extra SM)."""
        return self.total_er_pairs * 2 + self.num_extra_sm_tables
    
    def __repr__(self):
        lines = ["DataGenConfig("]
        for k, v in asdict(self).items():
            lines.append(f"  {k}={v!r},")
        lines.append(")")
        return "\n".join(lines)
