"""
Генератор датасета V3 - фокус на разнообразии через Faker и шаблоны
Ключевые улучшения:
- Максимальное использование Faker для реалистичных данных
- Ground truth метаданные для тестирования
- Контролируемая вариативность названий столбцов
- Структурированный формат для валидации

Запуск:
    uv run python -m table_unifier.ds_gen_v3
    или
    uv run python -m table_unifier.ds_gen_v3 --config project_config.json
"""

import sys
import pandas as pd
import json
import random
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
import logging
from tqdm import tqdm
import hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import signal

# Отключаем verbose логирование от httpx (используется ollama)
logging.getLogger("httpx").setLevel(logging.WARNING)

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False
    print("⚠️  Faker не установлен! Запустите: pip install faker")
    print("   Генератор не будет работать без Faker.")
    exit(1)

from .models import OllamaEmbedding
from .config import AppConfig
from .core import TableUnifier

# --- НАСТРОЙКА ЛОГИРОВАНИЯ ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('ds_gen_v3.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


# --- КОНФИГУРАЦИЯ ---

@dataclass
class DatasetConfigV3:
    """Конфигурация генератора V3
    
    УСТАРЕЛО: Используйте AppConfig.dataset_generation из project_config.json
    Этот класс оставлен для обратной совместимости.
    """
    # Модели (DEPRECATED - используйте AppConfig.ollama)
    ollama_host: str = 'http://127.0.0.1:11434'
    embedding_model: str = 'embeddinggemma'
    
    # Генерация
    num_tables: int = 500
    min_rows_per_table: int = 5
    max_rows_per_table: int = 20
    min_optional_columns: int = 2  # Минимум дополнительных столбцов
    max_optional_columns: int = 8  # Максимум дополнительных столбцов
    
    # Производительность
    num_workers: int = 16  # Количество параллельных потоков
    
    # Вариативность
    column_name_variation_level: float = 0.7  # 0-1, насколько сильно варьировать названия
    include_typos: bool = True
    include_abbreviations: bool = True
    include_translations: bool = True  # Англо-русские вариации
    
    # Выходные данные
    output_dir: str = 'generated_dataset_v3'
    tables_dir: str = 'tables'
    dataset_file: str = 'dataset_v3.json'
    metadata_file: str = 'metadata_v3.json'
    
    # Батчинг
    batch_size: int = 20
    
    # Локали Faker
    locales: List[str] = None
    
    def __post_init__(self):
        if self.locales is None:
            self.locales = ['ru_RU', 'en_US']
    
    @classmethod
    def from_app_config(cls, app_config: AppConfig) -> 'DatasetConfigV3':
        """Создать DatasetConfigV3 из AppConfig"""
        ds_gen = app_config.dataset_generation
        return cls(
            ollama_host=app_config.ollama.host,
            embedding_model=app_config.ollama.embedding_model,
            num_tables=ds_gen.num_tables,
            min_rows_per_table=ds_gen.min_rows_per_table,
            max_rows_per_table=ds_gen.max_rows_per_table,
            min_optional_columns=getattr(ds_gen, 'min_optional_columns', 2),
            max_optional_columns=getattr(ds_gen, 'max_optional_columns', 8),
            num_workers=getattr(ds_gen, 'num_workers', 16),  # Default 16 if not in config
            column_name_variation_level=ds_gen.column_name_variation_level,
            include_typos=ds_gen.include_typos,
            include_abbreviations=ds_gen.include_abbreviations,
            include_translations=ds_gen.include_translations,
            output_dir=ds_gen.output_dir,
            tables_dir=ds_gen.tables_dir,
            dataset_file=ds_gen.dataset_file,
            metadata_file=ds_gen.metadata_file,
            locales=ds_gen.locales
        )
    
    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'DatasetConfigV3':
        with open(filepath, 'r', encoding='utf-8') as f:
            return cls(**json.load(f))


# --- БАЗА ЗНАНИЙ: СЕМАНТИЧЕСКИЕ СТОЛБЦЫ ---

# Каждый столбец имеет:
# - base_name: каноническое имя
# - variants: список возможных вариантов названия
# - faker_method: метод Faker для генерации данных
# - generator_func: пользовательская функция генерации (если нужна)

COLUMN_TEMPLATES = {
    'item_number': {
        'base_name': 'item_number',
        'description': 'Порядковый номер позиции в таблице',
        'variants': {
            'ru': ['№', '№ п/п', 'Номер', 'Поз.', 'Позиция', '#', 'N', 'Пункт', 'Номир', 'Позициа', '№№', 'Нумер'],
            'en': ['#', 'No.', 'Item No.', 'Position', 'Line', 'Seq', 'Positon', 'Iten No.']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: idx + 1
    },
    
    'article': {
        'base_name': 'article',
        'description': 'Артикул товара',
        'variants': {
            'ru': ['Артикул', 'Код товара', 'SKU', 'Номенклатурный номер', 'Код', 'Арт.', 'Артикуль', 'Код тавара', 'Артикл', 'Артик', 'Номенклатура'],
            'en': ['Article', 'SKU', 'Item Code', 'Product Code', 'Code', 'Articl', 'Artcle', 'Item Cod']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            f"{fake.random_uppercase_letter()}{fake.random_int(100, 999)}-{fake.random_int(10, 99)}",
            f"SKU-{fake.random_int(10000, 99999)}",
            f"{fake.random_int(1000, 9999)}.{fake.random_int(100, 999)}",
            f"{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}-{fake.random_int(1000, 9999)}",
            f"{random.choice(['АРТ', 'PRD', 'ITM'])}{fake.random_int(10000, 99999)}",
            f"{fake.random_int(100, 999)}/{fake.random_uppercase_letter()}/{fake.random_int(10, 99)}"
        ])
    },
    
    'product_name': {
        'base_name': 'product_name',
        'description': 'Наименование товара',
        'variants': {
            'ru': ['Наименование товара', 'Товар', 'Описание', 'Номенклатура', 'Наименование', 'Название', 'Наименавание', 'Тавар', 'Наим-е', 'Наимен.', 'Назв.'],
            'en': ['Product Name', 'Item', 'Description', 'Name', 'Item Description', 'Product Nam', 'Itm Name', 'Prod Name']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            f"{fake.word().capitalize()} {fake.company_suffix()}",
            f"{fake.word().capitalize()} {random.choice(['Premium', 'Standard', 'Economy', 'Pro', 'Basic', 'Deluxe'])}",
            f"{random.choice(['Комплект', 'Набор', 'Система', 'Модуль', 'Блок'])} {fake.word().capitalize()}",
            f"{fake.word().capitalize()} {random.choice(['2000', '3000', 'Ultra', 'Plus', 'Light', 'Max'])}",
            f"{random.choice(['Универсальный', 'Профессиональный', 'Бытовой', 'Промышленный'])} {fake.word()}"
        ])
    },
    
    'product_group': {
        'base_name': 'product_group',
        'description': 'Категория или группа товара',
        'variants': {
            'ru': ['Товарная группа', 'Категория', 'Группа', 'Раздел', 'Тип товара', 'Категориа', 'Катег.', 'Групп', 'Товар.группа'],
            'en': ['Product Group', 'Category', 'Type', 'Group', 'Class', 'Categry', 'Grp', 'Prod Group']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            'Электроника', 'Мебель', 'Инструменты', 'Канцелярия', 'Стройматериалы',
            'Сантехника', 'Автозапчасти', 'Офисная техника', 'Хозтовары', 'Текстиль'
        ])
    },
    
    'manufacturer': {
        'base_name': 'manufacturer',
        'description': 'Производитель товара',
        'variants': {
            'ru': ['Производитель', 'Бренд', 'Торговая марка', 'Изготовитель', 'Завод-изготовитель', 'Производиттель', 'Произв.', 'Произв-ль', 'Бренд/Производитель'],
            'en': ['Manufacturer', 'Brand', 'Producer', 'Maker', 'Vendor', 'Manufactur', 'Mnfr', 'Mfr']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            fake.company(),
            f"{fake.company()} {random.choice(['Group', 'Corp', 'Ltd', 'Inc', 'GmbH', 'ООО', 'ЗАО'])}",
            f"{fake.last_name()} {random.choice(['& Co', 'Industries', 'Manufacturing', 'Factory'])}",
            random.choice(['Samsung', 'LG', 'Sony', 'Panasonic', 'Bosch', 'Siemens', 'Philips', 'Hitachi', 'Toshiba']) + random.choice(['', ' Electronics', ' Industrial', ' Professional'])
        ])
    },
    
    'country_of_origin': {
        'base_name': 'country_of_origin',
        'description': 'Страна происхождения товара',
        'variants': {
            'ru': ['Страна происхождения', 'Страна', 'Производство', 'Страна изготовления'],
            'en': ['Country of Origin', 'Country', 'Made in', 'Origin']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: fake.country()
    },
    
    'quantity': {
        'base_name': 'quantity',
        'description': 'Количество единиц товара',
        'variants': {
            'ru': ['Количество', 'Кол-во', 'Qty', 'Общее кол-во', 'К-во', 'Объем', 'Колличество', 'Кол.', 'Колво', 'Кол', 'Кол-во шт'],
            'en': ['Quantity', 'Qty', 'Amount', 'Count', 'Total Qty', 'Qnty', 'Quanty', 'Qtt']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: random.choice([
            fake.random_int(1, 5),      # Малое количество
            fake.random_int(10, 50),    # Среднее количество  
            fake.random_int(50, 200),   # Большое количество
            fake.random_int(1, 10) * 10, # Круглые числа
        ])
    },
    
    'unit': {
        'base_name': 'unit',
        'description': 'Единица измерения',
        'variants': {
            'ru': ['Ед. изм.', 'Единица', 'Ед.', 'ЕИ', 'Размерность', 'Ед.изм', 'Едизм', 'Единиц.', 'Ед измер'],
            'en': ['Unit', 'UoM', 'Unit of Measure', 'Measurement', 'Unt', 'Mesure', 'UOM']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice(['шт.', 'кг', 'л', 'м', 'упак.', 'компл.', 'пар'])
    },
    
    'base_price': {
        'base_name': 'base_price',
        'description': 'Базовая цена за единицу',
        'variants': {
            'ru': ['Цена', 'Базовая цена', 'Цена за ед.', 'Стоимость', 'Прайс', 'Цена по прайсу', 'Ценна', 'Цена/ед', 'Цена ед.', 'Цена₽'],
            'en': ['Price', 'Base Price', 'Unit Price', 'Cost', 'List Price', 'Prise', 'Prce', 'Unit Prc']
        },
        'data_type': 'float',
        'generator': lambda fake, idx: round(
            random.choice([
                fake.random.uniform(50, 500),      # Дешевые товары
                fake.random.uniform(500, 5000),    # Средний ценовой сегмент
                fake.random.uniform(5000, 50000),  # Дорогие товары
                fake.random.uniform(99, 999),      # Цены с красивыми окончаниями
            ]) * random.choice([1, 0.99, 1.15, 0.95]),  # Вариации цен
            2
        )
    },
    
    'discount_percent': {
        'base_name': 'discount_percent',
        'description': 'Процент скидки',
        'variants': {
            'ru': ['Скидка %', '% скидки', 'Скидка', 'Дисконт %', 'Процент скидки', 'Скидко %', '% скидк', 'Скид %', 'Скидка%'],
            'en': ['Discount %', 'Discount', '% Off', 'Disc %', 'Discnt %', 'Disct', 'Disc']
        },
        'data_type': 'float',
        'generator': lambda fake, idx: random.choice([0, 0, 0, 5, 10, 15, 20])
    },
    
    'price_after_discount': {
        'base_name': 'price_after_discount',
        'description': 'Цена со скидкой',
        'variants': {
            'ru': ['Цена со скидкой', 'Отпускная цена', 'Цена продажи', 'Финальная цена'],
            'en': ['Discounted Price', 'Sale Price', 'Final Price', 'Net Price']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    
    'vat_rate': {
        'base_name': 'vat_rate',
        'description': 'Ставка НДС в процентах',
        'variants': {
            'ru': ['Ставка НДС', '% НДС', 'НДС %', 'Налог'],
            'en': ['VAT Rate', 'VAT %', 'Tax Rate', 'Tax %']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: random.choice([0, 10, 20])
    },
    
    'total_sum': {
        'base_name': 'total_sum',
        'description': 'Общая стоимость без НДС',
        'variants': {
            'ru': ['Сумма', 'Итого', 'Сумма без НДС', 'Стоимость', 'Всего', 'Сумм', 'Итого₽', 'Сум.', 'Стоим.'],
            'en': ['Total', 'Sum', 'Amount', 'Total Amount', 'Subtotal', 'Totl', 'Amnt', 'Tot']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    
    'vat_amount': {
        'base_name': 'vat_amount',
        'description': 'Сумма НДС',
        'variants': {
            'ru': ['Сумма НДС', 'НДС', 'В т.ч. НДС', 'Налог (сумма)'],
            'en': ['VAT Amount', 'VAT', 'Tax Amount', 'Tax']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    
    'total_with_vat': {
        'base_name': 'total_with_vat',
        'description': 'Общая стоимость с НДС',
        'variants': {
            'ru': ['Всего с НДС', 'Итого с НДС', 'К оплате', 'Сумма к оплате'],
            'en': ['Total with VAT', 'Grand Total', 'Total Incl. VAT', 'Amount Due']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    
    'notes': {
        'base_name': 'notes',
        'description': 'Дополнительные примечания',
        'variants': {
            'ru': ['Примечание', 'Комментарий', 'Доп. информация', 'Заметки', 'Прим.'],
            'en': ['Notes', 'Comments', 'Remarks', 'Additional Info']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: fake.sentence(nb_words=6) if random.random() > 0.5 else ''
    }
}


# Обязательные столбцы для каждой таблицы
MANDATORY_COLUMNS = [
    'item_number', 'article', 'product_name', 'product_group',
    'quantity', 'unit', 'base_price', 'total_sum'
]

# Дополнительные столбцы (добавляются случайным образом)
OPTIONAL_COLUMNS = [
    'manufacturer', 'country_of_origin', 'discount_percent',
    'price_after_discount', 'vat_rate', 'vat_amount', 'total_with_vat', 'notes'
]


# --- ГЕНЕРАТОР ВАРИАНТОВ НАЗВАНИЙ ---

class ColumnNameVariator:
    """Генерирует вариации названий столбцов"""
    
    def __init__(self, config: DatasetConfigV3):
        self.config = config
        self.variation_level = config.column_name_variation_level
    
    def add_typos(self, text: str) -> str:
        """Добавляет СЛУЧАЙНЫЕ опечатки"""
        if not self.config.include_typos or random.random() > 0.5:
            return text
        
        # Конвертируем в список для изменения
        chars = list(text)
        
        # Типы случайных опечаток
        typo_type = random.choice(['swap', 'double', 'skip', 'replace'])
        
        if len(chars) < 3:
            return text
        
        # Выбираем случайную позицию (не первая и не последняя буква)
        pos = random.randint(1, len(chars) - 2)
        
        if typo_type == 'swap' and pos < len(chars) - 1:
            # Поменять соседние буквы местами: "номер" → "нмоер"
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        
        elif typo_type == 'double':
            # Удвоить букву: "количество" → "колличество"
            if chars[pos].isalpha():
                chars.insert(pos, chars[pos])
        
        elif typo_type == 'skip':
            # Пропустить букву: "товар" → "тoвар"
            if chars[pos].isalpha() and pos > 0:
                chars.pop(pos)
        
        elif typo_type == 'replace':
            # Заменить на соседнюю клавишу
            if chars[pos].isalpha():
                # Русская и английская раскладки: соседние клавиши
                keyboard_neighbors = {
                    # Русская раскладка
                    'а': 'ф', 'о': 'р', 'е': 'у', 'и': 'т', 'т': 'и',
                    'н': 'т', 'р': 'о', 'с': 'в', 'в': 'а', 'л': 'д',
                    'к': 'е', 'м': 'и', 'д': 'л', 'п': 'р', 'у': 'к',
                    'я': 'ч', 'ч': 'с', 'ы': 'в', 'ц': 'ы', 'й': 'ц',
                    'ф': 'а', 'б': 'н', 'ь': 'б', 'ж': 'д', 'э': 'ю',
                    'х': 'ъ', 'ъ': 'х', 'ю': 'б', 'з': 'я', 'ш': 'щ',
                    'щ': 'ш', 'г': 'н', 'ё': 'й',
                    # Английская раскладка (QWERTY)
                    'q': 'w', 'w': 'q', 'e': 'r', 'r': 't', 't': 'y',
                    'y': 'u', 'u': 'i', 'i': 'o', 'o': 'p', 'p': 'o',
                    'a': 's', 's': 'd', 'd': 'f', 'f': 'g', 'g': 'h',
                    'h': 'j', 'j': 'k', 'k': 'l', 'l': 'k',
                    'z': 'x', 'x': 'c', 'c': 'v', 'v': 'b', 'b': 'n',
                    'n': 'm', 'm': 'n'
                }
                lower_char = chars[pos].lower()
                if lower_char in keyboard_neighbors:
                    replacement = keyboard_neighbors[lower_char]
                    if chars[pos].isupper():
                        replacement = replacement.upper()
                    chars[pos] = replacement
        
        return ''.join(chars)
    
    def add_abbreviation(self, text: str) -> str:
        """Добавляет сокращения"""
        if not self.config.include_abbreviations or random.random() > 0.5:  # Увеличил с 0.4 до 0.5
            return text
        
        abbrevs = {
            'Количество': 'Кол-во',
            'Номер': '№',
            'Единица измерения': 'Ед. изм.',
            'Примечание': 'Прим.',
            'Артикул': 'Арт.',
            'Производитель': 'Произв.',
            'Наименование': 'Наим.',
            'Товар': 'Тов.',
            'Цена': 'Ц.',
            'Скидка': 'Ск.'
        }
        
        for full, abbr in abbrevs.items():
            if full in text:
                return text.replace(full, abbr)
        
        return text
    
    def add_spacing_variations(self, text: str) -> str:
        """Добавляет вариации с пробелами/подчеркиваниями"""
        if random.random() > 0.4:  # Увеличил с 0.3 до 0.4
            return text
        
        variations = [
            text.replace(' ', '_'),
            text.replace(' ', ''),
            text.replace(' ', '.'),
            text.replace(' ', '-'),
        ]
        
        return random.choice([text] + variations)
    
    def add_case_variations(self, text: str) -> str:
        """Меняет регистр"""
        if random.random() > 0.4:  # Увеличил с 0.3 до 0.4
            return text
        
        return random.choice([
            text.upper(),
            text.lower(),
            text.title(),
            text.capitalize()
        ])
    
    def add_extra_chars(self, text: str) -> str:
        """Добавляет лишние символы или пунктуацию"""
        if random.random() > 0.3:
            return text
        
        variations = [
            f"{text}:",
            f"{text}.",
            f"({text})",
            f"[{text}]",
            f"{text} ",  # Пробел в конце
            f" {text}",  # Пробел в начале
            f"{text}*",
            f"{text}_"
        ]
        
        return random.choice(variations)
    
    def generate_variant(self, base_variants: List[str], locale: str = 'ru') -> str:
        """Генерирует вариант названия столбца"""
        # Выбираем базовый вариант
        variant = random.choice(base_variants)
        
        # Применяем трансформации в зависимости от уровня вариативности
        if random.random() < self.variation_level:
            variant = self.add_typos(variant)
        
        if random.random() < self.variation_level:
            variant = self.add_abbreviation(variant)
        
        if random.random() < self.variation_level * 0.6:  # Увеличил с 0.5
            variant = self.add_spacing_variations(variant)
        
        if random.random() < self.variation_level * 0.5:  # Увеличил с 0.3
            variant = self.add_case_variations(variant)
        
        if random.random() < self.variation_level * 0.4:  # НОВОЕ
            variant = self.add_extra_chars(variant)
        
        return variant


# --- ГЕНЕРАТОР ТАБЛИЦ ---

class TableGeneratorV3:
    """Генератор таблиц с использованием Faker"""
    
    def __init__(self, config: DatasetConfigV3):
        self.config = config
        self.fakers = {locale: Faker(locale) for locale in config.locales}
        self.name_variator = ColumnNameVariator(config)
        
        # Статистика
        self.stats = {
            'total_generated': 0,
            'rows_generated': 0,
            'columns_generated': 0
        }
    
    def select_columns(self) -> List[str]:
        """Выбирает столбцы для таблицы"""
        # Всегда включаем обязательные
        selected = MANDATORY_COLUMNS.copy()
        
        # Добавляем случайные опциональные (по настройкам конфига)
        min_opt = self.config.min_optional_columns
        max_opt = min(self.config.max_optional_columns, len(OPTIONAL_COLUMNS))
        num_optional = random.randint(min_opt, max_opt)
        selected.extend(random.sample(OPTIONAL_COLUMNS, num_optional))
        
        return selected
    
    def generate_column_name(self, base_name: str) -> Tuple[str, str]:
        """
        Генерирует название столбца
        
        Returns:
            (display_name, locale) - название для отображения и использованная локаль
        """
        template = COLUMN_TEMPLATES[base_name]
        
        # Выбираем локаль
        locale = random.choice(self.config.locales)
        locale_key = 'ru' if 'ru' in locale else 'en'
        
        # Выбираем базовые варианты
        base_variants = template['variants'][locale_key]
        
        # Генерируем вариант
        display_name = self.name_variator.generate_variant(base_variants, locale_key)
        
        return display_name, locale_key
    
    def generate_row_data(self, columns: List[str], fake: Faker, row_idx: int) -> Dict:
        """Генерирует данные одной строки"""
        row = {}
        
        # Генерируем базовые значения
        for col_base_name in columns:
            template = COLUMN_TEMPLATES[col_base_name]
            
            if template.get('is_calculated'):
                continue  # Вычисляемые поля обработаем отдельно
            
            if 'generator' in template:
                value = template['generator'](fake, row_idx)
                row[col_base_name] = value
        
        # Вычисляем производные поля (только те, что в columns)
        self._calculate_derived_fields(row, columns)
        
        return row
    
    def _calculate_derived_fields(self, row: Dict, columns: List[str]):
        """Вычисляет производные поля (только если они есть в columns)"""
        # Цена со скидкой
        if ('price_after_discount' in columns and 
            'base_price' in row and 'discount_percent' in row):
            base_price = row['base_price']
            discount = row['discount_percent']
            row['price_after_discount'] = round(base_price * (1 - discount / 100), 2)
        
        # Используем цену со скидкой если есть, иначе базовую
        price = row.get('price_after_discount', row.get('base_price', 0))
        quantity = row.get('quantity', 0)
        
        # Сумма без НДС
        if 'total_sum' in columns:
            row['total_sum'] = round(price * quantity, 2)
        
        # НДС
        if 'vat_rate' in row:
            vat_rate = row['vat_rate']
            if 'vat_amount' in columns:
                row['vat_amount'] = round(row.get('total_sum', price * quantity) * vat_rate / 100, 2)
            if 'total_with_vat' in columns:
                total = row.get('total_sum', price * quantity)
                vat = row.get('vat_amount', total * vat_rate / 100)
                row['total_with_vat'] = round(total + vat, 2)
    
    def generate_table(self) -> Tuple[pd.DataFrame, Dict]:
        """
        Генерирует таблицу
        
        Returns:
            (dataframe, ground_truth) где ground_truth содержит маппинг display_name -> base_name
        """
        # Выбираем столбцы
        selected_columns = self.select_columns()
        
        # Генерируем названия столбцов и создаем ground truth
        column_mapping = {}  # display_name -> base_name
        display_names = []
        
        for base_name in selected_columns:
            display_name, locale = self.generate_column_name(base_name)
            
            # Обеспечиваем уникальность display_name
            original_display = display_name
            counter = 1
            while display_name in display_names:
                display_name = f"{original_display}_{counter}"
                counter += 1
            
            display_names.append(display_name)
            column_mapping[display_name] = base_name
        
        # Генерируем строки
        num_rows = random.randint(self.config.min_rows_per_table, self.config.max_rows_per_table)
        
        # Выбираем Faker для этой таблицы
        locale = random.choice(self.config.locales)
        fake = self.fakers[locale]
        
        rows = []
        for i in range(num_rows):
            row_data = self.generate_row_data(selected_columns, fake, i)
            rows.append(row_data)
        
        # Создаем DataFrame с base_name столбцами
        df_base = pd.DataFrame(rows)
        
        # Оставляем только те столбцы, которые есть в selected_columns
        # (убираем случайно добавленные в _calculate_derived_fields)
        existing_cols = [col for col in selected_columns if col in df_base.columns]
        df_base = df_base[existing_cols]
        
        # Переименовываем столбцы в display_name
        rename_map = {base_name: display_name 
                     for display_name, base_name in column_mapping.items()
                     if base_name in existing_cols}
        df = df_base.rename(columns=rename_map)
        
        # Упорядочиваем столбцы (только те, что действительно есть)
        final_display_names = [name for name in display_names 
                              if column_mapping[name] in existing_cols]
        df = df[final_display_names]
        
        # Обновляем статистику
        self.stats['total_generated'] += 1
        self.stats['rows_generated'] += num_rows
        self.stats['columns_generated'] += len(selected_columns)
        
        # Ground truth содержит информацию о маппинге
        ground_truth = {
            'column_mapping': column_mapping,  # display_name -> base_name
            'num_rows': num_rows,
            'locale': locale,
            'selected_columns': selected_columns
        }
        
        return df, ground_truth
    
    def get_stats(self) -> Dict:
        """Возвращает статистику"""
        return self.stats.copy()


# --- ГЕНЕРАТОР ДАТАСЕТА ---

class DatasetGeneratorV3:
    """Главный класс генератора датасета V3"""
    
    def __init__(self, config: DatasetConfigV3):
        self.config = config
        self.setup_environment()
        self.table_generator = TableGeneratorV3(config)
        
        # Инициализация TableUnifier для создания эмбеддингов
        app_config = AppConfig()
        app_config.ollama.host = config.ollama_host
        app_config.ollama.embedding_model = config.embedding_model
        app_config.embedding.batch_size = config.batch_size
        
        self.unifier = TableUnifier(app_config)
        
        # Для потокобезопасности
        self.lock = threading.Lock()
        
        # Для graceful shutdown
        self.interrupted = False
        self.all_dataset_entries = []
        self.all_metadata = []
        
        logger.info("DatasetGeneratorV3 инициализирован")
        logger.info(f"  Потоков для генерации: {self.config.num_workers}")
    
    def setup_environment(self):
        """Создает необходимые директории"""
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        tables_path = os.path.join(self.config.output_dir, self.config.tables_dir)
        os.makedirs(tables_path, exist_ok=True)
        
        logger.info(f"Директории созданы: {self.config.output_dir}")
    
    def save_table(self, df: pd.DataFrame, ground_truth: Dict, table_id: int) -> str:
        """Сохраняет таблицу в Excel с метаданными"""
        filename = f"table_{table_id:06d}.xlsx"
        filepath = os.path.join(self.config.output_dir, self.config.tables_dir, filename)
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Лист 1: Ground Truth (метаданные)
            metadata_rows = [
                ['KEY', 'VALUE'],
                ['table_id', table_id],
                ['num_rows', ground_truth['num_rows']],
                ['locale', ground_truth['locale']],
                ['generated_at', datetime.now().isoformat()],
                ['', ''],
                ['COLUMN MAPPING', ''],
                ['Display Name', 'Base Name (Ground Truth)']
            ]
            
            for display_name, base_name in ground_truth['column_mapping'].items():
                metadata_rows.append([display_name, base_name])
            
            metadata_df = pd.DataFrame(metadata_rows)
            metadata_df.to_excel(writer, sheet_name='Metadata', index=False, header=False)
            
            # Лист 2: Данные
            df.to_excel(writer, sheet_name='Data', index=False)
        
        return filename
    
    def process_table_to_dataset(self, df: pd.DataFrame, ground_truth: Dict, 
                                 table_filename: str) -> List[Dict]:
        """
        Обрабатывает таблицу и создает записи для датасета
        
        Returns:
            Список записей, каждая содержит:
            - column_name: название столбца в таблице (display name)
            - ground_truth_name: каноническое название (base_name)
            - embedding: вектор эмбеддинга
            - description: описание столбца
            - data_type: тип данных
            - content_sample: примеры данных
            - table_id: путь к файлу таблицы
        """
        # Получаем эмбеддинги через TableUnifier (БЕЗ БЛОКИРОВКИ - параллельно)
        emb_columns = self.unifier.process_dataframe(df)
        
        entries = []
        
        for emb_col in emb_columns:
            display_name = emb_col.name
            
            # Находим ground truth
            base_name = ground_truth['column_mapping'].get(display_name)
            
            if not base_name:
                logger.warning(f"Не найден ground truth для столбца '{display_name}'")
                continue
            
            # Получаем шаблон для дополнительной информации
            template = COLUMN_TEMPLATES.get(base_name, {})
            
            entry = {
                'column_name': display_name,
                'ground_truth_name': base_name,
                'ground_truth_description': template.get('description', ''),
                'embedding': emb_col.embedding.tolist(),
                'llm_description': emb_col.description,
                'data_type': emb_col.data_type,
                'content_sample': emb_col.content[:10],  # Первые 10 элементов
                'table_id': table_filename
            }
            
            entries.append(entry)
        
        return entries
    
    def generate_single_table(self, table_id: int) -> Tuple[Optional[Dict], Optional[List]]:
        """Генерирует одну таблицу (для многопоточности)"""
        try:
            # Генерируем таблицу
            df, ground_truth = self.table_generator.generate_table()
            
            # Сохраняем таблицу
            table_filename = self.save_table(df, ground_truth, table_id)
            
            # Обрабатываем в датасет
            entries = self.process_table_to_dataset(df, ground_truth, table_filename)
            
            # Метаданные
            metadata_entry = {
                'table_id': table_id,
                'table_filename': table_filename,
                'num_rows': ground_truth['num_rows'],
                'num_columns': len(ground_truth['column_mapping']),
                'column_mapping': ground_truth['column_mapping'],
                'locale': ground_truth['locale'],
                'base_columns': ground_truth['selected_columns']
            }
            
            return metadata_entry, entries
            
        except Exception as e:
            logger.error(f"Ошибка генерации таблицы {table_id}: {e}")
            return None, None
    
    def save_partial_results(self, successful: int, failed: int):
        """Сохраняет промежуточные результаты"""
        logger.info("\n" + "="*70)
        logger.info("⚠️  ПРЕРЫВАНИЕ: Сохранение промежуточных результатов...")
        logger.info("="*70)
        
        # Сортируем по table_id
        self.all_metadata.sort(key=lambda x: x['table_id'])
        
        # Сохраняем датасет
        dataset_path = os.path.join(self.config.output_dir, self.config.dataset_file)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_dataset_entries, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Датасет сохранен: {dataset_path}")
        logger.info(f"  Записей: {len(self.all_dataset_entries)}")
        
        # Сохраняем метаданные
        metadata_path = os.path.join(self.config.output_dir, self.config.metadata_file)
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_metadata, f, ensure_ascii=False, indent=2)
        
        logger.info(f"✓ Метаданные сохранены: {metadata_path}")
        logger.info(f"  Таблиц успешно: {successful}")
        logger.info(f"  Таблиц с ошибками: {failed}")
        logger.info("="*70)
    
    def generate(self):
        """Генерирует полный датасет С МНОГОПОТОЧНОСТЬЮ"""
        logger.info(f"Начало генерации {self.config.num_tables} таблиц")
        logger.info(f"🚀 Используется {self.config.num_workers} параллельных потоков")
        logger.info(f"💡 Нажмите Ctrl+C для остановки с сохранением результатов")
        
        from time import time
        start_time = time()
        
        successful = 0
        failed = 0
        
        # Обработчик прерывания
        def signal_handler(signum, frame):
            logger.warning("\n⚠️  Получен сигнал прерывания (Ctrl+C)")
            self.interrupted = True
        
        # Устанавливаем обработчик (Windows и Unix)
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            # Многопоточная генерация
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                # Отправляем задачи
                futures = {executor.submit(self.generate_single_table, table_id): table_id 
                          for table_id in range(self.config.num_tables)}
                
                # Собираем результаты с прогресс-баром
                with tqdm(total=self.config.num_tables, desc="Генерация таблиц") as pbar:
                    for future in as_completed(futures):
                        # Проверка на прерывание
                        if self.interrupted:
                            logger.warning("Отменяем оставшиеся задачи...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        
                        metadata, entries = future.result()
                        
                        if metadata and entries:
                            with self.lock:
                                self.all_metadata.append(metadata)
                                self.all_dataset_entries.extend(entries)
                                successful += 1
                        else:
                            with self.lock:
                                failed += 1
                        
                        pbar.set_postfix({'✓': successful, '✗': failed})
                        pbar.update(1)
        
        except KeyboardInterrupt:
            logger.warning("\n⚠️  KeyboardInterrupt перехвачен")
            self.interrupted = True
        
        finally:
            # Всегда сохраняем результаты
            if self.interrupted:
                self.save_partial_results(successful, failed)
                logger.info("\n✅ Генерация прервана пользователем. Промежуточные данные сохранены.")
                return
            
            # Обычное сохранение
            # Сортируем по table_id для консистентности
            self.all_metadata.sort(key=lambda x: x['table_id'])
            
            # Сохраняем датасет
            dataset_path = os.path.join(self.config.output_dir, self.config.dataset_file)
            with open(dataset_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_dataset_entries, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Датасет сохранен: {dataset_path}")
            logger.info(f"  Записей в датасете: {len(self.all_dataset_entries)}")
            
            # Сохраняем метаданные
            metadata_path = os.path.join(self.config.output_dir, self.config.metadata_file)
            with open(metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.all_metadata, f, ensure_ascii=False, indent=2)
            
            logger.info(f"Метаданные сохранены: {metadata_path}")
            
            # Итоговая статистика
            elapsed_time = time() - start_time
            self.print_summary(successful, failed, elapsed_time, len(self.all_dataset_entries))
    
    def print_summary(self, successful: int, failed: int, 
                     elapsed_time: float, total_entries: int):
        """Выводит итоговую статистику"""
        total = successful + failed
        success_rate = successful / total * 100 if total > 0 else 0
        
        gen_stats = self.table_generator.get_stats()
        
        logger.info("\n" + "="*70)
        logger.info("ИТОГИ ГЕНЕРАЦИИ ДАТАСЕТА V3")
        logger.info("="*70)
        logger.info(f"Успешно создано таблиц:     {successful} ({success_rate:.1f}%)")
        logger.info(f"Не удалось:                 {failed}")
        logger.info(f"Время:                      {elapsed_time:.1f} сек ({elapsed_time/60:.1f} мин)")
        logger.info(f"Скорость:                   {successful/elapsed_time:.2f} таблиц/сек")
        logger.info("")
        logger.info(f"Всего строк:                {gen_stats['rows_generated']}")
        logger.info(f"Всего столбцов:             {gen_stats['columns_generated']}")
        logger.info(f"Записей в датасете:         {total_entries}")
        logger.info(f"Среднее столбцов на таблицу: {gen_stats['columns_generated']/successful:.1f}")
        logger.info(f"Среднее строк на таблицу:    {gen_stats['rows_generated']/successful:.1f}")
        logger.info("")
        logger.info(f"Выходная директория:        {self.config.output_dir}")
        logger.info(f"Датасет:                    {self.config.dataset_file}")
        logger.info(f"Метаданные:                 {self.config.metadata_file}")
        logger.info("="*70 + "\n")


# --- MAIN ---

def main():
    """Главная функция"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Генератор датасета V3 с Faker')
    parser.add_argument('--config', type=str, default='project_config.json',
                       help='Путь к конфигурационному файлу проекта')
    parser.add_argument('--legacy-config', type=str, default=None,
                       help='Путь к старому конфигу V3 (ds_gen_config_v3.json)')
    
    args = parser.parse_args()
    
    # Проверка Faker
    if not FAKER_AVAILABLE:
        logger.error("Faker обязателен для работы генератора V3!")
        logger.error("Установите: pip install faker")
        return
    
    # Загрузка конфигурации
    if args.legacy_config and os.path.exists(args.legacy_config):
        logger.info(f"⚠️  Используется устаревший формат конфигурации: {args.legacy_config}")
        logger.info("   Рекомендуется перейти на project_config.json")
        config = DatasetConfigV3.load(args.legacy_config)
    elif os.path.exists(args.config):
        logger.info(f"✅ Загрузка конфигурации из {args.config}")
        app_config = AppConfig.from_file(args.config)
        logger.info(f"   Ollama host: {app_config.ollama.host}")
        logger.info(f"   Embedding model: {app_config.ollama.embedding_model}")
        logger.info(f"   Количество таблиц: {app_config.dataset_generation.num_tables}")
        
        # Формируем имя папки датасета
        num_tables = app_config.dataset_generation.num_tables
        llm_model = app_config.ollama.llm_model.replace('/', '_').replace(':', '_')
        emb_model = app_config.ollama.embedding_model.replace('/', '_').replace(':', '_')
        dataset_dir_name = f"dataset_{num_tables}t_{llm_model}_{emb_model}"
        
        # Обновляем путь в конфиге
        app_config.dataset_generation.output_dir = dataset_dir_name
        
        config = DatasetConfigV3.from_app_config(app_config)
        
        # Проверка доступности Ollama
        logger.info("\n🔍 Проверка подключения к Ollama...")
        try:
            import ollama
            client = ollama.Client(host=app_config.ollama.host)
            # Попытка получить список моделей
            models_response = client.list()
            
            # Извлечение списка моделей (может быть dict с ключом 'models' или другой формат)
            if isinstance(models_response, dict):
                models_list = models_response.get('models', [])
            else:
                models_list = models_response if isinstance(models_response, list) else []
            
            logger.info(f"✅ Ollama доступен, найдено моделей: {len(models_list)}")
            
            # Проверка наличия нужной модели
            try:
                model_names = []
                for m in models_list:
                    if isinstance(m, dict):
                        # Попробуем разные возможные ключи
                        name = m.get('name') or m.get('model') or m.get('id') or str(m)
                        model_names.append(name)
                    else:
                        model_names.append(str(m))
                
                if model_names:
                    if not any(app_config.ollama.embedding_model in str(name) for name in model_names):
                        logger.warning(f"⚠️  Модель '{app_config.ollama.embedding_model}' не найдена!")
                        logger.warning(f"   Доступные модели: {', '.join(str(n) for n in model_names)}")
                        logger.warning(f"   Загрузите модель: ollama pull {app_config.ollama.embedding_model}")
                    else:
                        logger.info(f"✅ Модель '{app_config.ollama.embedding_model}' найдена")
            except Exception as e:
                logger.warning(f"⚠️  Не удалось проверить наличие модели: {e}")
                logger.warning(f"   Попробуйте: ollama pull {app_config.ollama.embedding_model}")
                
        except Exception as e:
            logger.error(f"\n❌ НЕ УДАЛОСЬ ПОДКЛЮЧИТЬСЯ К OLLAMA!")
            logger.error(f"   Ошибка: {e}")
            logger.error(f"   Host: {app_config.ollama.host}")
            logger.error(f"\n💡 Решение:")
            logger.error(f"   1. Убедитесь, что Ollama запущен: ollama serve")
            logger.error(f"   2. Проверьте host в {args.config}")
            logger.error(f"      Используйте 'http://127.0.0.1:11434' вместо '0.0.0.0'")
            logger.error(f"   3. Загрузите модель: ollama pull {app_config.ollama.embedding_model}")
            return
    else:
        logger.warning(f"⚠️  Конфигурационный файл не найден: {args.config}")
        logger.info("Создание конфигурации по умолчанию")
        config = DatasetConfigV3()
        logger.info("💡 Рекомендуется создать project_config.json для централизованного управления")
    
    try:
        generator = DatasetGeneratorV3(config)
        generator.generate()
    except KeyboardInterrupt:
        logger.warning("\nГенерация прервана пользователем")
    except Exception as e:
        logger.error(f"Критическая ошибка: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
