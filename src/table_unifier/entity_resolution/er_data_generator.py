"""
Генератор синтетических данных для Entity Resolution.

Создаёт пары таблиц с контролируемым пересечением сущностей:
    - Таблица A и Таблица B содержат ОБЩИЕ сущности (дубликаты) и УНИКАЛЬНЫЕ
    - Дубликаты описаны по-разному: разные названия столбцов, опечатки в значениях,
      пропуски, изменения формата
    - Ground truth: какие строки из A соответствуют строкам из B

Расширяет существующий генератор из ds_gen_v3 (Faker-based).

NOTE: Для объединённой генерации SM + ER данных используйте модуль
      table_unifier.data_generation.UnifiedDatasetGenerator.
      Этот файл сохранён для обратной совместимости.
"""

import random
import re
import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict

try:
    from faker import Faker
except ImportError:
    raise ImportError("Faker не установлен! pip install faker")

from .config import ERConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Шаблоны столбцов для генерации (совместимы с ds_gen_v3)
# ═══════════════════════════════════════════════════════════════

ENTITY_COLUMNS = {
    'article': {
        'variants_ru': ['Артикул', 'Код товара', 'SKU', 'Арт.', 'Код', 'Артикуль', 'Артикл'],
        'variants_en': ['Article', 'SKU', 'Item Code', 'Product Code', 'Code'],
        'generator': lambda fake: random.choice([
            f"{fake.random_uppercase_letter()}{fake.random_int(100, 999)}-{fake.random_int(10, 99)}",
            f"SKU-{fake.random_int(10000, 99999)}",
            f"{fake.random_int(1000, 9999)}.{fake.random_int(100, 999)}",
        ]),
    },
    'product_name': {
        'variants_ru': ['Наименование', 'Товар', 'Название', 'Описание', 'Наим.', 'Продукт'],
        'variants_en': ['Product Name', 'Item', 'Name', 'Description', 'Product'],
        'generator': lambda fake: random.choice([
            f"{fake.word().capitalize()} {random.choice(['Pro', 'Standard', 'Premium', 'Plus', 'Basic'])}",
            f"{random.choice(['Комплект', 'Набор', 'Модуль', 'Блок'])} {fake.word().capitalize()}",
            f"{fake.word().capitalize()} {fake.word().capitalize()} {fake.random_int(100, 9999)}",
        ]),
    },
    'manufacturer': {
        'variants_ru': ['Производитель', 'Бренд', 'Изготовитель', 'Произв.', 'Торговая марка'],
        'variants_en': ['Manufacturer', 'Brand', 'Producer', 'Maker', 'Vendor'],
        'generator': lambda fake: random.choice([
            fake.company(),
            random.choice(['Samsung', 'LG', 'Sony', 'Bosch', 'Siemens', 'Philips', 'Hitachi']),
        ]),
    },
    'category': {
        'variants_ru': ['Категория', 'Группа', 'Тип товара', 'Раздел', 'Товарная группа'],
        'variants_en': ['Category', 'Group', 'Type', 'Class', 'Product Group'],
        'generator': lambda fake: random.choice([
            'Электроника', 'Мебель', 'Инструменты', 'Канцелярия', 'Стройматериалы',
            'Сантехника', 'Автозапчасти', 'Хозтовары', 'Текстиль', 'Офисная техника',
        ]),
    },
    'quantity': {
        'variants_ru': ['Количество', 'Кол-во', 'К-во', 'Объем', 'Кол.'],
        'variants_en': ['Quantity', 'Qty', 'Amount', 'Count'],
        'generator': lambda fake: random.choice([
            fake.random_int(1, 5),
            fake.random_int(10, 50),
            fake.random_int(50, 200),
        ]),
    },
    'unit': {
        'variants_ru': ['Ед. изм.', 'Единица', 'Ед.', 'ЕИ'],
        'variants_en': ['Unit', 'UoM', 'Measure'],
        'generator': lambda fake: random.choice(['шт.', 'кг', 'л', 'м', 'упак.', 'компл.']),
    },
    'price': {
        'variants_ru': ['Цена', 'Стоимость', 'Цена за ед.', 'Прайс', 'Базовая цена'],
        'variants_en': ['Price', 'Cost', 'Unit Price', 'Base Price'],
        'generator': lambda fake: round(random.uniform(50, 50000), 2),
    },
    'country': {
        'variants_ru': ['Страна', 'Страна происхождения', 'Производство'],
        'variants_en': ['Country', 'Country of Origin', 'Made in', 'Origin'],
        'generator': lambda fake: fake.country(),
    },
    'notes': {
        'variants_ru': ['Примечание', 'Комментарий', 'Доп. информация', 'Заметки'],
        'variants_en': ['Notes', 'Comments', 'Remarks'],
        'generator': lambda fake: fake.sentence(nb_words=5) if random.random() > 0.5 else '',
    },
}

# Столбцы, которые всегда присутствуют в таблице  
REQUIRED_COLUMNS = ['article', 'product_name', 'price', 'quantity']

# Столбцы, которые добавляются случайно
OPTIONAL_COLUMNS = ['manufacturer', 'category', 'unit', 'country', 'notes']


# ═══════════════════════════════════════════════════════════════
# Функции пертурбации значений
# ═══════════════════════════════════════════════════════════════

def perturb_string(value: str, prob: float = 0.3) -> str:
    """Добавить пертурбации к строковому значению.
    
    Виды пертурбаций: опечатки, пропуск символов, удвоение, замена регистра.
    """
    if not isinstance(value, str) or not value or random.random() > prob:
        return value
    
    chars = list(value)
    if len(chars) < 3:
        return value
    
    perturbation = random.choice(['swap', 'double', 'skip', 'case', 'space'])
    pos = random.randint(1, len(chars) - 2)
    
    if perturbation == 'swap' and pos < len(chars) - 1:
        chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
    elif perturbation == 'double' and chars[pos].isalpha():
        chars.insert(pos, chars[pos])
    elif perturbation == 'skip' and chars[pos].isalpha():
        chars.pop(pos)
    elif perturbation == 'case':
        chars[pos] = chars[pos].swapcase()
    elif perturbation == 'space' and ' ' in value:
        # Удалить или добавить пробел
        if random.random() > 0.5:
            value = value.replace(' ', '', 1)
            return value
        else:
            chars.insert(pos, ' ')
    
    return ''.join(chars)


def perturb_number(value, prob: float = 0.3):
    """Добавить пертурбации к числовому значению.
    
    Виды: округление, малый шум, изменение формата.
    """
    if random.random() > prob:
        return value
    
    try:
        num = float(value)
    except (ValueError, TypeError):
        return value
    
    perturbation = random.choice(['round', 'noise', 'format'])
    
    if perturbation == 'round':
        return round(num)
    elif perturbation == 'noise':
        noise = num * random.uniform(-0.01, 0.01)  # ±1%
        return round(num + noise, 2)
    elif perturbation == 'format':
        if isinstance(value, float):
            return int(num) if num == int(num) else round(num, 1)
    
    return value


def perturb_value(value, col_type: str, prob: float = 0.3):
    """Применить подходящую пертурбацию к значению ячейки."""
    if pd.isna(value) or value is None or str(value).strip() == '':
        return value
    
    if col_type in ('quantity', 'price'):
        return perturb_number(value, prob)
    else:
        return perturb_string(str(value), prob)


# ═══════════════════════════════════════════════════════════════
# Генератор сущностей
# ═══════════════════════════════════════════════════════════════

class EntityPool:
    """Пул канонических сущностей для генерации пар таблиц.
    
    Каждая сущность — словарь {column_key: value} с полной информацией.
    """
    
    def __init__(self, size: int = 500, locales: List[str] = None):
        if locales is None:
            locales = ['ru_RU', 'en_US']
        self.fakers = {loc: Faker(loc) for loc in locales}
        self.entities = self._generate(size)
        logger.info(f"EntityPool: создано {len(self.entities)} сущностей")
    
    def _generate(self, size: int) -> List[Dict]:
        """Генерация пула канонических сущностей."""
        entities = []
        for i in range(size):
            fake = random.choice(list(self.fakers.values()))
            entity = {'_entity_id': i}
            for col_key, template in ENTITY_COLUMNS.items():
                entity[col_key] = template['generator'](fake)
            entities.append(entity)
        return entities
    
    def sample(self, n: int) -> List[Dict]:
        """Выбрать n случайных сущностей (без повторения)."""
        n = min(n, len(self.entities))
        return random.sample(self.entities, n)


# ═══════════════════════════════════════════════════════════════
# Генератор пар таблиц для ER
# ═══════════════════════════════════════════════════════════════

@dataclass
class TablePairData:
    """Данные пары таблиц с ground truth для Entity Resolution."""
    df_a: pd.DataFrame
    df_b: pd.DataFrame
    duplicate_pairs: List[Tuple[int, int]]  # [(row_idx_a, row_idx_b), ...]
    column_mapping_a: Dict[str, str]        # display_name → base_name
    column_mapping_b: Dict[str, str]        # display_name → base_name
    entity_ids_a: List[int]                 # entity_id для каждой строки таблицы A
    entity_ids_b: List[int]                 # entity_id для каждой строки таблицы B


class ERTablePairGenerator:
    """Генератор пар таблиц для обучения Entity Resolution.
    
    Для каждой пары:
    1. Выбираем общие сущности из пула (дубликаты)
    2. Добавляем уникальные сущности в каждую таблицу
    3. Применяем разные пертурбации к значениям
    4. Выбираем разные подмножества столбцов
    5. Назначаем разные имена столбцам
    6. Перемешиваем строки
    """
    
    def __init__(self, config: ERConfig, entity_pool: EntityPool = None):
        self.config = config
        self.pool = entity_pool or EntityPool(
            size=config.num_entity_pool,
            locales=['ru_RU', 'en_US'],
        )
        self.fakers = self.pool.fakers
    
    def generate_pair(self) -> TablePairData:
        """Сгенерировать одну пару таблиц с ground truth."""
        
        # 1. Выбираем общие сущности (дубликаты)
        n_common = random.randint(
            self.config.min_common_entities,
            self.config.max_common_entities,
        )
        common_entities = self.pool.sample(n_common)
        
        # 2. Выбираем уникальные сущности для каждой таблицы
        n_unique_a = random.randint(
            self.config.min_unique_entities,
            self.config.max_unique_entities,
        )
        n_unique_b = random.randint(
            self.config.min_unique_entities,
            self.config.max_unique_entities,
        )
        
        # Исключаем уже выбранные сущности
        common_ids = {e['_entity_id'] for e in common_entities}
        remaining = [e for e in self.pool.entities if e['_entity_id'] not in common_ids]
        
        unique_a = random.sample(remaining, min(n_unique_a, len(remaining)))
        remaining_ids_a = {e['_entity_id'] for e in unique_a}
        remaining2 = [e for e in remaining if e['_entity_id'] not in remaining_ids_a]
        unique_b = random.sample(remaining2, min(n_unique_b, len(remaining2)))
        
        # 3. Выбираем столбцы для каждой таблицы
        cols_a = self._select_columns()
        cols_b = self._select_columns()
        
        # 4. Генерируем имена столбцов
        col_names_a, col_mapping_a = self._generate_column_names(cols_a)
        col_names_b, col_mapping_b = self._generate_column_names(cols_b)
        
        # 5. Создаём строки таблиц
        rows_a = []
        entity_ids_a = []
        
        # Общие сущности в таблицу A (с пертурбациями)
        for entity in common_entities:
            row = self._entity_to_row(entity, cols_a, perturbation_prob=self.config.perturbation_prob)
            rows_a.append(row)
            entity_ids_a.append(entity['_entity_id'])
        
        # Уникальные сущности в таблицу A
        for entity in unique_a:
            row = self._entity_to_row(entity, cols_a, perturbation_prob=0.0)
            rows_a.append(row)
            entity_ids_a.append(entity['_entity_id'])
        
        rows_b = []
        entity_ids_b = []
        
        # Общие сущности в таблицу B (с ДРУГИМИ пертурбациями)
        for entity in common_entities:
            row = self._entity_to_row(entity, cols_b, perturbation_prob=self.config.perturbation_prob)
            rows_b.append(row)
            entity_ids_b.append(entity['_entity_id'])
        
        # Уникальные сущности в таблицу B
        for entity in unique_b:
            row = self._entity_to_row(entity, cols_b, perturbation_prob=0.0)
            rows_b.append(row)
            entity_ids_b.append(entity['_entity_id'])
        
        # 6. Перемешиваем строки
        indices_a = list(range(len(rows_a)))
        indices_b = list(range(len(rows_b)))
        random.shuffle(indices_a)
        random.shuffle(indices_b)
        
        rows_a = [rows_a[i] for i in indices_a]
        entity_ids_a = [entity_ids_a[i] for i in indices_a]
        rows_b = [rows_b[i] for i in indices_b]
        entity_ids_b = [entity_ids_b[i] for i in indices_b]
        
        # 7. Создаём DataFrames
        df_a = pd.DataFrame(rows_a, columns=col_names_a)
        df_b = pd.DataFrame(rows_b, columns=col_names_b)
        
        # 8. Вычисляем ground truth пары дубликатов (после перемешивания!)
        duplicate_pairs = []
        for i, eid_a in enumerate(entity_ids_a):
            for j, eid_b in enumerate(entity_ids_b):
                if eid_a == eid_b:
                    duplicate_pairs.append((i, j))
        
        return TablePairData(
            df_a=df_a,
            df_b=df_b,
            duplicate_pairs=duplicate_pairs,
            column_mapping_a=col_mapping_a,
            column_mapping_b=col_mapping_b,
            entity_ids_a=entity_ids_a,
            entity_ids_b=entity_ids_b,
        )
    
    def _select_columns(self) -> List[str]:
        """Выбрать подмножество столбцов для таблицы."""
        selected = list(REQUIRED_COLUMNS)
        # Случайные дополнительные столбцы
        n_opt = random.randint(1, len(OPTIONAL_COLUMNS))
        selected.extend(random.sample(OPTIONAL_COLUMNS, n_opt))
        return selected
    
    def _generate_column_names(
        self, columns: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Сгенерировать display-имена для столбцов.
        
        Returns:
            (list_of_display_names, {display_name: base_name})
        """
        locale_key = random.choice(['ru', 'en'])
        display_names = []
        mapping = {}
        
        for col_key in columns:
            template = ENTITY_COLUMNS[col_key]
            variants = template.get(f'variants_{locale_key}', template.get('variants_ru', [col_key]))
            display_name = random.choice(variants)
            
            # Иногда добавляем опечатку или вариацию
            if random.random() < self.config.perturbation_prob:
                display_name = perturb_string(display_name, prob=0.7)
            
            # Уникальность
            original = display_name
            counter = 1
            while display_name in display_names:
                display_name = f"{original}_{counter}"
                counter += 1
            
            display_names.append(display_name)
            mapping[display_name] = col_key
        
        return display_names, mapping
    
    def _entity_to_row(
        self,
        entity: Dict,
        columns: List[str],
        perturbation_prob: float = 0.3,
    ) -> List:
        """Преобразовать сущность в строку таблицы с опциональными пертурбациями."""
        row = []
        for col_key in columns:
            value = entity.get(col_key, '')
            
            # Пропуск значения с некоторой вероятностью
            if random.random() < self.config.missing_value_prob and col_key not in ('article', 'product_name'):
                row.append(None)
                continue
            
            # Пертурбация значения
            if perturbation_prob > 0:
                value = perturb_value(value, col_key, perturbation_prob)
            
            row.append(value)
        
        return row
    
    def generate_dataset(
        self,
        n_train: int = None,
        n_val: int = None,
        n_test: int = None,
    ) -> Dict[str, List[TablePairData]]:
        """Сгенерировать полный датасет из пар таблиц.
        
        Returns:
            {'train': [...], 'val': [...], 'test': [...]}
        """
        n_train = n_train or self.config.num_train_pairs
        n_val = n_val or self.config.num_val_pairs
        n_test = n_test or self.config.num_test_pairs
        
        dataset = {}
        for split, n in [('train', n_train), ('val', n_val), ('test', n_test)]:
            logger.info(f"Генерация {split}: {n} пар таблиц...")
            pairs = []
            for i in range(n):
                try:
                    pair = self.generate_pair()
                    pairs.append(pair)
                except Exception as e:
                    logger.error(f"Ошибка генерации пары {i} ({split}): {e}")
            dataset[split] = pairs
            logger.info(f"  {split}: {len(pairs)} пар готово")
        
        return dataset
