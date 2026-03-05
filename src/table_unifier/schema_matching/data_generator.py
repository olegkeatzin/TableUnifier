"""
Генератор данных для Schema Matching с Triplet Loss.

Генерирует пары таблиц с различными схемами и создаёт эмбеддинги столбцов:
    1. Генерация таблиц через Faker (аналогично ds_gen_v3)
    2. Генерация описаний столбцов через LLM (qwen3.5:9b)
    3. Эмбеддинг описаний через embeddinggemma
    4. Сохранение (embedding, base_column_type) для triplet mining

Каждая запись в датасете:
    {
        "column_name": "Цена за ед.",       # display name
        "base_name": "base_price",           # ground truth label
        "description": "Цена за ед.: ...",   # LLM description  
        "embedding": [...],                   # float vector
        "data_type": "float64",
        "content_sample": [123.45, ...],
        "table_id": 42
    }

NOTE: Для объединённой генерации SM + ER данных используйте модуль
      table_unifier.data_generation.UnifiedDatasetGenerator.
      Этот файл сохранён для standalone SM генерации.
"""

import os
import sys
import json
import random
import signal
import hashlib
import logging
import threading
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

try:
    from faker import Faker
    FAKER_AVAILABLE = True
except ImportError:
    FAKER_AVAILABLE = False

from .config import SMConfig

logging.getLogger("httpx").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Шаблоны столбцов (расширенный набор из ds_gen_v3)
# ═══════════════════════════════════════════════════════════════

COLUMN_TEMPLATES = {
    'item_number': {
        'description': 'Порядковый номер позиции в таблице',
        'variants': {
            'ru': ['№', '№ п/п', 'Номер', 'Поз.', 'Позиция', '#', 'N', 'Пункт',
                    'Номир', 'Позициа', '№№', 'Нумер', 'Порядковый номер'],
            'en': ['#', 'No.', 'Item No.', 'Position', 'Line', 'Seq', 'Row #',
                    'Positon', 'Iten No.', 'Line No.', 'Sequence']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: idx + 1
    },
    'article': {
        'description': 'Артикул товара',
        'variants': {
            'ru': ['Артикул', 'Код товара', 'SKU', 'Номенклатурный номер', 'Код',
                    'Арт.', 'Артикуль', 'Код тавара', 'Артикл', 'Артик', 'Номенклатура'],
            'en': ['Article', 'SKU', 'Item Code', 'Product Code', 'Code',
                    'Articl', 'Artcle', 'Item Cod', 'Part Number', 'PN']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            f"{fake.random_uppercase_letter()}{fake.random_int(100, 999)}-{fake.random_int(10, 99)}",
            f"SKU-{fake.random_int(10000, 99999)}",
            f"{fake.random_int(1000, 9999)}.{fake.random_int(100, 999)}",
            f"{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}-{fake.random_int(1000, 9999)}",
            f"{random.choice(['АРТ', 'PRD', 'ITM'])}{fake.random_int(10000, 99999)}",
        ])
    },
    'product_name': {
        'description': 'Наименование товара',
        'variants': {
            'ru': ['Наименование товара', 'Товар', 'Описание', 'Номенклатура',
                    'Наименование', 'Название', 'Наименавание', 'Тавар', 'Наим-е',
                    'Наимен.', 'Назв.', 'Продукт'],
            'en': ['Product Name', 'Item', 'Description', 'Name', 'Item Description',
                    'Product Nam', 'Itm Name', 'Prod Name', 'Product']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            f"{fake.word().capitalize()} {fake.company_suffix()}",
            f"{fake.word().capitalize()} {random.choice(['Premium', 'Standard', 'Economy', 'Pro', 'Basic'])}",
            f"{random.choice(['Комплект', 'Набор', 'Система', 'Модуль', 'Блок'])} {fake.word().capitalize()}",
            f"{fake.word().capitalize()} {random.choice(['2000', '3000', 'Ultra', 'Plus', 'Light', 'Max'])}",
        ])
    },
    'product_group': {
        'description': 'Категория или группа товара',
        'variants': {
            'ru': ['Товарная группа', 'Категория', 'Группа', 'Раздел', 'Тип товара',
                    'Категориа', 'Катег.', 'Групп', 'Товар.группа'],
            'en': ['Product Group', 'Category', 'Type', 'Group', 'Class',
                    'Categry', 'Grp', 'Prod Group']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            'Электроника', 'Мебель', 'Инструменты', 'Канцелярия', 'Стройматериалы',
            'Сантехника', 'Автозапчасти', 'Офисная техника', 'Хозтовары', 'Текстиль',
        ])
    },
    'manufacturer': {
        'description': 'Производитель товара',
        'variants': {
            'ru': ['Производитель', 'Бренд', 'Торговая марка', 'Изготовитель',
                    'Завод-изготовитель', 'Производиттель', 'Произв.', 'Произв-ль'],
            'en': ['Manufacturer', 'Brand', 'Producer', 'Maker', 'Vendor',
                    'Manufactur', 'Mnfr', 'Mfr']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice([
            fake.company(),
            random.choice(['Samsung', 'LG', 'Sony', 'Bosch', 'Siemens', 'Philips', 'Hitachi']),
        ])
    },
    'country_of_origin': {
        'description': 'Страна происхождения товара',
        'variants': {
            'ru': ['Страна происхождения', 'Страна', 'Производство',
                    'Страна изготовления', 'Страна-произв.'],
            'en': ['Country of Origin', 'Country', 'Made in', 'Origin',
                    'COO', 'Country of Mfg']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: fake.country()
    },
    'quantity': {
        'description': 'Количество единиц товара',
        'variants': {
            'ru': ['Количество', 'Кол-во', 'Qty', 'Общее кол-во', 'К-во',
                    'Объем', 'Колличество', 'Кол.', 'Колво', 'Кол', 'Кол-во шт'],
            'en': ['Quantity', 'Qty', 'Amount', 'Count', 'Total Qty',
                    'Qnty', 'Quanty', 'Qtt']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: random.choice([
            fake.random_int(1, 5),
            fake.random_int(10, 50),
            fake.random_int(50, 200),
            fake.random_int(1, 10) * 10,
        ])
    },
    'unit': {
        'description': 'Единица измерения',
        'variants': {
            'ru': ['Ед. изм.', 'Единица', 'Ед.', 'ЕИ', 'Размерность',
                    'Ед.изм', 'Едизм', 'Единиц.', 'Ед измер'],
            'en': ['Unit', 'UoM', 'Unit of Measure', 'Measurement',
                    'Unt', 'Mesure', 'UOM']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: random.choice(['шт.', 'кг', 'л', 'м', 'упак.', 'компл.', 'пар'])
    },
    'base_price': {
        'description': 'Базовая цена за единицу',
        'variants': {
            'ru': ['Цена', 'Базовая цена', 'Цена за ед.', 'Стоимость', 'Прайс',
                    'Цена по прайсу', 'Ценна', 'Цена/ед', 'Цена ед.', 'Цена₽'],
            'en': ['Price', 'Base Price', 'Unit Price', 'Cost', 'List Price',
                    'Prise', 'Prce', 'Unit Prc']
        },
        'data_type': 'float',
        'generator': lambda fake, idx: round(
            random.choice([
                fake.random.uniform(50, 500),
                fake.random.uniform(500, 5000),
                fake.random.uniform(5000, 50000),
            ]) * random.choice([1, 0.99, 1.15, 0.95]),
            2
        )
    },
    'discount_percent': {
        'description': 'Процент скидки',
        'variants': {
            'ru': ['Скидка %', '% скидки', 'Скидка', 'Дисконт %', 'Процент скидки',
                    'Скидко %', '% скидк', 'Скид %', 'Скидка%'],
            'en': ['Discount %', 'Discount', '% Off', 'Disc %', 'Discnt %',
                    'Disct', 'Disc']
        },
        'data_type': 'float',
        'generator': lambda fake, idx: random.choice([0, 0, 0, 5, 10, 15, 20])
    },
    'price_after_discount': {
        'description': 'Цена со скидкой',
        'variants': {
            'ru': ['Цена со скидкой', 'Отпускная цена', 'Цена продажи', 'Финальная цена'],
            'en': ['Discounted Price', 'Sale Price', 'Final Price', 'Net Price']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    'vat_rate': {
        'description': 'Ставка НДС в процентах',
        'variants': {
            'ru': ['Ставка НДС', '% НДС', 'НДС %', 'Налог'],
            'en': ['VAT Rate', 'VAT %', 'Tax Rate', 'Tax %']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: random.choice([0, 10, 20])
    },
    'total_sum': {
        'description': 'Общая стоимость без НДС',
        'variants': {
            'ru': ['Сумма', 'Итого', 'Сумма без НДС', 'Стоимость', 'Всего',
                    'Сумм', 'Итого₽', 'Сум.', 'Стоим.'],
            'en': ['Total', 'Sum', 'Amount', 'Total Amount', 'Subtotal',
                    'Totl', 'Amnt', 'Tot']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    'vat_amount': {
        'description': 'Сумма НДС',
        'variants': {
            'ru': ['Сумма НДС', 'НДС', 'В т.ч. НДС', 'Налог (сумма)'],
            'en': ['VAT Amount', 'VAT', 'Tax Amount', 'Tax']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    'total_with_vat': {
        'description': 'Общая стоимость с НДС',
        'variants': {
            'ru': ['Всего с НДС', 'Итого с НДС', 'К оплате', 'Сумма к оплате'],
            'en': ['Total with VAT', 'Grand Total', 'Total Incl. VAT', 'Amount Due']
        },
        'data_type': 'float',
        'is_calculated': True
    },
    'notes': {
        'description': 'Дополнительные примечания',
        'variants': {
            'ru': ['Примечание', 'Комментарий', 'Доп. информация', 'Заметки', 'Прим.'],
            'en': ['Notes', 'Comments', 'Remarks', 'Additional Info']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: fake.sentence(nb_words=6) if random.random() > 0.5 else ''
    },
}

MANDATORY_COLUMNS = [
    'item_number', 'article', 'product_name', 'product_group',
    'quantity', 'unit', 'base_price', 'total_sum'
]

OPTIONAL_COLUMNS = [
    'manufacturer', 'country_of_origin', 'discount_percent',
    'price_after_discount', 'vat_rate', 'vat_amount', 'total_with_vat', 'notes'
]


def _build_column_templates(config: SMConfig) -> Dict:
    """Собирает финальные шаблоны столбцов, объединяя встроенные и пользовательские.
    
    Расширяет:
      - extra_column_variants → дополнительные варианты названий
      - extra_product_groups → значения для product_group
      - extra_units → значения для unit
      - extra_manufacturers → значения для manufacturer
      - extra_vat_rates → значения для vat_rate
      - extra_article_prefixes → префиксы артикулов
      - extra_product_name_parts → модификаторы наименований
      - extra_countries → страны для country_of_origin
      - extra_notes_templates → шаблоны примечаний
      - custom_columns → полностью новые типы столбцов
    """
    import copy
    templates = copy.deepcopy(COLUMN_TEMPLATES)

    # --- Дополнительные варианты названий ---
    for base_name, locale_variants in config.extra_column_variants.items():
        if base_name in templates:
            for lang, names_list in locale_variants.items():
                if lang in templates[base_name]['variants']:
                    templates[base_name]['variants'][lang].extend(names_list)
                else:
                    templates[base_name]['variants'][lang] = list(names_list)

    # --- Расширение генераторов значений ---
    if config.extra_product_groups:
        base_groups = [
            'Электроника', 'Мебель', 'Инструменты', 'Канцелярия', 'Стройматериалы',
            'Сантехника', 'Автозапчасти', 'Офисная техника', 'Хозтовары', 'Текстиль',
        ]
        all_groups = base_groups + list(config.extra_product_groups)
        templates['product_group']['generator'] = lambda fake, idx, _g=all_groups: random.choice(_g)

    if config.extra_units:
        base_units = ['шт.', 'кг', 'л', 'м', 'упак.', 'компл.', 'пар']
        all_units = base_units + list(config.extra_units)
        templates['unit']['generator'] = lambda fake, idx, _u=all_units: random.choice(_u)

    if config.extra_manufacturers:
        base_mfrs = ['Samsung', 'LG', 'Sony', 'Bosch', 'Siemens', 'Philips', 'Hitachi']
        all_mfrs = base_mfrs + list(config.extra_manufacturers)
        templates['manufacturer']['generator'] = lambda fake, idx, _m=all_mfrs: random.choice(
            [fake.company()] + _m
        )

    if config.extra_vat_rates:
        base_vat = [0, 10, 20]
        all_vat = base_vat + list(config.extra_vat_rates)
        templates['vat_rate']['generator'] = lambda fake, idx, _v=all_vat: random.choice(_v)

    if config.extra_article_prefixes:
        base_prefixes = ['АРТ', 'PRD', 'ITM']
        all_prefixes = base_prefixes + list(config.extra_article_prefixes)
        templates['article']['generator'] = lambda fake, idx, _p=all_prefixes: random.choice([
            f"{fake.random_uppercase_letter()}{fake.random_int(100, 999)}-{fake.random_int(10, 99)}",
            f"SKU-{fake.random_int(10000, 99999)}",
            f"{fake.random_int(1000, 9999)}.{fake.random_int(100, 999)}",
            f"{fake.random_uppercase_letter()}{fake.random_uppercase_letter()}-{fake.random_int(1000, 9999)}",
            f"{random.choice(_p)}{fake.random_int(10000, 99999)}",
        ])

    if config.extra_product_name_parts:
        base_suffixes = ['Premium', 'Standard', 'Economy', 'Pro', 'Basic',
                         '2000', '3000', 'Ultra', 'Plus', 'Light', 'Max']
        base_prefixes_p = ['Комплект', 'Набор', 'Система', 'Модуль', 'Блок']
        all_suffixes = base_suffixes + list(config.extra_product_name_parts)
        templates['product_name']['generator'] = lambda fake, idx, _s=all_suffixes, _p=base_prefixes_p: random.choice([
            f"{fake.word().capitalize()} {fake.company_suffix()}",
            f"{fake.word().capitalize()} {random.choice(_s)}",
            f"{random.choice(_p)} {fake.word().capitalize()}",
            f"{fake.word().capitalize()} {random.choice(_s)}",
        ])

    if config.extra_countries:
        templates['country_of_origin']['generator'] = (
            lambda fake, idx, _c=list(config.extra_countries): 
                random.choice([fake.country()] + _c)
        )

    if config.extra_notes_templates:
        templates['notes']['generator'] = (
            lambda fake, idx, _n=list(config.extra_notes_templates):
                random.choice(_n) if random.random() > 0.3
                else (fake.sentence(nb_words=6) if random.random() > 0.5 else '')
        )

    return templates


def _build_column_lists(config: SMConfig):
    """Возвращает (mandatory, optional) списки, включая custom_columns."""
    mandatory = list(MANDATORY_COLUMNS)
    optional = list(OPTIONAL_COLUMNS)

    for col_name, col_def in config.custom_columns.items():
        if col_def.get('mandatory', False):
            mandatory.append(col_name)
        else:
            optional.append(col_name)

    return mandatory, optional


def _register_custom_columns(templates: Dict, config: SMConfig) -> Dict:
    """Регистрирует пользовательские столбцы из config.custom_columns."""
    for col_name, col_def in config.custom_columns.items():
        if col_name in templates:
            continue  # не перезаписываем встроенные

        variants = col_def.get('variants', {'ru': [col_name], 'en': [col_name]})
        data_type = col_def.get('data_type', 'str')
        values = col_def.get('values', [])
        description = col_def.get('description', col_name)

        entry = {
            'description': description,
            'variants': variants,
            'data_type': data_type,
        }

        if values:
            entry['generator'] = lambda fake, idx, _v=list(values): random.choice(_v)
        else:
            # Авто-генерация по типу данных
            if data_type == 'int':
                entry['generator'] = lambda fake, idx: fake.random_int(1, 1000)
            elif data_type == 'float':
                entry['generator'] = lambda fake, idx: round(fake.random.uniform(1, 10000), 2)
            else:
                entry['generator'] = lambda fake, idx: fake.word()

        templates[col_name] = entry

    return templates


# ═══════════════════════════════════════════════════════════════
# Вариатор названий столбцов
# ═══════════════════════════════════════════════════════════════

class ColumnNameVariator:
    """Генерирует разнообразные вариации названий столбцов."""
    
    def __init__(self, variation_level: float = 0.7,
                 include_typos: bool = True,
                 include_abbreviations: bool = True):
        self.variation_level = variation_level
        self.include_typos = include_typos
        self.include_abbreviations = include_abbreviations
    
    def generate_variant(self, base_variants: List[str]) -> str:
        variant = random.choice(base_variants)
        
        if random.random() < self.variation_level:
            variant = self._add_typos(variant)
        if random.random() < self.variation_level:
            variant = self._add_abbreviation(variant)
        if random.random() < self.variation_level * 0.5:
            variant = self._add_spacing(variant)
        if random.random() < self.variation_level * 0.4:
            variant = self._add_case_variation(variant)
        if random.random() < self.variation_level * 0.3:
            variant = self._add_extra_chars(variant)
        
        return variant
    
    def _add_typos(self, text: str) -> str:
        if not self.include_typos or random.random() > 0.5:
            return text
        chars = list(text)
        if len(chars) < 3:
            return text
        pos = random.randint(1, len(chars) - 2)
        typo_type = random.choice(['swap', 'double', 'skip'])
        if typo_type == 'swap' and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif typo_type == 'double' and chars[pos].isalpha():
            chars.insert(pos, chars[pos])
        elif typo_type == 'skip' and chars[pos].isalpha():
            chars.pop(pos)
        return ''.join(chars)
    
    def _add_abbreviation(self, text: str) -> str:
        if not self.include_abbreviations or random.random() > 0.5:
            return text
        abbrevs = {
            'Количество': 'Кол-во', 'Номер': '№', 'Единица измерения': 'Ед. изм.',
            'Примечание': 'Прим.', 'Артикул': 'Арт.', 'Производитель': 'Произв.',
            'Наименование': 'Наим.', 'Товар': 'Тов.', 'Цена': 'Ц.', 'Скидка': 'Ск.'
        }
        for full, abbr in abbrevs.items():
            if full in text:
                return text.replace(full, abbr)
        return text
    
    def _add_spacing(self, text: str) -> str:
        if random.random() > 0.4:
            return text
        return random.choice([
            text.replace(' ', '_'), text.replace(' ', ''),
            text.replace(' ', '.'), text.replace(' ', '-'),
        ])
    
    def _add_case_variation(self, text: str) -> str:
        if random.random() > 0.4:
            return text
        return random.choice([text.upper(), text.lower(), text.title(), text.capitalize()])
    
    def _add_extra_chars(self, text: str) -> str:
        if random.random() > 0.3:
            return text
        return random.choice([
            f"{text}:", f"{text}.", f"({text})", f"[{text}]",
            f"{text} ", f" {text}", f"{text}*",
        ])


# ═══════════════════════════════════════════════════════════════
# Генератор таблиц
# ═══════════════════════════════════════════════════════════════

class SMTableGenerator:
    """Генерирует таблицы с разнообразными названиями столбцов."""
    
    def __init__(self, config: SMConfig):
        self.config = config
        self.fakers = {locale: Faker(locale) for locale in config.locales}
        self.variator = ColumnNameVariator(
            variation_level=config.column_name_variation_level,
            include_typos=config.include_typos,
            include_abbreviations=config.include_abbreviations,
        )
        self.stats = {'total_generated': 0, 'rows_generated': 0, 'columns_generated': 0}
        
        # Собираем шаблоны столбцов с учётом расширений из конфига
        self.templates = _build_column_templates(config)
        self.templates = _register_custom_columns(self.templates, config)
        self.mandatory_columns, self.optional_columns = _build_column_lists(config)
        
        n_custom = len(config.custom_columns)
        n_extra_variants = sum(len(v) for lv in config.extra_column_variants.values() for v in lv.values())
        if n_custom or n_extra_variants:
            logger.info(f"Расширения: {n_custom} custom столбцов, {n_extra_variants} extra вариантов")
    
    def select_columns(self) -> List[str]:
        selected = self.mandatory_columns.copy()
        n_opt = random.randint(
            self.config.min_optional_columns,
            min(self.config.max_optional_columns, len(self.optional_columns))
        )
        selected.extend(random.sample(self.optional_columns, n_opt))
        return selected
    
    def generate_column_name(self, base_name: str) -> Tuple[str, str]:
        template = self.templates[base_name]
        locale_key = random.choice(['ru', 'en'])
        base_variants = template['variants'][locale_key]
        display_name = self.variator.generate_variant(base_variants)
        return display_name, locale_key
    
    def generate_row_data(self, columns: List[str], fake: Faker, row_idx: int) -> Dict:
        row = {}
        for col_key in columns:
            template = self.templates[col_key]
            if template.get('is_calculated'):
                continue
            if 'generator' in template:
                row[col_key] = template['generator'](fake, row_idx)
        self._calculate_derived(row, columns)
        return row
    
    def _calculate_derived(self, row: Dict, columns: List[str]):
        if 'price_after_discount' in columns and 'base_price' in row and 'discount_percent' in row:
            row['price_after_discount'] = round(row['base_price'] * (1 - row['discount_percent'] / 100), 2)
        
        price = row.get('price_after_discount', row.get('base_price', 0))
        quantity = row.get('quantity', 0)
        
        if 'total_sum' in columns:
            row['total_sum'] = round(price * quantity, 2)
        if 'vat_rate' in row:
            vat_rate = row['vat_rate']
            if 'vat_amount' in columns:
                row['vat_amount'] = round(row.get('total_sum', price * quantity) * vat_rate / 100, 2)
            if 'total_with_vat' in columns:
                total = row.get('total_sum', price * quantity)
                vat = row.get('vat_amount', total * vat_rate / 100)
                row['total_with_vat'] = round(total + vat, 2)
    
    def generate_table(self) -> Tuple[pd.DataFrame, Dict]:
        selected_columns = self.select_columns()
        column_mapping = {}
        display_names = []
        
        for base_name in selected_columns:
            display_name, locale = self.generate_column_name(base_name)
            original = display_name
            counter = 1
            while display_name in display_names:
                display_name = f"{original}_{counter}"
                counter += 1
            display_names.append(display_name)
            column_mapping[display_name] = base_name
        
        num_rows = random.randint(self.config.min_rows_per_table, self.config.max_rows_per_table)
        locale = random.choice(self.config.locales)
        fake = self.fakers[locale]
        
        rows = []
        for i in range(num_rows):
            rows.append(self.generate_row_data(selected_columns, fake, i))
        
        df_base = pd.DataFrame(rows)
        existing_cols = [c for c in selected_columns if c in df_base.columns]
        df_base = df_base[existing_cols]
        
        rename_map = {bn: dn for dn, bn in column_mapping.items() if bn in existing_cols}
        df = df_base.rename(columns=rename_map)
        final_display = [n for n in display_names if column_mapping[n] in existing_cols]
        df = df[final_display]
        
        self.stats['total_generated'] += 1
        self.stats['rows_generated'] += num_rows
        self.stats['columns_generated'] += len(final_display)
        
        ground_truth = {
            'column_mapping': column_mapping,
            'num_rows': num_rows,
            'locale': locale,
            'selected_columns': selected_columns,
        }
        return df, ground_truth


# ═══════════════════════════════════════════════════════════════
# Главный генератор датасета для Schema Matching
# ═══════════════════════════════════════════════════════════════

class SMDatasetGenerator:
    """Генератор датасета для обучения Schema Matching модели.
    
    Процесс:
        1. Генерация таблиц через Faker (разные схемы, разные имена столбцов)
        2. Для каждого столбца: генерация описания через LLM
        3. Эмбеддинг описания через embedding model
        4. Сохранение (embedding, base_name) для triplet loss обучения
    
    Usage:
        config = SMConfig(ollama_host="http://localhost:11434")
        generator = SMDatasetGenerator(config)
        generator.generate()
    """
    
    def __init__(self, config: SMConfig):
        if not FAKER_AVAILABLE:
            raise ImportError("Faker не установлен! pip install faker")
        
        self.config = config
        self.table_generator = SMTableGenerator(config)
        
        # Инициализация моделей Ollama
        import ollama
        self.ollama_client = ollama.Client(host=config.ollama_host)
        
        # Потокобезопасность
        self.lock = threading.Lock()
        self.interrupted = False
        self.all_entries: List[Dict] = []
        self.all_metadata: List[Dict] = []
        
        logger.info(f"SMDatasetGenerator инициализирован")
        logger.info(f"  LLM: {config.llm_model}")
        logger.info(f"  Embedding: {config.embedding_model}")
        logger.info(f"  Потоки: {config.num_workers}")
    
    def _generate_description(self, col_name: str, content: List, data_type: str) -> str:
        """Генерация описания столбца через LLM."""
        sample_size = min(self.config.sample_size, len(content))
        if sample_size > 0:
            sample = list(np.random.choice(content, size=sample_size, replace=False))
        else:
            sample = []
        
        type_info = f" Тип данных: {data_type}." if data_type else ""
        
        prompt = (
            f"Дай краткое описание для столбца таблицы с названием '{col_name}'.{type_info}\n"
            f"Если по названию не понятно, что это за столбец, попробуй угадать на основе "
            f"содержимого: {sample}.\n"
            f"Описание должно быть универсальным, чтобы подходить для любых значений в этом столбце.\n"
            f"Если столбец описывает что-то конкретное, думай шире - в столбце могут быть более "
            f"разнообразные данные.\n"
            f"Выведи только описание и ничего больше. /no_think"
        )
        
        try:
            response = self.ollama_client.generate(model=self.config.llm_model, prompt=prompt)
            desc = response['response']
            return f"{col_name}: {desc}"
        except Exception as e:
            logger.warning(f"Ошибка LLM для '{col_name}': {e}")
            return f"{col_name}: {', '.join(map(str, content[:3]))}"
    
    def _batch_embed(self, texts: List[str]) -> List[List[float]]:
        """Батчевый эмбеддинг текстов."""
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
                logger.error(f"Ошибка эмбеддинга батча {i}: {e}")
                for text in batch:
                    try:
                        resp = self.ollama_client.embed(
                            model=self.config.embedding_model,
                            input=[text]
                        )
                        all_embeddings.extend(resp['embeddings'])
                    except Exception as e2:
                        logger.error(f"Критическая ошибка эмбеддинга: {e2}")
                        all_embeddings.append([0.0] * self.config.input_dim)
        
        return all_embeddings
    
    def _process_table(self, table_id: int) -> Tuple[Optional[Dict], Optional[List[Dict]]]:
        """Обработка одной таблицы: генерация + описания + эмбеддинги."""
        try:
            # 1. Генерация таблицы
            df, ground_truth = self.table_generator.generate_table()
            
            # 2. Подготовка описаний для каждого столбца
            descriptions = []
            col_info = []
            
            for display_name in df.columns:
                base_name = ground_truth['column_mapping'].get(display_name)
                if not base_name:
                    continue
                
                content = df[display_name].dropna().tolist()[:self.config.sample_size]
                data_type = str(df[display_name].dtype)
                
                desc = self._generate_description(display_name, content, data_type)
                descriptions.append(desc)
                col_info.append({
                    'display_name': display_name,
                    'base_name': base_name,
                    'data_type': data_type,
                    'content_sample': [str(x) for x in content[:5]],
                    'description': desc,
                })
            
            if not descriptions:
                return None, None
            
            # 3. Батчевый эмбеддинг всех описаний
            embeddings = self._batch_embed(descriptions)
            
            # 4. Формирование записей
            entries = []
            for info, emb in zip(col_info, embeddings):
                entries.append({
                    'column_name': info['display_name'],
                    'base_name': info['base_name'],
                    'description': info['description'],
                    'embedding': emb,
                    'data_type': info['data_type'],
                    'content_sample': info['content_sample'],
                    'table_id': table_id,
                })
            
            metadata = {
                'table_id': table_id,
                'num_rows': ground_truth['num_rows'],
                'num_columns': len(entries),
                'column_mapping': ground_truth['column_mapping'],
                'locale': ground_truth['locale'],
                'base_columns': ground_truth['selected_columns'],
            }
            
            return metadata, entries
            
        except Exception as e:
            logger.error(f"Ошибка обработки таблицы {table_id}: {e}")
            return None, None
    
    def generate(self):
        """Генерация полного датасета для Schema Matching."""
        logger.info(f"Начало генерации {self.config.num_tables} таблиц")
        logger.info(f"Используется {self.config.num_workers} потоков")
        logger.info(f"Нажмите Ctrl+C для остановки с сохранением")
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        start_time = time()
        successful = 0
        failed = 0
        
        def signal_handler(signum, frame):
            logger.warning("\nПолучен сигнал прерывания (Ctrl+C)")
            self.interrupted = True
        
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        try:
            with ThreadPoolExecutor(max_workers=self.config.num_workers) as executor:
                futures = {
                    executor.submit(self._process_table, tid): tid
                    for tid in range(self.config.num_tables)
                }
                
                with tqdm(total=self.config.num_tables, desc="Schema Matching Dataset") as pbar:
                    for future in as_completed(futures):
                        if self.interrupted:
                            logger.warning("Отменяем оставшиеся задачи...")
                            executor.shutdown(wait=False, cancel_futures=True)
                            break
                        
                        metadata, entries = future.result()
                        
                        if metadata and entries:
                            with self.lock:
                                self.all_metadata.append(metadata)
                                self.all_entries.extend(entries)
                                successful += 1
                        else:
                            with self.lock:
                                failed += 1
                        
                        pbar.set_postfix({'ok': successful, 'err': failed})
                        pbar.update(1)
        
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt перехвачен")
            self.interrupted = True
        
        finally:
            self._save_results(successful, failed, time() - start_time)
    
    def _save_results(self, successful: int, failed: int, elapsed: float):
        """Сохранение результатов."""
        self.all_metadata.sort(key=lambda x: x['table_id'])
        
        # Датасет
        dataset_path = os.path.join(self.config.output_dir, self.config.dataset_file)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.all_entries, f, ensure_ascii=False, indent=2)
        
        # Метаданные
        metadata_path = os.path.join(self.config.output_dir, self.config.metadata_file)
        meta_info = {
            'config': asdict(self.config),
            'stats': {
                'successful_tables': successful,
                'failed_tables': failed,
                'total_entries': len(self.all_entries),
                'elapsed_seconds': elapsed,
                'unique_base_names': len(set(e['base_name'] for e in self.all_entries)),
            },
            'tables': self.all_metadata,
        }
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
        
        # Выводим статистику
        logger.info("\n" + "=" * 70)
        logger.info("ИТОГИ ГЕНЕРАЦИИ ДАТАСЕТА SCHEMA MATCHING")
        logger.info("=" * 70)
        logger.info(f"Успешно:            {successful} таблиц")
        logger.info(f"Ошибок:             {failed}")
        logger.info(f"Записей в датасете: {len(self.all_entries)}")
        logger.info(f"Время:              {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
        
        if self.all_entries:
            from collections import Counter
            base_names = Counter(e['base_name'] for e in self.all_entries)
            logger.info(f"Уникальных типов:   {len(base_names)}")
            logger.info(f"\nРаспределение по типам:")
            for bn, cnt in base_names.most_common():
                logger.info(f"  {bn:30s}: {cnt:5d}")
            
            # Определяем размерность эмбеддингов
            emb_dim = len(self.all_entries[0]['embedding'])
            logger.info(f"\nРазмерность эмбеддингов: {emb_dim}")
        
        logger.info(f"\nДатасет: {dataset_path}")
        logger.info(f"Метаданные: {metadata_path}")
        logger.info("=" * 70)
