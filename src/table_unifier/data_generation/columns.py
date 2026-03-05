"""
Единый реестр шаблонов столбцов + утилиты для генерации таблиц.

Объединяет шаблоны из Schema Matching (16 типов) и Entity Resolution (9 типов)
в один набор. Включает:
- COLUMN_TEMPLATES: все 16+ типов столбцов с генераторами, вариантами названий
- EntityPool: пул канонических сущностей для контролируемого пересечения (ER)
- ColumnNameVariator: вариации названий столбцов (опечатки, сокращения, регистр)
- Функции пертурбации значений ячеек (для ER дубликатов)
- Функции сборки шаблонов с учётом extra_* расширений из конфига
"""

import copy
import random
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

try:
    from faker import Faker
except ImportError:
    raise ImportError("Faker не установлен! pip install faker")

from .config import DataGenConfig

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Шаблоны столбцов (объединённый набор SM + ER)
# ═══════════════════════════════════════════════════════════════
#
# Столбцы делятся на 2 класса:
#   entity_column=True  — свойства сущности (article, product_name, ...)
#                         Хранимы в EntityPool, участвуют в ER дубликатах.
#   entity_column=False — табличные/вычисляемые (item_number, total_sum, ...)
#                         Генерируются per-row при создании DataFrame.

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
        'generator': lambda fake, idx: idx + 1,
        'entity_column': False,
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
        ]),
        'entity_column': True,
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
        ]),
        'entity_column': True,
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
        ]),
        'entity_column': True,
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
        ]),
        'entity_column': True,
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
        'generator': lambda fake, idx: fake.country(),
        'entity_column': True,
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
        ]),
        'entity_column': True,
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
        'generator': lambda fake, idx: random.choice(['шт.', 'кг', 'л', 'м', 'упак.', 'компл.', 'пар']),
        'entity_column': True,
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
        ),
        'entity_column': True,
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
        'generator': lambda fake, idx: random.choice([0, 0, 0, 5, 10, 15, 20]),
        'entity_column': False,
    },
    'price_after_discount': {
        'description': 'Цена со скидкой',
        'variants': {
            'ru': ['Цена со скидкой', 'Отпускная цена', 'Цена продажи', 'Финальная цена'],
            'en': ['Discounted Price', 'Sale Price', 'Final Price', 'Net Price']
        },
        'data_type': 'float',
        'is_calculated': True,
        'entity_column': False,
    },
    'vat_rate': {
        'description': 'Ставка НДС в процентах',
        'variants': {
            'ru': ['Ставка НДС', '% НДС', 'НДС %', 'Налог'],
            'en': ['VAT Rate', 'VAT %', 'Tax Rate', 'Tax %']
        },
        'data_type': 'int',
        'generator': lambda fake, idx: random.choice([0, 10, 20]),
        'entity_column': False,
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
        'is_calculated': True,
        'entity_column': False,
    },
    'vat_amount': {
        'description': 'Сумма НДС',
        'variants': {
            'ru': ['Сумма НДС', 'НДС', 'В т.ч. НДС', 'Налог (сумма)'],
            'en': ['VAT Amount', 'VAT', 'Tax Amount', 'Tax']
        },
        'data_type': 'float',
        'is_calculated': True,
        'entity_column': False,
    },
    'total_with_vat': {
        'description': 'Общая стоимость с НДС',
        'variants': {
            'ru': ['Всего с НДС', 'Итого с НДС', 'К оплате', 'Сумма к оплате'],
            'en': ['Total with VAT', 'Grand Total', 'Total Incl. VAT', 'Amount Due']
        },
        'data_type': 'float',
        'is_calculated': True,
        'entity_column': False,
    },
    'notes': {
        'description': 'Дополнительные примечания',
        'variants': {
            'ru': ['Примечание', 'Комментарий', 'Доп. информация', 'Заметки', 'Прим.'],
            'en': ['Notes', 'Comments', 'Remarks', 'Additional Info']
        },
        'data_type': 'str',
        'generator': lambda fake, idx: fake.sentence(nb_words=6) if random.random() > 0.5 else '',
        'entity_column': True,
    },
}

# Обязательные столбцы (всегда присутствуют в каждой таблице)
MANDATORY_COLUMNS = [
    'item_number', 'article', 'product_name', 'product_group',
    'quantity', 'unit', 'base_price', 'total_sum'
]

# Опциональные столбцы (добавляются случайно)
OPTIONAL_COLUMNS = [
    'manufacturer', 'country_of_origin', 'discount_percent',
    'price_after_discount', 'vat_rate', 'vat_amount', 'total_with_vat', 'notes'
]

# Обязательные entity-столбцы для ER (должны быть в каждой ER-таблице)
ER_REQUIRED_ENTITY_COLUMNS = ['article', 'product_name', 'base_price', 'quantity']


# ═══════════════════════════════════════════════════════════════
# Функции сборки шаблонов с учётом конфига
# ═══════════════════════════════════════════════════════════════

def build_column_templates(config: DataGenConfig) -> Dict:
    """Объединить встроенные шаблоны с extra_* расширениями из конфига."""
    templates = copy.deepcopy(COLUMN_TEMPLATES)

    # Дополнительные варианты названий
    for base_name, locale_variants in config.extra_column_variants.items():
        if base_name in templates:
            for lang, names_list in locale_variants.items():
                if lang in templates[base_name]['variants']:
                    templates[base_name]['variants'][lang].extend(names_list)
                else:
                    templates[base_name]['variants'][lang] = list(names_list)

    # Расширенные генераторы
    if config.extra_product_groups:
        base = ['Электроника', 'Мебель', 'Инструменты', 'Канцелярия', 'Стройматериалы',
                'Сантехника', 'Автозапчасти', 'Офисная техника', 'Хозтовары', 'Текстиль']
        all_groups = base + list(config.extra_product_groups)
        templates['product_group']['generator'] = lambda fake, idx, _g=all_groups: random.choice(_g)

    if config.extra_units:
        base = ['шт.', 'кг', 'л', 'м', 'упак.', 'компл.', 'пар']
        all_units = base + list(config.extra_units)
        templates['unit']['generator'] = lambda fake, idx, _u=all_units: random.choice(_u)

    if config.extra_manufacturers:
        base = ['Samsung', 'LG', 'Sony', 'Bosch', 'Siemens', 'Philips', 'Hitachi']
        all_mfrs = base + list(config.extra_manufacturers)
        templates['manufacturer']['generator'] = lambda fake, idx, _m=all_mfrs: random.choice(
            [fake.company()] + _m
        )

    if config.extra_vat_rates:
        base = [0, 10, 20]
        all_vat = base + list(config.extra_vat_rates)
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
        templates['product_name']['generator'] = (
            lambda fake, idx, _s=all_suffixes, _p=base_prefixes_p: random.choice([
                f"{fake.word().capitalize()} {fake.company_suffix()}",
                f"{fake.word().capitalize()} {random.choice(_s)}",
                f"{random.choice(_p)} {fake.word().capitalize()}",
                f"{fake.word().capitalize()} {random.choice(_s)}",
            ])
        )

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

    # Пользовательские столбцы
    for col_name, col_def in config.custom_columns.items():
        if col_name in templates:
            continue
        variants = col_def.get('variants', {'ru': [col_name], 'en': [col_name]})
        data_type = col_def.get('data_type', 'str')
        values = col_def.get('values', [])
        description = col_def.get('description', col_name)
        is_entity = col_def.get('entity_column', False)

        entry = {
            'description': description,
            'variants': variants,
            'data_type': data_type,
            'entity_column': is_entity,
        }
        if values:
            entry['generator'] = lambda fake, idx, _v=list(values): random.choice(_v)
        elif data_type == 'int':
            entry['generator'] = lambda fake, idx: fake.random_int(1, 1000)
        elif data_type == 'float':
            entry['generator'] = lambda fake, idx: round(fake.random.uniform(1, 10000), 2)
        else:
            entry['generator'] = lambda fake, idx: fake.word()
        templates[col_name] = entry

    return templates


def build_column_lists(config: DataGenConfig) -> Tuple[List[str], List[str]]:
    """Возвращает (mandatory, optional) списки столбцов, включая custom."""
    mandatory = list(MANDATORY_COLUMNS)
    optional = list(OPTIONAL_COLUMNS)
    for col_name, col_def in config.custom_columns.items():
        if col_name in mandatory or col_name in optional:
            continue
        if col_def.get('mandatory', False):
            mandatory.append(col_name)
        else:
            optional.append(col_name)
    return mandatory, optional


def get_entity_columns(templates: Dict) -> List[str]:
    """Вернуть имена столбцов, являющихся entity-атрибутами (для EntityPool)."""
    return [k for k, v in templates.items() if v.get('entity_column', False)]


# ═══════════════════════════════════════════════════════════════
# Пертурбации значений (для ER дубликатов)
# ═══════════════════════════════════════════════════════════════

def perturb_string(value: str, prob: float = 0.3) -> str:
    """Опечатки, пропуски, удвоения, замена регистра."""
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
        if random.random() > 0.5:
            return value.replace(' ', '', 1)
        else:
            chars.insert(pos, ' ')
    return ''.join(chars)


def perturb_number(value, prob: float = 0.3):
    """Округление, малый шум, изменение формата."""
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
        noise = num * random.uniform(-0.01, 0.01)
        return round(num + noise, 2)
    elif perturbation == 'format':
        if isinstance(value, float):
            return int(num) if num == int(num) else round(num, 1)
    return value


def perturb_value(value, data_type: str, prob: float = 0.3):
    """Применить подходящую пертурбацию к значению ячейки."""
    if pd.isna(value) or value is None or str(value).strip() == '':
        return value
    if data_type in ('int', 'float'):
        return perturb_number(value, prob)
    return perturb_string(str(value), prob)


# ═══════════════════════════════════════════════════════════════
# Вариатор названий столбцов
# ═══════════════════════════════════════════════════════════════

class ColumnNameVariator:
    """Генерирует вариации названий столбцов (опечатки, сокращения, регистр)."""

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
        typo = random.choice(['swap', 'double', 'skip'])
        if typo == 'swap' and pos < len(chars) - 1:
            chars[pos], chars[pos + 1] = chars[pos + 1], chars[pos]
        elif typo == 'double' and chars[pos].isalpha():
            chars.insert(pos, chars[pos])
        elif typo == 'skip' and chars[pos].isalpha():
            chars.pop(pos)
        return ''.join(chars)

    def _add_abbreviation(self, text: str) -> str:
        if not self.include_abbreviations or random.random() > 0.5:
            return text
        abbrevs = {
            'Количество': 'Кол-во', 'Номер': '№', 'Единица измерения': 'Ед. изм.',
            'Примечание': 'Прим.', 'Артикул': 'Арт.', 'Производитель': 'Произв.',
            'Наименование': 'Наим.', 'Товар': 'Тов.', 'Цена': 'Ц.', 'Скидка': 'Ск.',
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
# Пул канонических сущностей
# ═══════════════════════════════════════════════════════════════

class EntityPool:
    """Пул канонических сущностей для генерации пар таблиц (ER).

    Каждая сущность — словарь {base_column_name: value} с каноническими
    значениями для всех entity-столбцов. Используется для контролируемого
    создания дубликатов: одна сущность → разные пертурбации в двух таблицах.
    """

    def __init__(self, templates: Dict, config: DataGenConfig):
        self.fakers = {loc: Faker(loc) for loc in config.locales}
        self.entity_cols = get_entity_columns(templates)
        self.templates = templates
        self.entities = self._generate(config.num_entity_pool)
        logger.info(f"EntityPool: {len(self.entities)} сущностей, "
                     f"{len(self.entity_cols)} entity-столбцов")

    def _generate(self, size: int) -> List[Dict]:
        entities = []
        for i in range(size):
            fake = random.choice(list(self.fakers.values()))
            entity = {'_entity_id': i}
            for col_key in self.entity_cols:
                tmpl = self.templates[col_key]
                if 'generator' in tmpl:
                    entity[col_key] = tmpl['generator'](fake, i)
            entities.append(entity)
        return entities

    def sample(self, n: int) -> List[Dict]:
        """Выбрать n случайных сущностей (без повторения)."""
        n = min(n, len(self.entities))
        return random.sample(self.entities, n)


# ═══════════════════════════════════════════════════════════════
# Результат генерации пары таблиц
# ═══════════════════════════════════════════════════════════════

@dataclass
class TablePairData:
    """Данные пары таблиц с ground truth для Entity Resolution."""
    df_a: pd.DataFrame
    df_b: pd.DataFrame
    duplicate_pairs: List[Tuple[int, int]]
    column_mapping_a: Dict[str, str]   # display_name → base_name
    column_mapping_b: Dict[str, str]
    entity_ids_a: List[int]
    entity_ids_b: List[int]


# ═══════════════════════════════════════════════════════════════
# Генератор таблиц (общий для SM и ER)
# ═══════════════════════════════════════════════════════════════

class TableGenerator:
    """Генератор таблиц с разнообразными названиями столбцов.

    Используется в двух режимах:
    - standalone: генерация произвольной таблицы для SM
    - from_entities: заполнение таблицы данными из EntityPool для ER
    """

    def __init__(self, config: DataGenConfig, templates: Dict = None):
        self.config = config
        self.fakers = {locale: Faker(locale) for locale in config.locales}
        self.templates = templates or build_column_templates(config)
        self.mandatory_columns, self.optional_columns = build_column_lists(config)
        self.variator = ColumnNameVariator(
            variation_level=config.column_name_variation_level,
            include_typos=config.include_typos,
            include_abbreviations=config.include_abbreviations,
        )

    def select_columns(self) -> List[str]:
        """Выбрать подмножество столбцов для таблицы."""
        selected = list(self.mandatory_columns)
        n_opt = random.randint(
            self.config.min_optional_columns,
            min(self.config.max_optional_columns, len(self.optional_columns))
        )
        selected.extend(random.sample(self.optional_columns, n_opt))
        return selected

    def generate_column_name(self, base_name: str) -> Tuple[str, str]:
        """Вернуть (display_name, locale_key) для базового имени столбца."""
        template = self.templates[base_name]
        locale_key = random.choice(['ru', 'en'])
        base_variants = template['variants'].get(locale_key, template['variants'].get('ru', [base_name]))
        display_name = self.variator.generate_variant(base_variants)
        return display_name, locale_key

    def generate_column_names(
        self, columns: List[str]
    ) -> Tuple[List[str], Dict[str, str]]:
        """Сгенерировать display-имена для списка столбцов.

        Returns:
            (display_names_list, {display_name: base_name})
        """
        display_names = []
        mapping = {}
        for col_key in columns:
            display_name, _ = self.generate_column_name(col_key)
            # Уникальность
            original = display_name
            counter = 1
            while display_name in display_names:
                display_name = f"{original}_{counter}"
                counter += 1
            display_names.append(display_name)
            mapping[display_name] = col_key
        return display_names, mapping

    def generate_row_data(self, columns: List[str], fake: Faker, row_idx: int) -> Dict:
        """Сгенерировать одну строку таблицы."""
        row = {}
        for col_key in columns:
            tmpl = self.templates[col_key]
            if tmpl.get('is_calculated'):
                continue
            if 'generator' in tmpl:
                row[col_key] = tmpl['generator'](fake, row_idx)
        self._calculate_derived(row, columns)
        return row

    def entity_to_row(
        self, entity: Dict, columns: List[str], fake: Faker, row_idx: int,
        perturbation_prob: float = 0.0, missing_value_prob: float = 0.0,
    ) -> Dict:
        """Преобразовать сущность из EntityPool в строку таблицы.

        Entity-столбцы берутся из entity (с опциональной пертурбацией).
        Non-entity столбцы генерируются из шаблона.
        """
        row = {}
        for col_key in columns:
            tmpl = self.templates[col_key]
            if tmpl.get('is_calculated'):
                continue

            if tmpl.get('entity_column', False) and col_key in entity:
                value = entity[col_key]
                # Пропуск с вероятностью
                if (missing_value_prob > 0
                        and random.random() < missing_value_prob
                        and col_key not in ('article', 'product_name')):
                    row[col_key] = None
                    continue
                # Пертурбация
                if perturbation_prob > 0:
                    value = perturb_value(value, tmpl.get('data_type', 'str'), perturbation_prob)
                row[col_key] = value
            else:
                if 'generator' in tmpl:
                    row[col_key] = tmpl['generator'](fake, row_idx)

        self._calculate_derived(row, columns)
        return row

    def _calculate_derived(self, row: Dict, columns: List[str]):
        """Вычислить производные столбцы (price_after_discount, total_sum, etc.)."""
        if 'price_after_discount' in columns and 'base_price' in row and 'discount_percent' in row:
            base_price = row['base_price'] or 0
            discount_percent = row.get('discount_percent') or 0
            row['price_after_discount'] = round(
                base_price * (1 - discount_percent / 100), 2
            )

        price = row.get('price_after_discount') or row.get('base_price') or 0
        quantity = row.get('quantity') or 0

        if 'total_sum' in columns:
            row['total_sum'] = round(price * quantity, 2)
        if 'vat_rate' in row:
            vat_rate = row['vat_rate'] or 0
            if 'vat_amount' in columns:
                row['vat_amount'] = round(row.get('total_sum', price * quantity) * vat_rate / 100, 2)
            if 'total_with_vat' in columns:
                total = row.get('total_sum') or price * quantity
                vat = row.get('vat_amount') or total * vat_rate / 100
                row['total_with_vat'] = round(total + vat, 2)

    def generate_standalone_table(self) -> Tuple[pd.DataFrame, Dict]:
        """Сгенерировать standalone таблицу (для SM, без EntityPool)."""
        selected_columns = self.select_columns()
        display_names, col_mapping = self.generate_column_names(selected_columns)

        num_rows = random.randint(self.config.min_rows_per_table, self.config.max_rows_per_table)
        fake = random.choice(list(self.fakers.values()))

        rows = []
        for i in range(num_rows):
            rows.append(self.generate_row_data(selected_columns, fake, i))

        df_base = pd.DataFrame(rows)
        existing_cols = [c for c in selected_columns if c in df_base.columns]
        df_base = df_base[existing_cols]

        rename_map = {bn: dn for dn, bn in col_mapping.items() if bn in existing_cols}
        df = df_base.rename(columns=rename_map)
        final_display = [n for n in display_names if col_mapping[n] in existing_cols]
        df = df[final_display]

        meta = {
            'column_mapping': col_mapping,
            'num_rows': num_rows,
            'selected_columns': selected_columns,
        }
        return df, meta

    def generate_er_pair(self, entity_pool: EntityPool) -> TablePairData:
        """Сгенерировать пару таблиц с контролируемым overlap для ER."""
        config = self.config

        # 1. Общие сущности (дубликаты)
        n_common = random.randint(config.min_common_entities, config.max_common_entities)
        common_entities = entity_pool.sample(n_common)

        # 2. Уникальные сущности
        n_unique_a = random.randint(config.min_unique_entities, config.max_unique_entities)
        n_unique_b = random.randint(config.min_unique_entities, config.max_unique_entities)

        common_ids = {e['_entity_id'] for e in common_entities}
        remaining = [e for e in entity_pool.entities if e['_entity_id'] not in common_ids]
        unique_a = random.sample(remaining, min(n_unique_a, len(remaining)))
        remaining_ids = {e['_entity_id'] for e in unique_a}
        remaining2 = [e for e in remaining if e['_entity_id'] not in remaining_ids]
        unique_b = random.sample(remaining2, min(n_unique_b, len(remaining2)))

        # 3. Столбцы для каждой таблицы
        cols_a = self.select_columns()
        cols_b = self.select_columns()

        # 4. Display-имена
        disp_a, map_a = self.generate_column_names(cols_a)
        disp_b, map_b = self.generate_column_names(cols_b)

        fake = random.choice(list(self.fakers.values()))

        # 5. Строки таблицы A
        rows_a, eids_a = [], []
        for entity in common_entities:
            row = self.entity_to_row(
                entity, cols_a, fake, len(rows_a),
                perturbation_prob=config.perturbation_prob,
                missing_value_prob=config.missing_value_prob,
            )
            rows_a.append(row)
            eids_a.append(entity['_entity_id'])
        for entity in unique_a:
            row = self.entity_to_row(entity, cols_a, fake, len(rows_a))
            rows_a.append(row)
            eids_a.append(entity['_entity_id'])

        # 6. Строки таблицы B
        rows_b, eids_b = [], []
        for entity in common_entities:
            row = self.entity_to_row(
                entity, cols_b, fake, len(rows_b),
                perturbation_prob=config.perturbation_prob,
                missing_value_prob=config.missing_value_prob,
            )
            rows_b.append(row)
            eids_b.append(entity['_entity_id'])
        for entity in unique_b:
            row = self.entity_to_row(entity, cols_b, fake, len(rows_b))
            rows_b.append(row)
            eids_b.append(entity['_entity_id'])

        # 7. Перемешиваем
        idx_a = list(range(len(rows_a)))
        idx_b = list(range(len(rows_b)))
        random.shuffle(idx_a)
        random.shuffle(idx_b)
        rows_a = [rows_a[i] for i in idx_a]
        eids_a = [eids_a[i] for i in idx_a]
        rows_b = [rows_b[i] for i in idx_b]
        eids_b = [eids_b[i] for i in idx_b]

        # 8. DataFrames
        df_base_a = pd.DataFrame(rows_a)
        df_base_b = pd.DataFrame(rows_b)

        # Переименование в display-имена
        existing_a = [c for c in cols_a if c in df_base_a.columns]
        rename_a = {bn: dn for dn, bn in map_a.items() if bn in existing_a}
        df_a = df_base_a[existing_a].rename(columns=rename_a)
        final_disp_a = [n for n in disp_a if map_a[n] in existing_a]
        df_a = df_a[final_disp_a]

        existing_b = [c for c in cols_b if c in df_base_b.columns]
        rename_b = {bn: dn for dn, bn in map_b.items() if bn in existing_b}
        df_b = df_base_b[existing_b].rename(columns=rename_b)
        final_disp_b = [n for n in disp_b if map_b[n] in existing_b]
        df_b = df_b[final_disp_b]

        # 9. Ground truth дубликаты
        dup_pairs = []
        for i, ea in enumerate(eids_a):
            for j, eb in enumerate(eids_b):
                if ea == eb:
                    dup_pairs.append((i, j))

        return TablePairData(
            df_a=df_a, df_b=df_b,
            duplicate_pairs=dup_pairs,
            column_mapping_a=map_a, column_mapping_b=map_b,
            entity_ids_a=eids_a, entity_ids_b=eids_b,
        )
