"""Порча значений ячеек для генерации синтетических данных.

Этапы 2b/3b из схемы Синтетический датасет.canvas:
— Typo, Format, Drop.
"""

from __future__ import annotations

import random
import string

import pandas as pd


# ------------------------------------------------------------------ #
#  Элементарные операции
# ------------------------------------------------------------------ #

def add_typo(text: str, char_prob: float = 0.08) -> str:
    """Случайная перестановка соседних символов."""
    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if random.random() < char_prob:
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2  # не трогать те же символы повторно
        else:
            i += 1
    return "".join(chars)


def change_format(value: str) -> str:
    """Менять формат числовых / годовых значений."""
    stripped = value.strip()
    if stripped.isdigit() and len(stripped) == 4:
        return "'" + stripped[2:]
    if stripped.replace(".", "", 1).isdigit():
        try:
            num = float(stripped)
            return f"{num:,.1f}" if num > 100 else stripped
        except ValueError:
            pass
    return value


def drop_tokens(text: str, drop_prob: float = 0.3) -> str:
    """Случайное удаление токенов (слов)."""
    tokens = text.split()
    if len(tokens) <= 1:
        return text
    kept = [t for t in tokens if random.random() > drop_prob]
    return " ".join(kept) if kept else tokens[0]


# ------------------------------------------------------------------ #
#  Комбинированная порча
# ------------------------------------------------------------------ #

def corrupt_value(
    value: str,
    typo_prob: float = 0.3,
    format_prob: float = 0.2,
    drop_prob: float = 0.2,
) -> str:
    """Применить одну из стратегий порчи к значению ячейки."""
    value = str(value)
    r = random.random()
    if r < typo_prob:
        return add_typo(value)
    if r < typo_prob + format_prob:
        return change_format(value)
    if r < typo_prob + format_prob + drop_prob:
        return drop_tokens(value)
    return value  # без изменений


def corrupt_row(row: pd.Series, corruption_prob: float = 0.4) -> pd.Series:
    """Случайно испортить несколько ячеек строки."""
    new_row = row.copy()
    for col in new_row.index:
        if random.random() < corruption_prob:
            new_row[col] = corrupt_value(str(new_row[col]))
    return new_row


def corrupt_dataframe(
    df: pd.DataFrame,
    row_prob: float = 0.5,
    cell_prob: float = 0.4,
) -> pd.DataFrame:
    """Создать «шумную» копию таблицы, портя значения.

    Все столбцы приводятся к строковому типу, т.к. corrupt_value возвращает str.
    """
    noisy = df.astype(str).copy()
    for idx in noisy.index:
        if random.random() < row_prob:
            noisy.loc[idx] = corrupt_row(noisy.loc[idx], corruption_prob=cell_prob)
    return noisy
