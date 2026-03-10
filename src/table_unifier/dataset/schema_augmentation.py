"""Аугментация схемы: генерация синонимов названий столбцов через LLM.

Этап 2a/3a из схемы Синтетический датасет.canvas:
— LLM Paraphrasing → Schema Injection.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from table_unifier.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

SYNONYM_PROMPT = (
    "Given the column name '{col_name}' from a {domain} database table, "
    "generate {n} alternative column names that could be used in a different "
    "system to represent the same data.\n"
    "Output only the names, one per line, without numbering or explanation. /no_think"
)


def generate_column_synonyms(
    client: OllamaClient,
    col_name: str,
    domain: str = "general",
    n: int = 5,
) -> list[str]:
    """Сгенерировать *n* синонимов для имени столбца *col_name*."""
    prompt = SYNONYM_PROMPT.format(col_name=col_name, domain=domain, n=n)
    response = client.generate(prompt)
    synonyms = [line.strip() for line in response.strip().splitlines() if line.strip()]
    return synonyms[:n]


def augment_schema(
    client: OllamaClient,
    columns: list[str],
    domain: str = "general",
    n_variants: int = 3,
) -> dict[str, list[str]]:
    """Для каждого столбца сгенерировать несколько синонимов.

    Возвращает ``{orig_name: [synonym_1, …]}``.
    """
    mapping: dict[str, list[str]] = {}
    for col in columns:
        logger.info("Генерация синонимов для '%s' …", col)
        synonyms = generate_column_synonyms(client, col, domain=domain, n=n_variants)
        mapping[col] = synonyms
        logger.info("  %s → %s", col, synonyms)
    return mapping


def apply_schema_injection(
    columns: list[str],
    synonym_map: dict[str, list[str]],
    variant_idx: int = 0,
) -> dict[str, str]:
    """Заменить оригинальные имена столбцов на синонимы (variant_idx-й вариант).

    Возвращает ``{orig_name: new_name}``.
    """
    result: dict[str, str] = {}
    for col in columns:
        synonyms = synonym_map.get(col, [col])
        idx = min(variant_idx, len(synonyms) - 1)
        result[col] = synonyms[idx]
    return result
