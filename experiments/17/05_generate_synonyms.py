"""05_generate_synonyms.py — LLM-генерация синонимов для supplier-view аугментации.

Выход: data/raw_ru/<name>/synonyms.json
  {
    "columns": { "brand": ["Марка", "make", "производитель", "Бренд"] },
    "values":  { "bodyType": { "седан": ["sedan", "Sedan", "4-дверный"] } }
  }

Запускать на nvidia-server (нужен Ollama):
  cd experiments/17 && uv run python 05_generate_synonyms.py
  cd experiments/17 && uv run python 05_generate_synonyms.py --only cars_ru devices
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from table_unifier.ollama_client import OllamaClient

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_RU = ROOT / "data" / "raw_ru"

# Текстовое описание домена для каждого датасета (даём LLM контекст)
DOMAINS = {
    "lamoda":       "интернет-магазин одежды и аксессуаров Lamoda",
    "cars_ru":      "объявления о продаже автомобилей на российском рынке",
    "auto_ru":      "объявления о продаже автомобилей с сайта auto.ru",
    "auto_ru_2020": "объявления о продаже автомобилей с сайта auto.ru (2020 год)",
    "ozon":         "товары интернет-магазина Ozon",
    "devices":      "электронные устройства: мониторы, роутеры, сканеры, проекторы",
}

# Максимум уникальных значений на категориальную колонку (обрезаем топ-N)
MAX_UNIQUE_VALS = 40
# Максимум колонок в одном batch-промпте (чтобы не переполнить контекст)
MAX_COLS_PER_BATCH = 30
# Сколько синонимов просить для колонок и значений
N_COL_SYNS = 4
N_VAL_SYNS = 3


# ------------------------------------------------------------------ #
#  Namespace-префикс (about., technical_specs. и т.п.)
# ------------------------------------------------------------------ #

def _detect_namespace_prefix(cols: list[str], threshold: float = 0.3) -> str:
    """Определить доминирующий namespace-префикс вида 'about.' или 'technical_specs.'.

    Если >threshold колонок начинаются с одного и того же 'prefix.', возвращает его.
    Нужно чтобы не отправлять LLM 'about.Цвет' — она генерирует 'about.color' вместо 'color'.
    """
    from collections import Counter
    prefixes = Counter(
        col.split(".", 1)[0] + "."
        for col in cols
        if "." in col and not col.startswith("_")
    )
    if not prefixes:
        return ""
    top, count = prefixes.most_common(1)[0]
    return top if count / len(cols) >= threshold else ""


def _strip_prefix(text: str, prefix: str) -> str:
    if prefix and text.startswith(prefix):
        return text[len(prefix):]
    return text


# ------------------------------------------------------------------ #
#  JSON-парсинг из ответа LLM
# ------------------------------------------------------------------ #

def _strip_thinking(text: str) -> str:
    """Убрать <think>...</think> блоки (qwen3 thinking mode)."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def _extract_json(text: str) -> dict | list | None:
    """Извлечь JSON из ответа LLM, который может содержать markdown/thinking."""
    text = _strip_thinking(text).strip()
    # Прямой парс
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # JSON в markdown-блоке
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            pass
    # Первый объект в тексте
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    return None


def _llm_json(client: OllamaClient, prompt: str, retries: int = 2) -> dict:
    """Вызвать LLM и вернуть распарсенный dict. При неудаче — пустой dict."""
    for attempt in range(retries + 1):
        try:
            raw = client.generate(prompt)
            result = _extract_json(raw)
            if isinstance(result, dict):
                return result
            log.warning("LLM вернул не dict (attempt %d): %s", attempt, str(raw)[:200])
        except Exception as exc:
            log.warning("LLM ошибка (attempt %d): %s", attempt, exc)
    return {}


# ------------------------------------------------------------------ #
#  Промпты
# ------------------------------------------------------------------ #

def _col_synonyms_prompt(cols: list[str], domain: str) -> str:
    col_list = "\n".join(f"- {c}" for c in cols)
    half = N_COL_SYNS // 2
    return (
        f"Ты — эксперт по данным в области «{domain}».\n"
        f"Для каждого названия колонки придумай РОВНО {N_COL_SYNS} альтернативных названия "
        f"(строго {N_COL_SYNS} элемента в массиве — не меньше, не больше).\n\n"
        "Требования:\n"
        f"- 3 названия на РУССКОМ языке: как эту колонку назвал бы другой российский "
        "поставщик, магазин или CRM-система (синоним, сокращение, отраслевой термин)\n"
        f"- 1 название на АНГЛИЙСКОМ в нотации snake_case: как принято в IT-системах "
        "и зарубежных каталогах\n"
        "- НЕ меняй только регистр — это не синоним\n"
        "- НЕ добавляй namespace-префиксы исходной таблицы\n\n"
        f"Колонки:\n{col_list}\n\n"
        f'Ответь ТОЛЬКО JSON объектом (строго {N_COL_SYNS} элемента в каждом массиве):\n'
        '{"название_колонки": ["рус1", "рус2", "рус3", "en_1"], ...}\n'
        "Без пояснений. Только JSON."
    )


def _val_synonyms_prompt(col: str, vals: list[str], domain: str) -> str:
    val_list = "\n".join(f"- {v}" for v in vals)
    return (
        f"Ты — эксперт по данным в области «{domain}».\n"
        f"Колонка: «{col}».\n"
        f"Для каждого значения придумай РОВНО {N_VAL_SYNS} альтернативных варианта "
        f"(строго {N_VAL_SYNS} элемента — не меньше, не больше).\n\n"
        "Требования:\n"
        "- 2 варианта на РУССКОМ языке: как это значение записал бы другой российский "
        "поставщик, магазин или база данных (синоним, сокращение, отраслевой термин)\n"
        "- 1 вариант на АНГЛИЙСКОМ: перевод или общепринятое обозначение в IT-системах\n"
        "- НЕ меняй только регистр — это не вариант\n"
        "- Вариант должен называть ту же сущность, а не быть похожим словом\n\n"
        f"Значения:\n{val_list}\n\n"
        f'Ответь ТОЛЬКО JSON объектом (строго {N_VAL_SYNS} элемента в каждом массиве):\n'
        '{"значение": ["рус1", "рус2", "en_1"], ...}\n'
        "Без пояснений. Только JSON."
    )


# ------------------------------------------------------------------ #
#  Определение categorical колонок
# ------------------------------------------------------------------ #

def _categorical_cols(df: pd.DataFrame, max_unique: int = MAX_UNIQUE_VALS) -> list[str]:
    """Колонки с object-типом и ≤ max_unique уникальных непустых значений."""
    result = []
    for col in df.columns:
        if col == "id":
            continue
        if df[col].dtype != object:
            continue
        n_uniq = df[col].dropna().nunique()
        if 2 <= n_uniq <= max_unique:
            result.append(col)
    return result


# ------------------------------------------------------------------ #
#  Генерация синонимов для одного датасета
# ------------------------------------------------------------------ #

def generate_synonyms(name: str, client: OllamaClient) -> None:
    parquet_path = RAW_RU / name / "clean.parquet"
    out_path = RAW_RU / name / "synonyms.json"

    if not parquet_path.exists():
        log.warning("[%s] clean.parquet не найден — пропускаем", name)
        return

    # Загружаем существующий файл (идемпотентность)
    existing: dict = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except Exception:
            pass

    col_syns: dict = existing.get("columns", {})
    val_syns: dict = existing.get("values", {})

    df = pd.read_parquet(parquet_path)
    domain = DOMAINS.get(name, name)
    feature_cols = [c for c in df.columns if c != "id"]

    log.info("[%s] %d колонок, домен: %s", name, len(feature_cols), domain)

    # Определяем namespace-префикс (about., technical_specs. и т.п.)
    ns_prefix = _detect_namespace_prefix(feature_cols)
    if ns_prefix:
        log.info("[%s] namespace-префикс: '%s' — будет убран при запросе к LLM", name, ns_prefix)

    # ── Синонимы имён колонок ──────────────────────────────────────
    new_cols = [c for c in feature_cols if c not in col_syns]
    if new_cols:
        log.info("[%s] генерируем синонимы для %d колонок...", name, len(new_cols))
        for batch_start in range(0, len(new_cols), MAX_COLS_PER_BATCH):
            batch = new_cols[batch_start: batch_start + MAX_COLS_PER_BATCH]

            # Стрипаем prefix перед отправкой в LLM: "about.Цвет" → "Цвет"
            stripped_batch = [_strip_prefix(c, ns_prefix) for c in batch]
            # Карта stripped → original (для случая коллизий при strip)
            stripped_to_orig: dict[str, str] = {}
            for orig, stripped in zip(batch, stripped_batch):
                stripped_to_orig.setdefault(stripped, orig)

            prompt = _col_synonyms_prompt(stripped_batch, domain)
            result = _llm_json(client, prompt)

            for orig_col, stripped_col in zip(batch, stripped_batch):
                syns = result.get(stripped_col, [])
                # Фильтруем: непустые строки, не равные ни stripped-, ни original-имени
                # Также стрипаем prefix если LLM всё равно его добавил
                syns = [
                    _strip_prefix(str(s).strip(), ns_prefix)
                    for s in syns
                    if isinstance(s, str) and str(s).strip()
                ]
                syns = [s for s in syns if s and s != orig_col and s != stripped_col]
                col_syns[orig_col] = syns[:N_COL_SYNS]
                if syns:
                    log.info("  %s → %s", orig_col, syns[:2])
                else:
                    log.warning("  %s → нет синонимов", orig_col)
            _save(out_path, col_syns, val_syns)
    else:
        log.info("[%s] синонимы колонок уже есть, пропускаем", name)

    # ── Синонимы значений (только categorical) ────────────────────
    cat_cols = _categorical_cols(df)
    new_cat_cols = [c for c in cat_cols if c not in val_syns]
    if new_cat_cols:
        log.info("[%s] генерируем синонимы значений для %d categorical колонок...",
                 name, len(new_cat_cols))
        for col in new_cat_cols:
            vals = df[col].dropna().value_counts().head(MAX_UNIQUE_VALS).index.tolist()
            vals = [str(v) for v in vals]
            # Стрипаем prefix из имени колонки для контекста промпта
            prompt = _val_synonyms_prompt(_strip_prefix(col, ns_prefix), vals, domain)
            result = _llm_json(client, prompt)
            col_val_map: dict[str, list] = {}
            for v in vals:
                syns = result.get(v, [])
                syns = [s for s in syns if isinstance(s, str) and s.strip() and s != v]
                col_val_map[v] = syns[:N_VAL_SYNS]
            val_syns[col] = col_val_map
            n_covered = sum(1 for s in col_val_map.values() if s)
            log.info("  %s: %d/%d значений с синонимами", col, n_covered, len(vals))
            _save(out_path, col_syns, val_syns)
    else:
        log.info("[%s] синонимы значений уже есть, пропускаем", name)

    _save(out_path, col_syns, val_syns)
    log.info("[%s] сохранено: %s", name, out_path)


def _save(path: Path, col_syns: dict, val_syns: dict) -> None:
    path.write_text(
        json.dumps({"columns": col_syns, "values": val_syns}, ensure_ascii=False, indent=2)
    )


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

ALL_DATASETS = list(DOMAINS.keys())


def main() -> None:
    ap = argparse.ArgumentParser(description="LLM-генерация синонимов для exp 17")
    ap.add_argument("--only", nargs="*", choices=ALL_DATASETS)
    args = ap.parse_args()

    datasets = args.only or ALL_DATASETS
    client = OllamaClient()

    for name in datasets:
        log.info("\n=== %s ===", name)
        try:
            generate_synonyms(name, client)
        except Exception as exc:
            log.exception("[%s] FAILED: %s", name, exc)


if __name__ == "__main__":
    main()
