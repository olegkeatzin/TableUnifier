"""Разметка реальных данных: LLM-агент с веб-поиском.

Сравнение с 12_label_real_data.py:
  - Использует gemma4:26b вместо qwen3.5:9b
  - Агент может вызывать веб-поиск для уточнения информации о компонентах
  - Обрабатывает пары по одной (не батчами) для корректной работы tool calling
  - Результат сохраняется отдельно для сравнения качества

Требования:
  - Ollama с моделью gemma4:26b
  - Интернет-доступ для DuckDuckGo поиска

Логи:
  - Консоль: прогресс и статистика (INFO)
  - data/labeled/agent_labeling.log: полный лог с рассуждениями модели и tool calls (DEBUG)
  - data/labeled/agent_traces.jsonl: структурированный trace каждой пары (JSON Lines)
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import ollama
import pandas as pd
from duckduckgo_search import DDGS
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Логирование: консоль (INFO) + файл (DEBUG)
# ------------------------------------------------------------------ #
LOG_DIR = Path("data/labeled")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# Консоль — только INFO+
_console = logging.StreamHandler()
_console.setLevel(logging.INFO)
_console.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
logger.addHandler(_console)

# Файл — всё включая DEBUG (рассуждения модели, tool calls)
_file_handler = logging.FileHandler(LOG_DIR / "agent_labeling.log", encoding="utf-8")
_file_handler.setLevel(logging.DEBUG)
_file_handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
logger.addHandler(_file_handler)

# ------------------------------------------------------------------ #
#  Пути
# ------------------------------------------------------------------ #
DATA_DIR = Path("data/labeled")
DEDUP_PATH = DATA_DIR / "candidates_dedup.parquet"
OUTPUT_PATH = DATA_DIR / "labeled_pairs_agent.parquet"
TRACE_PATH = DATA_DIR / "agent_traces.jsonl"

# ------------------------------------------------------------------ #
#  Параметры
# ------------------------------------------------------------------ #
DEFAULT_MODEL = "gemma4:26b"
MAX_TOOL_ROUNDS = 3          # макс. итераций tool calling на одну пару
SAVE_EVERY = 50              # сохранять каждые N пар
SEARCH_MAX_RESULTS = 3       # результатов веб-поиска на запрос
OLLAMA_HOST = "http://localhost:11434"


# ------------------------------------------------------------------ #
#  JSONL trace writer
# ------------------------------------------------------------------ #
_trace_file = None


def _open_trace():
    global _trace_file
    _trace_file = open(TRACE_PATH, "a", encoding="utf-8")


def _write_trace(record: dict) -> None:
    """Записать одну запись trace в JSONL."""
    if _trace_file is None:
        return
    _trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    _trace_file.flush()


# ------------------------------------------------------------------ #
#  Web search tool
# ------------------------------------------------------------------ #

def web_search(query: str) -> str:
    """Search the web for information about electronic components, product codes, or technical specifications.

    Args:
        query: Search query in Russian or English. Use product codes, part numbers,
               or component names to find datasheets and specifications.
    """
    try:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=SEARCH_MAX_RESULTS))
        if not results:
            return "Ничего не найдено."
        parts = []
        for r in results:
            parts.append(f"**{r['title']}**\n{r['body']}\nURL: {r['href']}")
        return "\n\n".join(parts)
    except Exception as e:
        return f"Ошибка поиска: {e}"


# ------------------------------------------------------------------ #
#  Промпт
# ------------------------------------------------------------------ #

SYSTEM_PROMPT = """\
Ты эксперт по сопоставлению промышленной номенклатуры электронных компонентов.

Тебе дана пара записей: одна из спецификации, другая из номенклатуры.
Определи, описывают ли они ОДИН И ТОТ ЖЕ конкретный товар/компонент.

У тебя есть инструмент web_search — используй его, если:
- Не уверен, что за компонент описан (незнакомый код, децимальный номер, артикул)
- Нужно уточнить, являются ли два обозначения синонимами одного изделия
- Нужно проверить, отличаются ли модификации (-01, -02) по параметрам

Правила:
- Децимальные номера (ВАКШ.xxx, НЯИТ.xxx) — если совпадают, это почти наверняка match
- Коды продукции и артикулы тоже сильный сигнал
- Наименования могут различаться сокращениями и порядком слов
- Разные модификации одного изделия (-01, -02, -04) — это РАЗНЫЕ товары (не match)
- Разные допуски (±5% vs ±10%) — это РАЗНЫЕ товары (не match)
- Разные номиналы (100 пФ vs 1000 пФ) — это РАЗНЫЕ товары (не match)

В конце ОБЯЗАТЕЛЬНО ответь JSON (без markdown-блока):
{"match": true/false, "confidence": "high"/"medium"/"low", "reasoning": "краткое обоснование"}"""


def build_user_message(spec_text: str, nom_text: str, cosine_sim: float) -> str:
    return (
        f"Спецификация: \"{spec_text}\"\n"
        f"Номенклатура: \"{nom_text}\"\n"
        f"Cosine similarity: {cosine_sim:.3f}"
    )


# ------------------------------------------------------------------ #
#  Агентный цикл
# ------------------------------------------------------------------ #

def label_one_pair(
    client: ollama.Client,
    model: str,
    pair_idx: int,
    spec_text: str,
    nom_text: str,
    cosine_sim: float,
) -> dict:
    """Разметить одну пару через агента с tool calling.

    Returns:
        {"match": bool, "confidence": str, "reasoning": str,
         "searches": int, "raw_response": str}
    """
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_message(spec_text, nom_text, cosine_sim)},
    ]

    # Trace для JSONL — собираем все шаги
    trace = {
        "pair_idx": pair_idx,
        "spec_text": spec_text,
        "nom_text": nom_text,
        "cosine_sim": cosine_sim,
        "steps": [],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    logger.debug("=" * 80)
    logger.debug("PAIR #%d  sim=%.3f", pair_idx, cosine_sim)
    logger.debug("  SPEC: %s", spec_text)
    logger.debug("  NOM:  %s", nom_text)

    searches = 0

    for round_i in range(MAX_TOOL_ROUNDS):
        response = client.chat(
            model=model,
            messages=messages,
            tools=[web_search],
        )

        msg = response.message

        # Логируем текст ответа модели (рассуждения)
        if msg.content:
            logger.debug("  [round %d] MODEL: %s", round_i, msg.content)
            trace["steps"].append({
                "type": "model_response",
                "round": round_i,
                "content": msg.content,
            })

        # Если нет tool calls — финальный ответ
        if not msg.tool_calls:
            logger.debug("  [round %d] FINAL (no tool calls)", round_i)
            result = _parse_final(msg.content, searches)
            trace["result"] = result
            trace["total_rounds"] = round_i + 1
            _write_trace(trace)
            _log_result(pair_idx, result)
            return result

        # Выполняем tool calls
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = tool_call.function.arguments

            if fn_name == "web_search":
                query = fn_args.get("query", "")
                logger.debug("  [round %d] TOOL CALL: web_search(%r)", round_i, query)

                t_start = time.monotonic()
                search_result = web_search(query)
                elapsed_ms = (time.monotonic() - t_start) * 1000

                searches += 1
                # Логируем результат поиска (первые 500 символов в файловый лог)
                logger.debug(
                    "  [round %d] TOOL RESULT (%d chars, %.0fms): %s",
                    round_i, len(search_result), elapsed_ms,
                    search_result[:500],
                )

                trace["steps"].append({
                    "type": "tool_call",
                    "round": round_i,
                    "function": fn_name,
                    "query": query,
                    "result_length": len(search_result),
                    "result_preview": search_result[:1000],
                    "elapsed_ms": round(elapsed_ms),
                })

                messages.append({"role": "tool", "content": search_result})
            else:
                logger.warning("  [round %d] UNKNOWN TOOL: %s(%s)", round_i, fn_name, fn_args)
                trace["steps"].append({
                    "type": "unknown_tool",
                    "round": round_i,
                    "function": fn_name,
                    "arguments": fn_args,
                })

    # Исчерпали раунды — делаем финальный запрос без tools
    logger.debug("  MAX ROUNDS reached, forcing final answer")
    messages.append({
        "role": "user",
        "content": "Дай финальный ответ в JSON: {\"match\": true/false, \"confidence\": \"high\"/\"medium\"/\"low\", \"reasoning\": \"...\"}",
    })
    response = client.chat(model=model, messages=messages)

    if response.message.content:
        logger.debug("  FORCED FINAL: %s", response.message.content)
        trace["steps"].append({
            "type": "forced_final",
            "content": response.message.content,
        })

    result = _parse_final(response.message.content, searches)
    trace["result"] = result
    trace["total_rounds"] = MAX_TOOL_ROUNDS + 1
    _write_trace(trace)
    _log_result(pair_idx, result)
    return result


def _log_result(pair_idx: int, result: dict) -> None:
    """Лог-строка итога для пары."""
    match_str = "MATCH" if result["match"] else "NO MATCH"
    logger.debug(
        "  RESULT: %s (conf=%s, searches=%d) — %s",
        match_str, result["confidence"], result["searches"],
        result["reasoning"][:120] if result["reasoning"] else "",
    )


def _parse_final(text: str, searches: int) -> dict:
    """Извлечь JSON из финального ответа."""
    raw = text.strip()
    # Ищем JSON в ответе
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        try:
            data = json.loads(raw[start : end + 1])
            return {
                "match": bool(data.get("match", False)),
                "confidence": data.get("confidence", "medium"),
                "reasoning": data.get("reasoning", ""),
                "searches": searches,
                "raw_response": raw,
            }
        except json.JSONDecodeError:
            pass

    logger.warning("Не удалось распарсить ответ: %s", raw[:200])
    return {
        "match": False,
        "confidence": "low",
        "reasoning": f"PARSE_ERROR: {raw[:200]}",
        "searches": searches,
        "raw_response": raw,
    }


# ------------------------------------------------------------------ #
#  Основной цикл разметки
# ------------------------------------------------------------------ #

def label_candidates(
    candidates: pd.DataFrame,
    client: ollama.Client,
    model: str,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Разметить кандидатов через агента.

    Resume-логика: пропускает строки, где label уже != -1.
    """
    # Инициализация колонок
    for col, default in [
        ("label", -1), ("confidence", ""), ("reasoning", ""),
        ("searches", 0), ("raw_response", ""),
    ]:
        if col not in candidates.columns:
            candidates[col] = default

    todo_mask = candidates["label"] == -1
    todo_count = todo_mask.sum()
    logger.info("Всего: %d, уже размечено: %d, осталось: %d",
                len(candidates), len(candidates) - todo_count, todo_count)

    if todo_count == 0:
        logger.info("Все пары уже размечены.")
        return candidates

    labeled_count = 0
    errors = 0
    total_searches = 0

    todo_indices = candidates.index[todo_mask].tolist()

    for i, idx in enumerate(tqdm(todo_indices, desc="Agent labeling")):
        row = candidates.loc[idx]

        try:
            result = label_one_pair(
                client, model, idx,
                row["spec_text"], row["nom_text"], row["cosine_sim"],
            )
            candidates.at[idx, "label"] = 1 if result["match"] else 0
            candidates.at[idx, "confidence"] = result["confidence"]
            candidates.at[idx, "reasoning"] = result["reasoning"]
            candidates.at[idx, "searches"] = result["searches"]
            candidates.at[idx, "raw_response"] = result["raw_response"]
            labeled_count += 1
            total_searches += result["searches"]

        except Exception as e:
            logger.error("Ошибка на паре %d: %s", idx, e)
            errors += 1

        # Периодическое сохранение
        if (i + 1) % SAVE_EVERY == 0:
            candidates.to_parquet(output_path)
            done = (candidates["label"] != -1).sum()
            logger.info(
                "  Checkpoint %d/%d: размечено %d/%d, поисков %d, ошибок %d",
                i + 1, len(todo_indices), done, len(candidates),
                total_searches, errors,
            )

    candidates.to_parquet(OUTPUT_PATH)
    logger.info(
        "Разметка завершена: %d новых меток, %d поисков, %d ошибок",
        labeled_count, total_searches, errors,
    )
    return candidates


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="LLM-агент с веб-поиском для разметки пар",
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL,
        help=f"Ollama модель (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--host", type=str, default=OLLAMA_HOST,
        help=f"Ollama host (default: {OLLAMA_HOST})",
    )
    parser.add_argument(
        "--input", type=str, default=str(DEDUP_PATH),
        help="Путь к candidates_dedup.parquet (default: из эксперимента 12)",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Путь для сохранения результатов (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Открываем trace
    _open_trace()
    logger.info("Trace: %s", TRACE_PATH)
    logger.info("Лог:   %s", LOG_DIR / "agent_labeling.log")

    # Загрузка кандидатов
    input_path = Path(args.input)
    if not input_path.exists():
        logger.error("Файл %s не найден. Сначала запустите 12_label_real_data.py", input_path)
        return

    # Если есть частичный результат — продолжаем с него
    if output_path.exists():
        logger.info("Найден частичный результат %s, продолжаем...", output_path)
        candidates = pd.read_parquet(output_path)
    else:
        candidates = pd.read_parquet(input_path)
        logger.info("Загружено %d кандидатных пар из %s", len(candidates), input_path)

    # Клиент
    client = ollama.Client(host=args.host)
    logger.info("Модель: %s, host: %s", args.model, args.host)

    # Разметка
    candidates = label_candidates(candidates, client, args.model, output_path)
    candidates.to_parquet(output_path)

    # Статистика
    labeled = candidates[candidates["label"] != -1]
    logger.info("--- Итоговая статистика ---")
    logger.info("  Размечено: %d / %d", len(labeled), len(candidates))
    if len(labeled) > 0:
        n_match = (labeled["label"] == 1).sum()
        n_nomatch = (labeled["label"] == 0).sum()
        logger.info("  Match: %d (%.1f%%)", n_match, 100 * n_match / len(labeled))
        logger.info("  No match: %d (%.1f%%)", n_nomatch, 100 * n_nomatch / len(labeled))
        for conf in ["high", "medium", "low"]:
            n = (labeled["confidence"] == conf).sum()
            if n > 0:
                logger.info("  Confidence %s: %d (%.1f%%)", conf, n, 100 * n / len(labeled))
        avg_searches = labeled["searches"].mean()
        with_search = (labeled["searches"] > 0).sum()
        logger.info("  Пар с веб-поиском: %d (%.1f%%)", with_search, 100 * with_search / len(labeled))
        logger.info("  Среднее поисков на пару: %.2f", avg_searches)

    elapsed = time.time() - t0
    logger.info("Общее время: %.1f мин", elapsed / 60)

    # Закрываем trace
    if _trace_file is not None:
        _trace_file.close()
        logger.info("Trace сохранён: %s", TRACE_PATH)


if __name__ == "__main__":
    main()
