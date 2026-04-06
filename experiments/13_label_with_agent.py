"""Разметка реальных данных: LangChain ReAct-агент с веб-поиском.

Сравнение с 12_label_real_data.py:
  - Использует gemma4:26b через LangChain ReAct-агент (prompt-based tool calling)
  - Агент может вызывать веб-поиск для уточнения информации о компонентах
  - Не зависит от нативного tool calling модели — работает через Thought/Action/Observation
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
import re
import time
from pathlib import Path

import pandas as pd
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
from tqdm import tqdm

# ------------------------------------------------------------------ #
#  Логирование: консоль (INFO) + файл (DEBUG)
# ------------------------------------------------------------------ #
LOG_DIR = Path("data/labeled")
LOG_DIR.mkdir(parents=True, exist_ok=True)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
# Не наследовать от root logger
logger.propagate = False

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
MAX_AGENT_ITERATIONS = 5     # макс. шагов агента (Thought→Action→Observation)
SAVE_EVERY = 50              # сохранять каждые N пар
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
#  LangChain callback для логирования trace
# ------------------------------------------------------------------ #

class TraceCallback(BaseCallbackHandler):
    """Собирает шаги агента (мысли, tool calls, результаты) для логирования."""

    def __init__(self):
        self.steps: list[dict] = []
        self.searches = 0

    def on_agent_action(self, action, **kwargs):
        """Агент решил вызвать инструмент."""
        logger.debug("  ACTION: %s(%r)", action.tool, action.tool_input)
        self.steps.append({
            "type": "action",
            "tool": action.tool,
            "input": action.tool_input,
            "thought": action.log.strip(),
        })
        if action.tool == "web_search":
            self.searches += 1

    def on_tool_end(self, output, **kwargs):
        """Инструмент вернул результат."""
        output_str = str(output)
        logger.debug("  OBSERVATION (%d chars): %s", len(output_str), output_str[:500])
        self.steps.append({
            "type": "observation",
            "output_length": len(output_str),
            "output_preview": output_str[:1000],
        })

    def on_agent_finish(self, finish, **kwargs):
        """Агент дал финальный ответ."""
        logger.debug("  FINAL: %s", finish.log.strip())
        self.steps.append({
            "type": "final",
            "output": finish.return_values.get("output", ""),
            "log": finish.log.strip(),
        })


# ------------------------------------------------------------------ #
#  ReAct промпт
# ------------------------------------------------------------------ #

REACT_PROMPT = PromptTemplate.from_template("""\
Ты эксперт по сопоставлению номенклатуры электронных компонентов.
Определи, описывают ли две записи ОДИН И ТОТ ЖЕ товар.

Правила: совпадение децимальных номеров/артикулов = match. Разные модификации (-01/-02), допуски (±5%/±10%), номиналы = НЕ match. Сокращения и порядок слов могут отличаться.

Инструменты: {tools}
Названия: {tool_names}

Формат:
Thought: краткие рассуждения (2-3 предложения)
Action: <имя инструмента>
Action Input: <запрос>

Или финальный ответ:
Thought: краткий вывод
Final Answer: {{"match": true/false, "confidence": "high"/"medium"/"low", "reasoning": "одно предложение"}}

ВАЖНО: рассуждения должны быть КРАТКИМИ. Не пиши длинный анализ.

Вопрос: {input}
{agent_scratchpad}""")


# ------------------------------------------------------------------ #
#  Агентный цикл
# ------------------------------------------------------------------ #

def create_agent(model: str, host: str) -> AgentExecutor:
    """Создать LangChain ReAct-агента."""
    llm = ChatOllama(
        model=model,
        base_url=host,
        num_predict=4096,
        temperature=0,
    )
    search = DuckDuckGoSearchRun()
    search.name = "web_search"
    search.description = (
        "Поиск в интернете. Используй для поиска информации об электронных "
        "компонентах, кодах продукции, децимальных номерах и технических "
        "характеристиках. Input: поисковый запрос на русском или английском."
    )
    tools = [search]
    agent = create_react_agent(llm, tools, REACT_PROMPT)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        max_iterations=MAX_AGENT_ITERATIONS,
        handle_parsing_errors=True,
        verbose=False,  # логируем через callback
    )


def label_one_pair(
    executor: AgentExecutor,
    pair_idx: int,
    spec_text: str,
    nom_text: str,
    cosine_sim: float,
) -> dict:
    """Разметить одну пару через ReAct-агента.

    Returns:
        {"match": bool, "confidence": str, "reasoning": str,
         "searches": int, "raw_response": str}
    """
    question = (
        f'Спецификация: "{spec_text}"\n'
        f'Номенклатура: "{nom_text}"\n\n'
        f"Это один и тот же товар? Используй web_search если не уверен."
    )

    logger.debug("=" * 80)
    logger.debug("PAIR #%d  sim=%.3f", pair_idx, cosine_sim)
    logger.debug("  SPEC: %s", spec_text)
    logger.debug("  NOM:  %s", nom_text)

    cb = TraceCallback()
    t_start = time.monotonic()
    response = executor.invoke({"input": question}, config={"callbacks": [cb]})
    elapsed_ms = (time.monotonic() - t_start) * 1000

    raw_output = response.get("output", "")
    # Если в output нет JSON, ищем в полном логе агента (рассуждения + Final Answer)
    full_log = "\n".join(
        s.get("log", "") or s.get("content", "") or s.get("output", "")
        for s in cb.steps
    )
    result = _parse_final(raw_output, cb.searches)
    if result["reasoning"].startswith("PARSE_ERROR"):
        result = _parse_final(full_log + "\n" + raw_output, cb.searches)

    # Trace
    trace = {
        "pair_idx": pair_idx,
        "spec_text": spec_text,
        "nom_text": nom_text,
        "cosine_sim": cosine_sim,
        "steps": cb.steps,
        "result": result,
        "elapsed_ms": round(elapsed_ms),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_trace(trace)
    _log_result(pair_idx, result, elapsed_ms)
    return result


def _log_result(pair_idx: int, result: dict, elapsed_ms: float) -> None:
    """Лог-строка итога для пары."""
    match_str = "MATCH" if result["match"] else "NO MATCH"
    logger.debug(
        "  RESULT: %s (conf=%s, searches=%d, %.0fms) — %s",
        match_str, result["confidence"], result["searches"], elapsed_ms,
        result["reasoning"][:120] if result["reasoning"] else "",
    )


# ------------------------------------------------------------------ #
#  Парсинг JSON из ответа агента
# ------------------------------------------------------------------ #

def _parse_final(text: str, searches: int) -> dict:
    """Извлечь JSON из финального ответа.

    Устойчив к обрезанным ответам: пытается достроить незакрытый JSON.
    """
    raw = text.strip()
    start = raw.find("{")
    if start == -1:
        logger.warning("Нет JSON в ответе: %s", raw[:200])
        return _fallback_parse(raw, searches)

    fragment = raw[start:]

    # 1. Попытка парсинга как есть (полный JSON)
    end = fragment.rfind("}")
    if end != -1:
        try:
            data = json.loads(fragment[: end + 1])
            return _build_result(data, raw, searches)
        except json.JSONDecodeError:
            pass

    # 2. JSON обрезан — пробуем достроить
    repaired = fragment.rstrip()
    if repaired.count('"') % 2 == 1:
        repaired += '"'
    if not repaired.endswith("}"):
        repaired += "}"

    try:
        data = json.loads(repaired)
        logger.debug("  JSON repaired successfully")
        return _build_result(data, raw, searches)
    except json.JSONDecodeError:
        pass

    # 3. Regex-фолбэк
    return _fallback_parse(raw, searches)


def _build_result(data: dict, raw: str, searches: int) -> dict:
    return {
        "match": bool(data.get("match", False)),
        "confidence": data.get("confidence", "medium"),
        "reasoning": data.get("reasoning", ""),
        "searches": searches,
        "raw_response": raw,
    }


def _fallback_parse(raw: str, searches: int) -> dict:
    """Regex-фолбэк для извлечения match/confidence из сломанного JSON."""
    match_val = None
    m = re.search(r'"match"\s*:\s*(true|false)', raw, re.IGNORECASE)
    if m:
        match_val = m.group(1).lower() == "true"

    conf = "medium"
    m = re.search(r'"confidence"\s*:\s*"(high|medium|low)"', raw, re.IGNORECASE)
    if m:
        conf = m.group(1).lower()

    if match_val is not None:
        logger.debug("  Fallback parse OK: match=%s, confidence=%s", match_val, conf)
        return {
            "match": match_val,
            "confidence": conf,
            "reasoning": f"PARTIAL_PARSE: {raw[:200]}",
            "searches": searches,
            "raw_response": raw,
        }

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
    executor: AgentExecutor,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Разметить кандидатов через ReAct-агента.

    Resume-логика: пропускает строки, где label уже != -1.
    """
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
                executor, idx,
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
            logger.error("Ошибка на паре %d: %s", idx, e, exc_info=True)
            errors += 1

        # Периодическое сохранение (+ после первой пары для быстрого feedback)
        if (i + 1) % SAVE_EVERY == 0 or i == 0:
            candidates.to_parquet(output_path)
            done = (candidates["label"] != -1).sum()
            logger.info(
                "  Checkpoint %d/%d: размечено %d/%d, поисков %d, ошибок %d",
                i + 1, len(todo_indices), done, len(candidates),
                total_searches, errors,
            )

    candidates.to_parquet(output_path)
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
        description="LangChain ReAct-агент с веб-поиском для разметки пар",
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

    # Создаём агента
    logger.info("Модель: %s, host: %s", args.model, args.host)
    executor = create_agent(args.model, args.host)

    # Разметка
    candidates = label_candidates(candidates, executor, output_path)
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
