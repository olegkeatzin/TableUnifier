"""Разметка реальных данных: LangGraph ReAct-агент с веб-поиском (v2).

Улучшения по сравнению с v1:
  - Строгие параметры блокинга (как в exp 12): threshold=0.7, top_k=10, dedup=3
  - Улучшенный промпт с few-shot примерами из ручной верификации (12_verify_labels)
  - Акцент на поэлементном сравнении параметров (допуск, модификация, серия)
  - Переиспользует кандидатов из exp 12 (candidates_dedup.parquet) через --skip-blocking

Сравнение с 12_label_real_data.py:
  - Использует gemma4:26b через LangGraph create_react_agent (нативный tool calling)
  - Агент может вызывать веб-поиск для уточнения информации о компонентах
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
from typing import Literal

import faiss
import numpy as np
import pandas as pd
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from pydantic import BaseModel, Field
from tqdm import tqdm

from table_unifier.dataset.embedding_generation import TokenEmbedder

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
RAW_DATA_DIR = Path("experiments/07_real_data_test")
DATA_DIR = Path("data/labeled")
EMBEDDINGS_CACHE = DATA_DIR / "embeddings_cache.npz"  # общий с экспериментом 12
CANDIDATES_PATH = DATA_DIR / "candidates_agent.parquet"
DEDUP_PATH = DATA_DIR / "candidates_agent_dedup.parquet"
# Файлы из эксперимента 12 (для --skip-blocking)
EXP12_DEDUP_PATH = DATA_DIR / "candidates_dedup.parquet"
EXP12_CANDIDATES_PATH = DATA_DIR / "candidates.parquet"
OUTPUT_PATH = DATA_DIR / "labeled_pairs_agent.parquet"
TRACE_PATH = DATA_DIR / "agent_traces.jsonl"

# ------------------------------------------------------------------ #
#  Параметры блокинга (как в эксперименте 12 — строгие)
# ------------------------------------------------------------------ #
TOP_K_FAISS = 10             # как в exp 12
TOP_K_DEDUP = 3              # как в exp 12
SIMILARITY_THRESHOLD = 0.7   # как в exp 12

# ------------------------------------------------------------------ #
#  Параметры агента
# ------------------------------------------------------------------ #
DEFAULT_MODEL = "gemma4:26b"
MAX_AGENT_ITERATIONS = 5     # макс. шагов агента (Thought→Action→Observation)
SAVE_EVERY = 50              # сохранять каждые N пар
OLLAMA_HOST = "http://localhost:11434"


# ------------------------------------------------------------------ #
#  1. Загрузка и очистка данных
# ------------------------------------------------------------------ #

def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузить и очистить оба файла."""
    logger.info("Загрузка данных...")

    nom = pd.read_excel(RAW_DATA_DIR / "Номенклатура полная.xlsx")
    spec = pd.read_excel(RAW_DATA_DIR / "сводная_спецификация.xlsx")

    nom = nom[nom["Наименование"].notna() & (nom["Наименование"].str.strip() != "")]
    nom = nom.reset_index(drop=True)

    def nom_text(row: pd.Series) -> str:
        parts = [str(row["Наименование"]).strip()]
        if pd.notna(row.get("Артикул ")) and str(row["Артикул "]).strip():
            parts.append(f"Артикул: {str(row['Артикул ']).strip()}")
        if pd.notna(row.get("Полное наименование")) and str(row["Полное наименование"]).strip():
            full = str(row["Полное наименование"]).strip()
            if full != parts[0]:
                parts.append(full)
        return " | ".join(parts)

    nom["text"] = nom.apply(nom_text, axis=1)

    spec = spec[spec["наименование"].notna() & (spec["наименование"].str.strip() != "")]
    spec = spec.reset_index(drop=True)

    def spec_text(row: pd.Series) -> str:
        parts = [str(row["наименование"]).strip()]
        if pd.notna(row.get("кодпродукции")) and str(row["кодпродукции"]).strip():
            parts.append(f"Код: {str(row['кодпродукции']).strip()}")
        if pd.notna(row.get("обозначениедокументанапоставку")) and str(row["обозначениедокументанапоставку"]).strip():
            parts.append(str(row["обозначениедокументанапоставку"]).strip())
        return " | ".join(parts)

    spec["text"] = spec.apply(spec_text, axis=1)

    logger.info("Номенклатура: %d строк, Спецификация: %d строк", len(nom), len(spec))
    return nom, spec


# ------------------------------------------------------------------ #
#  2. Эмбеддинги (переиспользует кэш из эксперимента 12)
# ------------------------------------------------------------------ #

def get_embeddings(
    nom: pd.DataFrame, spec: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Генерация или загрузка кэшированных эмбеддингов."""
    if EMBEDDINGS_CACHE.exists():
        logger.info("Загрузка кэшированных эмбеддингов из %s", EMBEDDINGS_CACHE)
        data = np.load(EMBEDDINGS_CACHE)
        return data["nom_emb"], data["spec_emb"]

    logger.info("Генерация эмбеддингов rubert-tiny2...")
    embedder = TokenEmbedder()

    logger.info("  Номенклатура: %d текстов...", len(nom))
    nom_emb = embedder.embed_sentences(nom["text"].tolist(), batch_size=128)

    logger.info("  Спецификация: %d текстов...", len(spec))
    spec_emb = embedder.embed_sentences(spec["text"].tolist(), batch_size=128)

    np.savez(EMBEDDINGS_CACHE, nom_emb=nom_emb, spec_emb=spec_emb)
    logger.info("Эмбеддинги сохранены в %s", EMBEDDINGS_CACHE)
    return nom_emb, spec_emb


# ------------------------------------------------------------------ #
#  3. FAISS blocking (ослабленный)
# ------------------------------------------------------------------ #

def faiss_blocking(
    nom_emb: np.ndarray,
    spec_emb: np.ndarray,
    top_k: int = TOP_K_FAISS,
) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой строки спецификации найти top_k ближайших из номенклатуры."""
    logger.info("FAISS blocking: top-%d из %d для %d запросов...",
                top_k, nom_emb.shape[0], spec_emb.shape[0])

    nom_normed = nom_emb / (np.linalg.norm(nom_emb, axis=1, keepdims=True) + 1e-9)
    spec_normed = spec_emb / (np.linalg.norm(spec_emb, axis=1, keepdims=True) + 1e-9)

    nom_normed = nom_normed.astype(np.float32)
    spec_normed = spec_normed.astype(np.float32)

    dim = nom_normed.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(nom_normed)

    distances, indices = index.search(spec_normed, top_k)
    logger.info("Blocking завершён. Средний cosine sim top-1: %.3f", distances[:, 0].mean())
    return distances, indices


def build_candidates(
    nom: pd.DataFrame,
    spec: pd.DataFrame,
    distances: np.ndarray,
    indices: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Собрать DataFrame кандидатных пар с дедупликацией."""
    rows = []
    for spec_idx in range(len(spec)):
        for k in range(distances.shape[1]):
            sim = distances[spec_idx, k]
            if sim < SIMILARITY_THRESHOLD:
                continue
            nom_idx = indices[spec_idx, k]
            rows.append({
                "spec_idx": spec_idx,
                "nom_idx": int(nom_idx),
                "spec_text": spec.iloc[spec_idx]["text"],
                "nom_text": nom.iloc[nom_idx]["text"],
                "cosine_sim": float(sim),
            })

    candidates = pd.DataFrame(rows)
    logger.info("Кандидатов (до дедупликации): %d", len(candidates))

    dedup = (
        candidates
        .drop_duplicates(subset=["spec_text", "nom_text"])
        .sort_values("cosine_sim", ascending=False)
        .groupby("spec_text")
        .head(TOP_K_DEDUP)
        .reset_index(drop=True)
    )
    logger.info(
        "После дедупликации (top-%d на уник. spec): %d пар (%d уник. spec, %d уник. nom)",
        TOP_K_DEDUP, len(dedup), dedup["spec_text"].nunique(), dedup["nom_text"].nunique(),
    )

    # Статистика
    for lo, hi in [(0.50, 0.70), (0.70, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.001)]:
        n = ((dedup["cosine_sim"] >= lo) & (dedup["cosine_sim"] < hi)).sum()
        logger.info("  sim [%.2f, %.2f): %d (%.1f%%)", lo, hi, n, 100 * n / len(dedup))

    return candidates, dedup


# ------------------------------------------------------------------ #
#  JSONL trace writer
# ------------------------------------------------------------------ #
_trace_file = None


def _open_trace(path: Path = TRACE_PATH):
    global _trace_file
    _trace_file = open(path, "a", encoding="utf-8")


def _write_trace(record: dict) -> None:
    """Записать одну запись trace в JSONL."""
    if _trace_file is None:
        return
    _trace_file.write(json.dumps(record, ensure_ascii=False) + "\n")
    _trace_file.flush()


# ------------------------------------------------------------------ #
#  Submit answer tool (вместо structured output)
# ------------------------------------------------------------------ #

class MatchResult(BaseModel):
    """Результат сопоставления двух записей."""
    match: bool = Field(description="true если записи описывают один и тот же товар")
    confidence: Literal["high", "medium", "low"] = Field(description="уверенность в ответе")
    reasoning: str = Field(description="одно предложение с обоснованием")


# Хранилище последнего ответа — заполняется при вызове submit_answer tool
_last_answer: dict | None = None


def _submit_answer(match: bool, confidence: str, reasoning: str) -> str:
    """Зафиксировать финальный ответ. Вызови этот tool ОДИН раз после анализа."""
    global _last_answer
    _last_answer = {
        "match": match,
        "confidence": confidence,
        "reasoning": reasoning,
    }
    return "Ответ принят."


SYSTEM_PROMPT = """\
Ты эксперт по сопоставлению номенклатуры электронных компонентов.
Определи, описывают ли две записи ОДИН И ТОТ ЖЕ товар (конкретную позицию для закупки).

=== ПРАВИЛА ===
1. MATCH — записи описывают один и тот же товар, пригодный для взаимной замены в заказе.
2. NO MATCH — записи описывают разные товары, даже если они из одной серии.

=== КРИТЕРИИ MATCH ===
- Совпадение децимального номера (АЖЯР.xxx, ШКАБ.xxx, ОЮ0.xxx) при одинаковых параметрах.
- Одинаковый тип + номинал + допуск + напряжение + модификация.
- Различия ТОЛЬКО в форматировании: пробелы, регистр, тире/дефис, запятая/точка в числах,
  порядок слов, сокращения (мкФ/мкф, кОм/К, пФ/пф).

=== КРИТЕРИИ NO MATCH (даже при похожих названиях!) ===
- Разные допуски: ±5% vs ±10%, ±10% vs ±20%, ±10% vs ±30% — это РАЗНЫЕ товары.
- Разные модификации: -01 vs -02, тип 1 (т1) vs тип 13 (т13), -А vs -Б.
- Разные серии/подтипы: К10-17А vs К10-17В, МПО vs Н20.
- Разные номиналы: 100 пФ vs 1000 пФ, 1.5 кОм vs 9.1 кОм.
- Разное напряжение: 10В vs 6.3В, 100В vs 250В.
- Один текст — общее описание (только номинал), другой — конкретная марка с артикулом.

=== ПРИМЕРЫ ===

MATCH:
  A: "Р1-12-0,125-1,5 кОм± 5 % -М | ШКАБ.434110.002 ТУ"
  B: "Р1-12-0,125-1,5 кОм±5 %-М-А | Р1-12-0,125-1,5 кОм±5 %-М-А ШКАБ.434110.002 ТУ"
  → MATCH: тот же резистор, -М-А это расширенное обозначение -М, ТУ совпадает.

MATCH:
  A: "Вставка плавкая ВП1-1В 2,0 А 250 В | ОЮ0.480.003 ТУ-Р"
  B: "Вставка плавкая ВП1-1В 2,0 А 250 В – ОЮ0.480.003 ТУ-Р"
  → MATCH: идентичные параметры, различие только в разделителе (| vs –).

NO MATCH:
  A: "0,125-9,1 кОм ± 5 % -А-Д-В | ОЖ0.467.093 ТУ"
  B: "Р1-12-0,125-9,1 К 5% | Артикул: 412050"
  → NO MATCH: A — общее описание без марки, B — конкретный Р1-12. Невозможно подтвердить.

NO MATCH:
  A: "К53-56А-6,3 В-47 мкФ±10% | АЖЯР.673546.001 ТУ"
  B: "К53-56А-6.3 В-47 мкФ±30 % | АЖЯР.673546.001 ТУ"
  → NO MATCH: допуск ±10% ≠ ±30%, хотя всё остальное совпадает.

NO MATCH:
  A: "100 пФ+-5 % 100 В | Код: 27.90.60.000 | ТУРБ 14576608.003-96"
  B: "К10-17А-МПО-100В-100 5% т1 | Артикул: 420837"
  → NO MATCH: A — общее описание конденсатора, B — конкретный К10-17А-МПО тип 1.

NO MATCH:
  A: "Н20-6800 пФ ± 20 %-2 | АЕЯР.431260.734 ТУ"
  B: "К10-17В-Н20-6800 20% т2 | Артикул: 420456"
  → NO MATCH: A описывает характеристику (Н20), B — конкретный К10-17В. Разные уровни спецификации.

=== ИНСТРУКЦИИ ===
- Сначала выдели ключевые параметры из обеих записей: тип, серия, номинал, допуск, напряжение, ТУ.
- Сравни параметры поэлементно. Если ЛЮБОЙ параметр отличается — NO MATCH.
- Используй web_search только если не можешь расшифровать обозначение компонента.
- При сомнении — NO MATCH с confidence="medium".
- ОБЯЗАТЕЛЬНО вызови submit_answer с финальным результатом.
"""


# ------------------------------------------------------------------ #
#  Агентный цикл (LangGraph)
# ------------------------------------------------------------------ #

def create_agent(model: str, host: str):
    """Создать LangGraph ReAct-агента с web_search и submit_answer tools."""
    from langchain_core.tools import StructuredTool

    llm = ChatOllama(
        model=model,
        base_url=host,
        num_predict=1024,
        temperature=0,
    )
    search = DuckDuckGoSearchRun(max_results=3)
    search.name = "web_search"
    search.description = (
        "Поиск в интернете. Используй для поиска информации об электронных "
        "компонентах, кодах продукции, децимальных номерах и технических "
        "характеристиках. Input: поисковый запрос на русском или английском."
    )

    submit_tool = StructuredTool.from_function(
        func=_submit_answer,
        name="submit_answer",
        description=(
            "Зафиксировать финальный ответ после анализа пары. "
            "ОБЯЗАТЕЛЬНО вызови этот tool ровно один раз в конце."
        ),
        args_schema=MatchResult,
    )

    return create_react_agent(
        model=llm,
        tools=[search, submit_tool],
        prompt=SYSTEM_PROMPT,
        name="component_matcher",
    )


def _extract_trace(messages: list) -> tuple[list[dict], int]:
    """Извлечь шаги и число поисков из сообщений LangGraph."""
    steps = []
    searches = 0
    for msg in messages:
        if msg.type == "ai":
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    logger.debug("  TOOL CALL: %s(%r)", tc["name"], tc["args"])
                    steps.append({
                        "type": "action",
                        "tool": tc["name"],
                        "input": tc["args"],
                    })
                    if tc["name"] == "web_search":
                        searches += 1
                    elif tc["name"] == "submit_answer":
                        steps[-1]["type"] = "submit"
            elif msg.content:
                logger.debug("  AI: %s", str(msg.content)[:500])
                steps.append({"type": "ai", "output": str(msg.content)})
        elif msg.type == "tool":
            output_str = str(msg.content)
            logger.debug("  OBSERVATION (%d chars): %s", len(output_str), output_str[:500])
            steps.append({
                "type": "observation",
                "output_length": len(output_str),
                "output_preview": output_str[:1000],
            })
    return steps, searches


def label_one_pair(
    graph,
    pair_idx: int,
    spec_text: str,
    nom_text: str,
    cosine_sim: float,
) -> dict:
    """Разметить одну пару через LangGraph-агента со structured output.

    Returns:
        {"match": bool, "confidence": str, "reasoning": str,
         "searches": int, "raw_response": str}
    """
    question = (
        f'Запись A: "{spec_text}"\n'
        f'Запись B: "{nom_text}"\n'
        f"Проанализируй и вызови submit_answer с результатом."
    )

    logger.debug("=" * 80)
    logger.debug("PAIR #%d  sim=%.3f", pair_idx, cosine_sim)
    logger.debug("  SPEC: %s", spec_text)
    logger.debug("  NOM:  %s", nom_text)

    global _last_answer
    _last_answer = None

    t_start = time.monotonic()
    last_err = None
    response = None
    for attempt in range(3):
        try:
            _last_answer = None  # сбросить перед каждой попыткой
            response = graph.invoke(
                {"messages": [HumanMessage(content=question)]},
                config={"recursion_limit": MAX_AGENT_ITERATIONS * 2 + 1},
            )
            break
        except Exception as e:
            last_err = e
            logger.warning("  Attempt %d/3 failed: %s", attempt + 1, e)
            if attempt < 2:
                time.sleep(2 ** attempt)
    elapsed_ms = (time.monotonic() - t_start) * 1000

    if response is None:
        logger.error("  All 3 attempts failed for pair #%d", pair_idx)
        raise last_err  # noqa: TRY200

    messages = response["messages"]
    steps, searches = _extract_trace(messages)

    if _last_answer is not None:
        result = {
            "match": _last_answer["match"],
            "confidence": _last_answer["confidence"],
            "reasoning": _last_answer["reasoning"],
            "searches": searches,
            "raw_response": json.dumps(_last_answer, ensure_ascii=False),
        }
    else:
        # Фолбэк: агент не вызвал submit_answer
        raw = ""
        for msg in reversed(messages):
            if msg.type == "ai" and msg.content:
                raw = str(msg.content)
                break
        logger.warning("submit_answer не вызван, raw: %s", raw[:200])
        result = {
            "match": False,
            "confidence": "low",
            "reasoning": f"NO_SUBMIT: {raw[:200]}",
            "searches": searches,
            "raw_response": raw,
        }

    steps.append({"type": "final", "output": result["raw_response"]})

    # Trace
    trace = {
        "pair_idx": pair_idx,
        "spec_text": spec_text,
        "nom_text": nom_text,
        "cosine_sim": cosine_sim,
        "steps": steps,
        "result": result,
        "elapsed_ms": round(elapsed_ms),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    _write_trace(trace)

    match_str = "MATCH" if result["match"] else "NO MATCH"
    logger.debug(
        "  RESULT: %s (conf=%s, searches=%d, %.0fms) — %s",
        match_str, result["confidence"], searches, elapsed_ms,
        result["reasoning"][:120],
    )
    return result


# ------------------------------------------------------------------ #
#  Основной цикл разметки
# ------------------------------------------------------------------ #

def label_candidates(
    candidates: pd.DataFrame,
    graph,
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
                graph, idx,
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
#  Объединение шардов
# ------------------------------------------------------------------ #

def merge_shards(output_path: Path) -> pd.DataFrame | None:
    """Найти и объединить шард-файлы в единый parquet + trace + лог.

    Возвращает объединённый DataFrame или None если шардов нет.
    """
    shard_files = sorted(output_path.parent.glob("labeled_pairs_agent_shard*.parquet"))
    if not shard_files:
        return None

    logger.info("Найдено %d шард-файлов: %s", len(shard_files), [f.name for f in shard_files])

    # Объединяем parquet
    dfs = []
    for f in shard_files:
        df = pd.read_parquet(f)
        logger.info("  %s: %d строк, размечено %d", f.name, len(df),
                     (df["label"] != -1).sum() if "label" in df.columns else 0)
        dfs.append(df)
    merged = pd.concat(dfs, ignore_index=True)

    # Дедупликация по spec_text + nom_text (на случай пересечений)
    before = len(merged)
    merged = merged.drop_duplicates(subset=["spec_text", "nom_text"], keep="first").reset_index(drop=True)
    if len(merged) < before:
        logger.info("  Удалено %d дубликатов при объединении", before - len(merged))

    # Дополнить пары из dedup, которые не попали ни в один шард
    if DEDUP_PATH.exists():
        dedup = pd.read_parquet(DEDUP_PATH)
        merged_keys = set(zip(merged["spec_text"], merged["nom_text"]))
        missing = dedup[
            ~dedup.apply(lambda r: (r["spec_text"], r["nom_text"]) in merged_keys, axis=1)
        ]
        if len(missing) > 0:
            logger.info("  Добавлено %d пар из dedup, отсутствующих в шардах", len(missing))
            merged = pd.concat([merged, missing], ignore_index=True)

    logger.info("Объединено: %d строк, размечено %d",
                len(merged), (merged["label"] != -1).sum() if "label" in merged.columns else 0)

    # Сохраняем объединённый файл
    merged.to_parquet(output_path)
    logger.info("Сохранено в %s", output_path)

    # Объединяем trace JSONL
    trace_shards = sorted(output_path.parent.glob("agent_traces.shard*.jsonl"))
    if trace_shards:
        with open(TRACE_PATH, "a", encoding="utf-8") as out:
            for tf in trace_shards:
                out.write(tf.read_text(encoding="utf-8"))
        logger.info("Trace объединён из %d шардов в %s", len(trace_shards), TRACE_PATH)

    # Объединяем логи
    log_shards = sorted(output_path.parent.glob("agent_labeling_shard*.log"))
    if log_shards:
        main_log = LOG_DIR / "agent_labeling.log"
        with open(main_log, "a", encoding="utf-8") as out:
            for lf in log_shards:
                out.write(f"\n{'='*60} {lf.name} {'='*60}\n")
                out.write(lf.read_text(encoding="utf-8"))
        logger.info("Логи объединены из %d шардов в %s", len(log_shards), main_log)

    # Переименовываем шард-файлы чтобы не объединять повторно
    for f in shard_files + trace_shards + log_shards:
        f.rename(f.with_suffix(f.suffix + ".merged"))
        logger.info("  Переименован: %s → %s", f.name, f.name + ".merged")

    return merged


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
        "--skip-blocking", action="store_true",
        help="Пропустить блокинг, загрузить существующие кандидаты",
    )
    parser.add_argument(
        "--output", type=str, default=str(OUTPUT_PATH),
        help=f"Путь для сохранения результатов (default: {OUTPUT_PATH})",
    )
    parser.add_argument(
        "--shard", type=int, default=0,
        help="Номер шарда (0-indexed). Каждый шард обрабатывает свою часть пар.",
    )
    parser.add_argument(
        "--num-shards", type=int, default=1,
        help="Общее количество шардов. При >1 каждый процесс пишет свой файл.",
    )
    args = parser.parse_args()

    # Шардирование: каждый процесс пишет в свой файл
    if args.num_shards > 1:
        output_path = Path(str(args.output).replace(".parquet", f"_shard{args.shard}.parquet"))
        trace_path = TRACE_PATH.with_suffix(f".shard{args.shard}.jsonl")
        log_path = LOG_DIR / f"agent_labeling_shard{args.shard}.log"
        # Перенастраиваем файловый лог на шард
        _file_handler.close()
        logger.removeHandler(_file_handler)
        fh = logging.FileHandler(log_path, encoding="utf-8")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-8s  %(message)s"))
        logger.addHandler(fh)
    else:
        output_path = Path(args.output)
        trace_path = TRACE_PATH
    output_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Открываем trace
    _open_trace(trace_path)
    logger.info("Shard: %d/%d", args.shard, args.num_shards)
    logger.info("Trace: %s", trace_path)
    logger.info("Output: %s", output_path)

    # При запуске одним агентом — объединить шарды если есть
    candidates = None
    if args.num_shards == 1:
        candidates = merge_shards(output_path)
        if candidates is not None:
            logger.info("Продолжаем разметку одним агентом после объединения шардов")

    # Если кандидаты ещё не загружены — стандартная логика
    # NB: --skip-blocking имеет приоритет над resume из output_path,
    # чтобы не подхватить старые результаты с другими параметрами блокинга.
    if candidates is None and args.skip_blocking:
        # Берём кандидатов из exp 12 (dedup или полные → дедуплицируем)
        if EXP12_DEDUP_PATH.exists():
            logger.info("Загрузка кандидатов из эксперимента 12: %s", EXP12_DEDUP_PATH)
            candidates = pd.read_parquet(EXP12_DEDUP_PATH)
        elif EXP12_CANDIDATES_PATH.exists():
            logger.info("Загрузка полных кандидатов из exp 12: %s (дедупликация...)", EXP12_CANDIDATES_PATH)
            all_cand = pd.read_parquet(EXP12_CANDIDATES_PATH)
            all_cand = all_cand[all_cand["cosine_sim"] >= SIMILARITY_THRESHOLD]
            candidates = (
                all_cand
                .drop_duplicates(subset=["spec_text", "nom_text"])
                .sort_values("cosine_sim", ascending=False)
                .groupby("spec_text")
                .head(TOP_K_DEDUP)
                .reset_index(drop=True)
            )
            logger.info("Дедуплицировано: %d пар", len(candidates))
        else:
            logger.error("Нет кандидатов из exp 12: ни %s, ни %s", EXP12_DEDUP_PATH, EXP12_CANDIDATES_PATH)
            return
    elif candidates is None:
        # Собственный блокинг с ослабленными параметрами
        logger.info("=== Блокинг (threshold=%.2f, top_k=%d, dedup=%d) ===",
                     SIMILARITY_THRESHOLD, TOP_K_FAISS, TOP_K_DEDUP)
        nom, spec = load_and_clean()
        nom_emb, spec_emb = get_embeddings(nom, spec)
        distances, indices = faiss_blocking(nom_emb, spec_emb)
        all_candidates, candidates = build_candidates(nom, spec, distances, indices)
        all_candidates.to_parquet(CANDIDATES_PATH)
        candidates.to_parquet(DEDUP_PATH)
        logger.info("Кандидаты сохранены: %s", DEDUP_PATH)

    # Resume: подтянуть уже проставленные метки из предыдущего запуска
    if output_path.exists() and args.skip_blocking:
        prev = pd.read_parquet(output_path)
        if len(prev) == len(candidates):
            # Тот же набор кандидатов — подхватываем метки
            logger.info("Resume: подхватываем метки из %s", output_path)
            candidates = prev
        else:
            logger.info(
                "Размер кандидатов изменился (%d → %d), начинаем разметку заново",
                len(prev), len(candidates),
            )

    # Шардирование: берём свою часть
    if args.num_shards > 1 and not output_path.exists():
        n = len(candidates)
        shard_size = n // args.num_shards
        start = args.shard * shard_size
        end = n if args.shard == args.num_shards - 1 else start + shard_size
        candidates = candidates.iloc[start:end].reset_index(drop=True)
        logger.info("Шард %d: строки %d-%d (%d пар)", args.shard, start, end - 1, len(candidates))

    # Разметка
    logger.info("Модель: %s, host: %s", args.model, args.host)
    graph = create_agent(args.model, args.host)
    candidates = label_candidates(candidates, graph, output_path)
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
