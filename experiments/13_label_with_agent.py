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

import faiss
import numpy as np
import pandas as pd
from langchain_classic.agents import AgentExecutor, create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.prompts import PromptTemplate
from langchain_ollama import ChatOllama
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
OUTPUT_PATH = DATA_DIR / "labeled_pairs_agent.parquet"
TRACE_PATH = DATA_DIR / "agent_traces.jsonl"

# ------------------------------------------------------------------ #
#  Параметры блокинга (ослабленные vs эксперимент 12)
# ------------------------------------------------------------------ #
TOP_K_FAISS = 30             # было 10 — больше кандидатов
TOP_K_DEDUP = 10             # было 3 — больше пар после дедупликации
SIMILARITY_THRESHOLD = 0.5   # было 0.7 — мягче порог

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
        shard_suffix = f"_shard{args.shard}"
        output_path = Path(args.output).with_suffix("") / f"shard_{args.shard}.parquet"
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
    global TRACE_PATH
    TRACE_PATH = trace_path
    _open_trace()
    logger.info("Shard: %d/%d", args.shard, args.num_shards)
    logger.info("Trace: %s", trace_path)
    logger.info("Output: %s", output_path)

    # Если есть частичный результат разметки — продолжаем с него
    if output_path.exists():
        logger.info("Найден частичный результат %s, продолжаем...", output_path)
        candidates = pd.read_parquet(output_path)
    elif args.skip_blocking and DEDUP_PATH.exists():
        logger.info("Загрузка существующих кандидатов из %s", DEDUP_PATH)
        candidates = pd.read_parquet(DEDUP_PATH)
    else:
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
    executor = create_agent(args.model, args.host)
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
