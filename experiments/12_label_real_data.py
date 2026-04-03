"""Разметка реальных данных: FAISS blocking + LLM labeling.

Pipeline:
  1. Загрузка и очистка номенклатуры (40K) и спецификации (18K)
  2. Генерация эмбеддингов rubert-tiny2 (CLS, 312-dim)
  3. FAISS blocking: для каждой строки спецификации → top-K кандидатов из номенклатуры
  4. Дедупликация: уникальные пары (spec_text, nom_text), top-3 на spec
  5. LLM-разметка кандидатных пар через Ollama (~26K пар, ~3 ч)
  6. Развёртка меток обратно на все дублированные строки
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from tqdm import tqdm

from table_unifier.config import OllamaConfig
from table_unifier.dataset.embedding_generation import TokenEmbedder
from table_unifier.ollama_client import OllamaClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Пути
# ------------------------------------------------------------------ #
DATA_DIR = Path("experiments/07_real_data_test")
OUTPUT_DIR = Path("data/labeled")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

EMBEDDINGS_CACHE = OUTPUT_DIR / "embeddings_cache.npz"
CANDIDATES_PATH = OUTPUT_DIR / "candidates.parquet"
DEDUP_PATH = OUTPUT_DIR / "candidates_dedup.parquet"
LABELED_PATH = OUTPUT_DIR / "labeled_pairs.parquet"
FULL_LABELED_PATH = OUTPUT_DIR / "labeled_pairs_full.parquet"

# ------------------------------------------------------------------ #
#  Параметры
# ------------------------------------------------------------------ #
TOP_K_FAISS = 10  # кандидатов из FAISS на строку спецификации
TOP_K_DEDUP = 3  # оставляем top-N после дедупликации
LLM_BATCH_SIZE = 5  # пар в одном промпте для LLM
SIMILARITY_THRESHOLD = 0.7  # минимальный cosine similarity для кандидата
SAVE_EVERY = 10  # сохранять каждые N батчей


# ------------------------------------------------------------------ #
#  1. Загрузка и очистка
# ------------------------------------------------------------------ #

def load_and_clean() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Загрузить и очистить оба файла."""
    logger.info("Загрузка данных...")

    nom = pd.read_excel(DATA_DIR / "Номенклатура полная.xlsx")
    spec = pd.read_excel(DATA_DIR / "сводная_спецификация.xlsx")

    # --- Номенклатура: непустые наименования ---
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

    # --- Спецификация: непустые наименования ---
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
#  2. Эмбеддинги
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
#  3. FAISS blocking
# ------------------------------------------------------------------ #

def faiss_blocking(
    nom_emb: np.ndarray,
    spec_emb: np.ndarray,
    top_k: int = TOP_K_FAISS,
) -> tuple[np.ndarray, np.ndarray]:
    """Для каждой строки спецификации найти top_k ближайших из номенклатуры.

    Returns:
        (distances, indices) — оба shape [n_spec, top_k]
    """
    logger.info("FAISS blocking: top-%d из %d для %d запросов...",
                top_k, nom_emb.shape[0], spec_emb.shape[0])

    # L2-нормализация → cosine similarity через inner product
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
    threshold: float = SIMILARITY_THRESHOLD,
) -> pd.DataFrame:
    """Собрать DataFrame кандидатных пар с дедупликацией."""
    rows = []
    for spec_idx in range(len(spec)):
        for k in range(distances.shape[1]):
            sim = distances[spec_idx, k]
            if sim < threshold:
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

    # Дедупликация: уникальные пары текстов, top-K по similarity
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
    return candidates, dedup


# ------------------------------------------------------------------ #
#  4. LLM-разметка
# ------------------------------------------------------------------ #

LABEL_PROMPT = """Ты эксперт по сопоставлению промышленной номенклатуры.
Для каждой пары определи: описывают ли они ОДИН И ТОТ ЖЕ товар/компонент?

Отвечай СТРОГО в формате JSON-массива, по одному объекту на пару:
[{{"pair_id": 0, "match": true/false, "confidence": "high"/"medium"/"low"}}]

Учитывай:
- Децимальные номера (ВАКШ.xxx, НЯИТ.xxx) — если совпадают, это почти наверняка match
- Коды продукции и артикулы тоже сильный сигнал
- Наименования могут различаться сокращениями и порядком слов
- Разные модификации одного изделия (-01, -02, -04) — это РАЗНЫЕ товары (не match)

Пары:
{pairs}

JSON: /no_think"""


def format_pairs_for_prompt(candidates_batch: pd.DataFrame) -> str:
    """Форматировать батч пар для промпта."""
    lines = []
    for i, (_, row) in enumerate(candidates_batch.iterrows()):
        lines.append(
            f"Пара {i}: [Спецификация] \"{row['spec_text']}\" ↔ "
            f"[Номенклатура] \"{row['nom_text']}\" (cosine={row['cosine_sim']:.3f})"
        )
    return "\n".join(lines)


def parse_llm_response(response: str) -> list[dict]:
    """Извлечь JSON из ответа LLM."""
    text = response.strip()
    start = text.find("[")
    end = text.rfind("]")
    if start != -1 and end != -1:
        try:
            return json.loads(text[start : end + 1])
        except json.JSONDecodeError:
            pass

    logger.warning("Не удалось распарсить ответ LLM: %s", text[:200])
    return []


def label_candidates(
    candidates: pd.DataFrame,
    client: OllamaClient,
    batch_size: int = LLM_BATCH_SIZE,
) -> pd.DataFrame:
    """Разметить дедуплицированных кандидатов через LLM.

    Resume-логика: каждой строке присваивается batch_id.
    При повторном запуске пропускаются батчи, где ВСЕ строки уже размечены.
    Сохранение на диск каждые SAVE_EVERY батчей.
    """
    # Инициализация колонок
    if "label" not in candidates.columns:
        candidates["label"] = -1
    if "confidence" not in candidates.columns:
        candidates["confidence"] = ""
    if "batch_id" not in candidates.columns:
        candidates["batch_id"] = candidates.index // batch_size

    # Определяем какие батчи ещё не размечены (хотя бы одна строка с label == -1)
    all_batch_ids = sorted(candidates["batch_id"].unique())
    done_batches = set()
    for bid in all_batch_ids:
        mask = candidates["batch_id"] == bid
        if (candidates.loc[mask, "label"] != -1).all():
            done_batches.add(bid)

    todo_batches = [b for b in all_batch_ids if b not in done_batches]
    logger.info(
        "LLM-разметка: %d батчей всего, %d уже готово, %d осталось",
        len(all_batch_ids), len(done_batches), len(todo_batches),
    )

    if not todo_batches:
        logger.info("Все батчи уже размечены, пропускаем.")
        return candidates

    labeled_count = 0
    errors = 0

    for i, bid in enumerate(tqdm(todo_batches, desc="LLM labeling")):
        batch = candidates[candidates["batch_id"] == bid]

        prompt = LABEL_PROMPT.format(pairs=format_pairs_for_prompt(batch))

        try:
            response = client.generate(prompt)
            results = parse_llm_response(response)

            for result in results:
                pair_id = result.get("pair_id", -1)
                if 0 <= pair_id < len(batch):
                    real_idx = batch.index[pair_id]
                    candidates.at[real_idx, "label"] = 1 if result.get("match") else 0
                    candidates.at[real_idx, "confidence"] = result.get("confidence", "")
                    labeled_count += 1

        except Exception as e:
            logger.error("Ошибка LLM на батче %d: %s", bid, e)
            errors += 1

        # Периодическое сохранение
        if (i + 1) % SAVE_EVERY == 0:
            candidates.to_parquet(LABELED_PATH)
            total_done = (candidates["label"] != -1).sum()
            logger.info(
                "  Checkpoint: батч %d/%d, размечено %d/%d, ошибок %d",
                i + 1, len(todo_batches), total_done, len(candidates), errors,
            )

    # Финальное сохранение
    candidates.to_parquet(LABELED_PATH)
    logger.info("Разметка завершена: %d новых меток, %d ошибок", labeled_count, errors)
    return candidates


def propagate_labels(
    all_candidates: pd.DataFrame,
    labeled_dedup: pd.DataFrame,
) -> pd.DataFrame:
    """Развернуть метки с дедуплицированных пар на все строки."""
    label_map = labeled_dedup.set_index(["spec_text", "nom_text"])[["label", "confidence"]]
    merged = all_candidates.merge(
        label_map, on=["spec_text", "nom_text"], how="left", suffixes=("_old", ""),
    )
    # Берём новые метки
    if "label_old" in merged.columns:
        merged.drop(columns=["label_old"], inplace=True)
    if "confidence_old" in merged.columns:
        merged.drop(columns=["confidence_old"], inplace=True)
    return merged


# ------------------------------------------------------------------ #
#  5. Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="FAISS blocking + LLM labeling")
    parser.add_argument(
        "--model", type=str, default=None,
        help="Ollama LLM model (default: из OllamaConfig, qwen3.5:9b)",
    )
    args = parser.parse_args()

    t0 = time.time()

    # 1. Загрузка
    nom, spec = load_and_clean()

    # 2. Эмбеддинги
    nom_emb, spec_emb = get_embeddings(nom, spec)

    # 3. FAISS blocking + дедупликация
    distances, indices = faiss_blocking(nom_emb, spec_emb, top_k=TOP_K_FAISS)
    all_candidates, dedup = build_candidates(nom, spec, distances, indices)
    all_candidates.to_parquet(CANDIDATES_PATH)
    dedup.to_parquet(DEDUP_PATH)

    # Статистика
    logger.info("--- Статистика blocking ---")
    logger.info("  Cosine sim (dedup): mean=%.3f, median=%.3f, min=%.3f, max=%.3f",
                dedup["cosine_sim"].mean(), dedup["cosine_sim"].median(),
                dedup["cosine_sim"].min(), dedup["cosine_sim"].max())
    for lo, hi in [(0.70, 0.85), (0.85, 0.90), (0.90, 0.95), (0.95, 1.001)]:
        n = ((dedup["cosine_sim"] >= lo) & (dedup["cosine_sim"] < hi)).sum()
        logger.info("    [%.2f, %.2f): %d (%.1f%%)", lo, hi, n, 100 * n / len(dedup))

    # 4. LLM-разметка (только дедуплицированных пар)
    if LABELED_PATH.exists():
        logger.info("Найден частично размеченный файл, загружаем для продолжения...")
        dedup = pd.read_parquet(LABELED_PATH)
        done = (dedup["label"] != -1).sum()
        logger.info("  Уже размечено: %d / %d (%.1f%%)", done, len(dedup), 100 * done / len(dedup))

    config = OllamaConfig()
    if args.model:
        config.llm_model = args.model
        logger.info("Используем модель: %s", args.model)
    client = OllamaClient(config)
    dedup = label_candidates(dedup, client)

    # 5. Сохранение
    dedup.to_parquet(LABELED_PATH)

    # Развёртка меток на все строки
    full = propagate_labels(all_candidates, dedup)
    full.to_parquet(FULL_LABELED_PATH)

    # Итоговая статистика
    labeled = dedup[dedup["label"] != -1]
    logger.info("--- Итоговая статистика (дедупл.) ---")
    logger.info("  Всего размечено: %d / %d", len(labeled), len(dedup))
    if len(labeled) > 0:
        n_match = (labeled["label"] == 1).sum()
        n_nomatch = (labeled["label"] == 0).sum()
        logger.info("  Match: %d (%.1f%%)", n_match, 100 * n_match / len(labeled))
        logger.info("  No match: %d (%.1f%%)", n_nomatch, 100 * n_nomatch / len(labeled))
        for conf in ["high", "medium", "low"]:
            n = (labeled["confidence"] == conf).sum()
            logger.info("  Confidence %s: %d (%.1f%%)", conf, n, 100 * n / len(labeled))

    full_labeled = full[full["label"].notna() & (full["label"] != -1)]
    logger.info("--- Итоговая статистика (полная) ---")
    logger.info("  Всего пар с метками: %d / %d", len(full_labeled), len(full))

    elapsed = time.time() - t0
    logger.info("Общее время: %.1f мин", elapsed / 60)


if __name__ == "__main__":
    main()
