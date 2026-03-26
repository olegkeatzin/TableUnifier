"""Эксперимент 7 — Тестирование модели ER на реальных данных.

Сопоставление номенклатуры предприятия (table A, 40K строк)
со сводной спецификацией (table B, 18K строк) — без разметки.

Метрики (unsupervised):
  1. Category consistency — совпадение Категории спецификации с наименованием
     номенклатуры в top-1 матче
  2. Reciprocal matching — доля взаимных nearest-neighbor пар
  3. Confidence separation — разрыв sim(top-1) vs sim(top-2)
  4. GNN vs Baseline — сравнение всех метрик для GNN и raw CLS эмбеддингов

Использование:
    # Полный прогон (генерация эмбеддингов + граф + оценка)
    python -m experiments.07_real_data_test

    # Пропустить генерацию (если уже есть кеш)
    python -m experiments.07_real_data_test --skip-embeddings --skip-graph
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from table_unifier.config import EntityResolutionConfig, OllamaConfig
from table_unifier.dataset.embedding_generation import (
    TokenEmbedder,
    generate_column_embeddings,
    serialize_row,
)
from table_unifier.dataset.graph_builder import build_graph
from table_unifier.ollama_client import OllamaClient
from table_unifier.training.er_trainer import get_row_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DATA_DIR = Path("experiments/07_real_data_test")
OUTPUT_DIR = Path("output/07_real_data_test")

BEST_CONFIG = EntityResolutionConfig(
    hidden_dim=128,
    edge_dim=128,
    num_gnn_layers=3,
    dropout=0.0,
    bidirectional=True,
    lr=7.5e-4,
    weight_decay=0.0,
    margin=0.1,
)

# Столбцы номенклатуры (table A) — только информативные
COLS_A = [
    "Наименование",
    "Артикул ",  # с пробелом — так в исходном файле
    "Вид номенклатуры",
    "Базовая единица измерения",
    "Производитель",
]

# Столбцы спецификации (table B) — без Файл, лист
COLS_B_IGNORE = {"Файл", "лист"}

# Маппинг категорий → ключевые подстроки для проверки consistency
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "Резисторы": ["резистор", "сопротивлен", "ом"],
    "Конденсаторы": ["конденсатор", "ёмкост", "емкост", "мкф", "пф", "нф"],
    "Микросхемы": ["микросхем", "мс ", "чип", "ic"],
    "Соединители": ["соединител", "разъём", "разъем", "коннектор", "вилк", "розетк"],
    "Диоды": ["диод", "стабилитрон", "варикап"],
    "Транзисторы": ["транзистор"],
    "Индуктивности": ["индуктивност", "дроссел", "катушк"],
    "Трансформаторы": ["трансформатор"],
    "Реле": ["реле"],
    "Кварцевые резонаторы": ["кварц", "резонатор", "генератор"],
    "Светодиоды": ["светодиод", "led", "сид"],
    "Предохранители": ["предохранител"],
    "Крепежные изделия": ["болт", "гайк", "винт", "шайб", "шуруп", "крепёж", "крепеж"],
}


# ------------------------------------------------------------------ #
#  Загрузка и предобработка
# ------------------------------------------------------------------ #

def load_and_preprocess() -> tuple[pd.DataFrame, pd.DataFrame, list[str], list[str]]:
    """Загрузить и почистить обе таблицы."""
    logger.info("Загрузка таблиц...")

    # --- Table A: Номенклатура ---
    df_a = pd.read_excel(DATA_DIR / "Номенклатура полная.xlsx")
    # Берём только информативные столбцы (которые реально есть)
    cols_a = [c for c in COLS_A if c in df_a.columns]
    df_a = df_a[cols_a].copy()
    # Убираем строки где Наименование пусто
    df_a = df_a.dropna(subset=["Наименование"]).reset_index(drop=True)
    df_a["id"] = [f"a_{i}" for i in range(len(df_a))]
    df_a = df_a.fillna("")
    logger.info("Номенклатура: %d строк, столбцы: %s", len(df_a), cols_a)

    # --- Table B: Спецификация ---
    df_b = pd.read_excel(DATA_DIR / "сводная_спецификация.xlsx")
    cols_b = [c for c in df_b.columns if c not in COLS_B_IGNORE]
    df_b = df_b[cols_b].copy()
    # Убираем строки где наименование пусто
    name_col_b = "наименование" if "наименование" in df_b.columns else None
    if name_col_b:
        df_b = df_b.dropna(subset=[name_col_b]).reset_index(drop=True)
    df_b["id"] = [f"b_{i}" for i in range(len(df_b))]
    df_b = df_b.fillna("")
    columns_b = [c for c in cols_b if c != "id"]
    logger.info("Спецификация: %d строк, столбцы: %s", len(df_b), columns_b)

    return df_a, df_b, cols_a, columns_b


# ------------------------------------------------------------------ #
#  Генерация и кеширование эмбеддингов
# ------------------------------------------------------------------ #

def generate_and_cache_embeddings(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    cols_a: list[str],
    cols_b: list[str],
    ollama_config: OllamaConfig,
    device: str,
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Сгенерировать col/row эмбеддинги, закешировать на диск."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # --- Column embeddings (Ollama) ---
    col_emb_path = OUTPUT_DIR / "column_embeddings.npz"
    if col_emb_path.exists():
        logger.info("Загрузка column embeddings из кеша...")
        data = np.load(col_emb_path)
        col_embeddings = {k: data[k] for k in data.files}
    else:
        logger.info("Генерация column embeddings через Ollama...")
        client = OllamaClient(ollama_config)
        col_emb_a = generate_column_embeddings(client, df_a, cols_a)
        col_emb_b = generate_column_embeddings(client, df_b, cols_b)
        col_embeddings = {**col_emb_a, **col_emb_b}
        np.savez(col_emb_path, **col_embeddings)
        logger.info("Column embeddings сохранены: %d столбцов", len(col_embeddings))

    # --- Row embeddings (rubert-tiny2 CLS) ---
    row_a_path = OUTPUT_DIR / "row_embeddings_a.npy"
    row_b_path = OUTPUT_DIR / "row_embeddings_b.npy"
    if row_a_path.exists() and row_b_path.exists():
        logger.info("Загрузка row embeddings из кеша...")
        row_emb_a = np.load(row_a_path)
        row_emb_b = np.load(row_b_path)
    else:
        logger.info("Генерация row embeddings через rubert-tiny2...")
        embedder = TokenEmbedder(device=device)

        texts_a = [serialize_row(row, cols_a) for _, row in df_a.iterrows()]
        texts_b = [serialize_row(row, cols_b) for _, row in df_b.iterrows()]
        logger.info("Эмбеддинг %d + %d строк...", len(texts_a), len(texts_b))

        row_emb_a = embedder.embed_sentences(texts_a)
        row_emb_b = embedder.embed_sentences(texts_b)

        np.save(row_a_path, row_emb_a)
        np.save(row_b_path, row_emb_b)

        del embedder.model
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        logger.info("Row embeddings сохранены: A=%s, B=%s", row_emb_a.shape, row_emb_b.shape)

    return col_embeddings, row_emb_a, row_emb_b


# ------------------------------------------------------------------ #
#  Построение графа
# ------------------------------------------------------------------ #

def build_and_cache_graph(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    cols_a: list[str],
    cols_b: list[str],
    col_embeddings: dict[str, np.ndarray],
    row_emb_a: np.ndarray,
    row_emb_b: np.ndarray,
    device: str,
):
    """Построить граф, закешировать."""
    graph_path = OUTPUT_DIR / "graph.pt"
    id_a_path = OUTPUT_DIR / "id_to_global_a.json"
    id_b_path = OUTPUT_DIR / "id_to_global_b.json"

    if graph_path.exists() and id_a_path.exists() and id_b_path.exists():
        logger.info("Загрузка графа из кеша...")
        graph = torch.load(graph_path, weights_only=False)
        with open(id_a_path) as f:
            id_to_global_a = json.load(f)
        with open(id_b_path) as f:
            id_to_global_b = json.load(f)
        return graph, id_to_global_a, id_to_global_b

    logger.info("Построение графа...")
    token_embedder = TokenEmbedder(device=device)

    graph, id_to_global_a, id_to_global_b = build_graph(
        df_a, df_b,
        col_embeddings, token_embedder,
        columns_a=cols_a, columns_b=cols_b,
        precomputed_row_embeddings_a=row_emb_a,
        precomputed_row_embeddings_b=row_emb_b,
    )

    torch.save(graph, graph_path)
    with open(id_a_path, "w") as f:
        json.dump(id_to_global_a, f)
    with open(id_b_path, "w") as f:
        json.dump(id_to_global_b, f)

    del token_embedder
    import gc
    gc.collect()
    torch.cuda.empty_cache()

    logger.info(
        "Граф: %d row nodes, %d token nodes, %d edges",
        graph["row"].x.shape[0],
        graph["token"].x.shape[0],
        graph["token", "in_row", "row"].edge_index.shape[1],
    )
    return graph, id_to_global_a, id_to_global_b


# ------------------------------------------------------------------ #
#  Загрузка модели
# ------------------------------------------------------------------ #

def load_model(graph, device: str, model_path: Path | None = None):
    """Загрузить обученную модель."""
    candidates = [
        model_path,
        Path("output/er_model_hpo_best.pt"),
        Path("output/er_model_multidataset.pt"),
        Path("output/er_model_unified.pt"),
    ]
    found = None
    for p in candidates:
        if p is not None and p.exists():
            found = p
            break

    if found is None:
        logger.error("Модель не найдена. Проверьте output/")
        return None

    logger.info("Загрузка модели: %s", found)

    config = BEST_CONFIG
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    from table_unifier.models.entity_resolution import EntityResolutionGNN

    model = EntityResolutionGNN(
        row_dim=config.row_dim,
        token_dim=config.token_dim,
        col_dim=config.col_dim,
        hidden_dim=config.hidden_dim,
        edge_dim=config.edge_dim,
        output_dim=config.output_dim,
        num_gnn_layers=config.num_gnn_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    )
    state = torch.load(found, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


# ------------------------------------------------------------------ #
#  Поиск top-K ближайших (чанками, чтобы не взорвать RAM)
# ------------------------------------------------------------------ #

def find_top_k(
    emb_query: torch.Tensor,  # [N_b, D] — спецификация
    emb_index: torch.Tensor,  # [N_a, D] — номенклатура
    k: int = 10,
    chunk_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Для каждой строки query найти top-K из index.

    Returns: (sims [N_b, K], indices [N_b, K])
    """
    all_sims, all_idxs = [], []
    for i in range(0, len(emb_query), chunk_size):
        chunk = emb_query[i : i + chunk_size]  # [chunk, D]
        sim = chunk @ emb_index.T               # [chunk, N_a]
        vals, idxs = sim.topk(min(k, sim.shape[1]), dim=1)
        all_sims.append(vals)
        all_idxs.append(idxs)
    return torch.cat(all_sims), torch.cat(all_idxs)


# ------------------------------------------------------------------ #
#  Метрики (unsupervised)
# ------------------------------------------------------------------ #

def eval_category_consistency(
    top1_idx: torch.Tensor,   # [N_b] — индекс лучшего матча из ном-ры
    df_a: pd.DataFrame,       # номенклатура
    df_b: pd.DataFrame,       # спецификация
    idx_a_sorted: list[int],  # маппинг: позиция в emb_a → позиция в df_a
) -> dict:
    """Проверка: Категория спецификации ↔ Наименование номенклатуры."""
    cat_col = "Категория" if "Категория" in df_b.columns else None
    if cat_col is None:
        return {"error": "Нет столбца Категория"}

    total, match, per_cat = 0, 0, {}

    for b_idx in range(len(df_b)):
        cat = str(df_b.iloc[b_idx].get(cat_col, "")).strip()
        if not cat:
            continue

        nom_local = idx_a_sorted[top1_idx[b_idx].item()]
        nom_name = str(df_a.iloc[nom_local].get("Наименование", "")).lower()

        # Поиск ключевых слов
        keywords = CATEGORY_KEYWORDS.get(cat)
        if keywords is None:
            # Фоллбэк: стемминг категории (убираем окончание)
            stem = cat.lower().rstrip("ыие")
            keywords = [stem]

        found = any(kw in nom_name for kw in keywords)
        total += 1
        if found:
            match += 1

        if cat not in per_cat:
            per_cat[cat] = {"total": 0, "match": 0}
        per_cat[cat]["total"] += 1
        if found:
            per_cat[cat]["match"] += 1

    precision = match / total if total else 0.0
    logger.info("Category consistency: %d/%d = %.1f%%", match, total, precision * 100)
    return {
        "precision": precision,
        "matched": match,
        "total": total,
        "per_category": {
            k: {**v, "precision": v["match"] / v["total"] if v["total"] else 0}
            for k, v in sorted(per_cat.items(), key=lambda x: -x[1]["total"])
        },
    }


def eval_reciprocal(
    top1_b2a: torch.Tensor,  # [N_b] — для каждой строки spec, top-1 из nom
    top1_a2b: torch.Tensor,  # [N_a] — для каждой строки nom, top-1 из spec
) -> dict:
    """Доля взаимных nearest-neighbor пар."""
    reciprocal = 0
    for b_idx in range(len(top1_b2a)):
        a_idx = top1_b2a[b_idx].item()
        if a_idx < len(top1_a2b) and top1_a2b[a_idx].item() == b_idx:
            reciprocal += 1
    rate = reciprocal / len(top1_b2a) if len(top1_b2a) else 0.0
    logger.info("Reciprocal rate: %d/%d = %.1f%%", reciprocal, len(top1_b2a), rate * 100)
    return {"rate": rate, "count": reciprocal, "total": len(top1_b2a)}


def eval_confidence(
    top_sims: torch.Tensor,  # [N_b, K]
    label: str,
) -> dict:
    """Анализ разрыва sim(top-1) vs sim(top-2)."""
    if top_sims.shape[1] < 2:
        return {"error": "K < 2"}
    gaps = (top_sims[:, 0] - top_sims[:, 1]).numpy()

    stats = {
        "mean_gap": float(np.mean(gaps)),
        "median_gap": float(np.median(gaps)),
        "std_gap": float(np.std(gaps)),
        "pct_gap_gt_01": float((gaps > 0.1).mean()),
        "pct_gap_gt_02": float((gaps > 0.2).mean()),
        "mean_top1_sim": float(top_sims[:, 0].mean()),
    }
    logger.info(
        "%s confidence: mean_gap=%.3f, median=%.3f, >0.1: %.1f%%, mean_sim=%.3f",
        label, stats["mean_gap"], stats["median_gap"],
        stats["pct_gap_gt_01"] * 100, stats["mean_top1_sim"],
    )

    # Гистограмма
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(gaps, bins=50, alpha=0.7, edgecolor="black")
    ax.set_xlabel("sim(top-1) − sim(top-2)")
    ax.set_ylabel("Количество")
    ax.set_title(f"Confidence gap — {label}")
    ax.axvline(np.median(gaps), color="red", linestyle="--", label=f"median={np.median(gaps):.3f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / f"confidence_gap_{label}.png", dpi=150)
    plt.close(fig)

    return stats


# ------------------------------------------------------------------ #
#  Сохранение примеров матчей
# ------------------------------------------------------------------ #

def save_sample_matches(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    top_sims_gnn: torch.Tensor,
    top_idx_gnn: torch.Tensor,
    top_sims_base: torch.Tensor,
    top_idx_base: torch.Tensor,
    idx_a_sorted: list[int],
    n_samples: int = 300,
    top_k: int = 3,
):
    """Сохранить CSV с примерами матчей для визуальной проверки."""
    rows = []
    sample_idxs = np.random.default_rng(42).choice(len(df_b), min(n_samples, len(df_b)), replace=False)
    sample_idxs.sort()

    cat_col = "Категория" if "Категория" in df_b.columns else None
    name_col_b = "наименование" if "наименование" in df_b.columns else None

    for b_idx in sample_idxs:
        spec_name = str(df_b.iloc[b_idx].get(name_col_b, "")) if name_col_b else ""
        spec_cat = str(df_b.iloc[b_idx].get(cat_col, "")) if cat_col else ""

        for rank in range(min(top_k, top_idx_gnn.shape[1])):
            a_local = idx_a_sorted[top_idx_gnn[b_idx, rank].item()]
            rows.append({
                "method": "GNN",
                "spec_idx": int(b_idx),
                "spec_наименование": spec_name,
                "spec_Категория": spec_cat,
                "rank": rank + 1,
                "nom_idx": a_local,
                "nom_Наименование": str(df_a.iloc[a_local].get("Наименование", "")),
                "nom_Артикул": str(df_a.iloc[a_local].get("Артикул ", "")),
                "similarity": f"{top_sims_gnn[b_idx, rank].item():.4f}",
            })

        for rank in range(min(top_k, top_idx_base.shape[1])):
            a_local = idx_a_sorted[top_idx_base[b_idx, rank].item()]
            rows.append({
                "method": "Baseline",
                "spec_idx": int(b_idx),
                "spec_наименование": spec_name,
                "spec_Категория": spec_cat,
                "rank": rank + 1,
                "nom_idx": a_local,
                "nom_Наименование": str(df_a.iloc[a_local].get("Наименование", "")),
                "nom_Артикул": str(df_a.iloc[a_local].get("Артикул ", "")),
                "similarity": f"{top_sims_base[b_idx, rank].item():.4f}",
            })

    result = pd.DataFrame(rows)
    out_path = OUTPUT_DIR / "sample_matches.csv"
    result.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Примеры матчей сохранены: %s (%d строк)", out_path, len(result))


# ------------------------------------------------------------------ #
#  Основной pipeline
# ------------------------------------------------------------------ #

def run_evaluation(
    graph,
    model,
    id_to_global_a: dict[str, int],
    id_to_global_b: dict[str, int],
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    device: str,
    top_k: int = 10,
) -> dict:
    """Прогнать GNN и baseline, посчитать все метрики."""

    # --- Извлечение эмбеддингов ---
    # Маппинг: sorted global indices → позиция в df
    sorted_a = sorted(id_to_global_a.items(), key=lambda x: x[1])
    sorted_b = sorted(id_to_global_b.items(), key=lambda x: x[1])
    global_indices_a = [v for _, v in sorted_a]
    global_indices_b = [v for _, v in sorted_b]
    # позиция в df_a по позиции в emb_a
    idx_a_sorted = list(range(len(sorted_a)))  # id = a_0, a_1, ... → df iloc

    # GNN embeddings
    if model is not None:
        gnn_emb = get_row_embeddings(model, graph, device=device)  # [N_total, D]
        emb_a_gnn = gnn_emb[global_indices_a]
        emb_b_gnn = gnn_emb[global_indices_b]
    else:
        emb_a_gnn = emb_b_gnn = None

    # Baseline embeddings (raw CLS, L2-normalized)
    raw_row = graph["row"].x  # [N_total, 312]
    baseline_emb = F.normalize(raw_row.float(), p=2, dim=-1)
    emb_a_base = baseline_emb[global_indices_a]
    emb_b_base = baseline_emb[global_indices_b]

    results = {}

    for label, emb_a, emb_b in [("GNN", emb_a_gnn, emb_b_gnn), ("Baseline", emb_a_base, emb_b_base)]:
        if emb_a is None:
            logger.warning("Пропуск %s (нет модели)", label)
            continue

        logger.info("=" * 60)
        logger.info("Оценка: %s", label)
        logger.info("=" * 60)

        # Top-K: spec → nom
        sims_b2a, idx_b2a = find_top_k(emb_b, emb_a, k=top_k)
        # Top-1: nom → spec (для reciprocal)
        _, idx_a2b = find_top_k(emb_a, emb_b, k=1)

        # Метрики
        cat = eval_category_consistency(idx_b2a[:, 0], df_a, df_b, idx_a_sorted)
        recip = eval_reciprocal(idx_b2a[:, 0], idx_a2b[:, 0])
        conf = eval_confidence(sims_b2a, label)

        results[label] = {
            "category_consistency": cat,
            "reciprocal": recip,
            "confidence": conf,
        }

        # Сохраняем top-K для sample_matches
        if label == "GNN":
            results["_gnn_topk"] = (sims_b2a, idx_b2a)
        else:
            results["_base_topk"] = (sims_b2a, idx_b2a)

    return results


# ------------------------------------------------------------------ #
#  Точка входа
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Тест ER на реальных данных")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--top-k", default=10, type=int)
    parser.add_argument("--skip-embeddings", action="store_true",
                        help="Не генерировать эмбеддинги (загрузить из кеша)")
    parser.add_argument("--skip-graph", action="store_true",
                        help="Не строить граф (загрузить из кеша)")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Загрузка таблиц
    df_a, df_b, cols_a, cols_b = load_and_preprocess()

    # 2. Эмбеддинги
    if args.skip_embeddings:
        logger.info("Загрузка эмбеддингов из кеша...")
        data = np.load(OUTPUT_DIR / "column_embeddings.npz")
        col_embeddings = {k: data[k] for k in data.files}
        row_emb_a = np.load(OUTPUT_DIR / "row_embeddings_a.npy")
        row_emb_b = np.load(OUTPUT_DIR / "row_embeddings_b.npy")
    else:
        ollama_cfg = OllamaConfig(host=args.ollama_host)
        col_embeddings, row_emb_a, row_emb_b = generate_and_cache_embeddings(
            df_a, df_b, cols_a, cols_b, ollama_cfg, device,
        )

    # 3. Граф
    if args.skip_graph:
        logger.info("Загрузка графа из кеша...")
        graph = torch.load(OUTPUT_DIR / "graph.pt", weights_only=False)
        with open(OUTPUT_DIR / "id_to_global_a.json") as f:
            id_to_global_a = json.load(f)
        with open(OUTPUT_DIR / "id_to_global_b.json") as f:
            id_to_global_b = json.load(f)
    else:
        graph, id_to_global_a, id_to_global_b = build_and_cache_graph(
            df_a, df_b, cols_a, cols_b,
            col_embeddings, row_emb_a, row_emb_b, device,
        )

    # 4. Модель
    model = load_model(graph, device, args.model_path)

    # 5. Оценка
    results = run_evaluation(
        graph, model, id_to_global_a, id_to_global_b,
        df_a, df_b, device, top_k=args.top_k,
    )

    # 6. Примеры матчей
    gnn_topk = results.pop("_gnn_topk", None)
    base_topk = results.pop("_base_topk", None)
    if gnn_topk and base_topk:
        save_sample_matches(
            df_a, df_b,
            gnn_topk[0], gnn_topk[1],
            base_topk[0], base_topk[1],
            idx_a_sorted=list(range(len(df_a))),
        )
    elif base_topk:
        # Только baseline (нет модели)
        save_sample_matches(
            df_a, df_b,
            base_topk[0], base_topk[1],
            base_topk[0], base_topk[1],
            idx_a_sorted=list(range(len(df_a))),
        )

    # 7. Сохранение результатов
    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Результаты сохранены: %s", out_path)

    # 8. Сводка
    print("\n" + "=" * 60)
    print("СВОДКА РЕЗУЛЬТАТОВ")
    print("=" * 60)
    for method in ("GNN", "Baseline"):
        if method not in results:
            continue
        r = results[method]
        cat_p = r["category_consistency"].get("precision", 0)
        rec_r = r["reciprocal"]["rate"]
        conf_g = r["confidence"]["mean_gap"]
        sim_m = r["confidence"]["mean_top1_sim"]
        print(f"\n{method}:")
        print(f"  Category consistency: {cat_p:.1%}")
        print(f"  Reciprocal rate:      {rec_r:.1%}")
        print(f"  Mean confidence gap:  {conf_g:.4f}")
        print(f"  Mean top-1 similarity:{sim_m:.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
