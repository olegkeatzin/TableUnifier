"""04_synth_pairs.py — Формирование ER-датасетов из русских источников.

Выходной формат (совместим с pipeline эксп. 14/18):
  data/synthetic/<name>/tableA_synth.csv   (id, col1, col2, ...)
  data/synthetic/<name>/tableB_synth.csv   (id, col1, col2, ...)
  data/synthetic/<name>/train.csv          (ltable_id, rtable_id, label)
  data/synthetic/<name>/valid.csv
  data/synthetic/<name>/test.csv

Датасеты:
  auto_ru   — natural labels (sell_id пары, sell_id дропается из признаков)
  ozon      — natural labels (URL cross-file пары, split по группам файлов)
  lamoda    — синтетика sparse-режим: фильтр min_notnull, no dropout, profile-aware negatives
  cars_ru   — синтетика: column_dropout + value_corruption
  devices   — синтетика: column_dropout + value_corruption
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

# value_corruption живёт в src/table_unifier/
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from table_unifier.dataset.value_corruption import corrupt_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Параметры по умолчанию
# ------------------------------------------------------------------ #
NEG_RATIO = 5          # негативов на одну позитивную пару
SPLIT = (0.7, 0.15, 0.15)
OZON_MAX_POSITIVES = 50_000   # ozon огромный — берём сэмпл
SYNTH_MAX_ROWS = 100_000      # потолок строк для синтетических датасетов

# column dropout: разные колонки выпадают у разных «поставщиков»
DROP_RATE_A = 0.15     # tableA роняет 15 % колонок
DROP_RATE_B = 0.30     # tableB роняет 30 % колонок (другие!)
CORR_ROW_PROB = 0.5    # доля строк tableB с порчей значений
CORR_CELL_PROB = 0.35  # вероятность порчи ячейки внутри строки


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


ROOT = _project_root()
RAW_RU = ROOT / "data" / "raw_ru"
SYNTH_OUT = ROOT / "data" / "synthetic"


# ------------------------------------------------------------------ #
#  Вспомогательные функции
# ------------------------------------------------------------------ #

def _load_parquet(name: str) -> pd.DataFrame | None:
    path = RAW_RU / name / "clean.parquet"
    if not path.exists():
        log.warning("[%s] clean.parquet не найден: %s", name, path)
        return None
    df = pd.read_parquet(path)
    log.info("[%s] загружено %d × %d", name, *df.shape)
    return df


def _sample_negatives(
    positives_set: set[tuple],
    ids_a: Sequence,
    ids_b: Sequence,
    n: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """Случайная выборка n негативных пар (не пересекается с positives_set)."""
    ids_a = list(ids_a)
    ids_b = list(ids_b)
    negatives: list[tuple] = []
    max_tries = n * 20
    tries = 0
    while len(negatives) < n and tries < max_tries:
        a = ids_a[int(rng.integers(len(ids_a)))]
        b = ids_b[int(rng.integers(len(ids_b)))]
        if (a, b) not in positives_set:
            negatives.append((a, b))
            positives_set.add((a, b))  # не дублировать
        tries += 1
    if len(negatives) < n:
        log.warning("Собрали только %d/%d негативов (недостаточно уникальных пар)", len(negatives), n)
    return negatives


def _build_profile_groups(
    df: pd.DataFrame,
    top_n: int = 30,
) -> tuple[dict[int, frozenset], dict[frozenset, list[int]]]:
    """Разбить строки на группы по набору непустых колонок (coarse profile).

    Profile = frozenset из top_n наиболее заполненных колонок, которые
    у данной строки не NaN. Строки с одинаковым профилем — «похожие» по схеме.
    """
    feature_cols = [c for c in df.columns if c != "id"]
    notnull_freq = df[feature_cols].notna().mean().sort_values(ascending=False)
    top_cols = notnull_freq.head(top_n).index.tolist()

    notnull_mask = df.set_index("id")[top_cols].notna()
    id_to_profile: dict[int, frozenset] = {}
    for row_id in df["id"]:
        id_to_profile[row_id] = frozenset(
            c for c in top_cols if notnull_mask.at[row_id, c]
        )

    profile_to_ids: dict[frozenset, list[int]] = {}
    for row_id, profile in id_to_profile.items():
        profile_to_ids.setdefault(profile, []).append(row_id)

    profile_sizes = sorted(len(v) for v in profile_to_ids.values())
    log.info(
        "  profile groups: %d unique, sizes median=%d max=%d",
        len(profile_to_ids),
        profile_sizes[len(profile_sizes) // 2] if profile_sizes else 0,
        max(profile_sizes) if profile_sizes else 0,
    )
    return id_to_profile, profile_to_ids


def _sample_negatives_profile_aware(
    positives_set: set[tuple],
    ids: list[int],
    id_to_profile: dict[int, frozenset],
    profile_to_ids: dict[frozenset, list[int]],
    n: int,
    rng: np.random.Generator,
) -> list[tuple]:
    """Негативы из строк с похожим column-профилем.

    Для каждого негатива (a, b) берём b из той же profile-группы что и a,
    чтобы модель не могла отличить match от non-match по наличию/отсутствию колонок.
    Если группа мала (< 2 строк), fallback на случайный id.
    """
    seen = set(positives_set)
    negatives: list[tuple] = []
    max_tries = n * 20
    tries = 0
    while len(negatives) < n and tries < max_tries:
        a = ids[int(rng.integers(len(ids)))]
        profile = id_to_profile.get(a, frozenset())
        candidates = profile_to_ids.get(profile, [])
        # Fallback: если в группе только сам a → случайный id
        valid_candidates = [x for x in candidates if x != a]
        if not valid_candidates:
            valid_candidates = [x for x in ids if x != a]
        b = valid_candidates[int(rng.integers(len(valid_candidates)))]
        if (a, b) not in seen:
            negatives.append((a, b))
            seen.add((a, b))
        tries += 1
    if len(negatives) < n:
        log.warning("Собрали только %d/%d профильных негативов", len(negatives), n)
    return negatives


def _split_pairs(
    pairs: list[tuple],   # (ltable_id, rtable_id, label)
    ratios: tuple[float, float, float] = SPLIT,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(pairs))
    n = len(idx)
    n_test = max(1, int(n * ratios[2]))
    n_val = max(1, int(n * ratios[1]))
    n_train = n - n_val - n_test

    df = pd.DataFrame(pairs, columns=["ltable_id", "rtable_id", "label"])
    train = df.iloc[idx[:n_train]].reset_index(drop=True)
    valid = df.iloc[idx[n_train:n_train + n_val]].reset_index(drop=True)
    test = df.iloc[idx[n_train + n_val:]].reset_index(drop=True)
    return train, valid, test


def _save_dataset(
    name: str,
    table_a: pd.DataFrame,
    table_b: pd.DataFrame,
    train: pd.DataFrame,
    valid: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    out = SYNTH_OUT / name
    out.mkdir(parents=True, exist_ok=True)

    table_a.to_csv(out / "tableA_synth.csv", index=False)
    table_b.to_csv(out / "tableB_synth.csv", index=False)
    train.to_csv(out / "train.csv", index=False)
    valid.to_csv(out / "valid.csv", index=False)
    test.to_csv(out / "test.csv", index=False)

    # labeled_data.csv = объединение всех сплитов (для совместимости)
    labeled = pd.concat([train, valid, test], ignore_index=True)
    labeled.to_csv(out / "labeled_data.csv", index=False)

    # Краткая статистика
    n_pos = int(labeled["label"].sum())
    n_neg = len(labeled) - n_pos
    stats = {
        "rows_a": len(table_a),
        "rows_b": len(table_b),
        "cols_a": len(table_a.columns),
        "cols_b": len(table_b.columns),
        "positives": n_pos,
        "negatives": n_neg,
        "train": len(train),
        "valid": len(valid),
        "test": len(test),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    log.info(
        "[%s] сохранено: A=%d×%d  B=%d×%d  pos=%d neg=%d  (train/val/test=%d/%d/%d)",
        name,
        len(table_a), len(table_a.columns),
        len(table_b), len(table_b.columns),
        n_pos, n_neg,
        len(train), len(valid), len(test),
    )


# ------------------------------------------------------------------ #
#  Синтетический датасет (column_dropout + value_corruption)
# ------------------------------------------------------------------ #

def _apply_column_dropout(df: pd.DataFrame, drop_rate: float, seed: int) -> pd.DataFrame:
    """Случайно дропнуть drop_rate долю колонок (кроме 'id')."""
    rng = np.random.default_rng(seed)
    feature_cols = [c for c in df.columns if c != "id"]
    n_drop = max(0, int(len(feature_cols) * drop_rate))
    drop_cols = rng.choice(feature_cols, n_drop, replace=False).tolist()
    return df.drop(columns=drop_cols)


def make_synth_dataset(
    name: str,
    drop_rate_a: float = DROP_RATE_A,
    drop_rate_b: float = DROP_RATE_B,
    corr_row_prob: float = CORR_ROW_PROB,
    corr_cell_prob: float = CORR_CELL_PROB,
    neg_ratio: int = NEG_RATIO,
    max_rows: int = SYNTH_MAX_ROWS,
    min_notnull: int = 0,
    profile_aware_negatives: bool = False,
    seed: int = 42,
) -> None:
    """Синтетический ER-датасет из одного clean.parquet.

    min_notnull: оставить только строки с ≥ N непустых feature-колонок.
                 Полезно для sparse датасетов (lamoda), где после column_dropout
                 строки вырождаются в пустые.
    profile_aware_negatives: негативы из строк с похожим набором непустых
                 колонок (harder negatives — модель не может ориентироваться
                 только на структуру NaN).
    """
    df = _load_parquet(name)
    if df is None:
        return

    # Добавляем integer id
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "id", df.index)

    # Фильтр по минимуму непустых feature-колонок
    if min_notnull > 0:
        feature_cols = [c for c in df.columns if c != "id"]
        notnull_counts = df[feature_cols].notna().sum(axis=1)
        before = len(df)
        df = df[notnull_counts >= min_notnull].reset_index(drop=True)
        df["id"] = df.index
        log.info("[%s] фильтр min_notnull=%d: %d → %d строк", name, min_notnull, before, len(df))

    if len(df) > max_rows:
        df = df.sample(max_rows, random_state=seed).reset_index(drop=True)
        df["id"] = df.index
        log.info("[%s] обрезано до %d строк", name, max_rows)

    # tableA: дропаем drop_rate_a колонок
    table_a = _apply_column_dropout(df, drop_rate_a, seed=seed)

    # tableB: дропаем drop_rate_b других колонок, затем портим значения
    table_b_raw = _apply_column_dropout(df, drop_rate_b, seed=seed + 1)
    table_b = corrupt_dataframe(
        table_b_raw,
        row_prob=corr_row_prob,
        cell_prob=corr_cell_prob,
        skip_columns=("id",),
    )
    # corrupt_dataframe приводит к str — вернём id как int
    table_b["id"] = table_b["id"].astype(int)
    table_a["id"] = table_a["id"].astype(int)

    # Позитивы: каждая строка → пара (id_a == id_b == row_id)
    ids = df["id"].tolist()
    positives = [(i, i) for i in ids]
    positives_set = set(positives)

    rng = np.random.default_rng(seed)
    n_neg = len(positives) * neg_ratio

    if profile_aware_negatives:
        id_to_profile, profile_to_ids = _build_profile_groups(df)
        negatives = _sample_negatives_profile_aware(
            positives_set, ids, id_to_profile, profile_to_ids, n_neg, rng,
        )
    else:
        negatives = _sample_negatives(positives_set, ids, ids, n_neg, rng)

    all_pairs = [(a, b, 1) for a, b in positives] + [(a, b, 0) for a, b in negatives]
    train, valid, test = _split_pairs(all_pairs, seed=seed)
    _save_dataset(name, table_a, table_b, train, valid, test)


# ------------------------------------------------------------------ #
#  auto_ru × auto_ru_2020 — natural labels (sell_id)
# ------------------------------------------------------------------ #

def make_auto_ru_dataset(neg_ratio: int = NEG_RATIO, seed: int = 42) -> None:
    ar = _load_parquet("auto_ru")
    ar2 = _load_parquet("auto_ru_2020")
    if ar is None or ar2 is None:
        return

    # sell_id — тривиальный leakage, убираем из признаков но используем как ключ
    if "sell_id" not in ar.columns or "sell_id" not in ar2.columns:
        log.error("[auto_ru] sell_id не найден в обоих датасетах")
        return

    ar["sell_id"] = ar["sell_id"].astype(str).str.strip()
    ar2["sell_id"] = ar2["sell_id"].astype(str).str.strip()

    # Дедупликация внутри каждого датасета по sell_id (берём первое вхождение)
    ar = ar.drop_duplicates("sell_id").reset_index(drop=True)
    ar2 = ar2.drop_duplicates("sell_id").reset_index(drop=True)

    # Integer id для pipeline
    ar.insert(0, "id", range(len(ar)))
    ar2.insert(0, "id", range(len(ar2)))

    # Маппинг sell_id → integer id
    sid_to_id_a: dict[str, int] = dict(zip(ar["sell_id"], ar["id"]))
    sid_to_id_b: dict[str, int] = dict(zip(ar2["sell_id"], ar2["id"]))

    overlap = set(sid_to_id_a) & set(sid_to_id_b)
    log.info("[auto_ru] sell_id overlap: %d / (%d, %d)", len(overlap), len(ar), len(ar2))

    positives = [(sid_to_id_a[sid], sid_to_id_b[sid]) for sid in overlap]
    positives_set = set(positives)

    rng = np.random.default_rng(seed)
    negatives = _sample_negatives(
        positives_set, ar["id"].tolist(), ar2["id"].tolist(),
        len(positives) * neg_ratio, rng,
    )

    # Убираем sell_id из tableA и tableB (leakage)
    table_a = ar.drop(columns=["sell_id"])
    table_b = ar2.drop(columns=["sell_id"])

    all_pairs = [(a, b, 1) for a, b in positives] + [(a, b, 0) for a, b in negatives]
    train, valid, test = _split_pairs(all_pairs, seed=seed)
    _save_dataset("auto_ru", table_a, table_b, train, valid, test)


# ------------------------------------------------------------------ #
#  ozon — natural labels (URL cross-file пары)
# ------------------------------------------------------------------ #

def make_ozon_dataset(
    neg_ratio: int = NEG_RATIO,
    max_positives: int = OZON_MAX_POSITIVES,
    seed: int = 42,
) -> None:
    ozon = _load_parquet("ozon")
    if ozon is None:
        return

    url_col = "Ссылка на товар"
    src_col = "__source_file"

    if url_col not in ozon.columns:
        log.error("[ozon] URL-колонка '%s' не найдена. Колонки: %s", url_col, list(ozon.columns))
        return
    if src_col not in ozon.columns:
        log.error("[ozon] source_file-колонка '%s' не найдена", src_col)
        return

    ozon[url_col] = ozon[url_col].astype(str).str.strip()

    # Разбиваем файлы на две группы (хронологически: первая половина → A, вторая → B)
    files_sorted = sorted(ozon[src_col].unique())
    n_files = len(files_sorted)
    split_idx = n_files // 2
    files_a = set(files_sorted[:split_idx])
    files_b = set(files_sorted[split_idx:])
    log.info("[ozon] файлы: %d (A=%d, B=%d)", n_files, len(files_a), len(files_b))

    group_a = ozon[ozon[src_col].isin(files_a)]
    group_b = ozon[ozon[src_col].isin(files_b)]

    # Дедупликация по URL в каждой группе (берём первое вхождение)
    dedup_a = group_a.drop_duplicates(url_col).reset_index(drop=True)
    dedup_b = group_b.drop_duplicates(url_col).reset_index(drop=True)

    dedup_a.insert(0, "id", range(len(dedup_a)))
    dedup_b.insert(0, "id", range(len(dedup_b)))

    url_to_id_a: dict[str, int] = dict(zip(dedup_a[url_col], dedup_a["id"]))
    url_to_id_b: dict[str, int] = dict(zip(dedup_b[url_col], dedup_b["id"]))

    overlap_urls = set(url_to_id_a) & set(url_to_id_b)
    log.info("[ozon] URL overlap: %d / (A=%d, B=%d)", len(overlap_urls), len(dedup_a), len(dedup_b))

    rng = np.random.default_rng(seed)
    overlap_list = list(overlap_urls)
    if len(overlap_list) > max_positives:
        chosen = rng.choice(len(overlap_list), max_positives, replace=False)
        overlap_list = [overlap_list[i] for i in chosen]
        log.info("[ozon] сэмплировано %d позитивных пар из %d", max_positives, len(overlap_urls))

    positives = [(url_to_id_a[u], url_to_id_b[u]) for u in overlap_list]
    positives_set = set(positives)

    negatives = _sample_negatives(
        positives_set, dedup_a["id"].tolist(), dedup_b["id"].tolist(),
        len(positives) * neg_ratio, rng,
    )

    # Убираем URL-ключ (leakage) и source_file из обеих таблиц
    drop_cols = [url_col, src_col]
    table_a = dedup_a.drop(columns=[c for c in drop_cols if c in dedup_a.columns])
    table_b = dedup_b.drop(columns=[c for c in drop_cols if c in dedup_b.columns])

    all_pairs = [(a, b, 1) for a, b in positives] + [(a, b, 0) for a, b in negatives]
    train, valid, test = _split_pairs(all_pairs, seed=seed)
    _save_dataset("ozon", table_a, table_b, train, valid, test)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

MAKERS = {
    "auto_ru": make_auto_ru_dataset,
    "ozon": make_ozon_dataset,
    # lamoda: sparse (много NaN) — no column dropout, фильтр по заполненности,
    #         profile-aware negatives чтобы модель не угадывала match по NaN-структуре
    "lamoda": lambda: make_synth_dataset(
        "lamoda",
        drop_rate_a=0.0,
        drop_rate_b=0.0,
        min_notnull=5,
        profile_aware_negatives=True,
    ),
    "cars_ru": lambda: make_synth_dataset("cars_ru"),
    "devices": lambda: make_synth_dataset("devices"),
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Формирование ER-датасетов для exp 17")
    ap.add_argument(
        "--only", nargs="*", choices=list(MAKERS),
        help="Обработать только указанные датасеты",
    )
    args = ap.parse_args()

    keys = args.only or list(MAKERS)
    failures: list[tuple[str, str]] = []
    for key in keys:
        log.info("\n=== %s ===", key)
        try:
            MAKERS[key]()
        except Exception as exc:
            log.exception("[%s] FAILED: %s", key, exc)
            failures.append((key, f"{type(exc).__name__}: {exc}"))

    if failures:
        log.error("\n--- failures ---")
        for k, msg in failures:
            log.error("  %s: %s", k, msg)


if __name__ == "__main__":
    main()
