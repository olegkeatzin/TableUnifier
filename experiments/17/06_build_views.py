"""06_build_views.py — Генерация N supplier views с synonym-аугментацией.

Каждый «view» — это вариант таблицы с независимым выбором:
  1. Синонима для каждого имени колонки (из synonyms.json)
  2. Синонима для каждого категориального значения
  3. Column dropout (разный набор колонок)
  4. Value corruption для числовых/текстовых полей

Из N=4 views на датасет получаем C(4,2)=6 пар (view_i, view_j).
Каждая пара — отдельный ER-датасет совместимый с pipeline (exp 14/18).

Выход:
  data/synthetic/<name>_v{i}v{j}/tableA_synth.csv
  data/synthetic/<name>_v{i}v{j}/tableB_synth.csv
  data/synthetic/<name>_v{i}v{j}/{train,valid,test}.csv

Запускать после 04_synth_pairs.py и 05_generate_synonyms.py:
  cd experiments/17 && uv run python 06_build_views.py
  cd experiments/17 && uv run python 06_build_views.py --only lamoda cars_ru
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from table_unifier.dataset.value_corruption import corrupt_dataframe

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parents[2]
RAW_RU = ROOT / "data" / "raw_ru"
SYNTH_OUT = ROOT / "data" / "synthetic"

N_VIEWS = 4
NEG_RATIO = 5
SPLIT = (0.7, 0.15, 0.15)
SYNTH_MAX_ROWS = 100_000
OZON_MAX_POSITIVES = 50_000

# Вероятность применить синоним (per-view, per-column)
COL_RENAME_PROB = 0.75
VAL_SYN_PROB = 0.60

# Column dropout per view
DROP_RATE = 0.20

# Value corruption
CORR_ROW_PROB = 0.50
CORR_CELL_PROB = 0.35


# ------------------------------------------------------------------ #
#  Вспомогательные функции
# ------------------------------------------------------------------ #

def _load_parquet(name: str) -> pd.DataFrame | None:
    p = RAW_RU / name / "clean.parquet"
    if not p.exists():
        log.warning("[%s] clean.parquet не найден", name)
        return None
    df = pd.read_parquet(p)
    log.info("[%s] загружено %d × %d", name, *df.shape)
    return df


def _load_synonyms(name: str) -> dict:
    p = RAW_RU / name / "synonyms.json"
    if not p.exists():
        log.warning("[%s] synonyms.json не найден — работаем без синонимов", name)
        return {}
    return json.loads(p.read_text())


def _apply_synonyms(
    df: pd.DataFrame,
    synonyms: dict,
    rng: np.random.Generator,
    col_rename_prob: float = COL_RENAME_PROB,
    val_syn_prob: float = VAL_SYN_PROB,
) -> pd.DataFrame:
    """Применить синонимы: имена колонок (per-view) + значения (per-cell)."""
    col_syns = synonyms.get("columns", {})
    val_syns = synonyms.get("values", {})

    result = df.copy()
    col_rename_map: dict[str, str] = {}

    for col in df.columns:
        if col == "id":
            continue
        # Одно синонимное имя на всю колонку в этом view (консистентно)
        syns = col_syns.get(col, [])
        if syns and rng.random() < col_rename_prob:
            col_rename_map[col] = str(syns[int(rng.integers(len(syns)))])

        # Синонимы значений — per-cell (разная терминология в разных строках)
        col_val_syns = val_syns.get(col, {})
        if col_val_syns:
            def _replace(v, cvs=col_val_syns, p=val_syn_prob, r=rng):
                if pd.isna(v):
                    return v
                vs = cvs.get(str(v), [])
                if vs and r.random() < p:
                    return str(vs[int(r.integers(len(vs)))])
                return v
            result[col] = result[col].map(_replace)

    result = result.rename(columns=col_rename_map)
    return result


def _apply_column_dropout(df: pd.DataFrame, drop_rate: float, rng: np.random.Generator) -> pd.DataFrame:
    feature_cols = [c for c in df.columns if c != "id"]
    n_drop = max(0, int(len(feature_cols) * drop_rate))
    if n_drop == 0:
        return df
    drop_cols = rng.choice(feature_cols, n_drop, replace=False).tolist()
    return df.drop(columns=drop_cols)


MIN_NOTNULL_FRAC = 0.05


def _make_view(
    df: pd.DataFrame,
    synonyms: dict,
    rng: np.random.Generator,
    drop_rate: float = DROP_RATE,
    apply_corruption: bool = True,
    min_notnull_frac: float = MIN_NOTNULL_FRAC,
) -> pd.DataFrame:
    """Один supplier view: synonyms → dropout → corruption."""
    # Дропаем sparse колонки (кроме id) перед всем остальным
    feat_cols = [c for c in df.columns if c != "id"]
    dense_cols = [c for c in feat_cols if df[c].notna().mean() >= min_notnull_frac]
    df = df[["id"] + dense_cols]

    view = _apply_synonyms(df, synonyms, rng)
    view = _apply_column_dropout(view, drop_rate, rng)
    if apply_corruption:
        view = corrupt_dataframe(
            view,
            row_prob=CORR_ROW_PROB,
            cell_prob=CORR_CELL_PROB,
            skip_columns=("id",),
        )
        view["id"] = view["id"].astype(int)
    return view


def _ensure_id(df: pd.DataFrame) -> pd.DataFrame:
    if "id" not in df.columns:
        df = df.reset_index(drop=True)
        df.insert(0, "id", df.index)
    return df


def _sample_negatives(
    positives_set: set[tuple],
    ids_a: list[int],
    ids_b: list[int],
    n: int,
    rng: np.random.Generator,
) -> list[tuple]:
    negatives: list[tuple] = []
    seen = set(positives_set)
    max_tries = n * 20
    tries = 0
    while len(negatives) < n and tries < max_tries:
        a = ids_a[int(rng.integers(len(ids_a)))]
        b = ids_b[int(rng.integers(len(ids_b)))]
        if (a, b) not in seen:
            negatives.append((a, b))
            seen.add((a, b))
        tries += 1
    if len(negatives) < n:
        log.warning("  Собрали %d/%d негативов", len(negatives), n)
    return negatives


def _split_and_save(
    dataset_name: str,
    view_a: pd.DataFrame,
    view_b: pd.DataFrame,
    positives: list[tuple],
    negatives: list[tuple],
    seed: int = 42,
) -> None:
    all_pairs = [(a, b, 1) for a, b in positives] + [(a, b, 0) for a, b in negatives]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(all_pairs))
    n = len(idx)
    n_test = max(1, int(n * SPLIT[2]))
    n_val = max(1, int(n * SPLIT[1]))

    df_pairs = pd.DataFrame(all_pairs, columns=["ltable_id", "rtable_id", "label"])
    train = df_pairs.iloc[idx[:n - n_val - n_test]].reset_index(drop=True)
    valid = df_pairs.iloc[idx[n - n_val - n_test: n - n_test]].reset_index(drop=True)
    test = df_pairs.iloc[idx[n - n_test:]].reset_index(drop=True)
    labeled = pd.concat([train, valid, test], ignore_index=True)

    out = SYNTH_OUT / dataset_name
    out.mkdir(parents=True, exist_ok=True)
    view_a.to_csv(out / "tableA_synth.csv", index=False)
    view_b.to_csv(out / "tableB_synth.csv", index=False)
    train.to_csv(out / "train.csv", index=False)
    valid.to_csv(out / "valid.csv", index=False)
    test.to_csv(out / "test.csv", index=False)
    labeled.to_csv(out / "labeled_data.csv", index=False)

    n_pos = int(labeled["label"].sum())
    stats = {
        "source": dataset_name,
        "rows_a": len(view_a), "cols_a": len(view_a.columns),
        "rows_b": len(view_b), "cols_b": len(view_b.columns),
        "positives": n_pos, "negatives": len(labeled) - n_pos,
        "train": len(train), "valid": len(valid), "test": len(test),
    }
    (out / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2))
    log.info(
        "  [%s] A=%d×%d B=%d×%d pos=%d neg=%d",
        dataset_name, len(view_a), len(view_a.columns),
        len(view_b), len(view_b.columns), n_pos, len(labeled) - n_pos,
    )


# ------------------------------------------------------------------ #
#  Синтетические датасеты: N views → C(N,2) пар
# ------------------------------------------------------------------ #

def build_synth_views(
    name: str,
    n_views: int = N_VIEWS,
    min_notnull: int = 0,
    seed: int = 42,
) -> None:
    df = _load_parquet(name)
    if df is None:
        return
    synonyms = _load_synonyms(name)

    df = _ensure_id(df)

    if min_notnull > 0:
        feature_cols = [c for c in df.columns if c != "id"]
        mask = df[feature_cols].notna().sum(axis=1) >= min_notnull
        before = len(df)
        df = df[mask].reset_index(drop=True)
        df["id"] = df.index
        log.info("[%s] min_notnull=%d: %d → %d строк", name, min_notnull, before, len(df))

    if len(df) > SYNTH_MAX_ROWS:
        df = df.sample(SYNTH_MAX_ROWS, random_state=seed).reset_index(drop=True)
        df["id"] = df.index

    # Генерируем N views
    views: list[pd.DataFrame] = []
    for k in range(n_views):
        view_rng = np.random.default_rng(seed + k * 137)
        view = _make_view(df, synonyms, view_rng)
        views.append(view)
        log.info("[%s] view %d: %d×%d", name, k, len(view), len(view.columns))

    # Все пары view_i × view_j
    ids = df["id"].tolist()
    positives = [(i, i) for i in ids]

    for i, j in combinations(range(n_views), 2):
        dataset_name = f"{name}_v{i}v{j}"
        rng = np.random.default_rng(seed + i * 13 + j * 7)
        positives_set = set(positives)
        negatives = _sample_negatives(positives_set, ids, ids, len(ids) * NEG_RATIO, rng)
        _split_and_save(dataset_name, views[i], views[j], positives, negatives, seed=seed + i + j)


# ------------------------------------------------------------------ #
#  auto_ru × auto_ru_2020: natural pairs + N view-вариантов
# ------------------------------------------------------------------ #

def build_auto_ru_views(n_views: int = N_VIEWS, seed: int = 42) -> None:
    ar = _load_parquet("auto_ru")
    ar2 = _load_parquet("auto_ru_2020")
    if ar is None or ar2 is None:
        return

    syns_a = _load_synonyms("auto_ru")
    syns_b = _load_synonyms("auto_ru_2020")

    if "sell_id" not in ar.columns or "sell_id" not in ar2.columns:
        log.error("[auto_ru] sell_id не найден")
        return

    ar["sell_id"] = ar["sell_id"].astype(str).str.strip()
    ar2["sell_id"] = ar2["sell_id"].astype(str).str.strip()
    ar = ar.drop_duplicates("sell_id").reset_index(drop=True)
    ar2 = ar2.drop_duplicates("sell_id").reset_index(drop=True)
    ar.insert(0, "id", range(len(ar)))
    ar2.insert(0, "id", range(len(ar2)))

    sid_to_a = dict(zip(ar["sell_id"], ar["id"]))
    sid_to_b = dict(zip(ar2["sell_id"], ar2["id"]))
    overlap = set(sid_to_a) & set(sid_to_b)
    log.info("[auto_ru] overlap: %d пар", len(overlap))

    positives = [(sid_to_a[s], sid_to_b[s]) for s in overlap]
    positives_set = set(positives)

    ar_feat = ar.drop(columns=["sell_id"])
    ar2_feat = ar2.drop(columns=["sell_id"])

    # N вариантов: каждый раз разные синонимы для каждой стороны
    for k in range(n_views):
        rng_a = np.random.default_rng(seed + k * 137)
        rng_b = np.random.default_rng(seed + k * 137 + 1000)

        view_a = _make_view(ar_feat, syns_a, rng_a)
        view_b = _make_view(ar2_feat, syns_b, rng_b)

        rng_neg = np.random.default_rng(seed + k)
        negatives = _sample_negatives(
            set(positives_set), ar_feat["id"].tolist(), ar2_feat["id"].tolist(),
            len(positives) * NEG_RATIO, rng_neg,
        )
        _split_and_save(f"auto_ru_syn{k}", view_a, view_b, positives, negatives, seed=seed + k)


# ------------------------------------------------------------------ #
#  ozon: natural pairs + N view-вариантов
# ------------------------------------------------------------------ #

def build_ozon_views(n_views: int = N_VIEWS, seed: int = 42) -> None:
    ozon = _load_parquet("ozon")
    if ozon is None:
        return
    synonyms = _load_synonyms("ozon")

    url_col = "Ссылка на товар"
    src_col = "__source_file"
    if url_col not in ozon.columns or src_col not in ozon.columns:
        log.error("[ozon] нужные колонки не найдены")
        return

    ozon[url_col] = ozon[url_col].astype(str).str.strip()
    files_sorted = sorted(ozon[src_col].unique())
    split_idx = len(files_sorted) // 2
    group_a = ozon[ozon[src_col].isin(set(files_sorted[:split_idx]))]
    group_b = ozon[ozon[src_col].isin(set(files_sorted[split_idx:]))]

    dedup_a = group_a.drop_duplicates(url_col).reset_index(drop=True)
    dedup_b = group_b.drop_duplicates(url_col).reset_index(drop=True)
    dedup_a.insert(0, "id", range(len(dedup_a)))
    dedup_b.insert(0, "id", range(len(dedup_b)))

    url_to_a = dict(zip(dedup_a[url_col], dedup_a["id"]))
    url_to_b = dict(zip(dedup_b[url_col], dedup_b["id"]))
    overlap_urls = set(url_to_a) & set(url_to_b)

    rng_main = np.random.default_rng(seed)
    if len(overlap_urls) > OZON_MAX_POSITIVES:
        chosen = rng_main.choice(list(overlap_urls), OZON_MAX_POSITIVES, replace=False)
        overlap_urls = set(chosen)

    positives = [(url_to_a[u], url_to_b[u]) for u in overlap_urls]
    log.info("[ozon] %d позитивных пар", len(positives))

    drop_cols = [url_col, src_col]
    base_a = dedup_a.drop(columns=[c for c in drop_cols if c in dedup_a.columns])
    base_b = dedup_b.drop(columns=[c for c in drop_cols if c in dedup_b.columns])

    for k in range(n_views):
        rng_a = np.random.default_rng(seed + k * 137)
        rng_b = np.random.default_rng(seed + k * 137 + 1000)

        view_a = _make_view(base_a, synonyms, rng_a)
        view_b = _make_view(base_b, synonyms, rng_b)

        rng_neg = np.random.default_rng(seed + k)
        positives_set = set(positives)
        negatives = _sample_negatives(
            positives_set, base_a["id"].tolist(), base_b["id"].tolist(),
            len(positives) * NEG_RATIO, rng_neg,
        )
        _split_and_save(f"ozon_syn{k}", view_a, view_b, positives, negatives, seed=seed + k)


# ------------------------------------------------------------------ #
#  CLI
# ------------------------------------------------------------------ #

def main() -> None:
    ap = argparse.ArgumentParser(description="Генерация synonym-view датасетов для exp 17")
    ap.add_argument(
        "--only", nargs="*",
        choices=["auto_ru", "ozon", "lamoda", "cars_ru", "devices"],
    )
    ap.add_argument("--n-views", type=int, default=N_VIEWS)
    args = ap.parse_args()

    targets = args.only or ["auto_ru", "ozon", "lamoda", "cars_ru", "devices"]
    nv = args.n_views

    runners = {
        "auto_ru":  lambda: build_auto_ru_views(n_views=nv),
        "ozon":     lambda: build_ozon_views(n_views=nv),
        "lamoda":   lambda: build_synth_views("lamoda", n_views=nv, min_notnull=5),
        "cars_ru":  lambda: build_synth_views("cars_ru", n_views=nv),
        "devices":  lambda: build_synth_views("devices", n_views=nv),
    }

    for name in targets:
        log.info("\n=== %s ===", name)
        try:
            runners[name]()
        except Exception as exc:
            log.exception("[%s] FAILED: %s", name, exc)


if __name__ == "__main__":
    main()
