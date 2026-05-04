"""Проверка качества natural labels — смотрим реальные примеры пар.

Запуск: cd experiments/17 && uv run python 05_check_natural_labels.py
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2] / "data" / "raw_ru"
N = 3  # примеров на каждый источник


def show(title: str, rows: list[dict], cols: list[str] | None = None) -> None:
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")
    for i, r in enumerate(rows, 1):
        print(f"\n--- пример {i} ---")
        if cols:
            r = {k: v for k, v in r.items() if k in cols}
        for k, v in r.items():
            val = str(v)[:120]
            print(f"  {k:35s}: {val}")


# ─────────────────────────────────────────────────────────────────────
# OZON — группы по URL (одинаковый URL = один товар)
# ─────────────────────────────────────────────────────────────────────
def check_ozon() -> None:
    path = ROOT / "ozon" / "clean.parquet"
    df = pd.read_parquet(path)
    print(f"\nOzon shape: {df.shape}")
    print("Columns:", list(df.columns))

    url_col = next((c for c in df.columns if "url" in c.lower() or "link" in c.lower()), None)
    if url_col is None:
        print("URL-колонка не найдена, ищем по всем строковым колонкам...")
        for c in df.select_dtypes("object").columns[:5]:
            sample = df[c].dropna().iloc[0] if len(df[c].dropna()) else ""
            print(f"  {c}: {str(sample)[:80]}")
        return

    print(f"\nURL-колонка: {url_col}")
    groups = df.groupby(url_col).filter(lambda g: len(g) >= 2)
    sample_urls = groups[url_col].unique()[:N]
    rows = []
    for url in sample_urls:
        g = df[df[url_col] == url]
        for _, row in g.head(2).iterrows():
            r = row.to_dict()
            r["__url"] = url
            rows.append(r)
        rows.append({"__url": "--- ^^^^ та же сущность ^^^^", **{}})

    key_cols = [url_col] + [c for c in df.columns if c != url_col][:8]
    show("OZON — пары по URL", rows, cols=key_cols)


# ─────────────────────────────────────────────────────────────────────
# AUTO_RU × AUTO_RU_2020 — пересечение по sell_id
# ─────────────────────────────────────────────────────────────────────
def check_auto_cross() -> None:
    a = pd.read_parquet(ROOT / "auto_ru" / "clean.parquet")
    b = pd.read_parquet(ROOT / "auto_ru_2020" / "clean.parquet")
    print(f"\nauto_ru shape: {a.shape}")
    print(f"auto_ru_2020 shape: {b.shape}")

    # Ищем общие колонки-кандидаты
    common = set(a.columns) & set(b.columns)
    print(f"Общих колонок: {len(common)}")
    print("Общие:", sorted(common)[:20])

    # sell_id
    if "sell_id" in common:
        ids_a = set(a["sell_id"].dropna().astype(str))
        ids_b = set(b["sell_id"].dropna().astype(str))
        overlap = ids_a & ids_b
        print(f"\nsell_id overlap: {len(overlap)}")
        if overlap:
            sample_ids = list(overlap)[:N]
            rows = []
            for sid in sample_ids:
                ra = a[a["sell_id"].astype(str) == sid].head(1).to_dict("records")
                rb = b[b["sell_id"].astype(str) == sid].head(1).to_dict("records")
                if ra and rb:
                    rows.append({"__source": "auto_ru", **ra[0]})
                    rows.append({"__source": "auto_ru_2020", **rb[0]})
                    rows.append({"__source": "---"})
            show_cols = ["__source", "sell_id"] + [c for c in sorted(common) if c != "sell_id"][:10]
            show("AUTO_RU × AUTO_RU_2020 — пары по sell_id", rows, cols=show_cols)
    else:
        print("sell_id не найден, пробуем другие общие колонки...")
        for c in sorted(common)[:5]:
            vals_a = set(a[c].dropna().astype(str))
            vals_b = set(b[c].dropna().astype(str))
            ov = vals_a & vals_b
            print(f"  {c}: overlap={len(ov)}")


# ─────────────────────────────────────────────────────────────────────
# LAMODA — группы по Title+Brand
# ─────────────────────────────────────────────────────────────────────
def check_lamoda() -> None:
    df = pd.read_parquet(ROOT / "lamoda" / "clean.parquet")
    print(f"\nLamoda shape: {df.shape}")
    print("Columns:", list(df.columns[:20]))

    title_col = next((c for c in df.columns if c.lower() in {"title", "name", "название", "product"}), None)
    brand_col = next((c for c in df.columns if c.lower() in {"brand", "бренд", "марка"}), None)
    print(f"title_col={title_col}, brand_col={brand_col}")

    if title_col is None:
        print("Title-колонка не найдена")
        return

    key = [title_col] + ([brand_col] if brand_col else [])
    groups = df.groupby(key).filter(lambda g: len(g) >= 2)
    print(f"Строк в группах с дублями: {len(groups)}")

    sample_keys = groups[key].drop_duplicates().head(N)
    rows = []
    for _, sk in sample_keys.iterrows():
        mask = pd.Series([True] * len(df))
        for k in key:
            mask &= df[k] == sk[k]
        g = df[mask].head(2)
        for _, row in g.iterrows():
            rows.append(row.to_dict())
        rows.append({title_col: "--- ^^^^ та же сущность ^^^^"})

    show_cols = key + [c for c in df.columns if c not in key][:8]
    show("LAMODA — пары по Title+Brand", rows, cols=show_cols)


# ─────────────────────────────────────────────────────────────────────
# DEVICES — группы по device_model
# ─────────────────────────────────────────────────────────────────────
def check_devices() -> None:
    df = pd.read_parquet(ROOT / "devices" / "clean.parquet")
    print(f"\nDevices shape: {df.shape}")
    print("Columns:", list(df.columns[:20]))

    model_col = next(
        (c for c in df.columns if "model" in c.lower() or "device" in c.lower()),
        None,
    )
    print(f"model_col={model_col}")
    if model_col is None:
        return

    groups = df.groupby(model_col).filter(lambda g: len(g) >= 2)
    print(f"Строк в группах с дублями: {len(groups)}")

    sample_keys = groups[model_col].unique()[:N]
    rows = []
    for key_val in sample_keys:
        g = df[df[model_col] == key_val].head(2)
        for _, row in g.iterrows():
            rows.append(row.to_dict())
        rows.append({model_col: "--- ^^^^ та же сущность ^^^^"})

    show_cols = [model_col] + [c for c in df.columns if c != model_col][:10]
    show("DEVICES — пары по device_model", rows, cols=show_cols)


# ─────────────────────────────────────────────────────────────────────
# CARS_RU — внутри датасета (VIN или brand+model+year+km)
# ─────────────────────────────────────────────────────────────────────
def check_cars_ru() -> None:
    df = pd.read_parquet(ROOT / "cars_ru" / "clean.parquet")
    print(f"\ncars_ru shape: {df.shape}")
    print("Columns:", list(df.columns[:20]))

    vin_col = next((c for c in df.columns if "vin" in c.lower()), None)
    print(f"VIN col: {vin_col}")
    if vin_col:
        vc = df[vin_col].dropna()
        dups = vc[vc.duplicated(keep=False)]
        print(f"VIN дублей: {len(dups)}, уникальных VIN с дублями: {dups.nunique()}")
        if len(dups):
            sample_vin = dups.unique()[:N]
            rows = []
            for v in sample_vin:
                g = df[df[vin_col] == v].head(2)
                for _, row in g.iterrows():
                    rows.append(row.to_dict())
                rows.append({vin_col: "--- ^^^^ та же сущность ^^^^"})
            show_cols = [vin_col] + [c for c in df.columns if c != vin_col][:10]
            show("CARS_RU — пары по VIN", rows, cols=show_cols)

    # Попробуем brand+model+year
    b_col = next((c for c in df.columns if c.lower() in {"brand", "mark", "марка", "make"}), None)
    m_col = next((c for c in df.columns if c.lower() in {"model"}), None)
    y_col = next((c for c in df.columns if "year" in c.lower() or "год" in c.lower()), None)
    print(f"\nbrand={b_col}, model={m_col}, year={y_col}")


if __name__ == "__main__":
    check_ozon()
    check_lamoda()
    check_devices()
    check_auto_cross()
    check_cars_ru()
