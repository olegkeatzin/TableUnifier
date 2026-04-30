"""Подготовка 6 русских датасетов для эксперимента 17.

Каждая prepare_* функция:
- качает датасет через kagglehub (идемпотентно, тянет из кэша)
- парсит JSON-колонки (`model_info`, `super_gen`, `About`, `Владение`, …) в плоские поля
- массивы опций (`options`, `equipment_dict`) → multi-hot по top-30
- дропает суррогатные id и константные колонки
- пишет `data/raw_ru/<name>/clean.parquet`
- возвращает DataFrame для показа в ноутбуке

Шумные per-listing колонки (price, seller_type, pts, address, description, favs и т.п.)
остаются — модель должна сама научиться их взвешивать.
"""

from __future__ import annotations

import argparse
import ast
import json
from collections import Counter
from pathlib import Path

import kagglehub
import pandas as pd

# None = без отсечки (multi-hot для ВСЕХ уникальных опций).
# Идея: реальные поставщики добавляют/убирают колонки по своему усмотрению,
# а модель должна учиться обрабатывать редкие колонки и пропуски — top-N
# искусственно сглаживает разнообразие, поэтому держим всё.
TOP_N_OPTIONS: int | None = None


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for parent in [p, *p.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path.cwd()


OUT_ROOT = _project_root() / "data" / "raw_ru"


def _parse_flex(v):
    """json.loads с fallback на ast.literal_eval (для python-repr с одинарными кавычками)."""
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    if not isinstance(v, str):
        return v
    s = v.strip()
    if not s:
        return None
    try:
        return json.loads(s)
    except Exception:
        try:
            return ast.literal_eval(s)
        except Exception:
            return None


def _read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suf == ".json":
        try:
            return pd.read_json(path, lines=True)
        except ValueError:
            return pd.read_json(path)
    raise ValueError(f"Unknown extension: {path}")


def _flatten_dict_col(series: pd.Series, prefix: str) -> pd.DataFrame:
    parsed = series.map(_parse_flex).map(lambda x: x if isinstance(x, dict) else {})
    flat = pd.json_normalize(parsed.to_list(), sep=".")
    flat.columns = [f"{prefix}.{c}" for c in flat.columns]
    flat.index = series.index
    return flat


def _multi_hot_from_lists(lists: pd.Series, n: int | None, prefix: str) -> pd.DataFrame:
    counter: Counter = Counter()
    for lst in lists:
        if isinstance(lst, list):
            counter.update(lst)
    if n is None:
        keys = list(counter.keys())
    else:
        keys = [k for k, _ in counter.most_common(n)]
    out = pd.DataFrame(index=lists.index)
    for opt in keys:
        out[f"{prefix}_{opt}"] = lists.map(
            lambda lst, o=opt: isinstance(lst, list) and o in lst
        )
    return out


def _multi_hot_array(series: pd.Series, n: int | None, prefix: str) -> pd.DataFrame:
    def to_list(v):
        obj = _parse_flex(v)
        if isinstance(obj, list):
            return obj
        if isinstance(obj, dict):
            return list(obj.keys())
        return []
    return _multi_hot_from_lists(series.map(to_list), n=n, prefix=prefix)


def _drop_dead(df: pd.DataFrame, extra_drop: tuple[str, ...] = ()) -> pd.DataFrame:
    drop = set(extra_drop)
    for col in df.columns:
        try:
            if df[col].nunique(dropna=True) <= 1:
                drop.add(col)
        except TypeError:
            pass  # list/dict values — column is not constant, keep it
    return df.drop(columns=[c for c in drop if c in df.columns])


def _save(df: pd.DataFrame, name: str) -> Path:
    out = OUT_ROOT / name
    out.mkdir(parents=True, exist_ok=True)
    # parquet может упасть на list-колонках с разными типами внутри —
    # такие колонки превращаем в JSON-строки перед записью
    df = df.copy()
    for col in df.columns:
        if df[col].dtype == object:
            sample = df[col].dropna().head(1)
            if len(sample) and isinstance(sample.iloc[0], (list, dict)):
                df[col] = df[col].map(
                    lambda v: json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v
                )
            else:
                types_seen = {type(v) for v in df[col].dropna()}
                if len(types_seen) > 1:
                    df[col] = df[col].map(lambda v: str(v) if pd.notna(v) else v)
    path = out / "clean.parquet"
    df.to_parquet(path, index=False)
    return path


def prepare_lamoda() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download("kirill57678/lamoda-products"))
    csv = next(root.rglob("*.csv"))
    df = pd.read_csv(csv)

    about_col = next((c for c in df.columns if c.lower() == "about"), None)
    if about_col:
        flat = _flatten_dict_col(df[about_col], prefix="about")
        df = pd.concat([df.drop(columns=[about_col]), flat], axis=1)

    df = df.drop(columns=[c for c in ("Alsobuy", "Similar") if c in df.columns])
    df = _drop_dead(df, extra_drop=("id", "Id", "ID"))

    _save(df, "lamoda")
    return df


def prepare_cars_ru() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download("egorledyaev/russian-car-market-dataset"))
    by_name = {p.stem: p for p in root.rglob("*.csv")}
    parts = [pd.read_csv(by_name[k]) for k in ("train", "test") if k in by_name]
    if not parts:
        parts = [pd.read_csv(next(iter(by_name.values())))]
    df = pd.concat(parts, ignore_index=True)

    if "options" in df.columns:
        mh = _multi_hot_array(df["options"], n=TOP_N_OPTIONS, prefix="opt")
        df = pd.concat([df.drop(columns=["options"]), mh], axis=1)

    df = _drop_dead(df, extra_drop=("id", "custom"))

    _save(df, "cars_ru")
    return df


def prepare_ozon() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download(
        "fiftin/ozon-what-products-do-users-add-to-favs"
    ))
    xlsx = sorted(p for p in root.rglob("*") if p.suffix.lower() in {".xlsx", ".xls"})
    parts = []
    for p in xlsx:
        d = pd.read_excel(p)
        d["__source_file"] = p.name
        parts.append(d)
    df = pd.concat(parts, ignore_index=True)
    df = _drop_dead(df)

    _save(df, "ozon")
    return df


def prepare_auto_ru() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download("rzabolotin/auto-ru-car-ads-parsed"))
    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".parquet"}
    )
    df = _read_any(files[0])

    for col in ("model_info", "super_gen", "complectation_dict"):
        if col in df.columns:
            flat = _flatten_dict_col(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), flat], axis=1)

    if "equipment_dict" in df.columns:
        mh = _multi_hot_array(df["equipment_dict"], n=TOP_N_OPTIONS, prefix="equip")
        df = pd.concat([df.drop(columns=["equipment_dict"]), mh], axis=1)

    df = _drop_dead(df, extra_drop=("id",))

    _save(df, "auto_ru")
    return df


def prepare_auto_ru_2020() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download("snezhanausova/all-auto-ru-14-11-2020csv"))
    files = sorted(
        p for p in root.rglob("*") if p.suffix.lower() in {".csv", ".parquet"}
    )
    df = _read_any(files[0])

    for col in ("model_info", "super_gen", "complectation_dict", "Владение"):
        if col in df.columns:
            flat = _flatten_dict_col(df[col], prefix=col)
            df = pd.concat([df.drop(columns=[col]), flat], axis=1)

    if "equipment_dict" in df.columns:
        mh = _multi_hot_array(df["equipment_dict"], n=TOP_N_OPTIONS, prefix="equip")
        df = pd.concat([df.drop(columns=["equipment_dict"]), mh], axis=1)

    df = _drop_dead(df, extra_drop=("id",))

    _save(df, "auto_ru_2020")
    return df


def prepare_devices() -> pd.DataFrame:
    root = Path(kagglehub.dataset_download("ruslanusov/dataset-of-electronics-with-lifecycle-and-specs"))
    ru_file = next(root.rglob("device_dataset_with_status_15*.json"))
    raw = json.loads(ru_file.read_text(encoding="utf-8"))
    df = pd.json_normalize(raw, sep=".")
    df = _drop_dead(df, extra_drop=("id",))

    _save(df, "devices")
    return df


PREPARATORS = {
    "lamoda": prepare_lamoda,
    "cars_ru": prepare_cars_ru,
    "ozon": prepare_ozon,
    "auto_ru": prepare_auto_ru,
    "auto_ru_2020": prepare_auto_ru_2020,
    "devices": prepare_devices,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", choices=list(PREPARATORS))
    args = ap.parse_args()
    keys = args.only or list(PREPARATORS)
    failures: list[tuple[str, str]] = []
    for key in keys:
        print(f"\n=== {key} ===")
        try:
            df = PREPARATORS[key]()
            print(f"saved: {OUT_ROOT / key / 'clean.parquet'}  shape={df.shape}")
        except Exception as e:
            print(f"FAILED: {type(e).__name__}: {e}")
            failures.append((key, f"{type(e).__name__}: {e}"))
    if failures:
        print("\n--- failures ---")
        for k, msg in failures:
            print(f"  {k}: {msg}")


if __name__ == "__main__":
    main()
