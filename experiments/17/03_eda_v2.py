"""EDA v2 — фокус на качестве natural labels и анализе расхождений в парах.

Ключевые вопросы:
  1. Какие natural labels реально валидны? (не мусор)
  2. Есть ли проблема sell_id leakage в auto_ru парах?
  3. Насколько сильно расходятся форматы в реальных парах?
  4. Какие колонки «выпадают» у одной из сторон пары?
  5. Итоговый план: что брать как реальные пары, что генерировать синтетически.

Запуск: cd experiments/17 && uv run python 03_eda_v2.py
"""
from __future__ import annotations

import json
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd

RAW_RU = Path(__file__).resolve().parents[2] / "data" / "raw_ru"
SEP = "=" * 72


def section(title: str) -> None:
    print(f"\n{SEP}\n  {title}\n{SEP}")


def subsection(title: str) -> None:
    print(f"\n--- {title} ---")


def load_all() -> dict[str, pd.DataFrame]:
    names = ["lamoda", "cars_ru", "ozon", "auto_ru", "auto_ru_2020", "devices"]
    out = {}
    for name in names:
        p = RAW_RU / name / "clean.parquet"
        if p.exists():
            out[name] = pd.read_parquet(p)
        else:
            print(f"[WARN] {name}: файл не найден ({p})")
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Секция 1: обзор датасетов
# ─────────────────────────────────────────────────────────────────────────────

def sec1_overview(dfs: dict[str, pd.DataFrame]) -> None:
    section("1. Обзор датасетов")
    rows = []
    for name, df in dfs.items():
        n_id = sum(1 for c in df.columns if any(
            h in c.lower() for h in ("id", "url", "ссылка", "артикул", "sell_id")
        ))
        rows.append({
            "датасет": name,
            "строк": len(df),
            "колонок": len(df.columns),
            "id-like колонок": n_id,
        })
    print(pd.DataFrame(rows).to_string(index=False))


# ─────────────────────────────────────────────────────────────────────────────
# Секция 2: качество natural labels
# ─────────────────────────────────────────────────────────────────────────────

def _false_positive_rate(df: pd.DataFrame, key: list[str], check_col: str) -> float:
    """Доля «дубль-пар» по ключу, у которых check_col различается (= ложные пары)."""
    groups = df.dropna(subset=key).groupby(key)
    fp = total = 0
    for _, g in groups:
        if len(g) < 2:
            continue
        pairs = 0
        false_pairs = 0
        vals = g[check_col].fillna("").astype(str).tolist()
        for i in range(len(vals)):
            for j in range(i + 1, len(vals)):
                pairs += 1
                if vals[i] != vals[j]:
                    false_pairs += 1
        total += pairs
        fp += false_pairs
    return fp / total if total else 0.0


def sec2_label_quality(dfs: dict[str, pd.DataFrame]) -> None:
    section("2. Качество natural labels")

    # ── 2.1 LAMODA ──────────────────────────────────────────────────────────
    subsection("2.1 Lamoda — Title+Brand как ключ")
    lam = dfs.get("lamoda")
    if lam is not None:
        key = ["Title", "Brand"]
        groups = lam.dropna(subset=key).groupby(key)
        dup_groups = {k: g for k, g in groups if len(g) >= 2}
        n_pairs = sum(len(g) * (len(g) - 1) // 2 for g in dup_groups.values())
        print(f"Групп с ≥2 строками: {len(dup_groups)}  Потенциальных пар: {n_pairs}")

        # Проверяем: одинаковый Title+Brand → всегда ли одна и та же вещь?
        # Метрика: различается ли состав (about.Состав, %) у «дублей»
        comp_col = next((c for c in lam.columns if "состав" in c.lower()), None)
        if comp_col:
            fp = _false_positive_rate(lam, key, comp_col)
            print(f"False-positive rate по «{comp_col}»: {fp:.1%}  (>10% → мусор)")

        # Несколько примеров
        print("\nПримеры «дублей» по Title+Brand:")
        shown = 0
        for (title, brand), g in dup_groups.items():
            if shown >= 2:
                break
            print(f"\n  Title={title!r}  Brand={brand!r}  ({len(g)} строк)")
            show_cols = [c for c in ["Price", comp_col, "about.Артикул"] if c and c in lam.columns]
            print(g[show_cols].head(3).to_string(index=False))
            shown += 1

        # Проверяем Артикул как альтернативный ключ
        art_col = next((c for c in lam.columns if "артикул" in c.lower()), None)
        if art_col:
            art_uniq = lam[art_col].dropna().nunique()
            art_total = lam[art_col].dropna().count()
            dup_art = (lam[art_col].dropna().duplicated(keep=False).sum())
            print(f"\nАртикул: уникальных={art_uniq}, заполненных={art_total}, "
                  f"дублей={dup_art}  → {'плохой ключ' if dup_art > art_total * 0.05 else 'потенциально хороший ключ'}")
        else:
            print("\nАртикул-колонка не найдена")

    # ── 2.2 OZON ────────────────────────────────────────────────────────────
    subsection("2.2 Ozon — URL-пары ('Ссылка на товар')")
    ozon = dfs.get("ozon")
    if ozon is not None:
        url_col = "Ссылка на товар"
        src_col = "__source_file"
        if url_col in ozon.columns:
            url_counts = ozon[url_col].value_counts()
            dup_urls = url_counts[url_counts >= 2]
            n_rows_dup = ozon[ozon[url_col].isin(dup_urls.index)]
            n_pairs = (dup_urls * (dup_urls - 1) // 2).sum()
            print(f"URL с ≥2 строками: {len(dup_urls)}  Потенциальных пар: {n_pairs}")

            # Cross-file пары (разные месяцы) vs внутри одного файла
            if src_col in ozon.columns:
                cross = 0
                same = 0
                for url, g in ozon[ozon[url_col].isin(dup_urls.index[:500])].groupby(url_col):
                    srcs = g[src_col].tolist()
                    for i in range(len(srcs)):
                        for j in range(i + 1, len(srcs)):
                            if srcs[i] != srcs[j]:
                                cross += 1
                            else:
                                same += 1
                print(f"Из первых 500 URL: cross-file пар={cross}, same-file пар={same}")
                print("→ cross-file = одна и та же вещь в разных месячных выгрузках")

            # Пример пары
            ex_url = dup_urls.index[0]
            ex_g = ozon[ozon[url_col] == ex_url]
            show_cols = [c for c in ozon.columns if c != url_col][:6]
            print(f"\nПример пары (URL={ex_url}):")
            print(ex_g[show_cols].head(2).to_string(index=False))
        else:
            print(f"Колонка '{url_col}' не найдена")

    # ── 2.3 auto_ru × auto_ru_2020 ──────────────────────────────────────────
    subsection("2.3 auto_ru × auto_ru_2020 — пары по sell_id")
    ar = dfs.get("auto_ru")
    ar2 = dfs.get("auto_ru_2020")
    if ar is not None and ar2 is not None:
        if "sell_id" in ar.columns and "sell_id" in ar2.columns:
            ids_a = set(ar["sell_id"].dropna().astype(str))
            ids_b = set(ar2["sell_id"].dropna().astype(str))
            overlap = ids_a & ids_b
            print(f"auto_ru: {len(ids_a)} sell_ids  |  auto_ru_2020: {len(ids_b)} sell_ids")
            print(f"Overlap: {len(overlap)} пар  ({len(overlap)/len(ids_a):.1%} от auto_ru)")
        else:
            print("sell_id не найден в одном из датасетов")

    # ── 2.4 DEVICES ─────────────────────────────────────────────────────────
    subsection("2.4 Devices — синтетические данные?")
    dev = dfs.get("devices")
    if dev is not None:
        model_col = "device_model"
        if model_col in dev.columns:
            sample_models = dev[model_col].dropna().unique()[:5]
            print(f"Примеры model_id: {list(sample_models)}")
            # Та же модель у разных производителей?
            if "manufacturer" in dev.columns:
                multi_mfr = (
                    dev.groupby(model_col)["manufacturer"].nunique()
                    .pipe(lambda s: (s > 1).mean())
                )
                print(f"Доля model_id с >1 производителем: {multi_mfr:.1%}")
                print("→ >50% = model_id не уникален для производителя = синтетика")

    # ── 2.5 CARS_RU ─────────────────────────────────────────────────────────
    subsection("2.5 cars_ru — поиск natural pairs")
    cars = dfs.get("cars_ru")
    if cars is not None:
        vin_col = next((c for c in cars.columns if "vin" in c.lower()), None)
        print(f"VIN колонка: {vin_col or 'не найдена'}")

        # Дубли по описанию
        for desc_col in ("description", "url", "link"):
            if desc_col in cars.columns:
                dup = cars[desc_col].dropna().duplicated(keep=False).sum()
                uniq = cars[desc_col].dropna().duplicated(keep=False).sum()
                total = cars[desc_col].dropna().count()
                print(f"{desc_col}: дублей={dup}/{total}")

        # Блокинг brand+model+year: много ли групп > 50 строк (слишком грубо)?
        key_cols = [c for c in ["mark", "model", "year"] if c in cars.columns]
        if key_cols:
            gs = cars.groupby(key_cols).size()
            n_large = (gs > 50).sum()
            n_pairs = sum(s * (s - 1) // 2 for s in gs if s >= 2)
            print(f"Блокинг {'+'.join(key_cols)}: {len(gs)} групп, {n_large} больших (>50), "
                  f"~{n_pairs:,} пар-кандидатов (слишком много = нет настоящих дублей)")


# ─────────────────────────────────────────────────────────────────────────────
# Секция 3: расхождения форматов в реальных парах (auto_ru × auto_ru_2020)
# ─────────────────────────────────────────────────────────────────────────────

def sec3_pair_divergence(dfs: dict[str, pd.DataFrame]) -> None:
    section("3. Расхождения форматов в реальных парах (auto_ru × auto_ru_2020)")
    ar = dfs.get("auto_ru")
    ar2 = dfs.get("auto_ru_2020")
    if ar is None or ar2 is None:
        print("[SKIP] датасеты не загружены")
        return

    if "sell_id" not in ar.columns or "sell_id" not in ar2.columns:
        print("[SKIP] sell_id не найден")
        return

    # Строим matched pairs
    ar_idx = ar.set_index(ar["sell_id"].astype(str))
    ar2_idx = ar2.set_index(ar2["sell_id"].astype(str))
    common_ids = ar_idx.index.intersection(ar2_idx.index)
    print(f"Matched пар: {len(common_ids)}")

    # Общие колонки (кроме sell_id)
    common_cols = [c for c in ar.columns if c in ar2.columns and c != "sell_id"]
    print(f"Общих колонок (без sell_id): {len(common_cols)}")

    # Для каждой общей колонки: % пар где значение ОТЛИЧАЕТСЯ
    subsection("3.1 Колонки с наибольшим расхождением значений в парах")
    divergence = {}
    n_sample = min(len(common_ids), 5000)
    sample_ids = common_ids[:n_sample]

    a_vals = ar_idx.loc[sample_ids, common_cols]
    b_vals = ar2_idx.loc[sample_ids, common_cols]

    for col in common_cols:
        a = a_vals[col].fillna("__NA__").astype(str)
        b = b_vals[col].fillna("__NA__").astype(str)
        diff_rate = (a != b).mean()
        divergence[col] = diff_rate

    div_series = pd.Series(divergence).sort_values(ascending=False)

    print("\nТоп-20 колонок с наибольшим расхождением:")
    print(div_series.head(20).map(lambda x: f"{x:.1%}").to_string())

    print("\nКолонки, где пары ВСЕГДА совпадают (diff=0%, safe keys):")
    zero_div = div_series[div_series == 0].index.tolist()
    print(zero_div[:20])

    # Детальные примеры расхождений в «важных» колонках
    subsection("3.2 Примеры расхождений в семантических колонках")
    key_semantic = ["bodyType", "color", "engineDisplacement", "brand", "mileage", "year"]
    key_semantic = [c for c in key_semantic if c in common_cols]

    print(f"\n{'Колонка':<30} {'auto_ru':<40} {'auto_ru_2020':<40}")
    print("-" * 110)
    for sid in sample_ids[:5]:
        print(f"\n  sell_id={sid}")
        for col in key_semantic:
            v_a = ar_idx.loc[sid, col] if sid in ar_idx.index else "—"
            v_b = ar2_idx.loc[sid, col] if sid in ar2_idx.index else "—"
            marker = " !!!" if str(v_a) != str(v_b) else "    "
            print(f"  {marker} {col:<28} {str(v_a):<40} {str(v_b):<40}")


# ─────────────────────────────────────────────────────────────────────────────
# Секция 4: sell_id leakage
# ─────────────────────────────────────────────────────────────────────────────

def sec4_sellid_leakage(dfs: dict[str, pd.DataFrame]) -> None:
    section("4. sell_id Leakage — риск тривиального матчинга")
    ar = dfs.get("auto_ru")
    ar2 = dfs.get("auto_ru_2020")
    if ar is None or ar2 is None:
        return

    print("sell_id присутствует как обычная колонка в обоих датасетах.")
    print("Если оставить его в признаках — GNN может научиться матчить по нему напрямую,")
    print("игнорируя семантику остальных колонок.")

    # Насколько уникален sell_id?
    if "sell_id" in ar.columns:
        total = len(ar)
        nuniq = ar["sell_id"].nunique()
        print(f"\nauto_ru: sell_id уникальность = {nuniq}/{total} ({nuniq/total:.1%})")
        print(f"auto_ru_2020: sell_id уникальность = "
              f"{ar2['sell_id'].nunique()}/{len(ar2)}" if ar2 is not None else "")

    print("\n>>> Вывод: sell_id нужно ДРОПАТЬ перед построением графа,")
    print("    использовать только как ключ для генерации лейблов.")
    print("    Альтернатива: симулировать поставщика, который не включает sell_id,")
    print("    через column_dropout в синтетической генерации пар.")


# ─────────────────────────────────────────────────────────────────────────────
# Секция 5: column presence divergence (кто какие колонки «включает»)
# ─────────────────────────────────────────────────────────────────────────────

def sec5_column_presence(dfs: dict[str, pd.DataFrame]) -> None:
    section("5. Column presence: какие колонки выпадают у одной из сторон пары")
    ar = dfs.get("auto_ru")
    ar2 = dfs.get("auto_ru_2020")
    if ar is None or ar2 is None:
        return

    only_a = set(ar.columns) - set(ar2.columns)
    only_b = set(ar2.columns) - set(ar.columns)
    common = set(ar.columns) & set(ar2.columns)

    print(f"Колонок только в auto_ru: {len(only_a)}")
    print(sorted(only_a)[:20])
    print(f"\nКолонок только в auto_ru_2020: {len(only_b)}")
    print(sorted(only_b)[:20])
    print(f"\nОбщих колонок: {len(common)}")

    # Для общих колонок: % строк где одна сторона NULL, другая нет
    if "sell_id" in ar.columns and "sell_id" in ar2.columns:
        subsection("Null-divergence для общих колонок в matched парах")
        ar_idx = ar.set_index(ar["sell_id"].astype(str))
        ar2_idx = ar2.set_index(ar2["sell_id"].astype(str))
        common_ids = ar_idx.index.intersection(ar2_idx.index)
        n_sample = min(len(common_ids), 3000)
        sample_ids = common_ids[:n_sample]

        common_cols = [c for c in ar.columns if c in ar2.columns and c != "sell_id"]
        null_div = {}
        a_vals = ar_idx.loc[sample_ids, common_cols]
        b_vals = ar2_idx.loc[sample_ids, common_cols]
        for col in common_cols:
            a_null = a_vals[col].isna()
            b_null = b_vals[col].isna()
            # один null, другой нет — «выпала» колонка у поставщика
            one_sided_null = (a_null != b_null).mean()
            null_div[col] = one_sided_null

        nd = pd.Series(null_div).sort_values(ascending=False)
        print("\nТоп-15 колонок с наибольшим «одностороннем null» в парах:")
        print(nd.head(15).map(lambda x: f"{x:.1%}").to_string())
        print("\n→ Эти колонки поставщики «включают/не включают» по-разному.")
        print("  Именно их хорошо симулировать через column_dropout в синтетике.")


# ─────────────────────────────────────────────────────────────────────────────
# Секция 6: итог и план
# ─────────────────────────────────────────────────────────────────────────────

def sec6_summary() -> None:
    section("6. ИТОГ — план лейблов и синтетики")
    print("""
┌─────────────────┬──────────────────────────┬───────────────────────────────────────────────────┐
│ Датасет         │ Natural labels            │ Вывод                                             │
├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────┤
│ auto_ru ×       │ sell_id overlap ~17k      │ ВАЛИДНЫ. sell_id ДРОПАТЬ из признаков перед        │
│ auto_ru_2020    │                           │ построением графа. Брать как real positives.       │
├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────┤
│ ozon            │ URL-пары (cross-file)     │ Условно валидны: один продукт, разные месяцы.     │
│                 │                           │ Проверить: есть ли расхождение счётчиков избранн. │
├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────┤
│ lamoda          │ Title+Brand → МУСОР       │ "Пиджак" + "Lime" ≠ одна вещь. Артикул проверить. │
│                 │                           │ Скорее всего — только синтетика.                  │
├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────┤
│ cars_ru         │ нет (нет VIN)             │ Только синтетика (column_dropout + synonyms).     │
├─────────────────┼──────────────────────────┼───────────────────────────────────────────────────┤
│ devices         │ СИНТЕТИКА (model_id не   │ Удалить из real labels вообще. Если нужен          │
│                 │ уникален по производит.) │ — только синтетические пары.                      │
└─────────────────┴──────────────────────────┴───────────────────────────────────────────────────┘

ПЛАН для 04_synth_pairs.py:
  Real positives  : auto_ru × auto_ru_2020 sell_id пары (sell_id дропается из признаков)
  Ozon            : проверить cross-file URL пары; если счётчики различаются — валидны
  Synthetic only  : lamoda, cars_ru, devices
                    → N «поставщиков» на датасет через column_dropout + column_name_synonyms
                    → value_corruption для text/categorical колонок

  Negatives       : random pairs из разных entities (для cars: разные mark+model+year)
""")


# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Загружаем датасеты...")
    dfs = load_all()
    print(f"Загружено: {list(dfs.keys())}")

    sec1_overview(dfs)
    sec2_label_quality(dfs)
    sec3_pair_divergence(dfs)
    sec4_sellid_leakage(dfs)
    sec5_column_presence(dfs)
    sec6_summary()


if __name__ == "__main__":
    main()
