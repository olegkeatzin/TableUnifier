"""Генерирует две тестовые xlsx-таблицы для веб-приложения.

Сценарий: два парсера auto.ru-подобной площадки с расхождением схем —
разные имена колонок, color text vs HEX, UPPERCASE bodyType, литры vs cc.

Запуск:
    uv run python scripts/gen_test_xlsx.py
Файлы появятся в data/test_webapp/.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Windows-консоль по умолчанию в cp1251 — переключаем stdout на UTF-8,
# иначе print со стрелками ↔ / · падает с UnicodeEncodeError.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

OUT = Path("data/test_webapp")
OUT.mkdir(parents=True, exist_ok=True)

# ---- Table A — clean format ---------------------------------------------
table_a = pd.DataFrame([
    ["77-A1042", "BMW",        "X5",        2018,  85000, "белый",       "внедорожник", 3.0, "АКПП", 3520000],
    ["77-A1043", "Toyota",     "Camry",     2020,  45000, "чёрный",      "седан",       2.5, "АКПП", 2780000],
    ["77-A1044", "Mercedes",   "E-Class",   2019,  60500, "серебристый", "седан",       2.0, "АКПП", 3210000],
    ["77-A1045", "Audi",       "A6",        2021,  28000, "серый",       "седан",       2.0, "АКПП", 3650000],
    ["77-A1046", "Volkswagen", "Tiguan",    2017, 112000, "красный",     "внедорожник", 1.4, "МКПП", 1490000],
    ["77-A1047", "Kia",        "Rio",       2019,  67000, "синий",       "седан",       1.6, "МКПП",  890000],
    ["77-A1048", "Hyundai",    "Solaris",   2018,  89000, "белый",       "седан",       1.6, "АКПП",  870000],
    ["77-A1049", "Lada",       "Vesta",     2020,  42000, "тёмно-синий", "седан",       1.6, "МКПП",  760000],
    ["77-A1050", "Skoda",      "Octavia",   2019,  71000, "белый",       "лифтбек",     1.4, "АКПП", 1340000],
    ["77-A1051", "Renault",    "Logan",     2017, 134000, "серебристый", "седан",       1.6, "МКПП",  540000],
    ["77-A1052", "BMW",        "X5",        2020,  41000, "чёрный",      "внедорожник", 3.0, "АКПП", 4890000],
    ["77-A1053", "Toyota",     "RAV4",      2019,  56000, "белый",       "внедорожник", 2.0, "АКПП", 2440000],
    ["77-A1054", "Mazda",      "CX-5",      2018,  78000, "красный",     "внедорожник", 2.0, "АКПП", 1980000],
    ["77-A1055", "Mitsubishi", "Outlander", 2017,  98000, "серый",       "внедорожник", 2.4, "АКПП", 1620000],
], columns=["sell_id", "mark", "model", "year", "mileage", "color",
            "bodyType", "engine", "transmission", "price"])

# ---- Table B — different parser: rename + UPPERCASE + HEX + cc + k -------
table_b = pd.DataFrame([
    # — duplicates of A (форматные расхождения) —
    ["o-7745021", "BMW",          "X5",      2018,  85120, "#FFFFFF", "ВНЕДОРОЖНИК", 2998, "AT", 3500],
    ["o-7745022", "TOYOTA",       "CAMRY",   2020,  45000, "#1A1A1A", "СЕДАН",       2487, "AT", 2800],
    ["o-7745023", "Mercedes-Benz","E 200",   2019,  60800, "#C0C0C0", "СЕДАН",       1991, "AT", 3250],
    ["o-7745024", "AUDI",         "A6",      2021,  27500, "#808080", "СЕДАН",       1984, "AT", 3700],
    ["o-7745025", "Hyundai",      "Solaris", 2018,  89400, "#F8F8F8", "СЕДАН",       1591, "AT",  860],
    ["o-7745026", "Skoda",        "Octavia", 2019,  71200, "#FAFAFA", "ЛИФТБЕК",     1395, "AT", 1320],
    ["o-7745027", "TOYOTA",       "RAV-4",   2019,  56400, "#FFFFFF", "ВНЕДОРОЖНИК", 1987, "AT", 2480],
    ["o-7745028", "Mazda",        "CX-5",    2018,  78000, "#B71C1C", "ВНЕДОРОЖНИК", 1998, "AT", 1950],
    # — уникальные —
    ["o-7745029", "Ford",         "Focus",   2016, 145000, "#1565C0", "СЕДАН",       1596, "MT",  680],
    ["o-7745030", "Nissan",       "Qashqai", 2019,  64000, "#FFFFFF", "ВНЕДОРОЖНИК", 1997, "AT", 1780],
    ["o-7745031", "BMW",          "X3",      2019,  58000, "#1A1A1A", "ВНЕДОРОЖНИК", 1998, "AT", 3120],
    # — tricky: тот же бренд/модель что A0/A10, но другой год → не дубль —
    ["o-7745032", "BMW",          "X5",      2022,  22000, "#0D47A1", "ВНЕДОРОЖНИК", 2998, "AT", 6450],
    # — Lada Vesta (дубль A7) —
    ["o-7745033", "LADA",         "Vesta",   2020,  42300, "#0D47A1", "СЕДАН",       1596, "MT",  755],
], columns=["offer_id", "brand", "model_name", "year", "probeg_km", "color_hex",
            "body_type", "engine_cc", "gearbox", "price_k"])

p_a = OUT / "auto_ru_2023_q4.xlsx"
p_b = OUT / "auto_ru_2024_q1.xlsx"
table_a.to_excel(p_a, index=False)
table_b.to_excel(p_b, index=False)
print(f"  wrote {p_a}  ({len(table_a)} rows, {len(table_a.columns)} cols)")
print(f"  wrote {p_b}  ({len(table_b)} rows, {len(table_b.columns)} cols)")
print("\nExpected duplicates (ground truth, 9 пар):")
print("  A0↔B0  · BMW X5 2018")
print("  A1↔B1  · Toyota Camry 2020")
print("  A2↔B2  · Mercedes E-Class 2019")
print("  A3↔B3  · Audi A6 2021")
print("  A6↔B4  · Hyundai Solaris 2018")
print("  A8↔B5  · Skoda Octavia 2019")
print("  A11↔B6 · Toyota RAV4 2019")
print("  A12↔B7 · Mazda CX-5 2018")
print("  A7↔B12 · Lada Vesta 2020")
