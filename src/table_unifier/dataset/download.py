"""Загрузка датасетов из Magellan Data Repository (CS784).

Источник: https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository

Формат CS784 (tar.gz):
    - csv_files/{source_a}.csv — таблица A
    - csv_files/{source_b}.csv — таблица B
    - csv_files/labeled_data.csv — размеченные пары (gold: 1=match, 0=no-match)
    - csv_files/candset.csv — пары-кандидаты
"""

from __future__ import annotations

import io
import logging
import tarfile
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Каталог CS784 Magellan-датасетов (tar.gz)
# ------------------------------------------------------------------ #

_784_BASE_URL = "http://pages.cs.wisc.edu/~anhai/data/784_data"

DATASETS: dict[str, dict] = {
    "restaurants1": {
        "url": f"{_784_BASE_URL}/restaurants1.tar.gz",
        "table_a": "zomato", "table_b": "yelp",
        "domain": "restaurants",
    },
    "bikes": {
        "url": f"{_784_BASE_URL}/bikes.tar.gz",
        "table_a": "bikedekho", "table_b": "bikewale",
        "domain": "bikes",
    },
    "movies1": {
        "url": f"{_784_BASE_URL}/movies1.tar.gz",
        "table_a": "rotten_tomatoes", "table_b": "imdb",
        "domain": "movies",
    },
    "movies2": {
        "url": f"{_784_BASE_URL}/movies2.tar.gz",
        "table_a": "imdb", "table_b": "tmd",
        "domain": "movies",
    },
    "movies3": {
        "url": f"{_784_BASE_URL}/movies3.tar.gz",
        "table_a": "imdb", "table_b": "rotten_tomatoes",
        "domain": "movies",
    },
    "movies4": {
        "url": f"{_784_BASE_URL}/movies4.tar.gz",
        "table_a": "amazon", "table_b": "rotten_tomatoes",
        "domain": "movies",
    },
    "movies5": {
        "url": f"{_784_BASE_URL}/movies5.tar.gz",
        "table_a": "roger_ebert", "table_b": "imdb",
        "domain": "movies",
    },
    "restaurants2": {
        "url": f"{_784_BASE_URL}/restaurants2.tar.gz",
        "table_a": "zomato", "table_b": "yelp",
        "domain": "restaurants",
    },
    "restaurants3": {
        "url": f"{_784_BASE_URL}/restaurants3.tar.gz",
        "table_a": "yelp", "table_b": "yellow_pages",
        "domain": "restaurants",
    },
    "restaurants4": {
        "url": f"{_784_BASE_URL}/restaurants4.tar.gz",
        "table_a": "yellow_pages", "table_b": "yelp",
        "domain": "restaurants",
    },
    "electronics": {
        "url": f"{_784_BASE_URL}/electronics.tar.gz",
        "table_a": "amazon", "table_b": "best_buy",
        "domain": "electronics",
    },
    "music": {
        "url": f"{_784_BASE_URL}/music.tar.gz",
        "table_a": "itunes", "table_b": "amazon_music",
        "domain": "music",
    },
    "cosmetics": {
        "url": f"{_784_BASE_URL}/cosmetics.tar.gz",
        "table_a": "amazon", "table_b": "sephora",
        "domain": "cosmetics",
    },
    "beer": {
        "url": f"{_784_BASE_URL}/beer.tar.gz",
        "table_a": "beer_advocate", "table_b": "rate_beer",
        "domain": "beer",
    },
    "anime": {
        "url": f"{_784_BASE_URL}/anime.tar.gz",
        "table_a": "my_anime_list", "table_b": "anime_planet",
        "domain": "anime",
    },
    "books1": {
        "url": f"{_784_BASE_URL}/books1.tar.gz",
        "table_a": "amazon", "table_b": "barnes_and_noble",
        "domain": "books",
    },
    "books2": {
        "url": f"{_784_BASE_URL}/books2.tar.gz",
        "table_a": "goodreads", "table_b": "barnes_and_noble",
        "domain": "books",
    },
    "books3": {
        "url": f"{_784_BASE_URL}/books3.tar.gz",
        "table_a": "barnes_and_noble", "table_b": "half",
        "domain": "books",
    },
    "books4": {
        "url": f"{_784_BASE_URL}/books4.tar.gz",
        "table_a": "amazon", "table_b": "barnes_and_noble",
        "domain": "books",
    },
    "books5": {
        "url": f"{_784_BASE_URL}/books5.tar.gz",
        "table_a": "amazon", "table_b": "barnes_and_noble",
        "domain": "books",
    },
    "ebooks1": {
        "url": f"{_784_BASE_URL}/ebooks1.tar.gz",
        "table_a": "itunes", "table_b": "ebooks",
        "domain": "ebooks",
    },
    "ebooks2": {
        "url": f"{_784_BASE_URL}/ebooks2.tar.gz",
        "table_a": "itunes", "table_b": "ebooks",
        "domain": "ebooks",
    },
    "citations": {
        "url": f"{_784_BASE_URL}/citations.tar.gz",
        "table_a": "google_scholar", "table_b": "dblp",
        "domain": "bibliography",
    },
    "baby_products": {
        "url": f"{_784_BASE_URL}/baby_products.tar.gz",
        "table_a": "babies_r_us", "table_b": "buy_buy_baby",
        "domain": "baby products",
    },
}


# ------------------------------------------------------------------ #
#  Скачивание и распаковка
# ------------------------------------------------------------------ #

def download_dataset(name: str, data_dir: Path) -> Path:
    """Скачать и распаковать датасет CS784 (tar.gz) в *data_dir/<name>/csv_files/*.

    Если папка уже существует — пропускает. Возвращает путь до csv_files/.
    """
    ds_info = DATASETS.get(name)
    if ds_info is None:
        raise ValueError(
            f"Неизвестный датасет '{name}'. "
            f"Доступные: {list(DATASETS.keys())}"
        )

    data_dir = Path(data_dir)
    csv_dir = data_dir / name / "csv_files"

    if csv_dir.exists() and any(csv_dir.iterdir()):
        logger.info("Датасет %s уже загружен: %s", name, csv_dir)
        return csv_dir

    data_dir.mkdir(parents=True, exist_ok=True)
    tgz_path = data_dir / f"{name}.tar.gz"

    if not tgz_path.exists():
        url = ds_info["url"]
        logger.info("Скачивание %s …", url)
        urllib.request.urlretrieve(url, str(tgz_path))

    logger.info("Распаковка %s …", tgz_path.name)
    with tarfile.open(str(tgz_path), "r:gz") as tf:
        tf.extractall(str(data_dir))

    tgz_path.unlink(missing_ok=True)
    logger.info("Датасет %s сохранён в %s", name, csv_dir)
    return csv_dir


# ------------------------------------------------------------------ #
#  Загрузка CSV
# ------------------------------------------------------------------ #

def _read_magellan_csv(path: Path) -> pd.DataFrame:
    """Читать Magellan CSV, пропуская строки-метаданные (начинающиеся с '#')."""
    with open(path, "r", encoding="latin-1") as f:
        lines = f.readlines()
    data_lines = [line for line in lines if not line.startswith("#")]
    return pd.read_csv(io.StringIO("".join(data_lines)), on_bad_lines="skip")


def _detect_id_column(df: pd.DataFrame) -> str:
    """Определить столбец ID в таблице Magellan."""
    for col in ("id", "ID", "Id", "_id"):
        if col in df.columns:
            return col
    return df.columns[0]


def load_dataset(
    dataset_path: Path,
    name: str | None = None,
    split_labels: bool = True,
    split_ratio: tuple[float, float, float] = (0.6, 0.2, 0.2),
    seed: int = 42,
) -> dict[str, pd.DataFrame]:
    """Загрузить CS784 датасет из *dataset_path* (csv_files/).

    Args:
        dataset_path: путь до csv_files/.
        name:         имя датасета (ключ из DATASETS) — нужно для
                      определения имён файлов tableA / tableB.
                      Если None — ищет автоматически.
        split_labels: если True, разбивает labeled_data на train/valid/test.
        split_ratio:  пропорции (train, valid, test).
        seed:         сид для разбиения.

    Returns:
        ``{'tableA': df, 'tableB': df, 'labeled_data': df,
          'train': df, 'valid': df, 'test': df}``
    """
    dataset_path = Path(dataset_path)

    # Определяем имена файлов таблиц
    table_a_name: str | None = None
    table_b_name: str | None = None

    if name and name in DATASETS:
        table_a_name = DATASETS[name]["table_a"]
        table_b_name = DATASETS[name]["table_b"]
    else:
        # Пробуем угадать по имени родительской папки
        parent = dataset_path.parent.name if dataset_path.name == "csv_files" else dataset_path.name
        if parent in DATASETS:
            table_a_name = DATASETS[parent]["table_a"]
            table_b_name = DATASETS[parent]["table_b"]

    result: dict[str, pd.DataFrame] = {}

    # Загрузка tableA и tableB
    if table_a_name and (dataset_path / f"{table_a_name}.csv").exists():
        result["tableA"] = _read_magellan_csv(dataset_path / f"{table_a_name}.csv")
    if table_b_name and (dataset_path / f"{table_b_name}.csv").exists():
        result["tableB"] = _read_magellan_csv(dataset_path / f"{table_b_name}.csv")

    # Fallback: если имена неизвестны, загрузить все CSV
    if "tableA" not in result or "tableB" not in result:
        for csv_file in sorted(dataset_path.glob("*.csv")):
            key = csv_file.stem
            if key not in ("labeled_data", "candset"):
                result.setdefault(key, _read_magellan_csv(csv_file))

    # Нормализация ID-столбца → 'id'
    for key in ("tableA", "tableB"):
        if key in result:
            df = result[key]
            id_col = _detect_id_column(df)
            if id_col != "id":
                df = df.rename(columns={id_col: "id"})
                result[key] = df

    # labeled_data
    labeled_path = dataset_path / "labeled_data.csv"
    if labeled_path.exists():
        labeled = _read_magellan_csv(labeled_path)
        result["labeled_data"] = labeled

        if split_labels:
            _split_labeled(result, labeled, split_ratio, seed)

    for key, df in result.items():
        logger.info("  %s: %d строк, столбцы=%s", key, len(df), list(df.columns))

    return result


def _split_labeled(
    result: dict[str, pd.DataFrame],
    labeled: pd.DataFrame,
    ratio: tuple[float, float, float],
    seed: int,
) -> None:
    """Разбить labeled_data на train/valid/test и нормализовать столбцы."""
    # Определяем столбцы внешних ключей и метки
    ltable_col = next(
        (c for c in labeled.columns if c.lower().startswith("ltable.")), None,
    )
    rtable_col = next(
        (c for c in labeled.columns if c.lower().startswith("rtable.")), None,
    )
    label_col = labeled.columns[-1]
    for c in labeled.columns:
        if c.lower().strip() in ("gold", "label", "match"):
            label_col = c
            break

    if ltable_col is None or rtable_col is None:
        logger.warning("Не удалось определить ltable/rtable столбцы: %s", list(labeled.columns))
        return

    # Нормализация: переименовать в ltable_id, rtable_id, label
    norm = labeled.rename(columns={
        ltable_col: "ltable_id",
        rtable_col: "rtable_id",
        label_col: "label",
    })[["ltable_id", "rtable_id", "label"]].copy()

    # Разбиение
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(norm))
    n = len(norm)
    n_test = max(1, int(n * ratio[2]))
    n_val = max(1, int(n * ratio[1]))

    result["train"] = norm.iloc[idx[: n - n_val - n_test]].reset_index(drop=True)
    result["valid"] = norm.iloc[idx[n - n_val - n_test : n - n_test]].reset_index(drop=True)
    result["test"] = norm.iloc[idx[n - n_test :]].reset_index(drop=True)


def load_local_dataset(
    table_a_path: str | Path,
    table_b_path: str | Path,
    labels_path: str | Path,
) -> dict[str, pd.DataFrame]:
    """Загрузить датасет из произвольных CSV файлов."""
    return {
        "tableA": pd.read_csv(table_a_path),
        "tableB": pd.read_csv(table_b_path),
        "labels": pd.read_csv(labels_path),
    }
