"""
Загрузчик данных из Magellan Data Repository для Entity Resolution.

Magellan Data Repository (https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository)
содержит реальные бенчмарк-датасеты для entity matching.

Каждый датасет содержит:
    - tableA.csv — первая таблица
    - tableB.csv — вторая таблица
    - train.csv / valid.csv / test.csv — пары (ltable_id, rtable_id, label)

Этот модуль:
    1. Загружает и распаковывает ZIP-архивы с датасетами
    2. Конвертирует в формат TablePairData (совместимый с Pipeline 01)
    3. Генерирует SM данные (описания + эмбеддинги столбцов через Ollama)
    4. Сохраняет в стандартном формате:
       - SM: sm_dataset.json
       - ER: raw/{train,val,test}/pair_XXXX_{table_a.csv, table_b.csv, meta.json}
"""

import io
import os
import json
import logging
import zipfile
from dataclasses import dataclass, field, asdict
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .columns import TablePairData

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Каталог доступных датасетов Magellan
# ═══════════════════════════════════════════════════════════════

MAGELLAN_DATASETS: Dict[str, Dict] = {
    "Abt-Buy": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Abt-Buy/exp_data.zip",
        "domain": "products",
        "description": "Abt.com vs Buy.com product listings",
    },
    "Amazon-Google": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Amazon-Google/exp_data.zip",
        "domain": "software",
        "description": "Amazon vs Google product listings",
    },
    "DBLP-ACM": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-ACM/exp_data.zip",
        "domain": "bibliography",
        "description": "DBLP vs ACM bibliographic records",
    },
    "DBLP-GoogleScholar": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/DBLP-GoogleScholar/exp_data.zip",
        "domain": "bibliography",
        "description": "DBLP vs Google Scholar bibliographic records",
    },
    "Walmart-Amazon": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Walmart-Amazon/exp_data.zip",
        "domain": "products",
        "description": "Walmart vs Amazon product listings",
    },
    "Beer": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Beer/exp_data.zip",
        "domain": "beer",
        "description": "Beer reviews dataset",
    },
    "iTunes-Amazon": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/iTunes-Amazon/exp_data.zip",
        "domain": "music",
        "description": "iTunes vs Amazon music tracks",
    },
    "Fodors-Zagats": {
        "url": "https://pages.cs.wisc.edu/~anhai/data1/deepmatcher_data/Structured/Fodors-Zagats/exp_data.zip",
        "domain": "restaurants",
        "description": "Fodors vs Zagats restaurant listings",
    },
}


@dataclass
class MagellanConfig:
    """Конфигурация загрузки реальных датасетов из Magellan Data Repository."""

    # ─────────────────── Ollama модели ───────────────────
    ollama_host: str = "http://127.0.0.1:11434"
    """Адрес Ollama сервера"""

    llm_model: str = "qwen3.5:9b"
    """LLM модель для генерации описаний столбцов"""

    embedding_model: str = "qwen3-embedding:8b"
    """Embedding модель для векторизации описаний"""

    # ─────────────────── Ollama ускорение ───────────────────
    auto_batch_size: bool = True
    initial_batch_size: int = 8
    max_batch_size: int = 512
    min_batch_size: int = 1
    keep_alive: str = "30m"
    num_predict: int = 150
    num_ctx: int = 2048
    warmup: bool = True
    auto_llm_parallel: bool = True
    max_llm_parallel: int = 16
    min_llm_parallel: int = 1
    initial_llm_parallel: int = 1

    # ─────────────────── Датасеты ───────────────────
    datasets: List[str] = field(default_factory=lambda: [
        "DBLP-ACM",
        "DBLP-GoogleScholar",
        "Amazon-Google",
        "Walmart-Amazon",
    ])
    """Список датасетов Magellan для загрузки.
    Доступные: Abt-Buy, Amazon-Google, DBLP-ACM, DBLP-GoogleScholar,
               Walmart-Amazon, Beer, iTunes-Amazon, Fodors-Zagats"""

    max_pairs_per_dataset: int = 0
    """Максимум пар на датасет (0 = все). Полезно для отладки."""

    sample_size: int = 5
    """Количество примеров значений для генерации описания столбца"""

    # ─────────────────── Выходные пути ───────────────────
    output_dir: str = "magellan_dataset"
    sm_dataset_file: str = "sm_dataset.json"
    sm_metadata_file: str = "sm_metadata.json"
    er_raw_dir: str = "raw"
    download_dir: str = "magellan_downloads"

    def save(self, filepath: str):
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(asdict(self), f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, filepath: str) -> 'MagellanConfig':
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        if isinstance(data, dict) and isinstance(data.get("magellan"), dict):
            data = data["magellan"]
        valid_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_fields}
        return cls(**filtered)


# ═══════════════════════════════════════════════════════════════
# Загрузчик датасетов
# ═══════════════════════════════════════════════════════════════


class MagellanDatasetLoader:
    """Загрузчик и конвертер датасетов из Magellan Data Repository.

    Процесс:
        1. Скачивание ZIP-архивов с датасетами
        2. Парсинг таблиц и matching-пар (train/valid/test)
        3. Конвертация matching-пар в формат TablePairData
        4. Генерация SM данных (LLM описания + эмбеддинги через Ollama)
        5. Сохранение в стандартном формате Pipeline 01

    Usage:
        config = MagellanConfig(datasets=["DBLP-ACM", "Amazon-Google"])
        loader = MagellanDatasetLoader(config)
        loader.load_and_convert()
    """

    def __init__(self, config: MagellanConfig):
        self.config = config
        self.sm_entries: List[Dict] = []
        self.sm_metadata: List[Dict] = []
        self.all_pairs: Dict[str, List[TablePairData]] = {
            'train': [], 'val': [], 'test': []
        }
        self._pair_table_ids: Dict[str, List[Tuple[int, int]]] = {
            'train': [], 'val': [], 'test': []
        }
        self._table_id_counter = 0
        self._all_tables: List[Tuple] = []  # (df, col_mapping, table_id, source)

    def load_and_convert(self):
        """Полный цикл: загрузка → конвертация → SM обработка → сохранение."""
        start_time = time()
        os.makedirs(self.config.output_dir, exist_ok=True)
        os.makedirs(self.config.download_dir, exist_ok=True)

        # Этап 1: Загрузка и парсинг датасетов
        logger.info("\n" + "═" * 70)
        logger.info("ЭТАП 1: Загрузка датасетов Magellan")
        logger.info("═" * 70)

        for ds_name in self.config.datasets:
            if ds_name not in MAGELLAN_DATASETS:
                logger.warning(f"Неизвестный датасет: {ds_name}. Пропускаем.")
                continue
            self._process_dataset(ds_name)

        total_pairs = sum(len(v) for v in self.all_pairs.values())
        logger.info(f"\nВсего пар: {total_pairs} "
                    f"(train={len(self.all_pairs['train'])}, "
                    f"val={len(self.all_pairs['val'])}, "
                    f"test={len(self.all_pairs['test'])})")
        logger.info(f"Таблиц для SM: {len(self._all_tables)}")

        # Этап 2: SM обработка (LLM описания + эмбеддинги)
        logger.info("\n" + "═" * 70)
        logger.info("ЭТАП 2: Генерация описаний и эмбеддингов столбцов")
        logger.info("═" * 70)

        self._process_sm_data()

        # Этап 3: Сохранение
        logger.info("\n" + "═" * 70)
        logger.info("ЭТАП 3: Сохранение датасета")
        logger.info("═" * 70)

        self._save_sm_data()
        self._save_er_data()

        elapsed = time() - start_time
        self._print_summary(elapsed)

    def _download_dataset(self, ds_name: str) -> Path:
        """Скачать и распаковать датасет."""
        import urllib.request

        ds_info = MAGELLAN_DATASETS[ds_name]
        url = ds_info["url"]
        zip_path = Path(self.config.download_dir) / f"{ds_name}.zip"
        extract_dir = Path(self.config.download_dir) / ds_name

        if extract_dir.exists() and any(extract_dir.iterdir()):
            logger.info(f"  {ds_name}: уже загружен → {extract_dir}")
            return extract_dir

        logger.info(f"  {ds_name}: скачивание {url} ...")
        urllib.request.urlretrieve(url, str(zip_path))

        logger.info(f"  {ds_name}: распаковка ...")
        with zipfile.ZipFile(str(zip_path), 'r') as zf:
            zf.extractall(str(extract_dir))

        # Удаляем ZIP после распаковки
        zip_path.unlink(missing_ok=True)

        return extract_dir

    def _find_csv_files(self, extract_dir: Path) -> Dict[str, Path]:
        """Найти CSV-файлы в распакованной директории.

        Magellan хранит данные в формате DeepMatcher:
            tableA.csv, tableB.csv, train.csv, valid.csv, test.csv

        Файлы могут быть в подпапке (например, exp_data/).
        """
        result = {}
        needed = ['tableA.csv', 'tableB.csv', 'train.csv', 'valid.csv', 'test.csv']

        # Рекурсивный поиск
        for csv_name in needed:
            found = list(extract_dir.rglob(csv_name))
            if found:
                result[csv_name] = found[0]

        return result

    def _process_dataset(self, ds_name: str):
        """Загрузить и сконвертировать один датасет."""
        logger.info(f"\n--- {ds_name} ---")

        extract_dir = self._download_dataset(ds_name)
        csv_files = self._find_csv_files(extract_dir)

        # Проверяем наличие ключевых файлов
        if 'tableA.csv' not in csv_files or 'tableB.csv' not in csv_files:
            logger.error(f"  {ds_name}: tableA.csv/tableB.csv не найдены!")
            return

        table_a = pd.read_csv(csv_files['tableA.csv'], encoding='utf-8')
        table_b = pd.read_csv(csv_files['tableB.csv'], encoding='utf-8')

        logger.info(f"  Table A: {table_a.shape}, столбцы: {list(table_a.columns)}")
        logger.info(f"  Table B: {table_b.shape}, столбцы: {list(table_b.columns)}")

        # Колонки ID (обычно 'id')
        id_col_a = self._detect_id_column(table_a)
        id_col_b = self._detect_id_column(table_b)
        logger.info(f"  ID столбцы: A={id_col_a}, B={id_col_b}")

        # Обработка каждого сплита
        split_map = {
            'train.csv': 'train',
            'valid.csv': 'val',
            'test.csv': 'test',
        }

        for csv_name, split in split_map.items():
            if csv_name not in csv_files:
                logger.warning(f"  {ds_name}: {csv_name} не найден, пропускаем сплит {split}")
                continue

            match_df = pd.read_csv(csv_files[csv_name], encoding='utf-8')
            pairs = self._convert_matches_to_pairs(
                table_a, table_b, match_df,
                id_col_a, id_col_b,
                ds_name, split,
            )

            logger.info(f"  {split}: {len(pairs)} пар")
            self.all_pairs[split].extend(pairs)

    def _detect_id_column(self, df: pd.DataFrame) -> str:
        """Определить столбец ID в таблице Magellan."""
        for col in ['id', 'ID', 'Id', '_id']:
            if col in df.columns:
                return col
        # Первый столбец как fallback
        return df.columns[0]

    def _convert_matches_to_pairs(
        self,
        table_a: pd.DataFrame,
        table_b: pd.DataFrame,
        match_df: pd.DataFrame,
        id_col_a: str,
        id_col_b: str,
        ds_name: str,
        split: str,
    ) -> List[TablePairData]:
        """Конвертировать Magellan matching pairs в формат TablePairData.

        Magellan match файлы содержат:
            ltable_id, rtable_id, label (0 или 1)

        Стратегия: группируем positive пары в «контекстные блоки»,
        добавляя негативные записи для формирования пар таблиц,
        похожих на синтетические из Pipeline 01.
        """
        # Определяем столбцы для ltable_id и rtable_id
        ltable_col = None
        rtable_col = None
        label_col = None

        for col in match_df.columns:
            cl = col.lower().strip()
            if cl in ('ltable_id', 'ltable_id'):
                ltable_col = col
            elif cl in ('rtable_id', 'rtable_id'):
                rtable_col = col
            elif cl in ('label', 'gold', 'match'):
                label_col = col

        if ltable_col is None or rtable_col is None or label_col is None:
            logger.warning(f"  Не удалось определить столбцы matching: {list(match_df.columns)}")
            return []

        # Положительные и отрицательные пары
        positive_pairs = match_df[match_df[label_col] == 1]
        negative_pairs = match_df[match_df[label_col] == 0]

        if len(positive_pairs) == 0:
            logger.warning(f"  Нет положительных пар в {ds_name}/{split}")
            return []

        # Столбцы данных (исключаем ID)
        data_cols_a = [c for c in table_a.columns if c != id_col_a]
        data_cols_b = [c for c in table_b.columns if c != id_col_b]

        # column_mapping: display_name → base_name (для совместимости с SM)
        # Для реальных данных base_name = display_name (нет синтетического маппинга)
        col_mapping_a = {c: c for c in data_cols_a}
        col_mapping_b = {c: c for c in data_cols_b}

        # Индексирование таблиц по ID
        table_a_indexed = table_a.set_index(id_col_a)
        table_b_indexed = table_b.set_index(id_col_b)

        # Стратегия создания пар:
        # Каждый положительный match → 1 пара таблиц, в которой:
        #   - Таблица A: строка из table_a + несколько случайных негативных строк
        #   - Таблица B: строка из table_b + несколько случайных негативных строк
        #   - duplicate_pairs: [(pos_a_idx, pos_b_idx)]
        #
        # Для более крупных пар группируем positive matches по «кластерам»

        pairs = []
        pos_list = list(positive_pairs.iterrows())

        if self.config.max_pairs_per_dataset > 0:
            pos_list = pos_list[:self.config.max_pairs_per_dataset]

        # Группируем по ~5 позитивных пар на одну TablePair для реалистичности
        group_size = 5
        neg_ids_a = set(table_a_indexed.index)
        neg_ids_b = set(table_b_indexed.index)

        for group_start in range(0, len(pos_list), group_size):
            group = pos_list[group_start:group_start + group_size]

            # Собираем ID положительных записей
            pos_ids_a = []
            pos_ids_b = []
            for _, row in group:
                lid = row[ltable_col]
                rid = row[rtable_col]
                if lid in table_a_indexed.index and rid in table_b_indexed.index:
                    pos_ids_a.append(lid)
                    pos_ids_b.append(rid)

            if not pos_ids_a:
                continue

            # Добавляем случайные не-matching строки
            import random
            available_neg_a = list(neg_ids_a - set(pos_ids_a))
            available_neg_b = list(neg_ids_b - set(pos_ids_b))

            n_neg = random.randint(3, 8)
            neg_sample_a = random.sample(available_neg_a, min(n_neg, len(available_neg_a)))
            neg_sample_b = random.sample(available_neg_b, min(n_neg, len(available_neg_b)))

            all_ids_a = pos_ids_a + neg_sample_a
            all_ids_b = pos_ids_b + neg_sample_b

            # Собираем DataFrame
            df_a = table_a_indexed.loc[all_ids_a, data_cols_a].reset_index(drop=True)
            df_b = table_b_indexed.loc[all_ids_b, data_cols_b].reset_index(drop=True)

            # Формируем duplicate_pairs (индексы в df_a/df_b)
            duplicate_pairs = [(i, i) for i in range(len(pos_ids_a))]

            # entity_ids: для реальных данных — оригинальные ID
            entity_ids_a = list(all_ids_a)
            entity_ids_b = list(all_ids_b)

            pair = TablePairData(
                df_a=df_a,
                df_b=df_b,
                duplicate_pairs=duplicate_pairs,
                column_mapping_a=col_mapping_a,
                column_mapping_b=col_mapping_b,
                entity_ids_a=entity_ids_a,
                entity_ids_b=entity_ids_b,
            )
            pairs.append(pair)

            # Также регистрируем таблицы для SM обработки
            tid_a = self._table_id_counter
            self._all_tables.append((df_a, col_mapping_a, tid_a, f"er_{split}"))
            self._table_id_counter += 1

            tid_b = self._table_id_counter
            self._all_tables.append((df_b, col_mapping_b, tid_b, f"er_{split}"))
            self._table_id_counter += 1

            self._pair_table_ids[split].append((tid_a, tid_b))

        return pairs

    def _process_sm_data(self):
        """Обработать все таблицы: LLM описания → эмбеддинги."""
        from .ollama_utils import OllamaAccelerator
        from .config import DataGenConfig

        # Создаём DataGenConfig с параметрами Ollama из MagellanConfig
        accelerator_config = DataGenConfig(
            ollama_host=self.config.ollama_host,
            llm_model=self.config.llm_model,
            embedding_model=self.config.embedding_model,
            auto_batch_size=self.config.auto_batch_size,
            initial_batch_size=self.config.initial_batch_size,
            max_batch_size=self.config.max_batch_size,
            min_batch_size=self.config.min_batch_size,
            keep_alive=self.config.keep_alive,
            num_predict=self.config.num_predict,
            num_ctx=self.config.num_ctx,
            warmup=self.config.warmup,
            auto_llm_parallel=self.config.auto_llm_parallel,
            max_llm_parallel=self.config.max_llm_parallel,
            min_llm_parallel=self.config.min_llm_parallel,
            initial_llm_parallel=self.config.initial_llm_parallel,
            sample_size=self.config.sample_size,
        )
        accelerator = OllamaAccelerator(accelerator_config)

        if self.config.warmup:
            accelerator.warmup_models()
        if self.config.auto_batch_size:
            accelerator.calibrate_batch_size()
        if self.config.auto_llm_parallel:
            accelerator.calibrate_llm_parallel()

        embed_dim = accelerator.get_embedding_dim()
        logger.info(f"Размерность эмбеддингов: {embed_dim}")

        # Собираем столбцы
        column_entries = []
        for df, col_mapping, table_id, source in self._all_tables:
            for display_name in df.columns:
                base_name = col_mapping.get(display_name)
                if not base_name:
                    continue
                content = df[display_name].dropna().tolist()[:self.config.sample_size]
                data_type = str(df[display_name].dtype)
                column_entries.append((
                    table_id, display_name, base_name,
                    content, data_type, source
                ))

        total_cols = len(column_entries)
        logger.info(f"Столбцов для обработки: {total_cols}")

        if not column_entries:
            return

        # Дедупликация: LLM вызывается только для уникальных (display_name, data_type)
        unique_key_to_content = {}
        for _, display_name, base_name, content, data_type, _ in column_entries:
            key = (display_name, data_type)
            if key not in unique_key_to_content:
                unique_key_to_content[key] = content

        unique_keys = list(unique_key_to_content.keys())
        saved = total_cols - len(unique_keys)
        logger.info(f"Уникальных (display_name, data_type): {len(unique_keys)} "
                    f"(экономия {saved} LLM вызовов)")

        # LLM описания с кэшем на диске
        desc_cache_path = os.path.join(self.config.output_dir, 'llm_desc_cache.json')
        desc_cache: Dict = {}

        if os.path.exists(desc_cache_path):
            try:
                with open(desc_cache_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                desc_cache = {(row[0], row[1]): row[2] for row in raw}
                logger.info(f"Загружен кэш LLM описаний: {len(desc_cache)} записей")
            except Exception as e:
                logger.warning(f"Не удалось загрузить кэш описаний: {e}")

        missing_keys = [k for k in unique_keys if k not in desc_cache]

        if missing_keys:
            missing_prompts = []
            for display_name, data_type in missing_keys:
                content = unique_key_to_content[(display_name, data_type)]
                sample = content[:self.config.sample_size]
                type_info = f" Data type: {data_type}." if data_type else ""
                missing_prompts.append(
                    f"Give a brief description for a table column named '{display_name}'.{type_info}\n"
                    f"If the name is unclear, try to infer from sample values: {sample}.\n"
                    f"The description should be universal, suitable for any values in this column.\n"
                    f"Output only the description and nothing else. /no_think"
                )

            logger.info(f"Генерация описаний через LLM ({len(missing_prompts)} уникальных)...")

            with tqdm(total=len(missing_prompts), desc="LLM описания") as pbar:
                def _llm_progress(done, total):
                    pbar.update(1)

                new_descriptions = accelerator.generate_descriptions_batch(
                    missing_prompts,
                    progress_callback=_llm_progress,
                )

            for key, desc in zip(missing_keys, new_descriptions):
                desc_cache[key] = desc

            # Сохраняем кэш
            os.makedirs(self.config.output_dir, exist_ok=True)
            try:
                with open(desc_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        [[k[0], k[1], v] for k, v in desc_cache.items()],
                        f, ensure_ascii=False, indent=2,
                    )
                logger.info(f"Кэш LLM описаний сохранён: {desc_cache_path}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить кэш описаний: {e}")
        else:
            logger.info(f"Все {len(unique_keys)} описаний уже есть в кэше")

        # Формируем полные описания
        full_descriptions = []
        for _, display_name, _, content, data_type, _ in column_entries:
            key = (display_name, data_type)
            desc = desc_cache.get(key, "")
            if desc:
                full_desc = f"{display_name}: {desc}"
            else:
                full_desc = f"{display_name}: {', '.join(map(str, content[:3]))}"
            full_descriptions.append(full_desc)

        # Batch embed
        logger.info(f"Batch embedding {len(full_descriptions)} описаний "
                    f"(batch_size={accelerator.embed_batch_size})...")

        with tqdm(total=len(full_descriptions), desc="Embedding") as pbar:
            def _emb_progress(done, total):
                pbar.n = done
                pbar.refresh()

            embeddings = accelerator.embed_batch(
                full_descriptions,
                progress_callback=_emb_progress,
            )

        # Формирование SM записей
        for idx, (entry, desc, emb) in enumerate(
            zip(column_entries, full_descriptions, embeddings)
        ):
            table_id, display_name, base_name, content, data_type, source = entry

            if emb is None:
                emb = [0.0] * embed_dim

            self.sm_entries.append({
                'column_name': display_name,
                'base_name': base_name,
                'description': desc,
                'embedding': emb,
                'data_type': data_type,
                'content_sample': [str(x) for x in content[:5]],
                'table_id': table_id,
                'source': source,
            })

        # Метаданные по таблицам
        table_meta = {}
        for df, col_mapping, table_id, source in self._all_tables:
            if table_id not in table_meta:
                table_meta[table_id] = {
                    'table_id': table_id,
                    'source': source,
                    'num_rows': len(df),
                    'num_columns': len(df.columns),
                    'column_mapping': col_mapping,
                }
        self.sm_metadata = sorted(table_meta.values(), key=lambda x: x['table_id'])

    def _save_sm_data(self):
        """Сохранить SM датасет."""
        if not self.sm_entries:
            logger.warning("SM датасет пуст")
            return

        dataset_path = os.path.join(self.config.output_dir, self.config.sm_dataset_file)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.sm_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"SM датасет: {dataset_path} ({len(self.sm_entries)} записей)")

        meta_path = os.path.join(self.config.output_dir, self.config.sm_metadata_file)
        meta_info = {
            'config': asdict(self.config),
            'stats': {
                'total_entries': len(self.sm_entries),
                'unique_base_names': len(set(e['base_name'] for e in self.sm_entries)),
                'tables_from_er': sum(
                    1 for m in self.sm_metadata if m['source'].startswith('er_')
                ),
                'source': 'magellan',
            },
            'tables': self.sm_metadata,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
        logger.info(f"SM метаданные: {meta_path}")

    def _save_er_data(self):
        """Сохранить ER данные: CSV пары + meta.json по сплитам."""
        for split, pairs in self.all_pairs.items():
            if not pairs:
                continue

            pair_tids = self._pair_table_ids.get(split, [])
            split_dir = os.path.join(
                self.config.output_dir, self.config.er_raw_dir, split
            )
            os.makedirs(split_dir, exist_ok=True)

            for i, pair in enumerate(pairs):
                prefix = f"pair_{i:04d}"

                pair.df_a.to_csv(
                    os.path.join(split_dir, f"{prefix}_table_a.csv"),
                    index=False, encoding='utf-8'
                )
                pair.df_b.to_csv(
                    os.path.join(split_dir, f"{prefix}_table_b.csv"),
                    index=False, encoding='utf-8'
                )

                tid_a, tid_b = pair_tids[i] if i < len(pair_tids) else (-1, -1)

                meta = {
                    'pair_id': i,
                    'split': split,
                    'duplicate_pairs': pair.duplicate_pairs,
                    'column_mapping_a': pair.column_mapping_a,
                    'column_mapping_b': pair.column_mapping_b,
                    'entity_ids_a': pair.entity_ids_a,
                    'entity_ids_b': pair.entity_ids_b,
                    'num_rows_a': len(pair.df_a),
                    'num_rows_b': len(pair.df_b),
                    'num_duplicates': len(pair.duplicate_pairs),
                    'table_id_a': tid_a,
                    'table_id_b': tid_b,
                    'source': 'magellan',
                }
                with open(
                    os.path.join(split_dir, f"{prefix}_meta.json"),
                    'w', encoding='utf-8'
                ) as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)

            logger.info(f"ER {split}: {len(pairs)} пар → {split_dir}")

    def _print_summary(self, elapsed: float):
        """Итоговая сводка."""
        mins, secs = divmod(elapsed, 60)
        logger.info("\n" + "═" * 70)
        logger.info("ИТОГО")
        logger.info("═" * 70)
        logger.info(f"Время: {int(mins)} мин {secs:.1f} сек")
        logger.info(f"Датасетов обработано: {len(self.config.datasets)}")
        for split in ['train', 'val', 'test']:
            logger.info(f"  ER {split}: {len(self.all_pairs[split])} пар")
        logger.info(f"SM записей: {len(self.sm_entries)}")
        logger.info(f"Выходная директория: {self.config.output_dir}")

    # ─────────────────── Статистика ───────────────────

    def get_dataset_stats(self) -> Dict:
        """Получить статистику по загруженным данным."""
        stats = {
            'total_pairs': sum(len(v) for v in self.all_pairs.values()),
            'splits': {},
            'sm_entries': len(self.sm_entries),
        }
        for split in ['train', 'val', 'test']:
            pairs = self.all_pairs[split]
            if not pairs:
                stats['splits'][split] = {'count': 0}
                continue
            rows_a = [len(p.df_a) for p in pairs]
            rows_b = [len(p.df_b) for p in pairs]
            dups = [len(p.duplicate_pairs) for p in pairs]
            stats['splits'][split] = {
                'count': len(pairs),
                'avg_rows_a': np.mean(rows_a),
                'avg_rows_b': np.mean(rows_b),
                'avg_duplicates': np.mean(dups),
            }
        return stats
