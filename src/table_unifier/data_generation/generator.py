"""
Унифицированный генератор датасета для Schema Matching + Entity Resolution.

Один процесс генерации создаёт данные для обоих модулей:

1. Создание EntityPool (пулл канонических сущностей)
2. Генерация пар таблиц для ER (train/val/test) с контролируемым overlap
3. Извлечение SM данных из каждой таблицы (описания + эмбеддинги столбцов)
4. Опциональная генерация дополнительных standalone-таблиц для SM
5. Сохранение объединённого датасета:
   - SM: sm_dataset.json (column embeddings + метки для triplet loss)
   - ER: raw/{train,val,test}/pair_XXXX_{table_a.csv, table_b.csv, meta.json}

Ускорение Ollama:
- Автоподбор embed batch_size (OllamaAccelerator.calibrate_batch_size)
- keep_alive для удержания моделей в GPU
- num_predict / num_ctx для быстрой LLM генерации
- Warmup моделей перед генерацией
"""

import os
import json
import signal
import logging
import threading
import numpy as np
import pandas as pd
from time import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import asdict
from tqdm import tqdm

from .config import DataGenConfig
from .columns import (
    build_column_templates,
    TableGenerator,
    EntityPool,
    TablePairData,
    COLUMN_TEMPLATES,
)
from .ollama_utils import OllamaAccelerator

logger = logging.getLogger(__name__)


class UnifiedDatasetGenerator:
    """Генератор объединённого датасета SM + ER.
    
    Процесс:
        1. Warmup моделей Ollama + калибровка batch_size
        2. Создание EntityPool
        3. Генерация ER пар таблиц (train/val/test)
        4. Для каждой таблицы (из пар + extra): LLM описание каждого столбца
        5. Batch embed всех описаний
        
        6. Сохранение SM датасета (sm_dataset.json)
        7. Сохранение ER данных (CSV файлы + meta.json)
    
    Usage:
        config = DataGenConfig(
            ollama_host="http://100.74.62.22:11434",
            num_train_pairs=300,
            num_extra_sm_tables=200,
        )
        gen = UnifiedDatasetGenerator(config)
        gen.generate()
    """
    
    def __init__(self, config: DataGenConfig):
        self.config = config
        self.templates = build_column_templates(config)
        self.table_gen = TableGenerator(config, self.templates)
        self.accelerator = OllamaAccelerator(config)
        
        # Состояние
        self.interrupted = False
        self._lock = threading.Lock()
        
        # Результаты
        self.sm_entries: List[Dict] = []
        self.sm_metadata: List[Dict] = []
        self.er_pairs: Dict[str, List[TablePairData]] = {}
        self._er_pair_table_ids: Dict[str, List[Tuple[int, int]]] = {}
        
        logger.info(f"UnifiedDatasetGenerator инициализирован")
        logger.info(f"  LLM: {config.llm_model}")
        logger.info(f"  Embedding: {config.embedding_model}")
        logger.info(f"  ER пар: {config.total_er_pairs} "
                     f"(train={config.num_train_pairs}, "
                     f"val={config.num_val_pairs}, "
                     f"test={config.num_test_pairs})")
        logger.info(f"  Extra SM таблиц: {config.num_extra_sm_tables}")
        logger.info(f"  Ollama ускорение: auto_batch={config.auto_batch_size}, "
                     f"keep_alive={config.keep_alive}")
    
    def generate(self):
        """Полный цикл генерации объединённого датасета."""
        start_time = time()
        
        # Обработка Ctrl+C
        def signal_handler(signum, frame):
            logger.warning("\nПолучен сигнал прерывания (Ctrl+C)")
            self.interrupted = True
        signal.signal(signal.SIGINT, signal_handler)
        if hasattr(signal, 'SIGTERM'):
            signal.signal(signal.SIGTERM, signal_handler)
        
        os.makedirs(self.config.output_dir, exist_ok=True)
        
        try:
            # ═══ Этап 1: Прогрев и калибровка ═══
            logger.info("\n" + "═" * 70)
            logger.info("ЭТАП 1: Прогрев моделей и калибровка batch_size / llm_parallel")
            logger.info("═" * 70)
            
            if self.config.warmup:
                self.accelerator.warmup_models()
            
            if self.config.auto_batch_size:
                self.accelerator.calibrate_batch_size()
            
            if self.config.auto_llm_parallel:
                self.accelerator.calibrate_llm_parallel()
            
            embed_dim = self.accelerator.get_embedding_dim()
            logger.info(f"Размерность эмбеддингов: {embed_dim}")
            
            # ═══ Этап 2: EntityPool + генерация таблиц ═══
            logger.info("\n" + "═" * 70)
            logger.info("ЭТАП 2: Генерация таблиц")
            logger.info("═" * 70)
            
            entity_pool = EntityPool(self.templates, self.config)
            
            # 2a. ER пары
            all_tables = []  # (df, col_mapping, table_id, source)
            table_id_counter = 0
            
            for split, n_pairs in [
                ('train', self.config.num_train_pairs),
                ('val', self.config.num_val_pairs),
                ('test', self.config.num_test_pairs),
            ]:
                logger.info(f"Генерация ER пар: {split} ({n_pairs})")
                pairs = []
                pair_tids = []
                for i in tqdm(range(n_pairs), desc=f"ER {split}"):
                    if self.interrupted:
                        break
                    try:
                        pair = self.table_gen.generate_er_pair(entity_pool)
                        pairs.append(pair)
                        
                        tid_a = table_id_counter
                        # Таблицы для SM обработки
                        all_tables.append((
                            pair.df_a, pair.column_mapping_a,
                            table_id_counter, f"er_{split}"
                        ))
                        table_id_counter += 1
                        tid_b = table_id_counter
                        all_tables.append((
                            pair.df_b, pair.column_mapping_b,
                            table_id_counter, f"er_{split}"
                        ))
                        table_id_counter += 1
                        pair_tids.append((tid_a, tid_b))
                    except Exception as e:
                        logger.error(f"Ошибка генерации ER пары {i} ({split}): {e}")
                
                self.er_pairs[split] = pairs
                self._er_pair_table_ids[split] = pair_tids
                if self.interrupted:
                    break
            
            # 2b. Extra standalone таблицы для SM
            if self.config.num_extra_sm_tables > 0 and not self.interrupted:
                logger.info(f"Генерация extra SM таблиц: {self.config.num_extra_sm_tables}")
                for i in tqdm(range(self.config.num_extra_sm_tables), desc="SM extra"):
                    if self.interrupted:
                        break
                    try:
                        df, meta = self.table_gen.generate_standalone_table()
                        all_tables.append((
                            df, meta['column_mapping'],
                            table_id_counter, "sm_extra"
                        ))
                        table_id_counter += 1
                    except Exception as e:
                        logger.error(f"Ошибка генерации SM таблицы {i}: {e}")
            
            logger.info(f"Всего таблиц для SM обработки: {len(all_tables)}")
            
            # ═══ Этап 3: LLM описания + эмбеддинги столбцов ═══
            logger.info("\n" + "═" * 70)
            logger.info("ЭТАП 3: Генерация описаний и эмбеддингов столбцов")
            logger.info("═" * 70)
            
            self._process_sm_data(all_tables, embed_dim)
            
            # ═══ Этап 4: Сохранение ═══
            logger.info("\n" + "═" * 70)
            logger.info("ЭТАП 4: Сохранение датасета")
            logger.info("═" * 70)
            
            self._save_sm_data()
            self._save_er_data()
            
        except KeyboardInterrupt:
            logger.warning("KeyboardInterrupt перехвачен")
            self.interrupted = True
        
        finally:
            elapsed = time() - start_time
            self._print_summary(elapsed)
    
    def _process_sm_data(self, all_tables: List[Tuple], embed_dim: int):
        """Обработать все таблицы: LLM описания → эмбеддинги.
        
        Оптимизация:
        1. Собираем ВСЕ столбцы из всех таблиц
        2. Дедуплицируем по (display_name, data_type) — LLM вызывается один раз на уникальную пару
        3. Batch embed всех описаний за один проход
        """
        # 1. Собираем информацию о всех столбцах
        column_entries = []  # (table_id, display_name, base_name, content_sample, data_type, source)
        
        for df, col_mapping, table_id, source in all_tables:
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
        
        # 2. Дедупликация: LLM вызывается только для уникальных (display_name, data_type)
        unique_key_to_content = {}  # (display_name, data_type) -> первый встреченный content
        unique_key_to_basenames: Dict[Tuple, str] = {}  # (display_name, data_type) -> base_name
        for _, display_name, base_name, content, data_type, _ in column_entries:
            key = (display_name, data_type)
            if key not in unique_key_to_content:
                unique_key_to_content[key] = content
                unique_key_to_basenames[key] = base_name
        
        unique_keys = list(unique_key_to_content.keys())
        saved = total_cols - len(unique_keys)
        logger.info(f"Уникальных (display_name, data_type): {len(unique_keys)} "
                    f"(экономия {saved} LLM вызовов, "
                    f"{saved / total_cols * 100:.1f}%)")
        
        # 3. Генерация промптов только для уникальных пар
        unique_prompts = []
        for display_name, data_type in unique_keys:
            content = unique_key_to_content[(display_name, data_type)]
            sample = content[:self.config.sample_size]
            type_info = f" Тип данных: {data_type}." if data_type else ""
            base_name = unique_key_to_basenames.get((display_name, data_type))
            canonical_desc = (
                COLUMN_TEMPLATES.get(base_name, {}).get('description', '')
                if base_name else ''
            )
            context_hint = (
                f" Семантика столбца: {canonical_desc}."
                if canonical_desc else ""
            )
            prompt = (
                f"Дай краткое описание для столбца таблицы с названием '{display_name}'.{type_info}{context_hint}\n"
                f"Если по названию не понятно, что это за столбец, попробуй угадать на основе "
                f"содержимого: {sample}.\n"
                f"Описание должно быть универсальным, чтобы подходить для любых значений в этом столбце.\n"
                f"Если столбец описывает что-то конкретное, думай шире - в столбце могут быть более "
                f"разнообразные данные.\n"
                f"Выведи только описание и ничего больше. /no_think"
            )
            unique_prompts.append(prompt)
        
        # 4. LLM генерация только уникальных описаний (с кэшем на диске для возобновления)
        desc_cache_path = os.path.join(self.config.output_dir, 'llm_desc_cache.json')
        desc_cache: Dict = {}
        
        if os.path.exists(desc_cache_path):
            try:
                with open(desc_cache_path, 'r', encoding='utf-8') as f:
                    raw = json.load(f)
                # Формат: [[display_name, data_type, description], ...]
                desc_cache = {(row[0], row[1]): row[2] for row in raw}
                logger.info(f"Загружен кэш LLM описаний: {len(desc_cache)} записей из {desc_cache_path}")
            except Exception as e:
                logger.warning(f"Не удалось загрузить кэш описаний: {e}")
                desc_cache = {}
        
        missing_keys = [k for k in unique_keys if k not in desc_cache]
        
        if missing_keys:
            missing_prompts = []
            for display_name, data_type in missing_keys:
                content = unique_key_to_content[(display_name, data_type)]
                sample = content[:self.config.sample_size]
                type_info = f" Тип данных: {data_type}." if data_type else ""
                missing_prompts.append(
                    f"Дай краткое описание для столбца таблицы с названием '{display_name}'.{type_info}\n"
                    f"Если по названию не понятно, что это за столбец, попробуй угадать на основе "
                    f"содержимого: {sample}.\n"
                    f"Описание должно быть универсальным, чтобы подходить для любых значений в этом столбце.\n"
                    f"Если столбец описывает что-то конкретное, думай шире - в столбце могут быть более "
                    f"разнообразные данные.\n"
                    f"Выведи только описание и ничего больше. /no_think"
                )
            
            logger.info(f"Генерация описаний через LLM "
                        f"(num_parallel={self.accelerator.llm_num_parallel}, "
                        f"уникальных пар: {len(missing_prompts)})...")
            
            with tqdm(total=len(missing_prompts), desc="LLM описания") as pbar:
                def _llm_progress(done, total):
                    pbar.update(1)
                
                new_descriptions = self.accelerator.generate_descriptions_batch(
                    missing_prompts,
                    progress_callback=_llm_progress,
                )
            
            for key, desc in zip(missing_keys, new_descriptions):
                desc_cache[key] = desc
            
            # Сохраняем кэш на диск — защита от повторного LLM при сбое embedding
            try:
                os.makedirs(self.config.output_dir, exist_ok=True)
                with open(desc_cache_path, 'w', encoding='utf-8') as f:
                    json.dump(
                        [[k[0], k[1], v] for k, v in desc_cache.items()],
                        f, ensure_ascii=False, indent=2,
                    )
                logger.info(f"Кэш LLM описаний сохранён: {desc_cache_path}")
            except Exception as e:
                logger.warning(f"Не удалось сохранить кэш описаний: {e}")
        else:
            logger.info(f"Все {len(unique_keys)} описаний уже есть в кэше, LLM пропускается")
        
        # Формируем полные описания для всех столбцов из кэша
        full_descriptions = []
        for _, display_name, _, content, data_type, _ in column_entries:
            key = (display_name, data_type)
            desc = desc_cache.get(key, "")
            if desc:
                full_desc = f"{display_name}: {desc}"
            else:
                full_desc = f"{display_name}: {', '.join(map(str, content[:3]))}"
            full_descriptions.append(full_desc)
        
        # 4. Batch embed всех описаний
        logger.info(f"Batch embedding {len(full_descriptions)} описаний "
                    f"(batch_size={self.accelerator.embed_batch_size})...")
        
        with tqdm(total=len(full_descriptions), desc="Embedding") as pbar:
            def _emb_progress(done, total):
                pbar.n = done
                pbar.refresh()
            
            embeddings = self.accelerator.embed_batch(
                full_descriptions,
                progress_callback=_emb_progress,
            )
        
        # 5. Формирование SM записей
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
        for df, col_mapping, table_id, source in all_tables:
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
        """Сохранить SM датасет и метаданные."""
        if not self.sm_entries:
            logger.warning("SM датасет пуст, пропускаем сохранение")
            return
        
        # Датасет
        dataset_path = os.path.join(self.config.output_dir, self.config.sm_dataset_file)
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(self.sm_entries, f, ensure_ascii=False, indent=2)
        logger.info(f"SM датасет: {dataset_path} ({len(self.sm_entries)} записей)")
        
        # Метаданные
        meta_path = os.path.join(self.config.output_dir, self.config.sm_metadata_file)
        meta_info = {
            'config': asdict(self.config),
            'stats': {
                'total_entries': len(self.sm_entries),
                'unique_base_names': len(set(e['base_name'] for e in self.sm_entries)),
                'tables_from_er': sum(
                    1 for m in self.sm_metadata if m['source'].startswith('er_')
                ),
                'tables_from_extra': sum(
                    1 for m in self.sm_metadata if m['source'] == 'sm_extra'
                ),
            },
            'accelerator': self.accelerator.get_status(),
            'tables': self.sm_metadata,
        }
        with open(meta_path, 'w', encoding='utf-8') as f:
            json.dump(meta_info, f, ensure_ascii=False, indent=2)
        logger.info(f"SM метаданные: {meta_path}")
    
    def _save_er_data(self):
        """Сохранить ER данные: CSV пары + meta.json по сплитам.
        
        Также сохраняет table_id_a/table_id_b для связи с SM датасетом,
        чтобы pipeline 03 мог загрузить готовые raw embeddings столбцов
        вместо повторного вызова LLM + Ollama.
        """
        for split, pairs in self.er_pairs.items():
            if not pairs:
                continue
            
            pair_tids = self._er_pair_table_ids.get(split, [])
            
            split_dir = os.path.join(
                self.config.output_dir, self.config.er_raw_dir, split
            )
            os.makedirs(split_dir, exist_ok=True)
            
            for i, pair in enumerate(pairs):
                prefix = f"pair_{i:04d}"
                
                # CSV таблиц
                pair.df_a.to_csv(
                    os.path.join(split_dir, f"{prefix}_table_a.csv"),
                    index=False, encoding='utf-8'
                )
                pair.df_b.to_csv(
                    os.path.join(split_dir, f"{prefix}_table_b.csv"),
                    index=False, encoding='utf-8'
                )
                
                # table_ids из SM датасета
                tid_a, tid_b = pair_tids[i] if i < len(pair_tids) else (-1, -1)
                
                # Метаданные пары
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
                }
                with open(
                    os.path.join(split_dir, f"{prefix}_meta.json"),
                    'w', encoding='utf-8'
                ) as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
            
            logger.info(f"ER {split}: {len(pairs)} пар → {split_dir}")
    
    def _print_summary(self, elapsed: float):
        """Итоговая статистика."""
        logger.info("\n" + "═" * 70)
        logger.info("ИТОГИ ГЕНЕРАЦИИ ОБЪЕДИНЁННОГО ДАТАСЕТА")
        logger.info("═" * 70)
        
        # ER статистика
        for split, pairs in self.er_pairs.items():
            logger.info(f"ER {split:>5s}: {len(pairs)} пар")
        total_pairs = sum(len(p) for p in self.er_pairs.values())
        logger.info(f"ER всего:  {total_pairs} пар ({total_pairs * 2} таблиц)")
        
        # SM статистика
        logger.info(f"\nSM записей: {len(self.sm_entries)}")
        if self.sm_entries:
            from collections import Counter
            base_names = Counter(e['base_name'] for e in self.sm_entries)
            sources = Counter(e['source'] for e in self.sm_entries)
            logger.info(f"SM уникальных типов: {len(base_names)}")
            logger.info(f"SM источники:")
            for src, cnt in sources.most_common():
                logger.info(f"  {src}: {cnt}")
            
            emb_dim = len(self.sm_entries[0]['embedding']) if self.sm_entries[0]['embedding'] else 0
            logger.info(f"Размерность эмбеддингов: {emb_dim}")
            
            logger.info(f"\nРаспределение по типам столбцов:")
            for bn, cnt in base_names.most_common():
                logger.info(f"  {bn:30s}: {cnt:5d}")
        
        # Ollama статистика
        status = self.accelerator.get_status()
        logger.info(f"\nOllama ускорение:")
        logger.info(f"  embed batch_size: {status['embed_batch_size']}")
        logger.info(f"  keep_alive:       {status['keep_alive']}")
        logger.info(f"  num_predict:      {status['num_predict']}")
        logger.info(f"  num_ctx:          {status['num_ctx']}")
        
        logger.info(f"\nВремя: {elapsed:.1f} сек ({elapsed/60:.1f} мин)")
        logger.info(f"Выходная директория: {self.config.output_dir}")
        
        if self.interrupted:
            logger.warning("Генерация была прервана! Данные сохранены частично.")
        
        logger.info("═" * 70)
