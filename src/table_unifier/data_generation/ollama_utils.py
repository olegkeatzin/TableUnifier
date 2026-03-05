"""Утилиты ускорения Ollama: автоподбор batch_size, keep_alive.

Поддерживаемые методы ускорения:
1. keep_alive — модель остаётся в GPU между запросами (без повторной загрузки)
2. num_predict — ограничение генерации (LLM не тратит время на лишние токены)
3. num_ctx — уменьшенный контекст для коротких промптов (экономия VRAM)
4. auto_batch_size — автоматический подбор оптимального batch_size для embed API
5. warmup — предзагрузка моделей в GPU перед началом генерации

LLM-генерация выполняется последовательно, т.к. Ollama по умолчанию обрабатывает
один LLM-запрос за раз (OLLAMA_NUM_PARALLEL=1), и параллельные потоки через
ThreadPoolExecutor лишь добавляют overhead без реального ускорения.
"""

import time
import logging
import numpy as np
from typing import List, Dict, Optional, Callable

from .config import DataGenConfig

logger = logging.getLogger(__name__)


class OllamaAccelerator:
    """Ускорение работы с Ollama: автоподбор batch_size, кэширование, конкурентность.
    
    Usage:
        config = DataGenConfig(ollama_host="http://100.74.62.22:11434")
        acc = OllamaAccelerator(config)
        acc.warmup_models()              # загрузить модели в GPU
        acc.calibrate_batch_size()       # автоподбор batch_size
        
        embeddings = acc.embed_batch(texts)  # оптимальный батчинг
        descriptions = acc.generate_descriptions_batch(prompts)  # параллельные LLM
    """
    
    def __init__(self, config: DataGenConfig):
        import ollama
        self.config = config
        self.client = ollama.Client(host=config.ollama_host)
        
        self._embed_batch_size: int = config.initial_batch_size
        self._calibrated = False
    
    @property
    def embed_batch_size(self) -> int:
        return self._embed_batch_size
    
    # ─────────────────── Валидация моделей ───────────────────
    
    def _validate_models(self):
        """Проверить доступность моделей на сервере Ollama.
        
        Raises:
            RuntimeError: если модель не найдена на сервере
        """
        try:
            available = self.client.list()
            available_names = [m.model for m in available.models]
        except Exception as e:
            logger.warning(f"Не удалось получить список моделей: {e}")
            return  # не блокируем, пусть упадёт на реальном запросе
        
        for model_name, role in [
            (self.config.embedding_model, "embedding"),
            (self.config.llm_model, "LLM"),
        ]:
            # Точное совпадение
            if model_name in available_names:
                logger.info(f"  ✓ {role} модель '{model_name}' найдена")
                continue
            
            # Проверяем с :latest
            if f"{model_name}:latest" in available_names:
                logger.info(f"  ✓ {role} модель '{model_name}' найдена (как {model_name}:latest)")
                continue
            
            # Ищем частичное совпадение
            base = model_name.split(':')[0]
            candidates = [n for n in available_names if n.startswith(base)]
            
            if candidates:
                logger.warning(
                    f"  ⚠ {role} модель '{model_name}' не найдена точно. "
                    f"Похожие: {candidates}. Попробуем использовать '{model_name}' как есть."
                )
            else:
                logger.error(
                    f"  ✗ {role} модель '{model_name}' не найдена на сервере!\n"
                    f"    Доступные модели: {available_names}"
                )
                raise RuntimeError(
                    f"{role} модель '{model_name}' не найдена на сервере Ollama. "
                    f"Доступные: {available_names}"
                )
    
    # ─────────────────── Прогрев моделей ───────────────────
    
    def warmup_models(self):
        """Проверить наличие и загрузить модели в GPU с keep_alive.
        
        1. Валидация — проверяем что обе модели есть на сервере
        2. Embedding warmup — загружаем embedding модель
        3. LLM warmup — загружаем LLM модель
        """
        logger.info("Прогрев моделей Ollama...")
        
        # 0. Валидация
        self._validate_models()
        
        # 1. Embedding model
        try:
            t0 = time.time()
            self.client.embed(
                model=self.config.embedding_model,
                input=["warmup test"],
                keep_alive=self.config.keep_alive,
            )
            t_emb = time.time() - t0
            logger.info(f"  Embedding '{self.config.embedding_model}' "
                        f"загружена ({t_emb:.1f}с), keep_alive={self.config.keep_alive}")
        except Exception as e:
            logger.error(f"  Ошибка прогрева embedding: {e}")
            raise
        
        # 2. LLM model
        try:
            t0 = time.time()
            self.client.generate(
                model=self.config.llm_model,
                prompt="test",
                options={"num_predict": 1, "num_ctx": 256},
                keep_alive=self.config.keep_alive,
            )
            t_llm = time.time() - t0
            logger.info(f"  LLM '{self.config.llm_model}' "
                        f"загружена ({t_llm:.1f}с), keep_alive={self.config.keep_alive}")
        except Exception as e:
            logger.error(f"  Ошибка прогрева LLM: {e}")
            raise
    
    # ─────────────────── Автоподбор batch_size ───────────────────
    
    def calibrate_batch_size(self) -> int:
        """Автоматический подбор оптимального batch_size для embed API.
        
        Алгоритм:
        1. Пробуем batch_size = 1, 2, 4, 8, 16, 32, 64, 128, 256, 512
        2. Измеряем throughput (текстов/сек) для каждого
        3. Останавливаемся если: ошибка, или throughput начал падать
        4. Возвращаем batch_size с максимальным throughput
        
        Returns:
            Оптимальный batch_size
        """
        if not self.config.auto_batch_size:
            self._embed_batch_size = self.config.initial_batch_size
            logger.info(f"Auto batch_size отключён, используется {self._embed_batch_size}")
            return self._embed_batch_size
        
        logger.info("Калибровка оптимального batch_size для embed API...")
        
        # Генерируем тестовые тексты разной длины (реалистичнее)
        test_texts = [
            f"Столбец таблицы '{i}': содержит числовые значения количества товаров"
            for i in range(self.config.max_batch_size)
        ]
        
        best_throughput = 0.0
        best_batch_size = self.config.min_batch_size
        prev_throughput = 0.0
        decline_count = 0
        
        sizes = []
        bs = self.config.min_batch_size
        while bs <= self.config.max_batch_size:
            sizes.append(bs)
            bs *= 2
        
        for batch_size in sizes:
            batch = test_texts[:batch_size]
            try:
                # 3 замера для стабильности
                times = []
                for _ in range(3):
                    t0 = time.time()
                    self.client.embed(
                        model=self.config.embedding_model,
                        input=batch,
                        keep_alive=self.config.keep_alive,
                    )
                    times.append(time.time() - t0)
                
                median_time = sorted(times)[1]
                throughput = batch_size / median_time
                
                logger.info(f"  batch_size={batch_size:>4d}: "
                            f"{throughput:>8.1f} texts/sec "
                            f"(median {median_time:.3f}s)")
                
                if throughput > best_throughput:
                    best_throughput = throughput
                    best_batch_size = batch_size
                    decline_count = 0
                else:
                    decline_count += 1
                
                # Если throughput падает 2 раза подряд — стоп
                if decline_count >= 2:
                    logger.info(f"  Throughput падает, останавливаем калибровку")
                    break
                
                prev_throughput = throughput
                
            except Exception as e:
                logger.warning(f"  batch_size={batch_size}: ошибка ({e}), "
                              f"используем предыдущий")
                break
        
        self._embed_batch_size = best_batch_size
        self._calibrated = True
        logger.info(f"Оптимальный batch_size: {best_batch_size} "
                    f"({best_throughput:.1f} texts/sec)")
        
        return best_batch_size
    
    # ─────────────────── Batch Embedding ───────────────────
    
    def embed_batch(
        self,
        texts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[List[float]]:
        """Эмбеддинг текстов с оптимальным batch_size и fallback на единичные.
        
        Args:
            texts: Список текстов для эмбеддинга
            progress_callback: Callable(processed, total)
            
        Returns:
            Список эмбеддинг-векторов
        """
        if not texts:
            return []
        
        if not self._calibrated and self.config.auto_batch_size:
            self.calibrate_batch_size()
        
        batch_size = self._embed_batch_size
        all_embeddings: List[List[float]] = []
        total = len(texts)
        
        for i in range(0, total, batch_size):
            batch = texts[i:i + batch_size]
            try:
                response = self.client.embed(
                    model=self.config.embedding_model,
                    input=batch,
                    keep_alive=self.config.keep_alive,
                )
                all_embeddings.extend(response['embeddings'])
            except Exception as e:
                logger.warning(f"Batch embed ошибка (offset {i}, size {len(batch)}): {e}")
                # Fallback: по одному
                for text in batch:
                    try:
                        resp = self.client.embed(
                            model=self.config.embedding_model,
                            input=[text],
                            keep_alive=self.config.keep_alive,
                        )
                        all_embeddings.extend(resp['embeddings'])
                    except Exception as e2:
                        logger.error(f"Критическая ошибка embed: {e2}")
                        # Нулевой вектор (размерность определится позже)
                        all_embeddings.append(None)
            
            if progress_callback:
                progress_callback(min(i + batch_size, total), total)
        
        # Заполняем None нулевыми векторами
        dim = None
        for emb in all_embeddings:
            if emb is not None:
                dim = len(emb)
                break
        if dim and any(e is None for e in all_embeddings):
            all_embeddings = [e if e is not None else [0.0] * dim for e in all_embeddings]
        
        return all_embeddings
    
    # ─────────────────── Последовательная LLM генерация ───────────────────
    
    def generate_description(self, prompt: str) -> str:
        """Генерация одного описания через LLM с ускорениями."""
        try:
            response = self.client.generate(
                model=self.config.llm_model,
                prompt=prompt,
                options={
                    "num_predict": self.config.num_predict,
                    "num_ctx": self.config.num_ctx,
                },
                keep_alive=self.config.keep_alive,
            )
            return response['response']
        except Exception as e:
            logger.warning(f"LLM ошибка: {e}")
            return ""
    
    def generate_descriptions_batch(
        self,
        prompts: List[str],
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> List[str]:
        """Последовательная генерация описаний через LLM.
        
        Ollama по умолчанию обрабатывает один LLM-запрос за раз
        (OLLAMA_NUM_PARALLEL=1). Параллельные потоки (ThreadPoolExecutor)
        лишь добавляют overhead без реального ускорения, поэтому
        генерация выполняется последовательно.
        
        Args:
            prompts: Список промптов
            progress_callback: Callable(processed, total)
            
        Returns:
            Список описаний (в том же порядке)
        """
        if not prompts:
            return []
        
        results: List[str] = []
        total = len(prompts)
        
        for i, prompt in enumerate(prompts):
            desc = self.generate_description(prompt)
            results.append(desc)
            
            if progress_callback:
                progress_callback(i + 1, total)
        
        return results
    
    # ─────────────────── Утилиты ───────────────────
    
    def get_embedding_dim(self) -> int:
        """Определить размерность эмбеддингов текущей модели."""
        try:
            resp = self.client.embed(
                model=self.config.embedding_model,
                input=["dim test"],
                keep_alive=self.config.keep_alive,
            )
            return len(resp['embeddings'][0])
        except Exception as e:
            logger.error(f"Не удалось определить размерность: {e}")
            return 768  # fallback
    
    def release_models(self):
        """Выгрузить модели из GPU (keep_alive=0)."""
        try:
            self.client.embed(
                model=self.config.embedding_model,
                input=["release"],
                keep_alive="0",
            )
            self.client.generate(
                model=self.config.llm_model,
                prompt="release",
                options={"num_predict": 1},
                keep_alive="0",
            )
            logger.info("Модели выгружены из GPU")
        except Exception:
            pass
    
    def get_status(self) -> Dict:
        """Статус акселератора."""
        return {
            'embedding_model': self.config.embedding_model,
            'llm_model': self.config.llm_model,
            'embed_batch_size': self._embed_batch_size,
            'calibrated': self._calibrated,
            'keep_alive': self.config.keep_alive,
            'num_predict': self.config.num_predict,
            'num_ctx': self.config.num_ctx,
        }
