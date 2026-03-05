"""
Эмбеддер токенов на основе предобученной модели FastText.

Ключевая идея:
    Используются готовые предобученные FastText-эмбеддинги для токенов.
    FastText поддерживает OOV-слова через субсловные n-граммы, что делает его
    устойчивым к опечаткам без необходимости обучения собственного эмбеддера.

Преимущества:
    1. Устойчивость к опечаткам: субсловные n-граммы FastText обеспечивают
       близкие векторы для похожих слов ("цена" ≈ "ценна")
    2. Не требует обучения — используются готовые веса из огромных корпусов
    3. Работает с любым языком (при наличии соответствующей модели)
    4. Bag-of-Words из FastText-векторов используется как эмбеддинг строки
"""

import numpy as np
import logging
from typing import List
from pathlib import Path

logger = logging.getLogger(__name__)


class FastTextEmbedder:
    """Эмбеддер токенов на основе предобученной модели FastText.

    Загружает предобученную модель FastText (формат .bin от Facebook)
    и предоставляет эмбеддинги для токенов. Благодаря субсловным
    n-граммам работает даже для OOV-слов (опечатки, новые слова).

    Пример:
        embedder = FastTextEmbedder("cc.ru.300.bin")
        vec = embedder.get_word_vector("цена")                     # [300]
        vecs = embedder.get_batch_vectors(["цена", "ценна"])       # [2, 300]
        # Векторы "цена" и "ценна" будут близки!

        bow = embedder.get_bow_embedding(["iphone", "128gb", "черный"])  # [300]
    """

    def __init__(self, model_path: str):
        """
        Args:
            model_path: Путь к предобученной FastText модели (.bin формат).
                        Скачать: https://fasttext.cc/docs/en/crawl-vectors.html
                        Например: cc.ru.300.bin для русского языка
        """
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(
                f"FastText модель не найдена: {model_path}\n"
                "Скачайте модель с https://fasttext.cc/docs/en/crawl-vectors.html\n"
                "Например: cc.ru.300.bin для русского языка"
            )

        logger.info(f"Загрузка FastText модели из {model_path}...")

        from gensim.models.fasttext import load_facebook_vectors
        self._model = load_facebook_vectors(str(path))
        self.dim = self._model.vector_size

        logger.info(f"FastText модель загружена: dim={self.dim}")

    def get_word_vector(self, word: str) -> np.ndarray:
        """Получить эмбеддинг одного слова.

        Работает и для OOV-слов благодаря субсловным n-граммам FastText.

        Args:
            word: Токен (слово)

        Returns:
            np.ndarray [dim] — вектор слова
        """
        return self._model[word.lower()].astype(np.float32)

    def get_batch_vectors(self, words: List[str]) -> np.ndarray:
        """Получить эмбеддинги для списка слов.

        Args:
            words: Список токенов

        Returns:
            np.ndarray [N, dim] — матрица эмбеддингов
        """
        if not words:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.stack([self.get_word_vector(w) for w in words])

    def get_bow_embedding(self, tokens: List[str]) -> np.ndarray:
        """Bag-of-Words эмбеддинг: среднее FastText-векторов токенов.

        Используется как стартовый эмбеддинг строки таблицы.

        Args:
            tokens: Список токенов строки

        Returns:
            np.ndarray [dim] — средний вектор
        """
        if not tokens:
            return np.zeros(self.dim, dtype=np.float32)
        vectors = self.get_batch_vectors(tokens)
        return vectors.mean(axis=0)
