import ollama
import logging

# Отключаем verbose логирование от httpx (используется ollama)
logging.getLogger("httpx").setLevel(logging.WARNING)

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple

logger = logging.getLogger(__name__)


class OllamaLLM:
    def __init__(self, model: str, host: str = "localhost"):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=self.host)
    def generate(self, prompt: str):
        response = self.client.generate(model=self.model, prompt=prompt)
        return response['response']
    
class OllamaEmbedding:
    def __init__(self, model: str, host: str = "localhost"):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=self.host)
    def embed(self, text: str):
        response = self.client.embed(model=self.model, input=[text])
        return response['embeddings'][0]
    def embed(self, text_list: list[str]):
        response = self.client.embed(model=self.model, input=text_list)
        return response['embeddings']


class VectorClassifier:
    """
    Класс для классификации векторов путем сопоставления их с опорными
    эмбеддингами классов и отсеивания лишних экземпляров.
    """
    def __init__(self, initial_reference_embeddings: np.ndarray):
        if not isinstance(initial_reference_embeddings, np.ndarray) or initial_reference_embeddings.ndim != 2:
            raise ValueError("Опорные эмбеддинги должны быть 2D numpy массивом.")
        self.reference_embeddings = initial_reference_embeddings
        self.n_classes = initial_reference_embeddings.shape[0]

    def classify(self, input_vectors: np.ndarray, similarity_threshold: float = 0.5) -> tuple[dict, np.ndarray]:
        if input_vectors.shape[0] < self.n_classes:
            raise ValueError(f"Количество входных векторов ({input_vectors.shape[0]}) не может быть меньше количества классов ({self.n_classes}).")
        
        similarity_matrix = cosine_similarity(input_vectors, self.reference_embeddings)
        cost_matrix = 1 - similarity_matrix
        
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        classified_vectors = {}
        assigned_inputs = set()

        for input_idx, class_idx in zip(row_ind, col_ind):
            if similarity_matrix[input_idx, class_idx] >= similarity_threshold:
                if class_idx not in classified_vectors:
                     classified_vectors[class_idx] = input_idx
                     assigned_inputs.add(input_idx)

        all_input_indices = set(range(input_vectors.shape[0]))
        outlier_indices = np.array(list(all_input_indices - assigned_inputs))

        return classified_vectors, outlier_indices

    def update_embeddings(self, classified_vectors: dict, new_vectors: np.ndarray, alpha: float = 0.5):
        if not (0 < alpha <= 1):
            raise ValueError("alpha должен быть в диапазоне (0, 1].")
        for class_idx, vector_idx in classified_vectors.items():
            old_embedding = self.reference_embeddings[class_idx]
            new_vector = new_vectors[vector_idx]
            updated_embedding = (1 - alpha) * old_embedding + alpha * new_vector
            self.reference_embeddings[class_idx] = updated_embedding / np.linalg.norm(updated_embedding)



class EmbSerialVectorClassifier:
    """
    Класс-адаптер для классификации объектов EmbSerial.
    Использует VectorClassifier для вычислений, но предоставляет удобный
    интерфейс для работы с вашими кастомными объектами.
    """
    def __init__(self, initial_reference_objects: List['EmbSerial']):
        """
        Инициализирует классификатор списком опорных объектов EmbSerial.
        """
        self.reference_objects = initial_reference_objects
        
        # Извлекаем эмбеддинги для передачи в ядро классификатора
        initial_embeddings_np = np.array([obj.embedding for obj in self.reference_objects])
        
        # Создаем экземпляр вычислительного ядра
        self._core_classifier = VectorClassifier(initial_embeddings_np)
        logger.debug(f"Классификатор EmbSerial инициализирован для {len(self.reference_objects)} классов.")

    def classify(self, input_objects: List['EmbSerial'], similarity_threshold: float = 0.5) -> Tuple[Dict['EmbSerial', 'EmbSerial'], List['EmbSerial']]:
        """
        Классифицирует список входных объектов EmbSerial.

        Возвращает:
            - Словарь, где ключ - опорный объект, а значение - сопоставленный ему входной объект.
            - Список объектов, которые были определены как "лишние".
        """
        self.last_input_objects = input_objects # Сохраняем для удобства обновления
        
        # 1. Извлекаем эмбеддинги из объектов в numpy массив
        input_vectors_np = np.array([obj.embedding for obj in input_objects])

        # 2. Вызываем метод классификации из ядра
        classified_indices, outlier_indices = self._core_classifier.classify(input_vectors_np, similarity_threshold)

        # 3. Преобразуем результаты (индексы) обратно в осмысленные объекты
        classified_objects = {}
        for class_idx, vector_idx in classified_indices.items():
            ref_obj = self.reference_objects[class_idx]
            input_obj = input_objects[vector_idx]
            classified_objects[ref_obj] = input_obj
            
        outlier_objects = [input_objects[i] for i in outlier_indices]

        return classified_objects, outlier_objects

    def update_embeddings(self, classified_objects: Dict['EmbSerial', 'EmbSerial'], alpha: float = 0.5):
        """
        Обновляет опорные эмбеддинги на основе результата классификации.
        """
        # 1. Преобразуем словарь объектов обратно в словарь индексов
        classified_indices = {}
        
        # Создаем список всех входных объектов, которые были использованы в классификации,
        # чтобы правильно найти их индексы.
        all_classified_inputs = list(classified_objects.values())
        
        for ref_obj, input_obj in classified_objects.items():
            try:
                class_idx = self.reference_objects.index(ref_obj)
                # Важно: ищем индекс в том списке, который был подан на вход `classify`
                vector_idx = self.last_input_objects.index(input_obj) 
                classified_indices[class_idx] = vector_idx
            except ValueError:
                # Пропускаем, если по какой-то причине объект не найден
                continue
        
        # 2. Извлекаем numpy массив из входных данных
        input_vectors_np = np.array([obj.embedding for obj in self.last_input_objects])

        # 3. Вызываем метод обновления ядра
        self._core_classifier.update_embeddings(classified_indices, input_vectors_np, alpha)
        
        # 4. ВАЖНО: Синхронизируем обновленные эмбеддинги обратно в наши объекты
        for i, ref_obj in enumerate(self.reference_objects):
            ref_obj.embedding = self._core_classifier.reference_embeddings[i]
        
        logger.debug("Опорные эмбеддинги в объектах EmbSerial успешно обновлены.")


    