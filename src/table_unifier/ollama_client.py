"""Клиент для Ollama API с поддержкой удалённого хоста."""

import logging

import ollama

from table_unifier.config import OllamaConfig

logger = logging.getLogger(__name__)


class OllamaClient:
    """Обёртка над ollama.Client для генерации текста и эмбеддингов."""

    def __init__(self, config: OllamaConfig | None = None):
        config = config or OllamaConfig()
        self.client = ollama.Client(host=config.host)
        self.llm_model = config.llm_model
        self.embedding_model = config.embedding_model
        logger.info("Ollama client: host=%s, llm=%s, embed=%s",
                     config.host, self.llm_model, self.embedding_model)

    # ------------------------------------------------------------------ #
    #  Генерация текста (LLM)
    # ------------------------------------------------------------------ #

    def generate(self, prompt: str, model: str | None = None) -> str:
        """Генерация текста через LLM."""
        model = model or self.llm_model
        response = self.client.generate(model=model, prompt=prompt)
        return response.response

    # ------------------------------------------------------------------ #
    #  Эмбеддинги
    # ------------------------------------------------------------------ #

    def embed(self, text: str, model: str | None = None) -> list[float]:
        """Получить эмбеддинг одного текста."""
        if not text or not text.strip():
            raise ValueError("Cannot embed empty text")
        model = model or self.embedding_model
        response = self.client.embed(model=model, input=text)
        return response.embeddings[0]

    def embed_batch(self, texts: list[str], model: str | None = None) -> list[list[float]]:
        """Получить эмбеддинги батча текстов."""
        model = model or self.embedding_model
        response = self.client.embed(model=model, input=texts)
        return response.embeddings
