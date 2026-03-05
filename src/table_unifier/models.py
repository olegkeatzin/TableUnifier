"""
Ollama API обёртки для LLM генерации и эмбеддингов.

Используется модулями schema_matching и entity_resolution
для взаимодействия с Ollama сервером.
"""

import ollama
import logging

logging.getLogger("httpx").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)


class OllamaLLM:
    """Обёртка для генерации текста через Ollama."""

    def __init__(self, model: str, host: str = "localhost"):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=self.host)

    def generate(self, prompt: str) -> str:
        response = self.client.generate(model=self.model, prompt=prompt)
        return response['response']


class OllamaEmbedding:
    """Обёртка для получения эмбеддингов через Ollama."""

    def __init__(self, model: str, host: str = "localhost"):
        self.host = host
        self.model = model
        self.client = ollama.Client(host=self.host)

    def embed(self, text_list: list[str]) -> list:
        response = self.client.embed(model=self.model, input=text_list)
        return response['embeddings']