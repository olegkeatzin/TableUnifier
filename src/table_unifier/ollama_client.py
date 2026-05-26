"""Клиент для Ollama API с поддержкой удалённого хоста."""

import logging

import httpx
import ollama

from table_unifier.config import OllamaConfig

logger = logging.getLogger(__name__)

# Таймаут на один запрос: учитываем возможную перезагрузку модели (~1-2 мин)
# плюс время генерации. 10 минут — безопасный запас для A100.
_DEFAULT_TIMEOUT = 600.0

# Размер контекста модели. У qwen3.5:9b дефолт 262144 (256k) → 9 GiB KV cache,
# но для наших коротких промптов (описание колонки) достаточно 4k.
# Урезание контекста ускоряет prompt processing в разы.
_DEFAULT_NUM_CTX = 4096


class OllamaClient:
    """Обёртка над ollama.Client для генерации текста и эмбеддингов."""

    def __init__(
        self,
        config: OllamaConfig | None = None,
        timeout: float = _DEFAULT_TIMEOUT,
        num_ctx: int = _DEFAULT_NUM_CTX,
    ):
        config = config or OllamaConfig()
        self.client = ollama.Client(host=config.host)
        # Устанавливаем таймаут напрямую на httpx-клиент — надёжнее, чем kwarg,
        # т.к. разные версии ollama по-разному его пробрасывают.
        self.client._client.timeout = httpx.Timeout(timeout)
        self.llm_model = config.llm_model
        self.embedding_model = config.embedding_model
        self.num_ctx = num_ctx
        actual = self.client._client.timeout
        logger.info("Ollama client: host=%s, llm=%s, embed=%s, timeout=%s, num_ctx=%d",
                     config.host, self.llm_model, self.embedding_model, actual, num_ctx)

    # ------------------------------------------------------------------ #
    #  Генерация текста (LLM)
    # ------------------------------------------------------------------ #

    def generate(
        self,
        prompt: str,
        model: str | None = None,
        *,
        num_predict: int | None = None,
        temperature: float | None = None,
        keep_alive: str | None = None,
        extra_options: dict | None = None,
    ) -> str:
        """Генерация текста через LLM.

        Args:
            num_predict: жёсткий лимит токенов на выход. Очень сильно влияет на
                время — для коротких описаний достаточно ~80.
            temperature: 0.0 — детерминированный greedy decoding, быстрее sampling.
            keep_alive: как долго держать модель в VRAM после запроса (например, "30m").
                По дефолту Ollama выгружает через 5 минут idle.
        """
        model = model or self.llm_model
        options: dict = {"num_ctx": self.num_ctx}
        if num_predict is not None:
            options["num_predict"] = num_predict
        if temperature is not None:
            options["temperature"] = temperature
        if extra_options:
            options.update(extra_options)
        kwargs: dict = {"model": model, "prompt": prompt, "options": options}
        if keep_alive is not None:
            kwargs["keep_alive"] = keep_alive
        response = self.client.generate(**kwargs)
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
