"""Генерация эмбеддингов для строк, токенов и столбцов.

- Столбцы: описание через LLM → эмбеддинг qwen3-embedding:8b (4096 dim)
- Строки: CLS-эмбеддинг из rubert-tiny2 (312 dim)
- Токены: vocabulary embeddings из rubert-tiny2 без контекста (312 dim)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

if TYPE_CHECKING:
    from table_unifier.ollama_client import OllamaClient

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ #
#  Промпт для описания столбца (из заметок Obsidian)
# ------------------------------------------------------------------ #
COLUMN_DESCRIPTION_PROMPT = (
    "Дай краткое описание для столбца таблицы с названием '{col_name}'.{type_info}\n"
    "Если по названию не понятно, что это за столбец, попробуй угадать "
    "на основе содержимого: {sample}.\n"
    "Описание должно быть универсальным, чтобы подходить для любых "
    "значений в этом столбце.\n"
    "Если столбец описывает что-то конкретное, думай шире — в столбце "
    "могут быть более разнообразные данные.\n"
    "Выведи только описание и ничего больше. /no_think"
)


# ------------------------------------------------------------------ #
#  Эмбеддинги столбцов (через Ollama)
# ------------------------------------------------------------------ #

def _describe_column(
    client: OllamaClient,
    col_name: str,
    df: pd.DataFrame,
    max_sample: int = 5,
) -> str:
    """Получить текстовое описание столбца через LLM."""
    sample_vals = df[col_name].dropna().astype(str).unique()[:max_sample]
    sample = ", ".join(sample_vals)

    dtype = str(df[col_name].dtype)
    type_info = f" Тип данных: {dtype}." if dtype not in ("object",) else ""

    prompt = COLUMN_DESCRIPTION_PROMPT.format(
        col_name=col_name, type_info=type_info, sample=sample,
    )
    for _ in range(3):
        description = client.generate(prompt)
        if description.strip():
            return description.strip()
    return col_name


def generate_column_embeddings(
    client: OllamaClient,
    df: pd.DataFrame,
    columns: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Для каждого столбца: описание → embedding.

    Returns:
        ``{col_name: np.ndarray[4096]}``.
    """
    columns = columns or [c for c in df.columns if c != "id"]
    result: dict[str, np.ndarray] = {}
    for col in tqdm(columns, desc="Column embeddings"):
        description = _describe_column(client, col, df)
        vec = client.embed(description)
        result[col] = np.array(vec, dtype=np.float32)
        logger.debug("  col '%s' → описание: %s", col, description[:60])
    return result


# ------------------------------------------------------------------ #
#  Эмбеддинги строк и токенов (через rubert-tiny2)
# ------------------------------------------------------------------ #

class TokenEmbedder:
    """Генерирует эмбеддинги строк (CLS/mean) и токенов (vocabulary).

    Args:
        model_name: HuggingFace-имя модели-кодировщика.
        device: 'cuda' или 'cpu'. По умолчанию — auto.
        pooling: 'cls' или 'mean' — как агрегировать ``last_hidden_state``
            в embedding предложения. 'mean' выполняется с учётом attention-маски.
            e5/MiniLM/SBERT обучались с mean-pooling, LaBSE/rubert — с CLS.
        row_prefix: префикс, добавляемый к каждому тексту перед токенизацией.
            Для ``intfloat/multilingual-e5-*`` ожидается ``"query: "``.
        trust_remote_code: для моделей с custom implementation
            (``Alibaba-NLP/gte-multilingual-base``).
    """

    def __init__(
        self,
        model_name: str = "cointegrated/rubert-tiny2",
        device: str | None = None,
        pooling: str = "cls",
        row_prefix: str = "",
        trust_remote_code: bool = False,
    ):
        if pooling not in ("cls", "mean"):
            raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.pooling = pooling
        self.row_prefix = row_prefix
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        )
        self.model = AutoModel.from_pretrained(
            model_name, trust_remote_code=trust_remote_code,
        ).to(self.device)
        self.model.eval()

        # Vocabulary embeddings (без контекста). У большинства BERT-подобных
        # моделей это ``model.embeddings.word_embeddings``; для экзотических
        # архитектур используем универсальный get_input_embeddings().
        input_emb = self.model.get_input_embeddings()
        self.vocab_embeddings: torch.Tensor = input_emb.weight.detach().clone()
        logger.info(
            "TokenEmbedder: model=%s, vocab=%d, dim=%d, pooling=%s, prefix=%r, device=%s",
            model_name, self.vocab_embeddings.shape[0],
            self.vocab_embeddings.shape[1], pooling, row_prefix, self.device,
        )

    @property
    def hidden_dim(self) -> int:
        return self.vocab_embeddings.shape[1]

    # ----- Sentence (Row) embeddings ----- #

    @torch.no_grad()
    def embed_sentences(self, texts: list[str], batch_size: int = 64) -> np.ndarray:
        """Эмбеддинги предложений. Возвращает [N, D]."""
        if self.row_prefix:
            texts = [self.row_prefix + t for t in texts]
        all_embs: list[np.ndarray] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            enc = self.tokenizer(
                batch, padding=True, truncation=True,
                max_length=512, return_tensors="pt",
            ).to(self.device)
            out = self.model(**enc)
            hidden = out.last_hidden_state  # [B, T, D]
            if self.pooling == "cls":
                pooled = hidden[:, 0]
            else:  # mean
                mask = enc["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
            all_embs.append(pooled.cpu().numpy())
        return np.concatenate(all_embs, axis=0)

    # ----- Token (vocabulary) embeddings ----- #

    def get_token_ids(self, text: str, max_length: int = 2048) -> list[int]:
        """Токенизировать текст, вернуть ID токенов (без спец-токенов)."""
        encoding = self.tokenizer(
            text, add_special_tokens=False,
            truncation=True, max_length=max_length,
        )
        return encoding["input_ids"]

    def get_vocab_embedding(self, token_id: int) -> np.ndarray:
        """Получить vocabulary embedding для token_id."""
        return self.vocab_embeddings[token_id].cpu().numpy()


# ------------------------------------------------------------------ #
#  Сериализация строки таблицы в текст
# ------------------------------------------------------------------ #

def serialize_row(row: pd.Series, columns: list[str] | None = None) -> str:
    """Преобразовать строку таблицы в текстовое представление.

    Пример: ``'title: iPhone 13 | brand: Apple | price: 999'``
    """
    columns = columns or [c for c in row.index if c != "id"]
    parts = []
    for col in columns:
        val = row.get(col, "")
        if pd.notna(val) and str(val).strip():
            parts.append(f"{col}: {val}")
    return " | ".join(parts)
