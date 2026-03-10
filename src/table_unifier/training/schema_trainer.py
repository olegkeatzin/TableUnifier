"""Цикл обучения модели Schema Matching (проекция эмбеддингов + Triplet Loss)."""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from table_unifier.config import SchemaMatchingConfig
from table_unifier.models.losses import TripletLoss
from table_unifier.models.schema_matching import SchemaProjector

logger = logging.getLogger(__name__)


def _build_sm_triplets(
    column_embeddings: dict[str, np.ndarray],
    ground_truth: list[tuple[str, str]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Сформировать тройки для Schema Matching.

    Args:
        column_embeddings: {col_name: raw_embedding}.
        ground_truth:      [(col_A, col_B), …] — пары эквивалентных столбцов.

    Returns:
        (anchors, positives, negatives) — массивы эмбеддингов.
    """
    gt_set = set(ground_truth) | {(b, a) for a, b in ground_truth}
    all_names = list(column_embeddings.keys())

    anchors, positives, negatives = [], [], []
    for col_a, col_b in ground_truth:
        if col_a not in column_embeddings or col_b not in column_embeddings:
            continue
        anchor = column_embeddings[col_a]
        positive = column_embeddings[col_b]
        # Негатив: случайный столбец, не являющийся парой
        for neg_name in all_names:
            if (col_a, neg_name) in gt_set or neg_name == col_a:
                continue
            neg = column_embeddings[neg_name]
            anchors.append(anchor)
            positives.append(positive)
            negatives.append(neg)

    return np.array(anchors), np.array(positives), np.array(negatives)


def train_schema_matching(
    column_embeddings: dict[str, np.ndarray],
    ground_truth: list[tuple[str, str]],
    config: SchemaMatchingConfig | None = None,
    device: str | None = None,
    save_path: Path | None = None,
) -> SchemaProjector:
    """Обучить проекционную модель Schema Matching.

    Args:
        column_embeddings: {col_name: raw_embedding (4096 dim)}.
        ground_truth:      [(col_A, col_B)] — пары эквивалентных столбцов.
        config:            конфигурация.
        device:            устройство (cuda / cpu).
        save_path:         путь для сохранения обученной модели.

    Returns:
        Обученная модель SchemaProjector.
    """
    config = config or SchemaMatchingConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Данные
    anc, pos, neg = _build_sm_triplets(column_embeddings, ground_truth)
    if len(anc) == 0:
        raise ValueError("Не удалось сформировать триплеты для Schema Matching.")

    dataset = TensorDataset(
        torch.tensor(anc, dtype=torch.float32),
        torch.tensor(pos, dtype=torch.float32),
        torch.tensor(neg, dtype=torch.float32),
    )
    loader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Модель
    model = SchemaProjector(
        input_dim=config.embedding_dim,
        hidden_dim=config.hidden_dim,
        output_dim=config.projection_dim,
        dropout=config.dropout,
    ).to(device)

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    # Обучение
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        for batch_a, batch_p, batch_n in loader:
            batch_a = batch_a.to(device)
            batch_p = batch_p.to(device)
            batch_n = batch_n.to(device)

            out_a = model(batch_a)
            out_p = model(batch_p)
            out_n = model(batch_n)

            loss = criterion(out_a, out_p, out_n)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_a)

        avg_loss = total_loss / len(dataset)
        if epoch % 5 == 0 or epoch == 1:
            logger.info("SM Epoch %d/%d — loss=%.4f", epoch, config.epochs, avg_loss)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info("Модель SM сохранена: %s", save_path)

    return model
