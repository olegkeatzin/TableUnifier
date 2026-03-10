"""Цикл обучения модели Entity Resolution (GNN + Triplet Loss)."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from table_unifier.config import EntityResolutionConfig
from table_unifier.models.entity_resolution import EntityResolutionGNN
from table_unifier.models.losses import TripletLoss, mine_semi_hard

logger = logging.getLogger(__name__)


def train_entity_resolution(
    data: HeteroData,
    triplet_indices: torch.Tensor,
    config: EntityResolutionConfig | None = None,
    val_triplet_indices: torch.Tensor | None = None,
    device: str | None = None,
    save_path: Path | None = None,
) -> EntityResolutionGNN:
    """Обучить GNN для Entity Resolution.

    Args:
        data:              HeteroData-граф (row, token, edges).
        triplet_indices:   [T, 3] — (anchor_idx, positive_idx, negative_idx).
        config:            конфигурация.
        val_triplet_indices: опциональные триплеты для валидации.
        device:            устройство (cuda / cpu).
        save_path:         путь для сохранения модели.

    Returns:
        Обученная модель EntityResolutionGNN.
    """
    config = config or EntityResolutionConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Модель
    model = EntityResolutionGNN(
        row_dim=config.row_dim,
        token_dim=config.token_dim,
        col_dim=config.col_dim,
        hidden_dim=config.hidden_dim,
        edge_dim=config.edge_dim,
        output_dim=config.output_dim,
        num_gnn_layers=config.num_gnn_layers,
        num_heads=config.num_heads,
        dropout=config.dropout,
    ).to(device)

    data = data.to(device)
    triplet_indices = triplet_indices.to(device)
    if val_triplet_indices is not None:
        val_triplet_indices = val_triplet_indices.to(device)

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        # ---- Train ---- #
        model.train()
        embeddings = model(data)  # [N_rows, D_output]

        # Semi-hard mining
        a, p, n = mine_semi_hard(embeddings, triplet_indices, margin=config.margin)
        if a.shape[0] == 0:
            # Fallback: все триплеты
            a = embeddings[triplet_indices[:, 0]]
            p = embeddings[triplet_indices[:, 1]]
            n = embeddings[triplet_indices[:, 2]]

        # Mini-batch по триплетам
        n_triplets = a.shape[0]
        perm = torch.randperm(n_triplets, device=device)
        total_loss = 0.0
        n_batches = 0

        for start in range(0, n_triplets, config.batch_size):
            idx = perm[start : start + config.batch_size]
            loss = criterion(a[idx], p[idx], n[idx])

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

            # Пересчитать эмбеддинги после обновления весов
            embeddings = model(data)
            a, p, n = mine_semi_hard(embeddings, triplet_indices, margin=config.margin)
            if a.shape[0] == 0:
                a = embeddings[triplet_indices[:, 0]]
                p = embeddings[triplet_indices[:, 1]]
                n = embeddings[triplet_indices[:, 2]]

        train_loss = total_loss / max(n_batches, 1)

        # ---- Validation ---- #
        val_info = ""
        if val_triplet_indices is not None:
            model.eval()
            with torch.no_grad():
                val_emb = model(data)
                va = val_emb[val_triplet_indices[:, 0]]
                vp = val_emb[val_triplet_indices[:, 1]]
                vn = val_emb[val_triplet_indices[:, 2]]
                val_loss = criterion(va, vp, vn).item()
            val_info = f", val_loss={val_loss:.4f}"

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "ER Epoch %d/%d — train_loss=%.4f%s",
                epoch, config.epochs, train_loss, val_info,
            )

    # Сохранить финальную модель (если не было валидации)
    if save_path and val_triplet_indices is None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info("Модель ER сохранена: %s", save_path)

    return model


# ------------------------------------------------------------------ #
#  Утилиты для инференса
# ------------------------------------------------------------------ #

@torch.no_grad()
def get_row_embeddings(
    model: EntityResolutionGNN,
    data: HeteroData,
    device: str | None = None,
) -> torch.Tensor:
    """Получить L2-нормализованные эмбеддинги строк."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)
    data = data.to(device)
    return model(data).cpu()


def find_duplicates(
    embeddings: torch.Tensor,
    id_to_global_a: dict[str, int],
    id_to_global_b: dict[str, int],
    threshold: float = 0.8,
) -> list[tuple[str, str, float]]:
    """Найти дубликаты между таблицами по косинусной близости.

    Returns:
        Список (id_A, id_B, similarity).
    """
    global_to_id_a = {v: k for k, v in id_to_global_a.items()}
    global_to_id_b = {v: k for k, v in id_to_global_b.items()}

    idx_a = sorted(id_to_global_a.values())
    idx_b = sorted(id_to_global_b.values())

    emb_a = embeddings[idx_a]  # [N_A, D]
    emb_b = embeddings[idx_b]  # [N_B, D]

    # Косинусное сходство (эмбеддинги уже L2-нормализованы)
    sim = emb_a @ emb_b.T  # [N_A, N_B]

    results: list[tuple[str, str, float]] = []
    for i, ga in enumerate(idx_a):
        for j, gb in enumerate(idx_b):
            s = sim[i, j].item()
            if s >= threshold:
                results.append((global_to_id_a[ga], global_to_id_b[gb], s))

    results.sort(key=lambda x: -x[2])
    return results
