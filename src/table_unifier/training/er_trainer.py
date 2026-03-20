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
) -> tuple["EntityResolutionGNN", dict]:
    """Обучить GNN для Entity Resolution.

    Args:
        data:              HeteroData-граф (row, token, edges).
        triplet_indices:   [T, 3] — (anchor_idx, positive_idx, negative_idx).
        config:            конфигурация.
        val_triplet_indices: опциональные триплеты для валидации.
        device:            устройство (cuda / cpu).
        save_path:         путь для сохранения модели.

    Returns:
        (model, history) где history = {"train_loss": [...], "val_loss": [...]}.
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
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        # ---- Train ---- #
        model.train()
        embeddings = model(data)  # [N_rows, D_output]

        # Semi-hard mining
        a, p, n = mine_semi_hard(embeddings, triplet_indices, margin=config.margin)
        if a.shape[0] == 0:
            a = embeddings[triplet_indices[:, 0]]
            p = embeddings[triplet_indices[:, 1]]
            n = embeddings[triplet_indices[:, 2]]

        loss = criterion(a, p, n)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss = loss.item()
        history["train_loss"].append(train_loss)

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
            history["val_loss"].append(val_loss)
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

    # Сохранить историю рядом с моделью
    if save_path:
        import json
        history_path = save_path.with_suffix(".history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        logger.info("История обучения: %s", history_path)

    return model, history


# ------------------------------------------------------------------ #
#  Round-robin обучение на нескольких датасетах
# ------------------------------------------------------------------ #


def train_entity_resolution_multidataset(
    datasets: list[dict],
    config: EntityResolutionConfig | None = None,
    device: str | None = None,
    save_path: Path | None = None,
) -> tuple["EntityResolutionGNN", dict]:
    """Round-robin обучение одной GNN на нескольких датасетах.

    Каждую эпоху модель проходит по всем датасетам по очереди,
    загружая по одному графу на устройство. Это позволяет обучить
    универсальную модель без объединения графов в один гигантский граф.

    Args:
        datasets: список словарей, каждый содержит:
            - "name":            имя датасета
            - "graph":           HeteroData-граф (на CPU)
            - "train_triplets":  [T, 3] тензор
            - "val_triplets":    [V, 3] тензор или None
        config:    конфигурация.
        device:    устройство (cuda / cpu).
        save_path: путь для сохранения лучшей модели.

    Returns:
        (model, history) где history содержит train_loss, val_loss
        и per-dataset потери.
    """
    import random

    config = config or EntityResolutionConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Определяем размерности по первому графу
    sample = datasets[0]["graph"]
    config.row_dim = int(sample["row"].x.shape[1])
    config.token_dim = int(sample["token"].x.shape[1])
    config.col_dim = int(sample.col_embeddings.shape[1])

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

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

    best_val_loss = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_loss": []}

    for epoch in range(1, config.epochs + 1):
        # Перемешиваем порядок датасетов каждую эпоху
        order = list(range(len(datasets)))
        random.shuffle(order)

        epoch_train_loss = 0.0
        epoch_n_batches = 0

        # ---- Train: round-robin по датасетам ---- #
        model.train()
        for ds_idx in order:
            ds = datasets[ds_idx]
            graph = ds["graph"].to(device)
            tri = ds["train_triplets"].to(device)

            if len(tri) == 0:
                graph.to("cpu")
                continue

            embeddings = model(graph)

            a, p, n = mine_semi_hard(embeddings, tri, margin=config.margin)
            if a.shape[0] == 0:
                a = embeddings[tri[:, 0]]
                p = embeddings[tri[:, 1]]
                n = embeddings[tri[:, 2]]

            loss = criterion(a, p, n)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_train_loss += loss.item()
            epoch_n_batches += 1

            # Выгрузить граф с GPU
            graph.to("cpu")
            torch.cuda.empty_cache()

        train_loss = epoch_train_loss / max(epoch_n_batches, 1)
        history["train_loss"].append(train_loss)

        # ---- Validation: по всем датасетам ---- #
        val_loss_sum = 0.0
        val_n = 0
        model.eval()
        with torch.no_grad():
            for ds in datasets:
                vtri = ds.get("val_triplets")
                if vtri is None or len(vtri) == 0:
                    continue
                graph = ds["graph"].to(device)
                vtri_dev = vtri.to(device)

                val_emb = model(graph)
                va = val_emb[vtri_dev[:, 0]]
                vp = val_emb[vtri_dev[:, 1]]
                vn = val_emb[vtri_dev[:, 2]]
                val_loss_sum += criterion(va, vp, vn).item() * len(vtri_dev)
                val_n += len(vtri_dev)

                graph.to("cpu")
                torch.cuda.empty_cache()

        val_loss = val_loss_sum / max(val_n, 1) if val_n > 0 else None
        if val_loss is not None:
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "ER [multi] Epoch %d/%d — train_loss=%.4f%s (%d datasets)",
                epoch, config.epochs, train_loss, val_info, len(datasets),
            )

    # Сохранить финальную модель (если не было валидации)
    if save_path and not history["val_loss"]:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info("Модель ER (multi) сохранена: %s", save_path)

    # Сохранить историю
    if save_path:
        import json
        history_path = save_path.with_suffix(".history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        logger.info("История обучения: %s", history_path)

    return model, history


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
