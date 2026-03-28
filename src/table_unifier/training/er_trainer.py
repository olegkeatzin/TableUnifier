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


def _build_model(config: EntityResolutionConfig, device: str) -> EntityResolutionGNN:
    """Создать модель из конфигурации."""
    return EntityResolutionGNN(
        row_dim=config.row_dim,
        token_dim=config.token_dim,
        col_dim=config.col_dim,
        hidden_dim=config.hidden_dim,
        edge_dim=config.edge_dim,
        output_dim=config.output_dim,
        num_gnn_layers=config.num_gnn_layers,
        dropout=config.dropout,
        bidirectional=config.bidirectional,
    ).to(device)


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
    model = _build_model(config, device)

    data = data.to(device)
    triplet_indices = triplet_indices.to(device)
    if val_triplet_indices is not None:
        val_triplet_indices = val_triplet_indices.to(device)

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

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

    # Загрузить лучший checkpoint (если валидация была и checkpoint существует)
    if save_path and save_path.exists() and val_triplet_indices is not None:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info("Загружен лучший checkpoint (val_loss=%.4f)", best_val_loss)

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
    epoch_callback: "Callable[[int, float | None], None] | None" = None,
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
        epoch_callback: вызывается после каждой эпохи как
            callback(epoch, val_loss). Если выбросит исключение —
            обучение прерывается (используется для Optuna pruning).

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

    model = _build_model(config, device)

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6,
    )

    best_val_loss = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

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
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        if epoch % 5 == 0 or epoch == 1:
            logger.info(
                "ER [multi] Epoch %d/%d — train_loss=%.4f%s, lr=%.2e (%d datasets)",
                epoch, config.epochs, train_loss, val_info, current_lr, len(datasets),
            )

        # Callback (Optuna pruning, early stopping и т.д.)
        if epoch_callback is not None:
            try:
                epoch_callback(epoch, val_loss)
            except StopIteration:
                logger.info("Обучение остановлено callback на эпохе %d", epoch)
                break

    # Сохранить финальную модель (если не было валидации)
    if save_path and not history["val_loss"]:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(model.state_dict(), save_path)
        logger.info("Модель ER (multi) сохранена: %s", save_path)

    # Загрузить лучший checkpoint (если валидация была и checkpoint существует)
    if save_path and save_path.exists() and history["val_loss"]:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info("Загружен лучший checkpoint (val_loss=%.4f)", best_val_loss)

    # Сохранить историю
    if save_path:
        import json
        history_path = save_path.with_suffix(".history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history, f)
        logger.info("История обучения: %s", history_path)

    return model, history


# ------------------------------------------------------------------ #
#  Mini-batch обучение на большом графе (NeighborLoader)
# ------------------------------------------------------------------ #


def train_entity_resolution_minibatch(
    graph: HeteroData,
    train_pairs: torch.Tensor,
    val_pairs: torch.Tensor | None = None,
    config: EntityResolutionConfig | None = None,
    device: str | None = None,
    save_path: Path | None = None,
    epoch_callback: "Callable[[int, float | None], None] | None" = None,
    model_class: str = "gat",
) -> tuple["nn.Module", dict]:
    """Mini-batch обучени�� ER на объеди��ённом графе.

    Граф хранится в CPU RAM. На каждом шаге NeighborLoader
    сэмплирует подграф для батча seed-узлов (строки из триплетов),
    подграф загружается на GPU для forward + backward.

    Args:
        graph:        HeteroData — объединённый граф (в CPU)
        train_pairs:  [N, 3] — (idx_a, idx_b, label), label=1 positive
        val_pairs:    [M, 3] — аналогично, для валидации
        config:       EntityResolutionConfig
        device:       cuda / cpu
        save_path:    путь сохранения лучшей модели
        epoch_callback: для early stopping / pruning
        model_class:  "gat" или "gnn"

    Returns:
        (model, history)
    """
    import torch.nn as nn
    from torch_geometric.loader import NeighborLoader

    config = config or EntityResolutionConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Размерности из графа
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    # Модель
    if model_class == "gat":
        from table_unifier.models.entity_resolution import EntityResolutionGAT
        model = EntityResolutionGAT(
            row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
            hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
            num_gnn_layers=config.num_gnn_layers, num_heads=config.num_heads,
            dropout=config.dropout, attention_dropout=config.attention_dropout,
            bidirectional=config.bidirectional,
        ).to(device)
    else:
        model = _build_model(config, device)

    criterion = TripletLoss(margin=config.margin)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

    # Формируем триплеты из labeled pairs
    pos_mask = train_pairs[:, 2] == 1
    neg_mask = train_pairs[:, 2] == 0
    pos_pairs = train_pairs[pos_mask]
    neg_pairs = train_pairs[neg_mask]

    # NeighborLoader: сколько соседей сэмплировать на каждом слое
    num_neighbors = {
        ("token", "in_row", "row"): [32] * config.num_gnn_layers,
        ("row", "has_token", "token"): [32] * config.num_gnn_layers,
    }

    best_val_loss = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        model.train()

        # Генерируем триплеты для этой эпохи
        rng = torch.Generator().manual_seed(epoch)
        if len(neg_pairs) > 0:
            neg_idx = torch.randint(0, len(neg_pairs), (len(pos_pairs),), generator=rng)
            triplet_a = pos_pairs[:, 0]
            triplet_p = pos_pairs[:, 1]
            triplet_n = neg_pairs[neg_idx, 1]
        else:
            all_b = torch.unique(train_pairs[:, 1])
            triplet_a = pos_pairs[:, 0]
            triplet_p = pos_pairs[:, 1]
            triplet_n = all_b[torch.randint(0, len(all_b), (len(pos_pairs),), generator=rng)]

        # Seed nodes = все строки из триплетов
        seed_rows = torch.unique(torch.cat([triplet_a, triplet_p, triplet_n]))

        # NeighborLoader для подграфа
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            input_nodes=("row", seed_rows),
            batch_size=min(config.batch_size, len(seed_rows)),
            shuffle=False,
        )

        epoch_loss = 0.0
        n_batches = 0

        for batch in loader:
            batch = batch.to(device)
            embeddings = model(batch)

            # Маппинг: global row idx → local idx в батче
            global_to_local = {gid.item(): local for local, gid in enumerate(batch["row"].n_id)}

            # Фильтруем триплеты где все 3 строки в батче
            batch_triplets = []
            for a, p, n in zip(triplet_a.tolist(), triplet_p.tolist(), triplet_n.tolist()):
                if a in global_to_local and p in global_to_local and n in global_to_local:
                    batch_triplets.append([global_to_local[a], global_to_local[p], global_to_local[n]])

            if not batch_triplets:
                batch.to("cpu")
                continue

            bt = torch.tensor(batch_triplets, device=device)
            a_emb = embeddings[bt[:, 0]]
            p_emb = embeddings[bt[:, 1]]
            n_emb = embeddings[bt[:, 2]]

            # Semi-hard mining
            a_mined, p_mined, n_mined = mine_semi_hard(
                embeddings, bt, margin=config.margin,
            )
            if a_mined.shape[0] == 0:
                a_mined, p_mined, n_mined = a_emb, p_emb, n_emb

            loss = criterion(a_mined, p_mined, n_mined)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches += 1
            batch.to("cpu")
            torch.cuda.empty_cache()

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # Validation
        val_loss = None
        if val_pairs is not None and len(val_pairs) > 0:
            model.eval()
            val_pos = val_pairs[val_pairs[:, 2] == 1]
            val_neg = val_pairs[val_pairs[:, 2] == 0]

            if len(val_pos) > 0 and len(val_neg) > 0:
                val_seed = torch.unique(torch.cat([val_pairs[:, 0], val_pairs[:, 1]]))
                val_loader = NeighborLoader(
                    graph, num_neighbors=num_neighbors,
                    input_nodes=("row", val_seed),
                    batch_size=min(512, len(val_seed)),
                    shuffle=False,
                )

                val_loss_sum = 0.0
                val_count = 0

                with torch.no_grad():
                    for batch in val_loader:
                        batch = batch.to(device)
                        val_emb = model(batch)
                        global_to_local = {gid.item(): local for local, gid in enumerate(batch["row"].n_id)}

                        for row in val_pairs:
                            a, b, label = row[0].item(), row[1].item(), row[2].item()
                            if a in global_to_local and b in global_to_local:
                                ea = val_emb[global_to_local[a]]
                                eb = val_emb[global_to_local[b]]
                                sim = (ea * eb).sum().item()
                                if label == 1:
                                    val_loss_sum += max(0, 1 - sim)
                                val_count += 1

                        batch.to("cpu")
                        torch.cuda.empty_cache()

                val_loss = val_loss_sum / max(val_count, 1) if val_count > 0 else None

        if val_loss is not None:
            history["val_loss"].append(val_loss)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        if epoch % 5 == 0 or epoch == 1:
            logger.info("ER [minibatch] Epoch %d/%d — train_loss=%.4f%s, lr=%.2e",
                        epoch, config.epochs, train_loss, val_info, current_lr)

        if epoch_callback is not None:
            try:
                epoch_callback(epoch, val_loss)
            except StopIteration:
                logger.info("Обучение остановлено callback на эпохе %d", epoch)
                break

    # Загрузить лучший checkpoint
    if save_path and save_path.exists() and history["val_loss"]:
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=True))
        logger.info("Загружен лучший checkpoint (val_loss=%.4f)", best_val_loss)

    # Сохранить историю
    if save_path:
        import json as _json
        history_path = save_path.with_suffix(".history.json")
        with open(history_path, "w", encoding="utf-8") as f:
            _json.dump(history, f)

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
