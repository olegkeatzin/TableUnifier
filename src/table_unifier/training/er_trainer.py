"""Цикл обучения модели Entity Resolution (GNN + NT-Xent / Triplet Loss)."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import torch
from torch_geometric.data import HeteroData
from tqdm import tqdm

from table_unifier.config import EntityResolutionConfig
from table_unifier.models.entity_resolution import EntityResolutionGNN
from table_unifier.models.losses import TripletLoss, mine_semi_hard, nt_xent_loss

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

        if True:  # log every epoch
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
        if True:  # log every epoch
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
    """Mini-batch обучение ER на объединённом графе с NT-Xent loss.

    Изменения по результатам ресёрча (GNN_Research_Report.docx):
      - NT-Xent вместо Triplet Loss: O(N) негативов на якорь
      - AdamW вместо Adam: decoupled weight decay
      - CosineAnnealingWarmRestarts: warmup + cosine decay
      - Gradient clipping: max_norm=1.0
      - Val loss = NT-Xent (та же формула что и train)

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
            use_input_projection=config.use_input_projection,
        ).to(device)
    else:
        model = _build_model(config, device)

    # AdamW с decoupled weight decay (раздел 2.1 ресёрча)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    # CosineAnnealingWarmRestarts (раздел 2.2 ресёрча)
    # Warmup через линейный рост lr первые warmup_ratio * epochs эпох
    warmup_epochs = max(1, int(config.epochs * config.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.epochs - warmup_epochs, T_mult=1, eta_min=1e-6,
    )

    # Позитивные пары для NT-Xent
    pos_mask = train_pairs[:, 2] == 1
    train_pos_pairs = train_pairs[pos_mask][:, :2]  # [P, 2] — (idx_a, idx_b)

    # Val позитивные пары
    val_pos_pairs = None
    if val_pairs is not None and len(val_pairs) > 0:
        val_pos_mask = val_pairs[:, 2] == 1
        if val_pos_mask.sum() > 0:
            val_pos_pairs = val_pairs[val_pos_mask][:, :2]

    # Seed nodes = все строки из позитивных пар.
    # Чередуем (a0,b0,a1,b1,...) — при shuffle=True NeighborLoader партнёры
    # чаще попадают в один батч, что важно для NT-Xent с multi-view парами.
    interleaved = torch.stack([train_pos_pairs[:, 0], train_pos_pairs[:, 1]], dim=1).reshape(-1)
    seen_set: set[int] = set()
    unique_ordered: list[int] = []
    for idx in interleaved.tolist():
        if idx not in seen_set:
            seen_set.add(idx)
            unique_ordered.append(idx)
    all_train_rows = torch.tensor(unique_ordered, dtype=torch.long)

    # NeighborLoader: сколько соседей сэмплировать на каждом слое
    # 16 вместо 32 — снижает размер подграфа в ~4x, экономит VRAM
    num_neighbors = {
        ("token", "in_row", "row"): [16] * config.num_gnn_layers,
        ("row", "has_token", "token"): [16] * config.num_gnn_layers,
    }

    # Mixed precision — x2 экономия VRAM на forward/backward
    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Предкомпьют для быстрой выборки batch_pos: тензоры на GPU (а не tolist + Python loop).
    # torch.isin + searchsorted на каждом батче работают в C, а не в Python.
    # 1.45M × int64 = ~12 MB, копеечный расход VRAM.
    train_pos_a = train_pos_pairs[:, 0].contiguous().to(device)
    train_pos_b = train_pos_pairs[:, 1].contiguous().to(device)

    best_val_loss = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        model.train()

        # Warmup: линейно увеличиваем lr от 0 до peak_lr
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor
        else:
            scheduler.step(epoch - warmup_epochs)

        # NeighborLoader для подграфа.
        # num_workers > 0 → параллельное сэмплирование (pyg-lib потоки).
        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            input_nodes=("row", all_train_rows),
            batch_size=min(config.batch_size, len(all_train_rows)),
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Ep {epoch} train", unit="batch",
                    dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = batch.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp):
                all_embeddings = model(batch)

                # Seed-строки = первые batch_size в n_id (гарантия NeighborLoader)
                n_seeds = batch["row"].batch_size
                seed_embeddings = all_embeddings[:n_seeds]
                seed_n_ids = batch["row"].n_id[:n_seeds]

                # Векторизованная выборка пар (а,p), где оба конца — seeds.
                # torch.isin + searchsorted в C, не Python.
                sorted_seeds, sort_idx = torch.sort(seed_n_ids)
                a_in = torch.isin(train_pos_a, sorted_seeds)
                b_in = torch.isin(train_pos_b, sorted_seeds)
                both = a_in & b_in
                if not both.any():
                    continue
                pa = train_pos_a[both]
                pb = train_pos_b[both]
                a_local = sort_idx[torch.searchsorted(sorted_seeds, pa)]
                b_local = sort_idx[torch.searchsorted(sorted_seeds, pb)]
                bp = torch.stack([a_local, b_local], dim=1).to(device)
                # NT-Xent по seed-строкам (~256), не по всему подграфу (~70K)
                loss = nt_xent_loss(seed_embeddings, bp, temperature=config.temperature)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # Validation: NT-Xent на val пар (та же формула что train)
        val_loss = None
        if val_pos_pairs is not None and len(val_pos_pairs) > 0:
            model.eval()
            val_seed = torch.unique(torch.cat([val_pos_pairs[:, 0], val_pos_pairs[:, 1]]))
            val_loader = NeighborLoader(
                graph, num_neighbors=num_neighbors,
                input_nodes=("row", val_seed),
                batch_size=min(512, len(val_seed)),
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            )

            val_loss_sum = 0.0
            val_count = 0
            val_pos_a = val_pos_pairs[:, 0].contiguous().to(device)
            val_pos_b = val_pos_pairs[:, 1].contiguous().to(device)

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Ep {epoch} val", unit="batch",
                                  dynamic_ncols=True, leave=False):
                    batch = batch.to(device)
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        val_all_emb = model(batch)

                    n_seeds = batch["row"].batch_size
                    val_seed_emb = val_all_emb[:n_seeds]
                    val_seed_ids = batch["row"].n_id[:n_seeds]

                    sorted_seeds, sort_idx = torch.sort(val_seed_ids)
                    both = torch.isin(val_pos_a, sorted_seeds) & torch.isin(val_pos_b, sorted_seeds)
                    if not both.any():
                        continue
                    pa = val_pos_a[both]
                    pb = val_pos_b[both]
                    a_local = sort_idx[torch.searchsorted(sorted_seeds, pa)]
                    b_local = sort_idx[torch.searchsorted(sorted_seeds, pb)]
                    bvp = torch.stack([a_local, b_local], dim=1)
                    vl = nt_xent_loss(val_seed_emb.float(), bvp, temperature=config.temperature)
                    n_pairs_b = bvp.shape[0]
                    val_loss_sum += vl.item() * n_pairs_b
                    val_count += n_pairs_b

            val_loss = val_loss_sum / max(val_count, 1) if val_count > 0 else None

        if val_loss is not None:
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        if True:  # log every epoch
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
#  BCE обучение (классификация пар)
# ------------------------------------------------------------------ #


def train_entity_resolution_bce(
    graph: HeteroData,
    train_pairs: torch.Tensor,
    val_pairs: torch.Tensor | None = None,
    config: EntityResolutionConfig | None = None,
    device: str | None = None,
    save_path: Path | None = None,
    epoch_callback: "Callable[[int, float | None], None] | None" = None,
    model_class: str = "gat",
) -> tuple["nn.Module", dict]:
    """Mini-batch обучение ER с BCE loss (классификация пар).

    Вместо метрического обучения (Triplet/NT-Xent) использует явный
    classification head: MLP(|emb_a - emb_b|) → Sigmoid → BCE.
    Подход из Ditto (VLDB 2021) и большинства SOTA методов ER.

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
        (model, history) — model это PairClassifier (backbone + head)
    """
    import torch.nn as nn
    from torch_geometric.loader import NeighborLoader
    from table_unifier.models.entity_resolution import PairClassifier

    config = config or EntityResolutionConfig()
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    # Размерности из графа
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    # Backbone
    if model_class == "gat":
        from table_unifier.models.entity_resolution import EntityResolutionGAT
        backbone = EntityResolutionGAT(
            row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
            hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
            num_gnn_layers=config.num_gnn_layers, num_heads=config.num_heads,
            dropout=config.dropout, attention_dropout=config.attention_dropout,
            bidirectional=config.bidirectional,
            use_input_projection=config.use_input_projection,
        )
    else:
        backbone = EntityResolutionGNN(
            row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
            hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
            num_gnn_layers=config.num_gnn_layers, dropout=config.dropout,
            bidirectional=config.bidirectional,
        )

    model = PairClassifier(backbone, embedding_dim=config.output_dim).to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )

    warmup_epochs = max(1, int(config.epochs * config.warmup_ratio))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=config.epochs - warmup_epochs, T_mult=1, eta_min=1e-6,
    )

    # Все строки из пар
    all_train_rows = torch.unique(torch.cat([train_pairs[:, 0], train_pairs[:, 1]]))

    num_neighbors = {
        ("token", "in_row", "row"): [16] * config.num_gnn_layers,
        ("row", "has_token", "token"): [16] * config.num_gnn_layers,
    }

    use_amp = device != "cpu"
    scaler = torch.amp.GradScaler(enabled=use_amp)

    # Предкомпьют для векторизованной фильтрации пар.
    tp_a = train_pairs[:, 0].contiguous().to(device)
    tp_b = train_pairs[:, 1].contiguous().to(device)
    tp_label = train_pairs[:, 2].to(torch.float32).contiguous().to(device)

    best_val_loss = float("inf")
    history: dict[str, list] = {"train_loss": [], "val_loss": [], "lr": []}

    for epoch in range(1, config.epochs + 1):
        model.train()

        # Warmup
        if epoch <= warmup_epochs:
            warmup_factor = epoch / warmup_epochs
            for pg in optimizer.param_groups:
                pg["lr"] = config.lr * warmup_factor
        else:
            scheduler.step(epoch - warmup_epochs)

        loader = NeighborLoader(
            graph,
            num_neighbors=num_neighbors,
            input_nodes=("row", all_train_rows),
            batch_size=min(config.batch_size, len(all_train_rows)),
            shuffle=True,
            num_workers=4,
            persistent_workers=True,
        )

        epoch_loss = 0.0
        n_batches = 0

        pbar = tqdm(loader, desc=f"Ep {epoch} train", unit="batch",
                    dynamic_ncols=True, leave=False)
        for batch in pbar:
            batch = batch.to(device)

            # Seed-строки (BCE использует все пары, не только pos)
            n_seeds = batch["row"].batch_size
            seed_n_ids = batch["row"].n_id[:n_seeds]

            sorted_seeds, sort_idx = torch.sort(seed_n_ids)
            both = torch.isin(tp_a, sorted_seeds) & torch.isin(tp_b, sorted_seeds)
            if not both.any():
                continue
            pa = tp_a[both]
            pb = tp_b[both]
            a_local = sort_idx[torch.searchsorted(sorted_seeds, pa)]
            b_local = sort_idx[torch.searchsorted(sorted_seeds, pb)]
            bp = torch.stack([a_local, b_local], dim=1)
            bl = tp_label[both]

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = model(batch, bp)
                loss = criterion(preds, bl)

            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            n_batches += 1
            pbar.set_postfix(loss=f"{epoch_loss / n_batches:.4f}")

        train_loss = epoch_loss / max(n_batches, 1)
        history["train_loss"].append(train_loss)

        # Validation
        val_loss = None
        if val_pairs is not None and len(val_pairs) > 0:
            model.eval()
            val_seed = torch.unique(torch.cat([val_pairs[:, 0], val_pairs[:, 1]]))
            val_loader = NeighborLoader(
                graph, num_neighbors=num_neighbors,
                input_nodes=("row", val_seed),
                batch_size=min(512, len(val_seed)),
                shuffle=False,
                num_workers=4,
                persistent_workers=True,
            )

            val_loss_sum = 0.0
            val_count = 0
            vp_a = val_pairs[:, 0].contiguous().to(device)
            vp_b = val_pairs[:, 1].contiguous().to(device)
            vp_label = val_pairs[:, 2].to(torch.float32).contiguous().to(device)

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Ep {epoch} val", unit="batch",
                                  dynamic_ncols=True, leave=False):
                    batch = batch.to(device)
                    n_seeds = batch["row"].batch_size
                    seed_n_ids = batch["row"].n_id[:n_seeds]

                    sorted_seeds, sort_idx = torch.sort(seed_n_ids)
                    both = torch.isin(vp_a, sorted_seeds) & torch.isin(vp_b, sorted_seeds)
                    if not both.any():
                        continue
                    a_local = sort_idx[torch.searchsorted(sorted_seeds, vp_a[both])]
                    b_local = sort_idx[torch.searchsorted(sorted_seeds, vp_b[both])]
                    bvp = torch.stack([a_local, b_local], dim=1)
                    bvl = vp_label[both]
                    with torch.amp.autocast("cuda", enabled=use_amp):
                        preds = model(batch, bvp)
                        vl = criterion(preds, bvl)
                    n_pairs_b = bvp.shape[0]
                    val_loss_sum += vl.item() * n_pairs_b
                    val_count += n_pairs_b

            val_loss = val_loss_sum / max(val_count, 1) if val_count > 0 else None

        if val_loss is not None:
            history["val_loss"].append(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                if save_path:
                    save_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(model.state_dict(), save_path)

        current_lr = optimizer.param_groups[0]["lr"]
        history["lr"].append(current_lr)

        val_info = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
        if True:  # log every epoch
            logger.info("ER [BCE] Epoch %d/%d — train_loss=%.4f%s, lr=%.2e",
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
