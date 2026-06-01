"""Эксперимент 17 — Шаг 12: обучение GAT на объединённом графе (bge-m3 + views).

Обёртка над экспериментом 14 с настройками для bge-m3 и v17_views графа.
Модель: EntityResolutionGAT, use_input_projection=False (MRL — dim везде 1024).

Выходной чекпоинт: output/bge-m3/v17_views_gat[_bce]_model.pt

Использование:
    uv run python -m experiments.17.12_train                              # ntxent
    uv run python -m experiments.17.12_train --loss bce                   # BCE
    uv run python -m experiments.17.12_train --max-epochs 500 --patience 30
    uv run python -m experiments.17.12_train --no-early-stopping --max-epochs 300
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.paths import output_dir_for, unified_dir
from table_unifier.training.er_trainer import (
    train_entity_resolution_bce,
    train_entity_resolution_minibatch,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TAG = "bge-m3"
DEFAULT_SUBDIR = "v17_views"


class EarlyStopping:
    def __init__(self, patience: int = 30, warmup_epochs: int = 0):
        self.patience = patience
        self.warmup_epochs = warmup_epochs
        self.best_val = float("inf")
        self.best_epoch = 0
        self.no_improve = 0

    def __call__(self, epoch: int, val_loss: float | None) -> None:
        if val_loss is None:
            return
        if epoch <= self.warmup_epochs:
            # Во время warmup LR ещё растёт линейно — val_loss шумит.
            # Трекаем лучшее значение, но patience не накручиваем.
            if val_loss < self.best_val:
                self.best_val = val_loss
                self.best_epoch = epoch
            return
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_epoch = epoch
            self.no_improve = 0
        else:
            self.no_improve += 1
        if self.no_improve >= self.patience:
            logger.info("Early stopping @ epoch %d (best val=%.4f)",
                        self.best_epoch, self.best_val)
            raise StopIteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение GAT exp17 (bge-m3 + views)")
    parser.add_argument("--loss", choices=["bce", "ntxent"], default="ntxent")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-early-stopping", action="store_true")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--graph-subdir", default=DEFAULT_SUBDIR)
    parser.add_argument("--model-tag", default=DEFAULT_TAG)
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4,
                        help="1024 / 4 = 256 — безопасное значение для bge-m3")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    parser.add_argument("--num-workers", type=int, default=4,
                        help="DataLoader workers. Поставь 0 если Bus error / нехватка /dev/shm.")
    parser.add_argument("--warmup-epochs", type=int, default=10,
                        help="Фикс. число warmup эпох (переопределяет warmup_ratio=0.1).")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    graph_dir = unified_dir(config.data_dir, args.model_tag, args.graph_subdir)
    suffix = "_bce" if args.loss == "bce" else ""
    save_path = (
        output_dir_for(config.output_dir, args.model_tag) /
        f"{args.graph_subdir}_gat{suffix}_model.pt"
    )
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Граф: %s", graph_dir)
    graph = torch.load(graph_dir / "graph.pt", weights_only=False)
    train_pairs = torch.load(graph_dir / "train_pairs.pt", weights_only=False)
    val_pairs = torch.load(graph_dir / "val_pairs.pt", weights_only=False)

    col_dim = int(graph.col_embeddings.shape[1])
    row_dim = int(graph["row"].x.shape[1])
    token_dim = int(graph["token"].x.shape[1])

    assert row_dim == token_dim == col_dim, (
        f"MRL-размерности должны совпадать: row={row_dim}, token={token_dim}, col={col_dim}. "
        "Пересобери граф через 11_build_graph.py с правильным --target-dim."
    )
    assert col_dim % args.num_heads == 0, (
        f"col_dim={col_dim} не делится на num_heads={args.num_heads}"
    )

    logger.info("Graph: %d rows, %d tokens, %d edges | dim=%d",
                graph["row"].x.shape[0], graph["token"].x.shape[0],
                graph["token", "in_row", "row"].edge_index.shape[1], col_dim)
    logger.info("Pairs: train=%d, val=%d", len(train_pairs), len(val_pairs))

    er_config = EntityResolutionConfig(
        row_dim=row_dim,
        token_dim=token_dim,
        col_dim=col_dim,
        hidden_dim=col_dim,
        edge_dim=col_dim,
        output_dim=col_dim,
        num_gnn_layers=args.num_gnn_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        attention_dropout=0.1,
        bidirectional=args.bidirectional,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_epochs / max(args.max_epochs, 1),
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        use_input_projection=False,
    )
    logger.info("loss=%s, dim=%d, layers=%d, heads=%d, dropout=%.2f, bidir=%s, lr=%.0e",
                args.loss, col_dim, args.num_gnn_layers, args.num_heads,
                args.dropout, args.bidirectional, args.lr)

    warmup_epochs = max(1, int(er_config.epochs * er_config.warmup_ratio))
    callback = (
        None if args.no_early_stopping
        else EarlyStopping(patience=args.patience, warmup_epochs=warmup_epochs)
    )
    logger.info("Early stopping: patience=%d, warmup_epochs=%d (отсчёт начнётся с эпохи %d)",
                args.patience, warmup_epochs, warmup_epochs + 1)

    if args.loss == "bce":
        model, history = train_entity_resolution_bce(
            graph=graph, train_pairs=train_pairs, val_pairs=val_pairs,
            config=er_config, device=device, save_path=save_path,
            epoch_callback=callback, model_class="gat",
            num_workers=args.num_workers,
        )
    else:
        model, history = train_entity_resolution_minibatch(
            graph=graph, train_pairs=train_pairs, val_pairs=val_pairs,
            config=er_config, device=device, save_path=save_path,
            epoch_callback=callback, model_class="gat",
            num_workers=args.num_workers,
        )

    config_path = save_path.with_suffix(".config.json")
    with open(config_path, "w") as f:
        json.dump({
            "row_dim": er_config.row_dim,
            "token_dim": er_config.token_dim,
            "col_dim": er_config.col_dim,
            "hidden_dim": er_config.hidden_dim,
            "edge_dim": er_config.edge_dim,
            "output_dim": er_config.output_dim,
            "num_gnn_layers": er_config.num_gnn_layers,
            "num_heads": er_config.num_heads,
            "dropout": er_config.dropout,
            "attention_dropout": er_config.attention_dropout,
            "bidirectional": er_config.bidirectional,
            "use_input_projection": er_config.use_input_projection,
            "graph_subdir": args.graph_subdir,
            "model_tag": args.model_tag,
        }, f, indent=2)

    logger.info("Готово. Модель: %s", save_path)


if __name__ == "__main__":
    main()
