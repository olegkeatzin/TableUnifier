"""Эксперимент 14 — Обучение GAT на MRL-графе без входных проекций.

Архитектурное отличие от exp 09/11:
  - col_dim == row_dim == hidden_dim == edge_dim == output_dim == 312
  - use_input_projection=False: row_proj/token_proj/edge_proj убраны.
    Учатся только GAT-слои (W_src, W_dst, W_edge, attention) + output_head.
  - num_heads подбирается чтобы 312 делилось нацело (по умолч. 4: 312/4=78).

Поддерживает оба loss'а:
  - --loss bce    (как exp 11) → v14_mrl_gat_bce_model.pt
  - --loss ntxent (как exp 09) → v14_mrl_gat_model.pt

Граф ожидается из exp 14 build (data/graphs/v14_mrl/).

Использование:
    # в screen:
    python -m experiments.14_train_gat_mrl --loss bce    --max-epochs 500 --patience 30
    python -m experiments.14_train_gat_mrl --loss ntxent --max-epochs 500 --patience 30
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


class EarlyStopping:
    def __init__(self, patience: int = 30):
        self.patience = patience
        self.best_val = float("inf")
        self.best_epoch = 0
        self.epochs_no_improve = 0

    def __call__(self, epoch: int, val_loss: float | None) -> None:
        if val_loss is None:
            return
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.best_epoch = epoch
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1
        if self.epochs_no_improve >= self.patience:
            logger.info("Early stopping: best=%.4f @ ep %d", self.best_val, self.best_epoch)
            raise StopIteration


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение GAT без проекций (MRL)")
    parser.add_argument("--loss", choices=["bce", "ntxent"], default="bce",
                        help="bce → exp 11-style classification, ntxent → exp 09-style metric")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--no-early-stopping", action="store_true",
                        help="Отключить early stopping — учить ровно --max-epochs эпох")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--graph-subdir", default="v14_mrl")
    parser.add_argument("--model-tag", default=None,
                        help="Namespace токен-модели. По умолч. из config.")
    parser.add_argument("--num-gnn-layers", type=int, default=2)
    parser.add_argument("--num-heads", type=int, default=4,
                        help="hidden_dim должен делиться на num_heads "
                             "(4 безопасно для 312/384/768/1024)")
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--bidirectional", action="store_true", default=True)
    parser.add_argument("--no-bidirectional", dest="bidirectional", action="store_false")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_tag = args.model_tag or config.entity_resolution.token_model_tag
    graph_dir = unified_dir(config.data_dir, model_tag, args.graph_subdir)
    suffix = "_bce" if args.loss == "bce" else ""
    save_path = output_dir_for(config.output_dir, model_tag) / f"v14_mrl_gat{suffix}_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Model=%s, граф=%s", model_tag, graph_dir)
    graph = torch.load(graph_dir / "graph.pt", weights_only=False)
    train_pairs = torch.load(graph_dir / "train_pairs.pt", weights_only=False)
    val_pairs = torch.load(graph_dir / "val_pairs.pt", weights_only=False)

    col_dim = int(graph.col_embeddings.shape[1])
    row_dim = int(graph["row"].x.shape[1])
    token_dim = int(graph["token"].x.shape[1])
    n_edges = int(graph["token", "in_row", "row"].edge_index.shape[1])

    logger.info("Graph: %d rows, %d tokens, %d edges, col_dim=%d, row_dim=%d, token_dim=%d",
                graph["row"].x.shape[0], graph["token"].x.shape[0],
                n_edges, col_dim, row_dim, token_dim)
    logger.info("Train: %d pairs, Val: %d pairs", len(train_pairs), len(val_pairs))

    # use_input_projection=False требует row_dim == token_dim == hidden_dim, col_dim == edge_dim
    assert row_dim == token_dim == col_dim, (
        f"Размерности должны совпадать: row={row_dim}, token={token_dim}, col={col_dim}. "
        f"Перестрой граф через 14_build_unified_graph_mrl.py с правильным --target-dim."
    )
    assert col_dim % args.num_heads == 0, (
        f"col_dim ({col_dim}) не делится на num_heads ({args.num_heads})"
    )

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
        warmup_ratio=0.1,
        epochs=args.max_epochs,
        batch_size=args.batch_size,
        use_input_projection=False,
    )

    logger.info("Конфигурация: loss=%s, hidden=%d, layers=%d, heads=%d, dropout=%.2f, "
                "bidir=%s, lr=%.0e, wd=%.0e, no_input_proj=True",
                args.loss, er_config.hidden_dim, er_config.num_gnn_layers, er_config.num_heads,
                er_config.dropout, er_config.bidirectional, er_config.lr, er_config.weight_decay)

    epoch_callback = None if args.no_early_stopping else EarlyStopping(patience=args.patience)
    if args.no_early_stopping:
        logger.info("Early stopping отключён — будет %d эпох", args.max_epochs)

    if args.loss == "bce":
        model, history = train_entity_resolution_bce(
            graph=graph,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            config=er_config,
            device=device,
            save_path=save_path,
            epoch_callback=epoch_callback,
            model_class="gat",
        )
    else:
        model, history = train_entity_resolution_minibatch(
            graph=graph,
            train_pairs=train_pairs,
            val_pairs=val_pairs,
            config=er_config,
            device=device,
            save_path=save_path,
            epoch_callback=epoch_callback,
            model_class="gat",
        )

    # Сохраняем конфиг рядом — нужен для eval
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
            "model_tag": model_tag,
        }, f, indent=2)

    logger.info("Обучение завершено. Модель: %s, конфиг: %s", save_path, config_path)


if __name__ == "__main__":
    main()
