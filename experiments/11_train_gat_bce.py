"""Эксперимент 11 — Обучение GAT с BCE loss (классификация пар).

Ablation study: BCE vs NT-Xent vs Triplet Loss.
Тот же backbone (EntityResolutionGAT), но вместо метрического обучения
используется явный classification head: MLP(|emb_a - emb_b|) → Sigmoid → BCE.
Подход из Ditto (VLDB 2021).

Оценка: backbone эмбеддинги сравниваются через cosine similarity (как в exp 10)
для честного сравнения с NT-Xent.

Использование:
    python -m experiments.11_train_gat_bce
    python -m experiments.11_train_gat_bce --max-epochs 500 --patience 30
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.training.er_trainer import train_entity_resolution_bce

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
    parser = argparse.ArgumentParser(description="Обучение GAT с BCE loss")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    unified_dir = config.data_dir / "graphs" / "v3_unified"
    save_path = config.output_dir / "v3_gat_bce_model.pt"

    # Загрузка
    logger.info("Загрузка unified графа...")
    graph = torch.load(unified_dir / "graph.pt", weights_only=False)
    train_pairs = torch.load(unified_dir / "train_pairs.pt", weights_only=False)
    val_pairs = torch.load(unified_dir / "val_pairs.pt", weights_only=False)

    logger.info("Graph: %d rows, %d tokens, %d edges",
                graph["row"].x.shape[0], graph["token"].x.shape[0],
                graph["token", "in_row", "row"].edge_index.shape[1])
    logger.info("Train: %d pairs, Val: %d pairs", len(train_pairs), len(val_pairs))

    # Тот же конфиг что в exp 09 для честного ablation
    hpo_arch = config.output_dir / "hpo_architecture.json"
    hpo_train = config.output_dir / "hpo_training.json"

    if hpo_arch.exists() and hpo_train.exists():
        with open(hpo_arch) as f:
            best_arch = json.load(f)["best_params"]
        with open(hpo_train) as f:
            best_training = json.load(f)["best_training"]
        er_config = EntityResolutionConfig(
            hidden_dim=best_arch["hidden_dim"],
            edge_dim=best_arch["edge_dim"],
            num_gnn_layers=best_arch["num_gnn_layers"],
            dropout=best_arch["dropout"],
            bidirectional=best_arch["bidirectional"],
            lr=best_training["lr"],
            weight_decay=best_training["weight_decay"],
            num_heads=4, attention_dropout=0.1,
            warmup_ratio=0.1,
        )
        logger.info("Конфигурация из HPO (BCE)")
    else:
        er_config = EntityResolutionConfig(
            hidden_dim=128, edge_dim=128, num_gnn_layers=2,
            dropout=0.3, bidirectional=True,
            lr=5e-4, weight_decay=5e-4,
            num_heads=4, attention_dropout=0.1,
            warmup_ratio=0.1,
        )
        logger.info("Конфигурация по умолчанию (BCE)")

    er_config.epochs = args.max_epochs
    er_config.batch_size = args.batch_size

    early_stop = EarlyStopping(patience=args.patience)

    model, history = train_entity_resolution_bce(
        graph=graph,
        train_pairs=train_pairs,
        val_pairs=val_pairs,
        config=er_config,
        device=device,
        save_path=save_path,
        epoch_callback=early_stop,
        model_class="gat",
    )

    logger.info("Обучение завершено. Модель: %s", save_path)


if __name__ == "__main__":
    main()
