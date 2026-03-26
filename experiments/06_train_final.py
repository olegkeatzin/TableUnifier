"""Эксперимент 6 — Обучение финальной модели ER с лучшими гиперпараметрами.

Берёт лучшие параметры из HPO (output/hpo_architecture.json + hpo_training.json)
и обучает модель без ограничения на эпохи — до сходимости (early stopping).

Сохраняет:
  - output/er_model_hpo_best.pt — веса лучшей модели (по val_loss)
  - output/er_final_history.json — кривые обучения
  - output/er_final_metrics.json — финальные метрики (in/cross domain)

Использование:
    # Обучение до сходимости (patience=20, max 500 эпох)
    python -m experiments.06_train_final

    # Задать свои лимиты
    python -m experiments.06_train_final --max-epochs 1000 --patience 30

    # Указать конкретные параметры вместо HPO
    python -m experiments.06_train_final --config '{"hidden_dim":128,"edge_dim":128,...}'
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import numpy as np
import torch

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.dataset.download import HOLDOUT_DATASETS
from table_unifier.training.er_trainer import (
    get_row_embeddings,
    train_entity_resolution_multidataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Загрузка данных (из 05_hpo.py)
# ------------------------------------------------------------------ #

def load_all_datasets(
    graphs_dir: Path,
    holdout_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """Загрузить train и holdout графы."""
    holdout_set = set(holdout_names)
    available = sorted([
        d.name for d in graphs_dir.iterdir()
        if d.is_dir() and d.name != "unified" and (d / "graph.pt").exists()
    ])

    train_datasets: list[dict] = []
    holdout_datasets: list[dict] = []

    for name in available:
        ds_dir = graphs_dir / name
        try:
            graph = torch.load(ds_dir / "graph.pt", weights_only=False)
            train_tri = torch.load(ds_dir / "train_triplets.pt", weights_only=False)
            val_tri = torch.load(ds_dir / "val_triplets.pt", weights_only=False)
            with open(ds_dir / "stats.json") as f:
                stats = json.load(f)

            test_pairs_path = ds_dir / "test_pairs.pt"
            test_pairs = (
                torch.load(test_pairs_path, weights_only=False)
                if test_pairs_path.exists()
                else torch.zeros((0, 3), dtype=torch.long)
            )

            ds = {
                "name": name,
                "domain": stats.get("domain", ""),
                "graph": graph,
                "train_triplets": train_tri,
                "val_triplets": val_tri if len(val_tri) > 0 else None,
                "test_pairs": test_pairs if len(test_pairs) > 0 else None,
            }
            if name in holdout_set:
                holdout_datasets.append(ds)
            else:
                train_datasets.append(ds)
        except Exception:
            logger.exception("Ошибка загрузки %s", name)

    logger.info("Загружено: %d train, %d holdout", len(train_datasets), len(holdout_datasets))
    return train_datasets, holdout_datasets


# ------------------------------------------------------------------ #
#  Оценка
# ------------------------------------------------------------------ #

def evaluate_from_pairs(
    embeddings: torch.Tensor,
    test_pairs: torch.Tensor,
) -> dict:
    """ROC-AUC и AP из тестовых пар."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    if len(test_pairs) == 0:
        return {}
    idx_a = test_pairs[:, 0]
    idx_b = test_pairs[:, 1]
    labels = test_pairs[:, 2].numpy()
    scores = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1).numpy()
    if len(np.unique(labels)) < 2:
        return {}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "avg_precision": float(average_precision_score(labels, scores)),
    }


def evaluate_model_on_datasets(
    model, datasets: list[dict], device: str,
) -> list[dict]:
    """Оценить модель на списке датасетов."""
    results = []
    for ds in datasets:
        if ds.get("test_pairs") is None:
            continue
        embeddings = get_row_embeddings(model, ds["graph"], device=device)
        metrics = evaluate_from_pairs(embeddings, ds["test_pairs"])
        if metrics:
            metrics["name"] = ds["name"]
            results.append(metrics)
    return results


# ------------------------------------------------------------------ #
#  Загрузка лучших параметров из HPO
# ------------------------------------------------------------------ #

def load_best_config(output_dir: Path) -> EntityResolutionConfig:
    """Собрать EntityResolutionConfig из результатов HPO."""
    arch_path = output_dir / "hpo_architecture.json"
    train_path = output_dir / "hpo_training.json"

    if not arch_path.exists():
        raise FileNotFoundError(f"Нет результатов архитектуры: {arch_path}")
    if not train_path.exists():
        raise FileNotFoundError(f"Нет результатов обучения: {train_path}")

    with open(arch_path) as f:
        best_arch = json.load(f)["best_params"]
    with open(train_path) as f:
        best_training = json.load(f)["best_training"]

    logger.info("Лучшая архитектура: %s", best_arch)
    logger.info("Лучшие параметры обучения: %s", best_training)

    return EntityResolutionConfig(
        hidden_dim=best_arch["hidden_dim"],
        edge_dim=best_arch["edge_dim"],
        num_gnn_layers=best_arch["num_gnn_layers"],
        dropout=best_arch["dropout"],
        bidirectional=best_arch["bidirectional"],
        lr=best_training["lr"],
        margin=best_training["margin"],
        weight_decay=best_training["weight_decay"],
    )


# ------------------------------------------------------------------ #
#  Early stopping callback
# ------------------------------------------------------------------ #

class EarlyStopping:
    """Останавливает обучение, если val_loss не улучшается patience эпох."""

    def __init__(self, patience: int = 20):
        self.patience = patience
        self.best_val = float("inf")
        self.epochs_no_improve = 0

    def __call__(self, epoch: int, val_loss: float | None) -> None:
        if val_loss is None:
            return
        if val_loss < self.best_val:
            self.best_val = val_loss
            self.epochs_no_improve = 0
        else:
            self.epochs_no_improve += 1

        if self.epochs_no_improve >= self.patience:
            logger.info(
                "Early stopping: val_loss не улучшался %d эпох (best=%.4f, current=%.4f)",
                self.patience, self.best_val, val_loss,
            )
            raise StopIteration("Early stopping")


# ------------------------------------------------------------------ #
#  Точка входа
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение финальной модели ER")
    parser.add_argument("--max-epochs", type=int, default=500,
                        help="Максимум эпох (по умолчанию 500)")
    parser.add_argument("--patience", type=int, default=20,
                        help="Early stopping patience (по умолчанию 20)")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--config", default=None, type=str,
                        help="JSON-строка с параметрами (вместо HPO)")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    graphs_dir = config.data_dir / "graphs"

    save_path = config.output_dir / "er_model_hpo_best.pt"

    # 1. Конфигурация
    if args.config:
        params = json.loads(args.config)
        er_config = EntityResolutionConfig(**params)
        logger.info("Конфигурация из CLI: %s", params)
    else:
        er_config = load_best_config(config.output_dir)

    er_config.epochs = args.max_epochs

    # 2. Данные
    logger.info("Загрузка графов...")
    train_datasets, holdout_datasets = load_all_datasets(graphs_dir, HOLDOUT_DATASETS)

    if not train_datasets:
        logger.error("Нет train графов в %s", graphs_dir)
        return

    # 3. Обучение с early stopping
    logger.info("=" * 60)
    logger.info("Обучение финальной модели (max_epochs=%d, patience=%d)", args.max_epochs, args.patience)
    logger.info("=" * 60)

    early_stop = EarlyStopping(patience=args.patience)

    try:
        model, history = train_entity_resolution_multidataset(
            datasets=train_datasets,
            config=er_config,
            device=device,
            save_path=save_path,
            epoch_callback=early_stop,
        )
    except StopIteration:
        # Early stopping сработал — модель уже сохранена по лучшему val_loss
        logger.info("Обучение остановлено early stopping")
        # Загрузим лучшую модель
        from table_unifier.models.entity_resolution import EntityResolutionGNN

        sample = train_datasets[0]["graph"]
        er_config.row_dim = int(sample["row"].x.shape[1])
        er_config.token_dim = int(sample["token"].x.shape[1])
        er_config.col_dim = int(sample.col_embeddings.shape[1])

        model = EntityResolutionGNN(
            row_dim=er_config.row_dim,
            token_dim=er_config.token_dim,
            col_dim=er_config.col_dim,
            hidden_dim=er_config.hidden_dim,
            edge_dim=er_config.edge_dim,
            output_dim=er_config.output_dim,
            num_gnn_layers=er_config.num_gnn_layers,
            dropout=er_config.dropout,
            bidirectional=er_config.bidirectional,
        )
        model.load_state_dict(torch.load(save_path, map_location=device, weights_only=False))
        model.to(device)
        model.eval()
        # history недоступен после исключения — считаем из early_stop
        history = {"note": "early_stopped", "best_val_loss": early_stop.best_val}

    # 4. Сохранение истории
    history_path = config.output_dir / "er_final_history.json"
    serializable_history = {
        k: [float(v) for v in vals] if isinstance(vals, list) else vals
        for k, vals in history.items()
    }
    with open(history_path, "w") as f:
        json.dump(serializable_history, f, indent=2)
    logger.info("История обучения: %s", history_path)

    # 5. Оценка
    logger.info("=" * 60)
    logger.info("Оценка финальной модели")
    logger.info("=" * 60)

    in_results = evaluate_model_on_datasets(model, train_datasets, device)
    cross_results = evaluate_model_on_datasets(model, holdout_datasets, device)

    in_auc = np.mean([r["roc_auc"] for r in in_results]) if in_results else 0.0
    in_ap = np.mean([r["avg_precision"] for r in in_results]) if in_results else 0.0
    cross_auc = np.mean([r["roc_auc"] for r in cross_results]) if cross_results else 0.0
    cross_ap = np.mean([r["avg_precision"] for r in cross_results]) if cross_results else 0.0

    metrics = {
        "in_domain": {"roc_auc": float(in_auc), "avg_precision": float(in_ap), "per_dataset": in_results},
        "cross_domain": {"roc_auc": float(cross_auc), "avg_precision": float(cross_ap), "per_dataset": cross_results},
        "config": {
            "hidden_dim": er_config.hidden_dim,
            "edge_dim": er_config.edge_dim,
            "num_gnn_layers": er_config.num_gnn_layers,
            "dropout": er_config.dropout,
            "bidirectional": er_config.bidirectional,
            "lr": er_config.lr,
            "margin": er_config.margin,
            "weight_decay": er_config.weight_decay,
            "max_epochs": args.max_epochs,
            "patience": args.patience,
        },
    }

    metrics_path = config.output_dir / "er_final_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    # Сводка
    print("\n" + "=" * 60)
    print("ФИНАЛЬНАЯ МОДЕЛЬ")
    print("=" * 60)
    print(f"Модель:           {save_path}")
    print(f"In-domain AUC:    {in_auc:.4f}")
    print(f"In-domain AP:     {in_ap:.4f}")
    print(f"Cross-domain AUC: {cross_auc:.4f}")
    print(f"Cross-domain AP:  {cross_ap:.4f}")

    if in_results:
        print("\nIn-domain (per dataset):")
        for r in sorted(in_results, key=lambda x: -x["roc_auc"]):
            print(f"  {r['name']:30s} AUC={r['roc_auc']:.3f}  AP={r['avg_precision']:.3f}")

    if cross_results:
        print("\nCross-domain (per dataset):")
        for r in sorted(cross_results, key=lambda x: -x["roc_auc"]):
            print(f"  {r['name']:30s} AUC={r['roc_auc']:.3f}  AP={r['avg_precision']:.3f}")

    print("=" * 60)

    del model
    torch.cuda.empty_cache()
    gc.collect()


if __name__ == "__main__":
    main()
