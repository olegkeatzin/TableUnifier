"""Эксперимент 5 — Подбор гиперпараметров (Optuna + MLflow).

Два этапа:
  1. Архитектура: hidden_dim, edge_dim, num_gnn_layers,
     bidirectional, dropout — при фиксированном lr/margin/epochs.
  2. Обучение: lr, weight_decay, margin — при лучшей архитектуре из этапа 1.

Optuna (TPE) подбирает параметры, MLflow логирует результаты.
MedianPruner убивает слабые trial-ы на ранних эпохах.

Использование:
    # Этап 1 — архитектура
    python -m experiments.05_hpo --stage architecture --n-trials 40

    # Этап 2 — обучение (подхватывает лучшую архитектуру из этапа 1)
    python -m experiments.05_hpo --stage training --n-trials 30

    # Оба этапа последовательно
    python -m experiments.05_hpo --stage both --n-trials 40
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
from pathlib import Path

import mlflow
import numpy as np
import optuna
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.dataset.download import HOLDOUT_DATASETS
from table_unifier.training.er_trainer import (
    get_row_embeddings,
    train_entity_resolution_multidataset,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

# Уменьшим шум от библиотек
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("optuna").setLevel(logging.WARNING)


# ------------------------------------------------------------------ #
#  Загрузка данных (один раз)
# ------------------------------------------------------------------ #

def load_all_datasets(
    graphs_dir: Path,
    holdout_names: list[str],
) -> tuple[list[dict], list[dict]]:
    """Загрузить train и holdout графы в RAM (на CPU)."""
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

            # test_pairs: [N, 3] — (global_a, global_b, label)
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
#  Оценка (threshold-independent)
# ------------------------------------------------------------------ #

def evaluate_from_pairs(
    embeddings: torch.Tensor,
    test_pairs: torch.Tensor,
) -> dict:
    """ROC-AUC и Average Precision из предразрешённых пар.

    Args:
        embeddings: [N_rows, D] — L2-нормализованные эмбеддинги.
        test_pairs: [M, 3] — (global_a, global_b, label).
    """
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
    """Оценить модель на списке датасетов (используя test_pairs тензоры)."""
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
#  Objective функции для Optuna
# ------------------------------------------------------------------ #

def make_architecture_objective(
    train_datasets: list[dict],
    holdout_datasets: list[dict],
    device: str,
    hpo_epochs: int,
):
    """Создать objective для подбора архитектуры."""

    def objective(trial: optuna.Trial) -> float:
        # ---- Сэмплирование архитектурных параметров ---- #
        hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
        edge_dim = trial.suggest_categorical("edge_dim", [32, 64, 128])
        num_gnn_layers = trial.suggest_int("num_gnn_layers", 1, 3)
        dropout = trial.suggest_float("dropout", 0.0, 0.5, step=0.05)
        bidirectional = trial.suggest_categorical("bidirectional", [False, True])

        # ---- Конфигурация ---- #
        cfg = EntityResolutionConfig(
            hidden_dim=hidden_dim,
            edge_dim=edge_dim,
            num_gnn_layers=num_gnn_layers,
            dropout=dropout,
            bidirectional=bidirectional,
            # Фиксированные параметры обучения
            lr=1e-3,
            margin=0.3,
            weight_decay=0.0,
            epochs=hpo_epochs,
        )

        # ---- Optuna pruning callback ---- #
        def epoch_callback(epoch: int, val_loss: float | None) -> None:
            if val_loss is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        # ---- Обучение ---- #
        logger.info(
            "--- Trial %d | hidden=%d edge=%d layers=%d dropout=%.2f bidir=%s ---",
            trial.number, hidden_dim, edge_dim, num_gnn_layers, dropout, bidirectional,
        )
        with mlflow.start_run(nested=True, run_name=f"arch_trial_{trial.number}"):
            mlflow.log_params({
                "hidden_dim": hidden_dim,
                "edge_dim": edge_dim, "num_gnn_layers": num_gnn_layers,
                "dropout": dropout, "bidirectional": bidirectional,
            })

            try:
                model, history = train_entity_resolution_multidataset(
                    datasets=train_datasets,
                    config=cfg,
                    device=device,
                    epoch_callback=epoch_callback,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("Trial %d: CUDA OOM — pruned", trial.number)
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

            # ---- Оценка ---- #
            in_results = evaluate_model_on_datasets(
                model, train_datasets, device,
            )
            cross_results = evaluate_model_on_datasets(
                model, holdout_datasets, device,
            )

            in_auc = np.mean([r["roc_auc"] for r in in_results if "roc_auc" in r]) or 0.0
            in_ap = np.mean([r["avg_precision"] for r in in_results if "avg_precision" in r]) or 0.0
            cross_auc = np.mean([r["roc_auc"] for r in cross_results if "roc_auc" in r]) or 0.0
            cross_ap = np.mean([r["avg_precision"] for r in cross_results if "avg_precision" in r]) or 0.0
            best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")

            mlflow.log_metrics({
                "best_val_loss": best_val,
                "in_domain_roc_auc": in_auc, "in_domain_avg_precision": in_ap,
                "cross_domain_roc_auc": cross_auc, "cross_domain_avg_precision": cross_ap,
            })

            logger.info(
                "Trial %d: val=%.4f | in AUC=%.3f AP=%.3f | cross AUC=%.3f AP=%.3f",
                trial.number, best_val, in_auc, in_ap, cross_auc, cross_ap,
            )

            # Очистка GPU
            del model
            torch.cuda.empty_cache()
            gc.collect()

            return best_val

    return objective


def make_training_objective(
    train_datasets: list[dict],
    holdout_datasets: list[dict],
    device: str,
    best_arch: dict,
    hpo_epochs: int,
):
    """Создать objective для подбора параметров обучения."""

    def objective(trial: optuna.Trial) -> float:
        # ---- Сэмплирование параметров обучения ---- #
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
        weight_decay = trial.suggest_float("weight_decay", 0.0, 1e-3, step=1e-4)
        margin = trial.suggest_float("margin", 0.1, 2.0, step=0.1)

        # ---- Конфигурация (архитектура фиксирована) ---- #
        cfg = EntityResolutionConfig(
            hidden_dim=best_arch["hidden_dim"],
            edge_dim=best_arch["edge_dim"],
            num_gnn_layers=best_arch["num_gnn_layers"],
            dropout=best_arch["dropout"],
            bidirectional=best_arch["bidirectional"],
            lr=lr,
            margin=margin,
            weight_decay=weight_decay,
            epochs=hpo_epochs,
        )

        def epoch_callback(epoch: int, val_loss: float | None) -> None:
            if val_loss is not None:
                trial.report(val_loss, epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

        logger.info(
            "--- Trial %d | lr=%.1e wd=%.1e margin=%.1f ---",
            trial.number, lr, weight_decay, margin,
        )
        with mlflow.start_run(nested=True, run_name=f"train_trial_{trial.number}"):
            mlflow.log_params({"lr": lr, "weight_decay": weight_decay, "margin": margin})
            mlflow.log_params({k: v for k, v in best_arch.items()})

            try:
                model, history = train_entity_resolution_multidataset(
                    datasets=train_datasets,
                    config=cfg,
                    device=device,
                    epoch_callback=epoch_callback,
                )
            except torch.cuda.OutOfMemoryError:
                logger.warning("Trial %d: CUDA OOM — pruned", trial.number)
                torch.cuda.empty_cache()
                gc.collect()
                raise optuna.TrialPruned()

            in_results = evaluate_model_on_datasets(
                model, train_datasets, device,
            )
            cross_results = evaluate_model_on_datasets(
                model, holdout_datasets, device,
            )

            in_auc = np.mean([r["roc_auc"] for r in in_results]) if in_results else 0.0
            cross_auc = np.mean([r["roc_auc"] for r in cross_results]) if cross_results else 0.0
            best_val = min(history["val_loss"]) if history["val_loss"] else float("inf")

            mlflow.log_metrics({
                "best_val_loss": best_val,
                "in_domain_roc_auc": in_auc,
                "cross_domain_roc_auc": cross_auc,
            })

            logger.info(
                "Trial %d: val=%.4f | in AUC=%.3f | cross AUC=%.3f",
                trial.number, best_val, in_auc, cross_auc,
            )

            del model
            torch.cuda.empty_cache()
            gc.collect()

            return best_val

    return objective


# ------------------------------------------------------------------ #
#  Запуск этапов HPO
# ------------------------------------------------------------------ #

def run_architecture_search(
    train_datasets: list[dict],
    holdout_datasets: list[dict],
    device: str,
    n_trials: int,
    hpo_epochs: int,
    output_dir: Path,
) -> dict:
    """Этап 1: подбор архитектуры."""
    study = optuna.create_study(
        study_name="er_architecture",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=5, n_warmup_steps=10,
        ),
    )

    with mlflow.start_run(run_name="architecture_search"):
        mlflow.log_params({"stage": "architecture", "n_trials": n_trials, "hpo_epochs": hpo_epochs})

        objective = make_architecture_objective(
            train_datasets, holdout_datasets, device, hpo_epochs,
        )
        study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_arch = best.params
    logger.info("=" * 60)
    logger.info("Лучшая архитектура (val_loss=%.4f):", best.value)
    for k, v in best_arch.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    # Сохраняем
    result = {"best_value": best.value, "best_params": best_arch}
    out_path = output_dir / "hpo_architecture.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Результат сохранён: %s", out_path)

    return best_arch


def run_training_search(
    train_datasets: list[dict],
    holdout_datasets: list[dict],
    device: str,
    best_arch: dict,
    n_trials: int,
    hpo_epochs: int,
    output_dir: Path,
) -> dict:
    """Этап 2: подбор параметров обучения."""
    study = optuna.create_study(
        study_name="er_training",
        direction="minimize",
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=3, n_warmup_steps=10,
        ),
    )

    with mlflow.start_run(run_name="training_search"):
        mlflow.log_params({"stage": "training", "n_trials": n_trials, "hpo_epochs": hpo_epochs})
        mlflow.log_params({f"arch_{k}": v for k, v in best_arch.items()})

        objective = make_training_objective(
            train_datasets, holdout_datasets, device, best_arch, hpo_epochs,
        )
        study.optimize(objective, n_trials=n_trials)

    best = study.best_trial
    best_training = best.params
    logger.info("=" * 60)
    logger.info("Лучшие параметры обучения (val_loss=%.4f):", best.value)
    for k, v in best_training.items():
        logger.info("  %s: %s", k, v)
    logger.info("=" * 60)

    result = {
        "best_value": best.value,
        "best_arch": best_arch,
        "best_training": best_training,
    }
    out_path = output_dir / "hpo_training.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    logger.info("Результат сохранён: %s", out_path)

    return best_training


# ------------------------------------------------------------------ #
#  Точка входа
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="HPO для Entity Resolution (Optuna + MLflow)")
    parser.add_argument("--stage", choices=["architecture", "training", "both"],
                        default="both", help="Какой этап запустить")
    parser.add_argument("--n-trials", type=int, default=40,
                        help="Число trial-ов Optuna на каждый этап")
    parser.add_argument("--hpo-epochs", type=int, default=50,
                        help="Эпох на один trial")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--arch-from", default=None,
                        help="Путь к hpo_architecture.json (для --stage training)")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    config.output_dir.mkdir(parents=True, exist_ok=True)
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    graphs_dir = config.data_dir / "graphs"

    mlflow.set_experiment("TableUnifier_ER_HPO")

    # Загрузка данных один раз
    logger.info("Загрузка графов...")
    train_datasets, holdout_datasets = load_all_datasets(graphs_dir, HOLDOUT_DATASETS)

    if not train_datasets:
        logger.error("Нет train графов в %s", graphs_dir)
        return

    # Этап 1: архитектура
    best_arch = None
    if args.stage in ("architecture", "both"):
        best_arch = run_architecture_search(
            train_datasets, holdout_datasets, device,
            n_trials=args.n_trials, hpo_epochs=args.hpo_epochs,
            output_dir=config.output_dir,
        )

    # Этап 2: обучение
    if args.stage in ("training", "both"):
        if best_arch is None:
            # Загружаем из файла
            arch_path = args.arch_from or (config.output_dir / "hpo_architecture.json")
            arch_path = Path(arch_path)
            if not arch_path.exists():
                logger.error("Нет результатов архитектуры: %s", arch_path)
                return
            with open(arch_path) as f:
                best_arch = json.load(f)["best_params"]
            logger.info("Загружена архитектура из %s", arch_path)

        run_training_search(
            train_datasets, holdout_datasets, device,
            best_arch=best_arch,
            n_trials=args.n_trials, hpo_epochs=args.hpo_epochs,
            output_dir=config.output_dir,
        )


if __name__ == "__main__":
    main()
