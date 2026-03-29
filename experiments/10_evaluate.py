"""Эксперимент 10 — Оценка GAT модели.

Production-like evaluation:
  1. Val pairs → подбор оптимального порога θ (max F1)
  2. Test pairs → F1, Precision, Recall @ θ  +  ROC-AUC, AP
  3. Cross-domain → те же метрики с тем же порогом (generalization)

Поддерживает оба типа моделей:
  - NT-Xent: EntityResolutionGAT (--model-path output/v3_gat_model.pt)
  - BCE: PairClassifier с backbone (--model-path output/v3_gat_bce_model.pt --bce)

Использование:
    python -m experiments.10_evaluate
    python -m experiments.10_evaluate --bce
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import torch

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.evaluation.clustering import (
    evaluate_pairs_at_threshold,
    evaluate_pairs_auc,
    find_best_threshold,
)
from table_unifier.models.entity_resolution import EntityResolutionGAT, PairClassifier
from table_unifier.training.er_trainer import get_row_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def load_gat_model(graph, config: EntityResolutionConfig, model_path: Path, device: str, bce: bool = False):
    """Загрузить обученную GAT модель (NT-Xent или BCE)."""
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    backbone = EntityResolutionGAT(
        row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
        hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
        num_gnn_layers=config.num_gnn_layers, num_heads=config.num_heads,
        dropout=config.dropout, attention_dropout=config.attention_dropout,
        bidirectional=config.bidirectional,
    )

    if bce:
        model = PairClassifier(backbone, embedding_dim=config.output_dim)
        state = torch.load(model_path, map_location=device, weights_only=True)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model.backbone
    else:
        state = torch.load(model_path, map_location=device, weights_only=True)
        backbone.load_state_dict(state)
        backbone.to(device)
        backbone.eval()
        return backbone


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка GAT модели")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--bce", action="store_true",
                        help="Загрузить BCE модель (PairClassifier)")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    if args.model_path:
        model_path = args.model_path
    elif args.bce:
        model_path = config.output_dir / "v3_gat_bce_model.pt"
    else:
        model_path = config.output_dir / "v3_gat_model.pt"

    # Загрузка конфигурации: HPO если есть, иначе defaults
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
            num_heads=4, attention_dropout=0.1,
            temperature=0.1,
        )
        logger.info("Конфигурация из HPO")
    else:
        er_config = EntityResolutionConfig(
            hidden_dim=128, edge_dim=128, num_gnn_layers=2,
            dropout=0.3, bidirectional=True,
            num_heads=4, attention_dropout=0.1,
            temperature=0.1,
        )
        logger.info("Конфигурация по умолчанию")

    results = {}

    # ===== 1. In-domain: val → threshold, test → metrics =====
    logger.info("=" * 60)
    logger.info("1. In-domain evaluation")
    logger.info("=" * 60)

    unified_dir = config.data_dir / "graphs" / "v3_unified"
    graph = torch.load(unified_dir / "graph.pt", weights_only=False)
    val_pairs = torch.load(unified_dir / "val_pairs.pt", weights_only=False)
    test_pairs = torch.load(unified_dir / "test_pairs.pt", weights_only=False)

    model = load_gat_model(graph, er_config, model_path, device, bce=args.bce)
    embeddings = get_row_embeddings(model, graph, device="cpu")

    # Подбор порога на val
    best_threshold, val_f1 = find_best_threshold(embeddings, val_pairs)
    logger.info("Optimal threshold: %.4f (val F1=%.4f)", best_threshold, val_f1)

    # Val metrics (для справки)
    val_metrics = evaluate_pairs_at_threshold(embeddings, val_pairs, best_threshold)
    val_auc = evaluate_pairs_auc(embeddings, val_pairs)
    val_metrics.update(val_auc)
    results["val"] = val_metrics
    logger.info("Val: %s", val_metrics)

    # Test metrics
    test_metrics = evaluate_pairs_at_threshold(embeddings, test_pairs, best_threshold)
    test_auc = evaluate_pairs_auc(embeddings, test_pairs)
    test_metrics.update(test_auc)
    results["test"] = test_metrics
    logger.info("Test: %s", test_metrics)

    results["threshold"] = best_threshold

    del graph, embeddings
    torch.cuda.empty_cache()

    # ===== 2. Cross-domain =====
    logger.info("=" * 60)
    logger.info("2. Cross-domain evaluation (threshold=%.4f)", best_threshold)
    logger.info("=" * 60)

    cross_dir = config.data_dir / "graphs" / "v3_cross"
    cross_results = []

    for ds_dir in sorted(cross_dir.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "graph.pt").exists():
            continue

        name = ds_dir.name
        logger.info("Cross-domain: %s", name)

        cg = torch.load(ds_dir / "graph.pt", weights_only=False)
        model_cd = load_gat_model(cg, er_config, model_path, device, bce=args.bce)
        cd_emb = get_row_embeddings(model_cd, cg, device="cpu")

        cd_metrics = {"name": name}

        lp_path = ds_dir / "labeled_pairs.pt"
        if lp_path.exists():
            lp = torch.load(lp_path, weights_only=False)
            pair_metrics = evaluate_pairs_at_threshold(cd_emb, lp, best_threshold)
            auc_metrics = evaluate_pairs_auc(cd_emb, lp)
            pair_metrics.update(auc_metrics)
            cd_metrics.update(pair_metrics)

        cross_results.append(cd_metrics)
        logger.info("  %s: %s", name, cd_metrics)

        del cg, cd_emb
        torch.cuda.empty_cache()

    results["cross_domain"] = cross_results

    # ===== 3. Сохранение =====
    suffix = "_bce" if args.bce else ""
    out_path = config.output_dir / f"v3_evaluation_results{suffix}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Результаты сохранены: %s", out_path)

    # Сводка
    model_type = "BCE" if args.bce else "NT-Xent"
    print(f"\n{'='*60}")
    print(f"РЕЗУЛЬТАТЫ v3 ({model_type})  |  threshold = {best_threshold:.4f}")
    print(f"{'='*60}")

    print(f"\nVal (threshold selection):")
    print(f"  F1:        {val_metrics.get('f1', 0):.4f}")
    print(f"  Precision: {val_metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {val_metrics.get('recall', 0):.4f}")

    print(f"\nTest (held-out):")
    print(f"  F1:        {test_metrics.get('f1', 0):.4f}")
    print(f"  Precision: {test_metrics.get('precision', 0):.4f}")
    print(f"  Recall:    {test_metrics.get('recall', 0):.4f}")
    print(f"  ROC-AUC:   {test_metrics.get('roc_auc', 0):.4f}")
    print(f"  AP:        {test_metrics.get('avg_precision', 0):.4f}")

    if cross_results:
        print(f"\nCross-domain (same threshold):")
        for r in cross_results:
            f1 = r.get("f1", "N/A")
            f1s = f"{f1:.4f}" if isinstance(f1, float) else f1
            auc = r.get("roc_auc", "N/A")
            aucs = f"{auc:.4f}" if isinstance(auc, float) else auc
            print(f"  {r['name']:15s}  F1={f1s}  AUC={aucs}")

    print("=" * 60)


if __name__ == "__main__":
    main()
