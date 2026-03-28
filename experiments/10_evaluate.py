"""Эксперимент 10 — Оценка GAT модели.

Три уровня оценки:
  1. In-domain test set — ROC-AUC, AP на held-out строках
  2. Cross-domain — ROC-AUC, AP + HDBSCAN ARI/NMI на unseen датасетах
  3. Real data — HDBSCAN кластеризация (unsupervised)

Использование:
    python -m experiments.10_evaluate
    python -m experiments.10_evaluate --model-path output/v3_gat_model.pt
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from table_unifier.config import Config, EntityResolutionConfig
from table_unifier.evaluation.clustering import cluster_embeddings, evaluate_clusters
from table_unifier.models.entity_resolution import EntityResolutionGAT
from table_unifier.training.er_trainer import get_row_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def evaluate_pairs(embeddings: torch.Tensor, pairs: torch.Tensor) -> dict:
    """ROC-AUC и AP из labeled pairs."""
    if len(pairs) == 0:
        return {}
    idx_a = pairs[:, 0]
    idx_b = pairs[:, 1]
    labels = pairs[:, 2].numpy()
    scores = (embeddings[idx_a] * embeddings[idx_b]).sum(dim=1).numpy()
    if len(np.unique(labels)) < 2:
        return {}
    return {
        "roc_auc": float(roc_auc_score(labels, scores)),
        "avg_precision": float(average_precision_score(labels, scores)),
    }


def load_gat_model(graph, config: EntityResolutionConfig, model_path: Path, device: str):
    """Загрузить обученную GAT модель."""
    config.row_dim = int(graph["row"].x.shape[1])
    config.token_dim = int(graph["token"].x.shape[1])
    config.col_dim = int(graph.col_embeddings.shape[1])

    model = EntityResolutionGAT(
        row_dim=config.row_dim, token_dim=config.token_dim, col_dim=config.col_dim,
        hidden_dim=config.hidden_dim, edge_dim=config.edge_dim, output_dim=config.output_dim,
        num_gnn_layers=config.num_gnn_layers, num_heads=config.num_heads,
        dropout=config.dropout, attention_dropout=config.attention_dropout,
        bidirectional=config.bidirectional,
    )
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description="Оценка GAT модели")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--device", default=None)
    parser.add_argument("--model-path", default=None, type=Path)
    parser.add_argument("--min-cluster-size", type=int, default=10)
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_path = args.model_path or (config.output_dir / "v3_gat_model.pt")

    er_config = EntityResolutionConfig(
        hidden_dim=128, edge_dim=128, num_gnn_layers=3,
        dropout=0.0, bidirectional=True,
        num_heads=4, attention_dropout=0.1,
    )

    results = {}

    # ===== 1. In-domain test =====
    logger.info("=" * 60)
    logger.info("1. In-domain test evaluation")
    logger.info("=" * 60)

    unified_dir = config.data_dir / "graphs" / "v3_unified"
    graph = torch.load(unified_dir / "graph.pt", weights_only=False)
    test_pairs = torch.load(unified_dir / "test_pairs.pt", weights_only=False)

    model = load_gat_model(graph, er_config, model_path, device)

    # Для большого графа: получаем эмбеддинги на CPU
    embeddings = get_row_embeddings(model, graph, device="cpu")

    in_domain_metrics = evaluate_pairs(embeddings, test_pairs)
    logger.info("In-domain: %s", in_domain_metrics)
    results["in_domain"] = in_domain_metrics

    # HDBSCAN на test строках
    test_rows = torch.unique(torch.cat([test_pairs[:, 0], test_pairs[:, 1]]))
    test_emb = embeddings[test_rows]
    test_labels = cluster_embeddings(test_emb, min_cluster_size=args.min_cluster_size)

    # Ground truth для кластеризации: строки из одной positive пары → один кластер
    gt = np.full(len(test_rows), -1)
    row_to_local = {r.item(): i for i, r in enumerate(test_rows)}
    cluster_id = 0
    for pair in test_pairs:
        if pair[2].item() == 1:
            a_local = row_to_local.get(pair[0].item())
            b_local = row_to_local.get(pair[1].item())
            if a_local is not None and b_local is not None:
                gt[a_local] = cluster_id
                gt[b_local] = cluster_id
                cluster_id += 1

    cluster_metrics = evaluate_clusters(test_labels, gt)
    results["in_domain_clustering"] = cluster_metrics
    logger.info("In-domain clustering: %s", cluster_metrics)

    del graph, embeddings
    torch.cuda.empty_cache()

    # ===== 2. Cross-domain =====
    logger.info("=" * 60)
    logger.info("2. Cross-domain evaluation")
    logger.info("=" * 60)

    cross_dir = config.data_dir / "graphs" / "v3_cross"
    cross_results = []

    for ds_dir in sorted(cross_dir.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "graph.pt").exists():
            continue

        name = ds_dir.name
        logger.info("Cross-domain: %s", name)

        cg = torch.load(ds_dir / "graph.pt", weights_only=False)
        model_cd = load_gat_model(cg, er_config, model_path, device)
        cd_emb = get_row_embeddings(model_cd, cg, device="cpu")

        cd_metrics = {"name": name}

        # Pair-level metrics
        lp_path = ds_dir / "labeled_pairs.pt"
        if lp_path.exists():
            lp = torch.load(lp_path, weights_only=False)
            pair_metrics = evaluate_pairs(cd_emb, lp)
            cd_metrics.update(pair_metrics)

        # HDBSCAN
        cd_labels = cluster_embeddings(cd_emb, min_cluster_size=args.min_cluster_size)
        cd_cluster = evaluate_clusters(cd_labels)
        cd_metrics["clustering"] = cd_cluster

        cross_results.append(cd_metrics)
        logger.info("  %s: %s", name, cd_metrics)

        del cg, cd_emb
        torch.cuda.empty_cache()

    results["cross_domain"] = cross_results

    # ===== 3. Сохранение =====
    out_path = config.output_dir / "v3_evaluation_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    logger.info("Результаты сохранены: %s", out_path)

    # Сводка
    print("\n" + "=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ v3")
    print("=" * 60)
    if in_domain_metrics:
        print(f"\nIn-domain test:")
        print(f"  ROC-AUC:       {in_domain_metrics.get('roc_auc', 0):.4f}")
        print(f"  Avg Precision: {in_domain_metrics.get('avg_precision', 0):.4f}")
    if cluster_metrics:
        print(f"  HDBSCAN ARI:   {cluster_metrics.get('ari', 'N/A')}")
        print(f"  HDBSCAN NMI:   {cluster_metrics.get('nmi', 'N/A')}")
        print(f"  Coverage:      {cluster_metrics.get('coverage', 0):.1%}")

    if cross_results:
        print(f"\nCross-domain:")
        for r in cross_results:
            auc = r.get("roc_auc", "N/A")
            auc_str = f"{auc:.4f}" if isinstance(auc, float) else auc
            print(f"  {r['name']:20s} AUC={auc_str}")

    print("=" * 60)


if __name__ == "__main__":
    main()
