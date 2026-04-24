"""Эксперимент 18 — оценка ER (bge-m3) через HDBSCAN с ГА-подбором параметров.

Идея: отказываемся от threshold θ на косинусной близости (exp 10) и вместо этого
кластеризуем row-эмбеддинги HDBSCAN'ом. Параметры кластеризатора подбирает
генетический алгоритм на DEAP по F1 на val pairs.

Pipeline::

    1. Загрузить bge-m3 unified граф (v14_mrl) + обученную GAT-модель
       (output/bge-m3/v14_mrl_gat_model.pt).
    2. Посчитать L2-нормализованные row эмбеддинги.
    3. DEAP GA подбирает (min_cluster_size, min_samples, cluster_selection_epsilon,
       cluster_selection_method, metric) по F1 на val_pairs.
    4. С лучшими параметрами считаем F1 / P / R на val, test и cross-domain.
    5. Параллельно — baseline threshold (как exp 10) для сравнения.
    6. Сохраняем всё в output/bge-m3/v18_ga_hdbscan_results.json.

Запуск (на удалённом сервере с GPU)::

    uv sync                                    # подтянет свежий deap
    uv run python -m experiments.18_ga_hdbscan_bge_m3

Все CLI-флаги необязательны: по умолчанию — bge-m3, v14_mrl, ntxent-чекпоинт.
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
from table_unifier.evaluation.ga_hdbscan import (
    GAHDBSCANConfig,
    evaluate_params_on_pairs,
    run_ga_hdbscan,
)
from table_unifier.models.entity_resolution import EntityResolutionGAT
from table_unifier.paths import output_dir_for, unified_dir
from table_unifier.training.er_trainer import get_row_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("exp18")


def _load_config_json(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def _build_model(graph, cfg_dict: dict, device: str) -> EntityResolutionGAT:
    model = EntityResolutionGAT(
        row_dim=cfg_dict["row_dim"],
        token_dim=cfg_dict["token_dim"],
        col_dim=cfg_dict["col_dim"],
        hidden_dim=cfg_dict["hidden_dim"],
        edge_dim=cfg_dict["edge_dim"],
        output_dim=cfg_dict["output_dim"],
        num_gnn_layers=cfg_dict["num_gnn_layers"],
        num_heads=cfg_dict["num_heads"],
        dropout=cfg_dict["dropout"],
        attention_dropout=cfg_dict["attention_dropout"],
        bidirectional=cfg_dict["bidirectional"],
        use_input_projection=cfg_dict.get("use_input_projection", False),
    )
    model.to(device)
    model.eval()
    return model


def load_bge_model(model_path: Path, graph, device: str) -> EntityResolutionGAT:
    """Загрузить GAT для bge-m3. Конфиг лежит рядом: *.config.json."""
    cfg_path = model_path.with_suffix(".config.json")
    if cfg_path.exists():
        cfg_dict = _load_config_json(cfg_path)
    else:
        col_dim = int(graph.col_embeddings.shape[1])
        row_dim = int(graph["row"].x.shape[1])
        token_dim = int(graph["token"].x.shape[1])
        cfg_dict = {
            "row_dim": row_dim, "token_dim": token_dim, "col_dim": col_dim,
            "hidden_dim": col_dim, "edge_dim": col_dim, "output_dim": col_dim,
            "num_gnn_layers": 2, "num_heads": 4, "dropout": 0.3,
            "attention_dropout": 0.1, "bidirectional": True,
            "use_input_projection": False,
        }
        logger.warning("config.json не найден рядом с чекпоинтом — использую defaults")

    model = _build_model(graph, cfg_dict, device)
    state = torch.load(model_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model


def _threshold_metrics(embeddings, val_pairs, test_pairs):
    """Baseline: single threshold sweep на val, метрики на val + test."""
    thr, val_f1 = find_best_threshold(embeddings, val_pairs)
    val = evaluate_pairs_at_threshold(embeddings, val_pairs, thr)
    val.update(evaluate_pairs_auc(embeddings, val_pairs))
    test = evaluate_pairs_at_threshold(embeddings, test_pairs, thr)
    test.update(evaluate_pairs_auc(embeddings, test_pairs))
    return {"threshold": thr, "val": val, "test": test}


def main() -> None:
    parser = argparse.ArgumentParser(description="ER-оценка bge-m3 через HDBSCAN + DEAP GA")
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--model-tag", default="bge-m3")
    parser.add_argument("--graph-subdir", default="v14_mrl")
    parser.add_argument("--cross-subdir", default="v14_mrl_cross")
    parser.add_argument("--model-filename", default="v14_mrl_gat_model.pt",
                        help="ntxent=v14_mrl_gat_model.pt, bce=v14_mrl_gat_bce_model.pt")
    parser.add_argument("--device", default=None)

    # GA hyperparameters
    parser.add_argument("--pop-size", type=int, default=40)
    parser.add_argument("--n-gen", type=int, default=25)
    parser.add_argument("--cxpb", type=float, default=0.6)
    parser.add_argument("--mutpb", type=float, default=0.3)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--mcs-min", type=int, default=2)
    parser.add_argument("--mcs-max", type=int, default=100)
    parser.add_argument("--ms-min", type=int, default=1)
    parser.add_argument("--ms-max", type=int, default=50)
    parser.add_argument("--eps-min", type=float, default=0.0)
    parser.add_argument("--eps-max", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=1,
                        help="core_dist_n_jobs для CPU-бэкенда hdbscan")
    parser.add_argument("--backend", choices=["cpu", "gpu", "auto"], default="auto",
                        help="cpu → hdbscan, gpu → cuml.cluster.HDBSCAN, auto → gpu если есть cuml")

    parser.add_argument("--skip-threshold-baseline", action="store_true")
    parser.add_argument("--skip-cross-domain", action="store_true")
    args = parser.parse_args()

    config = Config(data_dir=Path(args.data_dir), output_dir=Path(args.output_dir))
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    model_tag = args.model_tag
    graph_dir = unified_dir(config.data_dir, model_tag, args.graph_subdir)
    cross_root = unified_dir(config.data_dir, model_tag, args.cross_subdir)
    out_dir = output_dir_for(config.output_dir, model_tag)
    model_path = out_dir / args.model_filename

    logger.info("Модель: %s (device=%s)", model_path, device)
    logger.info("Граф:   %s", graph_dir)

    # 1. Загрузка графа + пар
    graph = torch.load(graph_dir / "graph.pt", weights_only=False)
    val_pairs = torch.load(graph_dir / "val_pairs.pt", weights_only=False)
    test_pairs = torch.load(graph_dir / "test_pairs.pt", weights_only=False)
    logger.info("Rows=%d, val=%d пар, test=%d пар",
                graph["row"].x.shape[0], len(val_pairs), len(test_pairs))

    # 2. Row embeddings
    model = load_bge_model(model_path, graph, device)
    embeddings = get_row_embeddings(model, graph, device=device)
    logger.info("Embeddings: %s (нормализованы моделью)", tuple(embeddings.shape))

    results: dict = {
        "model_tag": model_tag,
        "model_path": str(model_path),
        "graph_subdir": args.graph_subdir,
        "n_rows": int(embeddings.shape[0]),
        "embedding_dim": int(embeddings.shape[1]),
    }

    # 3. Baseline threshold (для сравнения)
    if not args.skip_threshold_baseline:
        logger.info("=" * 60)
        logger.info("Baseline: cosine threshold (как exp 10)")
        logger.info("=" * 60)
        results["threshold_baseline"] = _threshold_metrics(embeddings, val_pairs, test_pairs)
        logger.info("  θ=%.4f  val F1=%.4f  test F1=%.4f",
                    results["threshold_baseline"]["threshold"],
                    results["threshold_baseline"]["val"].get("f1", 0.0),
                    results["threshold_baseline"]["test"].get("f1", 0.0))

    # 4. GA-HDBSCAN
    logger.info("=" * 60)
    logger.info("DEAP GA → HDBSCAN params (pop=%d, gen=%d)", args.pop_size, args.n_gen)
    logger.info("=" * 60)
    ga_cfg = GAHDBSCANConfig(
        pop_size=args.pop_size,
        n_gen=args.n_gen,
        cxpb=args.cxpb,
        mutpb=args.mutpb,
        tournament_size=args.tournament_size,
        min_cluster_size_bounds=(args.mcs_min, args.mcs_max),
        min_samples_bounds=(args.ms_min, args.ms_max),
        epsilon_bounds=(args.eps_min, args.eps_max),
        seed=args.seed,
        n_jobs=args.n_jobs,
        backend=args.backend,
    )
    ga = run_ga_hdbscan(embeddings, val_pairs, ga_cfg)
    logger.info("GA backend=%s best: %s  val F1=%.4f (evaluated=%d)",
                ga.backend, ga.best_params, ga.best_fitness, ga.n_evaluated)

    # 5. Оценка с лучшими параметрами на val / test
    embeddings_np = embeddings.detach().cpu().numpy()
    val_m = evaluate_params_on_pairs(embeddings_np, val_pairs, ga.best_params,
                                     n_jobs=args.n_jobs, backend=ga.backend)
    test_m = evaluate_params_on_pairs(embeddings_np, test_pairs, ga.best_params,
                                      n_jobs=args.n_jobs, backend=ga.backend)

    results["ga_hdbscan"] = {
        "backend": ga.backend,
        "best_params": ga.best_params,
        "best_val_fitness": ga.best_fitness,
        "n_evaluated": ga.n_evaluated,
        "config": {
            "pop_size": ga_cfg.pop_size, "n_gen": ga_cfg.n_gen,
            "cxpb": ga_cfg.cxpb, "mutpb": ga_cfg.mutpb,
            "tournament_size": ga_cfg.tournament_size,
            "seed": ga_cfg.seed,
        },
        "history": ga.history,
        "val": val_m,
        "test": test_m,
    }

    # 6. Cross-domain: те же параметры, per-dataset кластеризация
    if not args.skip_cross_domain and cross_root.exists():
        logger.info("=" * 60)
        logger.info("Cross-domain (best params re-applied per graph)")
        logger.info("=" * 60)
        cross_results = []
        for ds_dir in sorted(cross_root.iterdir()):
            if not ds_dir.is_dir() or not (ds_dir / "graph.pt").exists():
                continue
            lp_path = ds_dir / "labeled_pairs.pt"
            if not lp_path.exists():
                continue
            name = ds_dir.name
            cg = torch.load(ds_dir / "graph.pt", weights_only=False)
            cd_model = load_bge_model(model_path, cg, device)
            cd_emb = get_row_embeddings(cd_model, cg, device=device).detach().cpu().numpy()
            labeled = torch.load(lp_path, weights_only=False)

            cd_ga = evaluate_params_on_pairs(cd_emb, labeled, ga.best_params,
                                             n_jobs=args.n_jobs, backend=ga.backend)
            entry = {"name": name, "n_pairs": int(len(labeled)), **cd_ga}

            # Заодно — threshold baseline на cross-domain (с его же val_pairs, если есть)
            if not args.skip_threshold_baseline:
                thr = results["threshold_baseline"]["threshold"]
                thr_m = evaluate_pairs_at_threshold(torch.from_numpy(cd_emb), labeled, thr)
                thr_m.update(evaluate_pairs_auc(torch.from_numpy(cd_emb), labeled))
                entry["threshold_baseline"] = thr_m

            cross_results.append(entry)
            logger.info("  %-15s  GA F1=%.4f  P=%.4f  R=%.4f  clusters=%d  noise=%d",
                        name, entry.get("f1", 0.0), entry.get("precision", 0.0),
                        entry.get("recall", 0.0), entry.get("n_clusters", 0),
                        entry.get("n_noise", 0))

            del cg, cd_emb, cd_model
            if device == "cuda":
                torch.cuda.empty_cache()
        results["ga_hdbscan"]["cross_domain"] = cross_results

    # 7. Сохранение
    out_path = out_dir / "v18_ga_hdbscan_results.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Результаты: %s", out_path)

    # 8. Краткая сводка
    print("\n" + "=" * 60)
    print(f"EXP 18  —  {model_tag}  |  GA-HDBSCAN vs threshold")
    print("=" * 60)
    if "threshold_baseline" in results:
        b = results["threshold_baseline"]
        print(f"Threshold θ={b['threshold']:.4f}")
        print(f"  val   F1={b['val'].get('f1', 0):.4f}  P={b['val'].get('precision', 0):.4f}  R={b['val'].get('recall', 0):.4f}")
        print(f"  test  F1={b['test'].get('f1', 0):.4f}  P={b['test'].get('precision', 0):.4f}  R={b['test'].get('recall', 0):.4f}")
    print(f"\nGA-HDBSCAN best params: {ga.best_params}")
    print(f"  val   F1={val_m.get('f1', 0):.4f}  P={val_m.get('precision', 0):.4f}  R={val_m.get('recall', 0):.4f}  clusters={val_m.get('n_clusters', 0)}  noise={val_m.get('n_noise', 0)}")
    print(f"  test  F1={test_m.get('f1', 0):.4f}  P={test_m.get('precision', 0):.4f}  R={test_m.get('recall', 0):.4f}  clusters={test_m.get('n_clusters', 0)}  noise={test_m.get('n_noise', 0)}")
    if results.get("ga_hdbscan", {}).get("cross_domain"):
        print("\nCross-domain (GA params):")
        for r in results["ga_hdbscan"]["cross_domain"]:
            print(f"  {r['name']:15s}  F1={r.get('f1', 0):.4f}  P={r.get('precision', 0):.4f}  R={r.get('recall', 0):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
