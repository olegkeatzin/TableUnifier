"""Эксперимент 15 — Оценка ER-модели на trusted-разметке реальных данных.

Источники меток:
  * data/labeled/verification.parquet — 398 пар, размечены ВРУЧНУЮ
    (см. 12_verify_labels.ipynb: human_label ∈ {0, 1, 2-skip})
  * data/labeled/labeled_pairs.parquet — 26K LLM-меток (см. 12_label_real_data.py)

Trusted subset (фильтрация по reliability из 12_verify_labels.ipynb):
  * gold     : verification.human_label ∈ {0, 1}                — 398 пар, ground truth
  * silver   : labeled_pairs, confidence == 'high',
               исключая (label == 0 AND cosine_sim > 0.93)      — нестабильная зона
               (стратегия nomatch_high_sim показала только 60% accuracy)
  * silver+  : silver ∪ gold (gold перетирает LLM, если есть пересечение)

Pipeline:
  1. Reload Excel-ов, согласовать индексы:
        labeled_pairs.{spec_idx,nom_idx} = iloc в df после exp 12 cleaning
        cached graph id_to_global       = iloc в df после exp 07 cleaning
     Разница: exp 12 дополнительно дропает whitespace-only "Наименование".
     Маппинг строится через позиции в исходном (нефильтрованном) DataFrame.
  2. Загрузить кэшированный реальный граф (output/07_real_data_test/).
     Опционально MRL-обрезать col_embeddings → использовать v14_mrl модель.
  3. Загрузить модель (NT-Xent backbone или BCE PairClassifier.backbone).
  4. Cosine sim по парам: GNN vs raw-rubert baseline.
  5. Threshold sweep → F1/P/R + ROC-AUC + AP, плюс per-strategy breakdown.

Использование:
    # v3 NT-Xent (по умолчанию)
    python -m experiments.15_test_on_real_labels --model output/v3_gat_model.pt
    # v3 BCE
    python -m experiments.15_test_on_real_labels --model output/v3_gat_bce_model.pt --bce
    # v14 MRL NT-Xent (требует --mrl + --no-input-projection)
    python -m experiments.15_test_on_real_labels \
        --model output/v14_mrl_gat_model.pt --mrl --target-dim 312 --no-input-projection
    # v14 MRL BCE
    python -m experiments.15_test_on_real_labels \
        --model output/v14_mrl_gat_bce_model.pt --bce --mrl --target-dim 312 --no-input-projection
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from table_unifier.config import EntityResolutionConfig
from table_unifier.evaluation.clustering import (
    evaluate_pairs_at_threshold,
    evaluate_pairs_auc,
    find_best_threshold,
)
from table_unifier.models.entity_resolution import (
    EntityResolutionGAT,
    EntityResolutionGNN,
    PairClassifier,
)
from table_unifier.training.er_trainer import get_row_embeddings

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

EXCEL_DIR = Path("experiments/07_real_data_test")
GRAPH_DIR = Path("output/07_real_data_test")
LABELED_DIR = Path("data/labeled")

# Порог cosine sim, выше которого LLM-разметка no-match становится ненадёжной
# (по 12_verify_labels nomatch_high_sim — 60% accuracy на топ-50 sim).
SILVER_NOMATCH_SIM_CAP = 0.93


# ------------------------------------------------------------------ #
#  1. Согласование индексов между exp 12 и exp 07 cleanings
# ------------------------------------------------------------------ #

def build_idx_maps() -> tuple[np.ndarray, np.ndarray]:
    """Вернуть (nom12_to_nom07, spec12_to_spec07) — массивы такой же длины,
    что и df после exp 12 cleaning; элемент = iloc в df после exp 07 cleaning
    (т. е. позиция, которая сидит в id_to_global_a/b кэшированного графа).
    """
    nom_raw = pd.read_excel(EXCEL_DIR / "Номенклатура полная.xlsx")
    spec_raw = pd.read_excel(EXCEL_DIR / "сводная_спецификация.xlsx")

    # exp 07: dropna по Наименование
    nom07_orig = nom_raw.dropna(subset=["Наименование"]).index.to_numpy()
    spec07_orig = spec_raw.dropna(subset=["наименование"]).index.to_numpy()

    # exp 12: dropna + дроп пустых строк
    nom12_mask = nom_raw["Наименование"].notna() & (
        nom_raw["Наименование"].astype(str).str.strip() != ""
    )
    spec12_mask = spec_raw["наименование"].notna() & (
        spec_raw["наименование"].astype(str).str.strip() != ""
    )
    nom12_orig = nom_raw[nom12_mask].index.to_numpy()
    spec12_orig = spec_raw[spec12_mask].index.to_numpy()

    nom07_pos = {orig: pos for pos, orig in enumerate(nom07_orig)}
    spec07_pos = {orig: pos for pos, orig in enumerate(spec07_orig)}

    missing_nom = [o for o in nom12_orig if o not in nom07_pos]
    missing_spec = [o for o in spec12_orig if o not in spec07_pos]
    if missing_nom or missing_spec:
        raise RuntimeError(
            f"exp12-индексы не подмножество exp07-индексов: "
            f"{len(missing_nom)} nom + {len(missing_spec)} spec"
        )

    nom12_to_nom07 = np.array([nom07_pos[o] for o in nom12_orig], dtype=np.int64)
    spec12_to_spec07 = np.array([spec07_pos[o] for o in spec12_orig], dtype=np.int64)

    logger.info(
        "Индексы согласованы: nom %d→%d, spec %d→%d",
        len(nom12_to_nom07), len(nom07_orig),
        len(spec12_to_spec07), len(spec07_orig),
    )
    return nom12_to_nom07, spec12_to_spec07


# ------------------------------------------------------------------ #
#  2. Trusted subset
# ------------------------------------------------------------------ #

def load_trusted_pairs() -> dict[str, pd.DataFrame]:
    """Собрать trusted-наборы: gold (human) и silver (LLM, отфильтрованные)."""
    labeled = pd.read_parquet(LABELED_DIR / "labeled_pairs.parquet")
    labeled = labeled[labeled["label"] != -1].copy()

    # silver: high-conf, no-match с cosine_sim > SILVER_NOMATCH_SIM_CAP убраны
    silver_mask = (labeled["confidence"] == "high") & ~(
        (labeled["label"] == 0) & (labeled["cosine_sim"] > SILVER_NOMATCH_SIM_CAP)
    )
    silver = labeled.loc[silver_mask, [
        "spec_idx", "nom_idx", "spec_text", "nom_text", "cosine_sim", "label", "confidence",
    ]].copy()
    silver["source"] = "silver_llm"

    out: dict[str, pd.DataFrame] = {"silver": silver}

    verify_path = LABELED_DIR / "verification.parquet"
    if verify_path.exists():
        verify = pd.read_parquet(verify_path)
        gold = verify[verify["human_label"].isin([0, 1])].copy()
        gold = gold.rename(columns={"label": "llm_label"})
        gold["label"] = gold["human_label"].astype(int)
        gold["source"] = "gold_human"
        cols = ["spec_idx", "nom_idx", "spec_text", "nom_text", "cosine_sim",
                "label", "confidence", "source"]
        if "strategy" in gold.columns:
            cols.append("strategy")
        out["gold"] = gold[cols].copy()

        # silver+ : берём gold + silver минус совпадающие пары (gold побеждает)
        gold_keys = set(zip(gold["spec_text"], gold["nom_text"]))
        silver_extra = silver[
            ~silver.set_index(["spec_text", "nom_text"]).index.isin(gold_keys)
        ]
        out["silver_plus_gold"] = pd.concat(
            [out["gold"][cols], silver_extra.assign(strategy=pd.NA)[cols]],
            ignore_index=True,
        )
    else:
        logger.warning("verification.parquet не найден — gold-уровня не будет")

    for name, df in out.items():
        npos = int((df["label"] == 1).sum())
        nneg = int((df["label"] == 0).sum())
        logger.info("Trusted [%s]: %d пар (pos=%d, neg=%d)", name, len(df), npos, nneg)
    return out


# ------------------------------------------------------------------ #
#  3. Loading model
# ------------------------------------------------------------------ #

def mrl_truncate(emb: torch.Tensor, target_dim: int) -> torch.Tensor:
    """MRL: префикс + L2-renorm. qwen3-embedding обучен с MRL → префиксы валидны."""
    truncated = emb[:, :target_dim]
    return F.normalize(truncated, p=2, dim=-1)


def load_model(
    graph,
    model_path: Path,
    bce: bool,
    arch: str,                  # "gnn" | "gat"
    use_input_projection: bool, # для v14 MRL — False
    er_config: EntityResolutionConfig,
    device: str,
):
    """Загрузить backbone модели (для cosine-similarity оценки)."""
    er_config.row_dim = int(graph["row"].x.shape[1])
    er_config.token_dim = int(graph["token"].x.shape[1])
    er_config.col_dim = int(graph.col_embeddings.shape[1])

    if arch == "gat":
        backbone = EntityResolutionGAT(
            row_dim=er_config.row_dim,
            token_dim=er_config.token_dim,
            col_dim=er_config.col_dim,
            hidden_dim=er_config.hidden_dim,
            edge_dim=er_config.edge_dim,
            output_dim=er_config.output_dim,
            num_gnn_layers=er_config.num_gnn_layers,
            num_heads=er_config.num_heads,
            dropout=er_config.dropout,
            attention_dropout=er_config.attention_dropout,
            bidirectional=er_config.bidirectional,
            use_input_projection=use_input_projection,
        )
    else:
        backbone = EntityResolutionGNN(
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

    state = torch.load(model_path, map_location=device, weights_only=True)
    if bce:
        wrapper = PairClassifier(backbone, embedding_dim=er_config.output_dim)
        wrapper.load_state_dict(state)
        wrapper.to(device).eval()
        return wrapper.backbone
    backbone.load_state_dict(state)
    backbone.to(device).eval()
    return backbone


def load_er_config(model_path: Path) -> tuple[EntityResolutionConfig, bool]:
    """Подобрать er_config: рядом с моделью .config.json (exp 14) или HPO json'ы."""
    cfg_path = model_path.with_suffix(".config.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg_dict = json.load(f)
        logger.info("Конфиг модели из %s", cfg_path)
        return EntityResolutionConfig(
            hidden_dim=cfg_dict["hidden_dim"],
            edge_dim=cfg_dict["edge_dim"],
            output_dim=cfg_dict["output_dim"],
            num_gnn_layers=cfg_dict["num_gnn_layers"],
            num_heads=cfg_dict.get("num_heads", 4),
            dropout=cfg_dict["dropout"],
            attention_dropout=cfg_dict.get("attention_dropout", 0.1),
            bidirectional=cfg_dict["bidirectional"],
            temperature=0.1,
        ), cfg_dict.get("use_input_projection", True)

    # HPO defaults (как в exp 10)
    hpo_arch_path = Path("output/hpo_architecture.json")
    hpo_train_path = Path("output/hpo_training.json")
    if hpo_arch_path.exists() and hpo_train_path.exists():
        with open(hpo_arch_path) as f:
            arch = json.load(f)["best_params"]
        logger.info("Конфиг из HPO (%s)", hpo_arch_path)
        return EntityResolutionConfig(
            hidden_dim=arch["hidden_dim"], edge_dim=arch["edge_dim"],
            num_gnn_layers=arch["num_gnn_layers"], dropout=arch["dropout"],
            bidirectional=arch["bidirectional"],
            num_heads=4, attention_dropout=0.1, temperature=0.1,
        ), True

    logger.info("Конфиг по умолчанию (hidden=128, layers=2)")
    return EntityResolutionConfig(
        hidden_dim=128, edge_dim=128, num_gnn_layers=2, dropout=0.3,
        bidirectional=True, num_heads=4, attention_dropout=0.1, temperature=0.1,
    ), True


# ------------------------------------------------------------------ #
#  4. Сборка пар → tensor для evaluate_pairs_*
# ------------------------------------------------------------------ #

def pairs_to_tensor(
    df: pd.DataFrame,
    nom12_to_nom07: np.ndarray,
    spec12_to_spec07: np.ndarray,
    id_to_global_a: dict[str, int],
    id_to_global_b: dict[str, int],
) -> tuple[torch.Tensor, np.ndarray]:
    """Маппинг (spec_idx, nom_idx, label) → (global_a, global_b, label).

    Returns:
        pairs: [N, 3] tensor (global_idx_nom, global_idx_spec, label) для evaluate_pairs_*
        keep_mask: bool mask для исходного df (какие строки удалось замапить)
    """
    nom_idx = df["nom_idx"].to_numpy()
    spec_idx = df["spec_idx"].to_numpy()
    labels = df["label"].astype(int).to_numpy()

    # exp12 idx → exp07 iloc → ID-строка → global graph idx
    valid = (
        (nom_idx >= 0) & (nom_idx < len(nom12_to_nom07))
        & (spec_idx >= 0) & (spec_idx < len(spec12_to_spec07))
    )
    pairs_list = []
    keep_mask = np.zeros(len(df), dtype=bool)
    for i in np.where(valid)[0]:
        a_id = f"a_{int(nom12_to_nom07[nom_idx[i]])}"
        b_id = f"b_{int(spec12_to_spec07[spec_idx[i]])}"
        ga = id_to_global_a.get(a_id)
        gb = id_to_global_b.get(b_id)
        if ga is None or gb is None:
            continue
        pairs_list.append([ga, gb, int(labels[i])])
        keep_mask[i] = True
    return torch.tensor(pairs_list, dtype=torch.long), keep_mask


# ------------------------------------------------------------------ #
#  5. Eval c подбором порога на pos-доле, отчётом по стратам
# ------------------------------------------------------------------ #

def eval_with_threshold_sweep(
    name: str, embeddings: torch.Tensor, pairs: torch.Tensor, df: pd.DataFrame,
) -> dict:
    """Найти лучший порог по F1, посчитать P/R/F1 + ROC-AUC + AP.
       Дополнительно — per-strategy breakdown (если колонка strategy есть).
    """
    if pairs.shape[0] == 0 or len(np.unique(pairs[:, 2].numpy())) < 2:
        logger.warning("[%s] нечего оценивать (пар=%d)", name, pairs.shape[0])
        return {"n_pairs": int(pairs.shape[0])}

    threshold, sweep_f1 = find_best_threshold(embeddings, pairs)
    metrics = evaluate_pairs_at_threshold(embeddings, pairs, threshold)
    metrics.update(evaluate_pairs_auc(embeddings, pairs))
    metrics["sweep_f1"] = sweep_f1

    # per-strategy F1 (на той же θ, для прозрачности)
    if "strategy" in df.columns and df["strategy"].notna().any():
        per_strat = {}
        scores = (embeddings[pairs[:, 0]] * embeddings[pairs[:, 1]]).sum(dim=1).numpy()
        labels_arr = pairs[:, 2].numpy()
        # df отфильтрован тем же keep_mask, что и pairs (одинаковая длина / порядок)
        for strat, sub_idx in df.reset_index(drop=True).groupby(
            "strategy", dropna=True,
        ).groups.items():
            sub_idx = list(sub_idx)
            if not sub_idx:
                continue
            preds = (scores[sub_idx] >= threshold).astype(int)
            from sklearn.metrics import f1_score, precision_score, recall_score
            try:
                per_strat[str(strat)] = {
                    "n": len(sub_idx),
                    "n_pos": int(labels_arr[sub_idx].sum()),
                    "f1": float(f1_score(labels_arr[sub_idx], preds, zero_division=0)),
                    "precision": float(precision_score(
                        labels_arr[sub_idx], preds, zero_division=0,
                    )),
                    "recall": float(recall_score(
                        labels_arr[sub_idx], preds, zero_division=0,
                    )),
                }
            except Exception as e:
                per_strat[str(strat)] = {"error": str(e)}
        metrics["per_strategy"] = per_strat

    logger.info(
        "[%s] θ=%.3f  F1=%.3f  P=%.3f  R=%.3f  AUC=%.3f  AP=%.3f  (n=%d, pos=%d)",
        name, threshold, metrics["f1"], metrics["precision"], metrics["recall"],
        metrics.get("roc_auc", 0), metrics.get("avg_precision", 0),
        metrics["n_pairs"], metrics["n_pos"],
    )
    return metrics


# ------------------------------------------------------------------ #
#  Main
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(description="Test ER on trusted real labels")
    parser.add_argument("--model", type=Path, required=True,
                        help="Путь к чекпоинту модели (.pt)")
    parser.add_argument("--bce", action="store_true",
                        help="Чекпоинт — PairClassifier (см. exp 11/14)")
    parser.add_argument("--arch", choices=["gnn", "gat"], default="gat",
                        help="Архитектура backbone (по умолч. gat)")
    parser.add_argument("--mrl", action="store_true",
                        help="MRL-обрезать col_embeddings перед инференсом (для v14)")
    parser.add_argument("--target-dim", type=int, default=312,
                        help="Размер MRL-обрезки (default 312 = rubert hidden)")
    parser.add_argument("--no-input-projection", action="store_true",
                        help="Для v14 MRL: модель без row/token/edge проекций")
    parser.add_argument("--device", default=None)
    parser.add_argument("--out-json", type=Path, default=None,
                        help="Куда сохранить результаты (по умолч. рядом с моделью)")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    # 1. trusted-метки
    trusted = load_trusted_pairs()
    nom12_to_nom07, spec12_to_spec07 = build_idx_maps()

    # 2. cached real-data graph (exp 07)
    logger.info("Загрузка кэшированного графа из %s", GRAPH_DIR)
    graph = torch.load(GRAPH_DIR / "graph.pt", weights_only=False)
    with open(GRAPH_DIR / "id_to_global_a.json") as f:
        id_to_global_a = json.load(f)
    with open(GRAPH_DIR / "id_to_global_b.json") as f:
        id_to_global_b = json.load(f)
    logger.info("Граф: %d row, %d token, col_dim=%d",
                graph["row"].x.shape[0], graph["token"].x.shape[0],
                graph.col_embeddings.shape[1])

    if args.mrl:
        logger.info("MRL truncate col_embeddings: %d → %d",
                    graph.col_embeddings.shape[1], args.target_dim)
        graph.col_embeddings = mrl_truncate(graph.col_embeddings, args.target_dim)

    # 3. модель + GNN-эмбеддинги
    er_config, default_use_proj = load_er_config(args.model)
    use_input_proj = not args.no_input_projection and default_use_proj
    model = load_model(
        graph, args.model, bce=args.bce, arch=args.arch,
        use_input_projection=use_input_proj,
        er_config=er_config, device=device,
    )
    logger.info("Получаем GNN-эмбеддинги...")
    gnn_emb = get_row_embeddings(model, graph, device=device).cpu()
    baseline_emb = F.normalize(graph["row"].x.float().cpu(), p=2, dim=-1)

    # 4. оценка для каждого тира
    results: dict[str, dict] = {}
    for tier_name, tier_df in trusted.items():
        pairs, keep_mask = pairs_to_tensor(
            tier_df, nom12_to_nom07, spec12_to_spec07,
            id_to_global_a, id_to_global_b,
        )
        kept_df = tier_df.iloc[keep_mask].reset_index(drop=True)
        n_dropped = len(tier_df) - len(kept_df)
        if n_dropped:
            logger.warning("[%s] не удалось замапить %d/%d пар",
                           tier_name, n_dropped, len(tier_df))

        results[tier_name] = {
            "n_input": int(len(tier_df)),
            "n_evaluated": int(pairs.shape[0]),
            "GNN": eval_with_threshold_sweep(f"{tier_name}/GNN", gnn_emb, pairs, kept_df),
            "Baseline": eval_with_threshold_sweep(
                f"{tier_name}/Baseline", baseline_emb, pairs, kept_df,
            ),
        }

    # 5. дамп
    out_path = args.out_json or args.model.with_name(
        args.model.stem + "_real_trusted_eval.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    logger.info("Результаты сохранены: %s", out_path)

    # 6. сводка в stdout
    print("\n" + "=" * 70)
    print(f"REAL-DATA EVAL  |  model={args.model}")
    print("=" * 70)
    for tier_name, r in results.items():
        print(f"\n[{tier_name}] evaluated {r['n_evaluated']} / {r['n_input']} pairs")
        for method in ("GNN", "Baseline"):
            m = r.get(method, {})
            if not m or "f1" not in m:
                continue
            print(f"  {method:9s}  θ={m['threshold']:.3f}  "
                  f"F1={m['f1']:.3f}  P={m['precision']:.3f}  R={m['recall']:.3f}  "
                  f"AUC={m.get('roc_auc', 0):.3f}  AP={m.get('avg_precision', 0):.3f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
