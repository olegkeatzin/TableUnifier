"""Сервис, оркеструющий построение графа и инференс GAT-модели."""

from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from table_unifier.config import Config, EntityResolutionConfig, OllamaConfig
from table_unifier.dataset.embedding_generation import (
    TokenEmbedder,
    generate_column_embeddings,
)
from table_unifier.dataset.graph_builder import build_graph
from table_unifier.models.entity_resolution import EntityResolutionGAT
from table_unifier.ollama_client import OllamaClient
from table_unifier.server.services.progress import bus
from table_unifier.server.services.storage import RunState, store

logger = logging.getLogger(__name__)


# ----------------------------- helpers ----------------------------------- #


def _publish(run_id: str, event: dict[str, Any]) -> None:
    bus.publish(run_id, event)


def _log(run_id: str, level: str, msg: str) -> None:
    _publish(run_id, {"type": "log", "level": level, "msg": msg})


def _graph_nbytes(graph) -> int:
    """Суммарный размер тензоров HeteroData в байтах (для метрики памяти графа)."""
    total = 0
    seen: set[int] = set()

    def _add(t: Any) -> None:
        nonlocal total
        if isinstance(t, torch.Tensor) and id(t) not in seen:
            seen.add(id(t))
            total += t.element_size() * t.nelement()

    for store in graph.stores:
        for value in store.values():
            _add(value)
    # Глобальные атрибуты, которые могут не попасть в stores.
    for attr in ("col_embeddings", "token_ids"):
        _add(getattr(graph, attr, None))
    return total


def _mrl_truncate(emb: np.ndarray, target_dim: int) -> np.ndarray:
    """MRL truncation + L2 renorm (qwen3-embedding обучен с MRL)."""
    if emb.ndim == 1:
        truncated = emb[:target_dim]
        norm = float(np.linalg.norm(truncated)) + 1e-12
        return (truncated / norm).astype(np.float32)
    truncated = emb[:, :target_dim]
    norm = np.linalg.norm(truncated, axis=-1, keepdims=True) + 1e-12
    return (truncated / norm).astype(np.float32)


# Кэш TokenEmbedder между runs (модель тяжёлая, ~2GB для bge-m3).
_token_embedder_cache: dict[str, TokenEmbedder] = {}


def _get_token_embedder(model_tag: str, device: str | None = None) -> TokenEmbedder:
    if model_tag in _token_embedder_cache:
        return _token_embedder_cache[model_tag]

    name_map = {
        "bge-m3": "BAAI/bge-m3",
        "rubert-tiny2": "cointegrated/rubert-tiny2",
    }
    pooling_map = {"bge-m3": "cls", "rubert-tiny2": "cls"}
    model_name = name_map.get(model_tag, model_tag)
    pooling = pooling_map.get(model_tag, "cls")

    emb = TokenEmbedder(
        model_name=model_name, device=device,
        pooling=pooling, trust_remote_code=False,
    )
    _token_embedder_cache[model_tag] = emb
    return emb


# ------------------------- schema divergence ----------------------------- #


def detect_schema_divergence(
    df_a: pd.DataFrame, df_b: pd.DataFrame,
    ollama_host: str | None = None,
    threshold_rename: float = 0.55,
) -> list[dict]:
    """Сравнить колонки двух таблиц по cosine на qwen3-embedding column embeddings.

    Возвращает список словарей с парами расхождений. Если Ollama недоступна,
    fallback на сравнение имён (LCS-подобное) — но точность будет ниже.
    """
    cols_a = [c for c in df_a.columns if c != "id"]
    cols_b = [c for c in df_b.columns if c != "id"]

    try:
        cfg = OllamaConfig()
        if ollama_host:
            cfg.host = ollama_host
        client = OllamaClient(cfg)
        col_emb_a = generate_column_embeddings(client, df_a, columns=cols_a)
        col_emb_b = generate_column_embeddings(client, df_b, columns=cols_b)
    except Exception as e:
        logger.warning("Ollama недоступна для schema-divergence: %s — fallback на имена", e)
        return _name_only_divergence(cols_a, cols_b, df_a, df_b)

    items: list[dict] = []
    for a_col in cols_a:
        ea = col_emb_a.get(a_col)
        if ea is None:
            continue
        ea_n = ea / (np.linalg.norm(ea) + 1e-12)
        best_b = None
        best_sim = -1.0
        for b_col in cols_b:
            eb = col_emb_b.get(b_col)
            if eb is None:
                continue
            eb_n = eb / (np.linalg.norm(eb) + 1e-12)
            sim = float(ea_n @ eb_n)
            if sim > best_sim:
                best_sim = sim
                best_b = b_col
        if best_b is None or best_sim < threshold_rename:
            continue
        if a_col == best_b:
            kind = "exact"
            note = "точное совпадение имени"
        else:
            kind = "renamed"
            note = "вероятно переименовано"
        # format-расхождение: type mismatch / hex vs text
        if a_col in df_a.columns and best_b in df_b.columns:
            a_vals = df_a[a_col].dropna().astype(str).head(20).tolist()
            b_vals = df_b[best_b].dropna().astype(str).head(20).tolist()
            if _format_differs(a_vals, b_vals):
                kind = "format"
                note = "разный формат значений"
        items.append({
            "a_col": a_col, "b_col": best_b,
            "similarity": best_sim, "kind": kind, "note": note,
        })
    return items


def _name_only_divergence(cols_a: list[str], cols_b: list[str],
                          df_a: pd.DataFrame, df_b: pd.DataFrame) -> list[dict]:
    items = []
    for a in cols_a:
        for b in cols_b:
            la, lb = a.lower(), b.lower()
            if la == lb:
                items.append({"a_col": a, "b_col": b, "similarity": 1.0,
                              "kind": "exact", "note": "точное совпадение"})
                break
            if la in lb or lb in la:
                items.append({"a_col": a, "b_col": b, "similarity": 0.7,
                              "kind": "renamed", "note": "имена пересекаются"})
                break
    return items


def _format_differs(a_vals: list[str], b_vals: list[str]) -> bool:
    def is_hex(s: str) -> bool:
        return bool(s) and s.startswith("#") and len(s) in (4, 7)
    def is_upper(s: str) -> bool:
        return bool(s) and s.isupper()
    if any(is_hex(v) for v in a_vals) != any(is_hex(v) for v in b_vals):
        return True
    a_up = sum(is_upper(v) for v in a_vals) / max(1, len(a_vals))
    b_up = sum(is_upper(v) for v in b_vals) / max(1, len(b_vals))
    if abs(a_up - b_up) > 0.5:
        return True
    return False


# --------------------------- build graph --------------------------------- #


def build_graph_run(run_id: str, *,
                    idf_min_df: int = 2,
                    max_token_df: float = 0.3,
                    max_tokens_per_cell: int = 16,
                    target_col_dim: int = 1024) -> None:
    """Полный pipeline построения графа из таблиц A и B. Вызывается из потока."""
    run = store.get_run(run_id)
    if run is None:
        _publish(run_id, {"type": "error", "msg": f"run {run_id} not found"})
        return

    try:
        run.status = "building"
        session = store.get_session(run.session_id)
        if session is None or len(run.source_ids) < 2:
            raise ValueError("Нужно как минимум 2 источника")
        srcs = [session.sources[s] for s in run.source_ids[:2]]
        run.source_a_id, run.source_b_id = srcs[0].id, srcs[1].id
        run.table_a, run.table_b = srcs[0].df, srcs[1].df
        device = "cuda" if torch.cuda.is_available() else "cpu"

        # ---- Phase: embed (row embeddings) ---- #
        _publish(run_id, {"type": "phase", "phase": "embed",
                          "label": f"TokenEmbedder · {run.model_tag}"})
        _log(run_id, "info", f"loading TokenEmbedder({run.model_tag}) on {device}")
        token_embedder = _get_token_embedder(run.model_tag, device=device)
        _log(run_id, "info", f"hidden_dim={token_embedder.hidden_dim}, "
                              f"vocab={token_embedder.vocab_embeddings.shape[0]}")

        from table_unifier.dataset.embedding_generation import serialize_row
        cols_a = srcs[0].cols
        cols_b = srcs[1].cols
        texts_a = [serialize_row(r, cols_a) for _, r in run.table_a.iterrows()]
        texts_b = [serialize_row(r, cols_b) for _, r in run.table_b.iterrows()]
        _publish(run_id, {"type": "progress", "phase": "embed", "progress": 0.1})

        _t_row = time.perf_counter()
        row_emb_a = token_embedder.embed_sentences(texts_a, batch_size=16,
                                                    desc="rows A")
        _publish(run_id, {"type": "progress", "phase": "embed", "progress": 0.5})
        _log(run_id, "ok", f"row embeddings A: {row_emb_a.shape}")
        row_emb_b = token_embedder.embed_sentences(texts_b, batch_size=16,
                                                    desc="rows B")
        t_row_embeddings_ms = (time.perf_counter() - _t_row) * 1000
        _publish(run_id, {"type": "progress", "phase": "embed", "progress": 1.0})
        _log(run_id, "ok", f"row embeddings B: {row_emb_b.shape}")

        # ---- Phase: tokenize (column embeddings via Ollama) ---- #
        _publish(run_id, {"type": "phase", "phase": "tokenize",
                          "label": "Ollama qwen3-embedding · column"})
        try:
            ollama = OllamaClient()
        except Exception as e:
            raise RuntimeError(f"Не удалось подключиться к Ollama: {e}") from e
        _log(run_id, "info", "computing column embeddings via Ollama qwen3-embedding:8b")
        # timings аккумулирует descriptions_ms (LLM) + embed_ms по обоим вызовам.
        col_timings: dict[str, float] = {}
        try:
            col_emb_a = generate_column_embeddings(ollama, run.table_a, columns=cols_a,
                                                   timings=col_timings)
            col_emb_b = generate_column_embeddings(ollama, run.table_b, columns=cols_b,
                                                   timings=col_timings)
        except Exception as e:
            raise RuntimeError(f"Ollama embedding failed: {e}") from e
        t_col_descriptions_ms = col_timings.get("descriptions_ms", 0.0)
        t_col_embeddings_ms = col_timings.get("embed_ms", 0.0)
        _publish(run_id, {"type": "progress", "phase": "tokenize", "progress": 1.0})
        _log(run_id, "ok", f"qwen3-emb: {len(col_emb_a)} (A) + {len(col_emb_b)} (B) колонок")

        # MRL truncation для совместимости с v14_mrl чекпоинтом.
        col_emb_a = {k: _mrl_truncate(v, target_col_dim) for k, v in col_emb_a.items()}
        col_emb_b = {k: _mrl_truncate(v, target_col_dim) for k, v in col_emb_b.items()}
        col_emb_all = {**col_emb_a, **col_emb_b}

        # Row embeddings тоже MRL-урезаем если их размерность > target.
        if row_emb_a.shape[1] > target_col_dim:
            row_emb_a = _mrl_truncate(row_emb_a, target_col_dim)
            row_emb_b = _mrl_truncate(row_emb_b, target_col_dim)

        # ---- Phase: build (HeteroData) ---- #
        _publish(run_id, {"type": "phase", "phase": "build",
                          "label": "HeteroData (row + token + edges)"})
        graph, id_to_global_a, id_to_global_b = build_graph(
            run.table_a, run.table_b,
            column_embeddings=col_emb_all,
            token_embedder=token_embedder,
            columns_a=cols_a, columns_b=cols_b,
            precomputed_row_embeddings_a=row_emb_a,
            precomputed_row_embeddings_b=row_emb_b,
            max_token_df=max_token_df,
            max_tokens_per_cell=max_tokens_per_cell,
            min_token_count=idf_min_df,
        )
        _publish(run_id, {"type": "progress", "phase": "build", "progress": 1.0})

        n_rows = int(graph["row"].x.shape[0])
        n_tokens = int(graph["token"].x.shape[0])
        n_edges = int(graph["token", "in_row", "row"].edge_index.shape[1])
        _log(run_id, "ok",
             f"graph ready · {n_rows} row · {n_tokens} token · {n_edges} edges")

        run.graph = graph
        run.id_to_global_a = id_to_global_a
        run.id_to_global_b = id_to_global_b
        run.status = "graph_ready"

        graph_bytes = _graph_nbytes(graph)
        run.metrics.update({
            "n_rows": n_rows, "n_tokens": n_tokens,
            "n_edges": n_edges, "col_dim": target_col_dim,
            # тайминги стадий сборки (мс) — для панели «время выполнения» на экране инференса
            "t_row_embeddings_ms": round(t_row_embeddings_ms, 1),
            "t_col_descriptions_ms": round(t_col_descriptions_ms, 1),
            "t_col_embeddings_ms": round(t_col_embeddings_ms, 1),
            "graph_bytes": graph_bytes,
            "graph_mem_mb": round(graph_bytes / 1048576, 3),
        })
        _log(run_id, "info",
             f"timings · rows {t_row_embeddings_ms:.0f}ms · "
             f"col-desc {t_col_descriptions_ms:.0f}ms · "
             f"col-emb {t_col_embeddings_ms:.0f}ms · "
             f"graph {graph_bytes / 1048576:.1f}MB")

        _publish(run_id, {"type": "graph_done",
                          "n_rows": n_rows, "n_tokens": n_tokens, "n_edges": n_edges})
    except Exception as e:
        logger.exception("build_graph_run failed")
        run.status = "error"
        run.error = str(e)
        _publish(run_id, {"type": "error", "msg": str(e)})


# ----------------------------- inference --------------------------------- #


def _load_checkpoint(checkpoint: str, graph) -> EntityResolutionGAT:
    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    cfg_path = ckpt_path.with_suffix(".config.json")
    if cfg_path.exists():
        with open(cfg_path) as f:
            cfg = json.load(f)
    else:
        # Дефолт под v14_mrl, bge-m3
        col_dim = int(graph.col_embeddings.shape[1])
        cfg = {
            "row_dim": col_dim, "token_dim": col_dim, "col_dim": col_dim,
            "hidden_dim": col_dim, "edge_dim": col_dim, "output_dim": col_dim,
            "num_gnn_layers": 2, "num_heads": 4,
            "dropout": 0.3, "attention_dropout": 0.1,
            "bidirectional": True, "use_input_projection": False,
        }
    model = EntityResolutionGAT(
        row_dim=cfg["row_dim"], token_dim=cfg["token_dim"], col_dim=cfg["col_dim"],
        hidden_dim=cfg["hidden_dim"], edge_dim=cfg["edge_dim"],
        output_dim=cfg["output_dim"],
        num_gnn_layers=cfg["num_gnn_layers"], num_heads=cfg["num_heads"],
        dropout=cfg["dropout"], attention_dropout=cfg["attention_dropout"],
        bidirectional=cfg["bidirectional"],
        use_input_projection=cfg["use_input_projection"],
    )
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state)
    return model


def _aggregate_edge_attention(attentions: list) -> np.ndarray | None:
    """Сводит attention со всех GAT-слоёв к одному скаляру на ребро.

    attentions — список тензоров [n_edges, num_heads] (token→row, по слою).
    Возвращает np.ndarray [n_edges] = mean по головам, затем mean по слоям,
    либо None если внимание недоступно.
    """
    try:
        per_layer = [a.mean(dim=1) for a in attentions if a is not None]
        if not per_layer:
            return None
        stacked = torch.stack(per_layer, dim=0)  # [n_layers, n_edges]
        return stacked.mean(dim=0).cpu().numpy()
    except Exception as e:  # noqa: BLE001 — внимание не критично для инференса
        logger.warning("attention aggregation failed: %s", e)
        return None


def _connected_components(sim: np.ndarray, threshold: float,
                          n_a: int, n_b: int) -> tuple[list[int], list[int]]:
    """Union-find по cross-edges sim >= thr. Возвращает (labels_a, labels_b)."""
    parent = list(range(n_a + n_b))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i in range(n_a):
        for j in range(n_b):
            if sim[i, j] >= threshold:
                union(i, n_a + j)

    roots: dict[int, int] = {}
    labels_a = []
    labels_b = []
    for i in range(n_a):
        r = find(i)
        if r not in roots:
            roots[r] = len(roots)
        labels_a.append(roots[r])
    for j in range(n_b):
        r = find(n_a + j)
        if r not in roots:
            roots[r] = len(roots)
        labels_b.append(roots[r])
    return labels_a, labels_b


def _verdict(sim: float, threshold: float) -> str:
    if sim >= threshold + 0.05:
        return "auto"
    if sim >= threshold - 0.1:
        return "review"
    return "reject"


def _detect_field_divergence(row_a: pd.Series, row_b: pd.Series,
                             cols_a: list[str], cols_b: list[str]) -> list[str]:
    div: list[str] = []
    for ca, cb in zip(cols_a, cols_b):
        va, vb = row_a.get(ca), row_b.get(cb)
        if pd.isna(va) or pd.isna(vb):
            continue
        sa = str(va).strip().lower().replace("#", "").replace(" ", "")
        sb = str(vb).strip().lower().replace("#", "").replace(" ", "")
        if sa != sb:
            div.append(ca)
    return div


def run_inference(run_id: str, *,
                  checkpoint: str = "output/bge-m3/v14_mrl_gat_model.pt",
                  similarity_threshold: float = 0.831,
                  use_ga_tuning: bool = False,
                  top_k: int = 10) -> None:
    run = store.get_run(run_id)
    if run is None or run.graph is None:
        _publish(run_id, {"type": "error", "msg": "graph not built"})
        return

    try:
        run.status = "infer"
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info("[%s] inference thread started · device=%s · ckpt=%s",
                    run_id, device, checkpoint)

        # PHASE: load
        _publish(run_id, {"type": "phase", "phase": "load",
                          "label": f"load checkpoint · {Path(checkpoint).name}"})
        _log(run_id, "info", f"loading checkpoint {checkpoint}")
        model = _load_checkpoint(checkpoint, run.graph).to(device)
        model.eval()
        n_params = sum(p.numel() for p in model.parameters())
        _log(run_id, "ok", f"model ready · {n_params/1e6:.1f}M params · device={device}")
        _publish(run_id, {"type": "progress", "phase": "load", "progress": 1.0})

        # PHASE: l1 / l2 (forward pass) — выполняется одним вызовом, события
        # эмиттим вокруг него для UX (анимация на фронте).
        graph = run.graph.to(device)
        _publish(run_id, {"type": "phase", "phase": "l1",
                          "label": "GATv2Conv[0] · token→row"})
        _publish(run_id, {"type": "progress", "phase": "l1", "progress": 0.0})
        t0 = time.time()
        with torch.no_grad():
            row_embeddings_dev, attentions = model(graph, return_attention=True)
            row_embeddings = row_embeddings_dev.cpu()
        forward_ms = (time.time() - t0) * 1000

        # Агрегируем внимание token→row: среднее по головам, затем по GAT-слоям.
        # Даёт по одному скаляру на ребро → толщина рёбер графа на фронте.
        run.edge_attention = _aggregate_edge_attention(attentions)
        _publish(run_id, {"type": "progress", "phase": "l1", "progress": 1.0})
        _publish(run_id, {"type": "phase", "phase": "l2",
                          "label": "GATv2Conv[1] · row→token"})
        _publish(run_id, {"type": "progress", "phase": "l2", "progress": 1.0})
        _log(run_id, "ok",
             f"forward pass complete · {row_embeddings.shape} · {forward_ms:.0f}ms")
        run.row_embeddings = row_embeddings

        # PHASE: sim
        _publish(run_id, {"type": "phase", "phase": "sim",
                          "label": "cosine similarity matrix"})
        n_a = len(run.id_to_global_a)
        n_b = len(run.id_to_global_b)
        idx_a = sorted(run.id_to_global_a.values())
        idx_b = sorted(run.id_to_global_b.values())
        emb_a = row_embeddings[idx_a]
        emb_b = row_embeddings[idx_b]
        sim = (emb_a @ emb_b.T).numpy()
        run.similarity_matrix = sim
        _publish(run_id, {"type": "progress", "phase": "sim", "progress": 1.0})
        n_over = int((sim >= similarity_threshold).sum())
        _log(run_id, "ok",
             f"{n_a * n_b} pairs · {n_over} over threshold {similarity_threshold:.3f}")

        # PHASE: cluster (CC по cross-edges)
        _publish(run_id, {"type": "phase", "phase": "cluster",
                          "label": "connected components"})
        # Опциональный GA-tuning порога. Если нет val pairs — просто сжимаем.
        if use_ga_tuning:
            _log(run_id, "info", "GA-tuning skipped (no validation pairs in inference run)")
        labels_a, labels_b = _connected_components(sim, similarity_threshold, n_a, n_b)

        # Cand pairs: все пары с sim >= 0.5 (нижний хвост на review)
        global_to_id_a = {v: k for k, v in run.id_to_global_a.items()}
        global_to_id_b = {v: k for k, v in run.id_to_global_b.items()}

        cols_a = [c for c in run.table_a.columns if c != "id"]
        cols_b = [c for c in run.table_b.columns if c != "id"]

        candidates: list[dict] = []
        candidate_threshold = max(0.5, similarity_threshold - 0.15)
        pair_counter = 0
        for i in range(n_a):
            for j in range(n_b):
                s = float(sim[i, j])
                if s < candidate_threshold:
                    continue
                pair_counter += 1
                pair_id = f"pair_{pair_counter:03d}"
                a_id = global_to_id_a[idx_a[i]]
                b_id = global_to_id_b[idx_b[j]]
                row_a = run.table_a[run.table_a["id"].astype(str) == str(a_id)].iloc[0]
                row_b = run.table_b[run.table_b["id"].astype(str) == str(b_id)].iloc[0]
                div = _detect_field_divergence(row_a, row_b, cols_a, cols_b)
                v = _verdict(s, similarity_threshold)
                cluster_id = (f"C-{labels_a[i] + 1:03d}"
                              if v in ("auto", "review") and labels_a[i] == labels_b[j]
                              else None)
                candidates.append({
                    "id": pair_id, "a": str(a_id), "b": str(b_id),
                    "a_idx": i, "b_idx": j,
                    "similarity": s, "verdict": v,
                    "cluster_id": cluster_id,
                    "field_divergence": div,
                })
        candidates.sort(key=lambda x: -x["similarity"])

        # Clusters: union по labels_a/labels_b
        clusters_map: dict[int, dict] = {}
        for i in range(n_a):
            lab = labels_a[i]
            clusters_map.setdefault(lab, {"members": []})["members"].append(
                {"source": "A", "row": i, "id": global_to_id_a[idx_a[i]]},
            )
        for j in range(n_b):
            lab = labels_b[j]
            clusters_map.setdefault(lab, {"members": []})["members"].append(
                {"source": "B", "row": j, "id": global_to_id_b[idx_b[j]]},
            )
        clusters: list[dict] = []
        for lab, c in clusters_map.items():
            members = c["members"]
            if len(members) > 1:
                # similarity внутри кластера — средняя cross-table
                sims = []
                a_idxs = [m["row"] for m in members if m["source"] == "A"]
                b_idxs = [m["row"] for m in members if m["source"] == "B"]
                for ai in a_idxs:
                    for bj in b_idxs:
                        sims.append(float(sim[ai, bj]))
                cluster_sim = float(np.mean(sims)) if sims else 1.0
            else:
                cluster_sim = 1.0
            clusters.append({
                "id": f"C-{lab + 1:03d}",
                "members": members,
                "similarity": cluster_sim,
                "needs_review": any(c["verdict"] == "review"
                                     for c in candidates
                                     if c.get("cluster_id") == f"C-{lab + 1:03d}"),
            })
        clusters.sort(key=lambda c: (-len(c["members"]), c["id"]))

        n_pairs = sum(1 for c in candidates if c["verdict"] in ("auto", "review"))
        n_multi = sum(1 for c in clusters if len(c["members"]) > 1)
        _log(run_id, "ok",
             f"{n_multi} multi-row clusters · {len(clusters) - n_multi} singletons")
        _log(run_id, "ok",
             f"{n_pairs} candidate pairs · "
             f"{sum(1 for c in candidates if c['verdict']=='review')} require review")
        _publish(run_id, {"type": "progress", "phase": "cluster", "progress": 1.0})

        run.candidates = candidates
        run.clusters = clusters
        # t_total_ms = сумма стадий сборки графа (из run.metrics) + GAT forward.
        t_gat_ms = round(forward_ms, 1)
        t_total_ms = round(
            run.metrics.get("t_row_embeddings_ms", 0.0)
            + run.metrics.get("t_col_descriptions_ms", 0.0)
            + run.metrics.get("t_col_embeddings_ms", 0.0)
            + t_gat_ms,
            1,
        )
        run.metrics.update({
            "n_pairs_found": n_pairs,
            "n_clusters": len(clusters),
            "n_input_rows": n_a + n_b,
            "latency_ms": int(forward_ms),
            "t_gat_ms": t_gat_ms,
            "t_total_ms": t_total_ms,
            "threshold": similarity_threshold,
        })
        run.status = "done"
        _publish(run_id, {"type": "metric", "key": "n_pairs", "value": n_pairs})
        _publish(run_id, {"type": "done", "result_url": f"/api/runs/{run_id}/clusters"})

    except FileNotFoundError as e:
        run.status = "error"
        run.error = str(e)
        _publish(run_id, {"type": "error", "msg": f"checkpoint missing: {e}"})
    except Exception as e:
        logger.exception("run_inference failed")
        run.status = "error"
        run.error = str(e)
        _publish(run_id, {"type": "error", "msg": str(e)})


# --------------------------- background launcher ------------------------- #


def launch_build_graph(run_id: str, **kwargs: Any) -> None:
    t = threading.Thread(target=build_graph_run, args=(run_id,), kwargs=kwargs,
                         daemon=True, name=f"build_{run_id}")
    t.start()


def launch_inference(run_id: str, **kwargs: Any) -> None:
    t = threading.Thread(target=run_inference, args=(run_id,), kwargs=kwargs,
                         daemon=True, name=f"infer_{run_id}")
    t.start()


# ------------------------------ utilities -------------------------------- #


def umap_project(embeddings: torch.Tensor | np.ndarray, n_components: int = 2,
                 method: str = "umap") -> np.ndarray:
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    n = len(embeddings)
    if n <= n_components + 1:
        # Тривиально: для < 4 точек просто возьмём первые компоненты.
        return embeddings[:, :n_components].astype(np.float32)

    if method == "umap":
        try:
            import umap  # type: ignore
            reducer = umap.UMAP(n_components=n_components, n_neighbors=min(15, n - 1),
                                metric="cosine", random_state=42)
            return reducer.fit_transform(embeddings).astype(np.float32)
        except Exception as e:
            logger.warning("UMAP failed (%s) — fallback на PCA", e)

    # PCA fallback
    from sklearn.decomposition import PCA
    return PCA(n_components=n_components, random_state=42).fit_transform(
        embeddings,
    ).astype(np.float32)
