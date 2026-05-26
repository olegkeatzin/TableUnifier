"""Маршруты построения и обзора графа."""

from __future__ import annotations

import logging
import math

import numpy as np
from fastapi import APIRouter, HTTPException

from table_unifier.server.models import BuildGraphRequest, RunStartedResponse
from table_unifier.server.services.inference import launch_build_graph, umap_project
from table_unifier.server.services.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["graph"])


@router.post("/graph/build", response_model=RunStartedResponse)
def build_graph(req: BuildGraphRequest) -> RunStartedResponse:
    session = store.get_session(req.session_id)
    if session is None:
        raise HTTPException(404, detail="session not found")
    src_ids = req.source_ids or list(session.sources.keys())
    if len(src_ids) < 2:
        raise HTTPException(400, detail="need at least 2 sources")
    run = store.new_run(req.session_id, src_ids[:2], model_tag=req.model_tag)
    launch_build_graph(
        run.id,
        idf_min_df=req.idf_min_df,
        max_token_df=req.max_token_df,
        max_tokens_per_cell=req.max_tokens_per_cell,
        target_col_dim=req.target_col_dim,
    )
    return RunStartedResponse(run_id=run.id, status="started")


@router.get("/runs/{run_id}/graph")
def get_graph(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    if run.graph is None:
        raise HTTPException(409, detail=f"graph not ready (status={run.status})")

    graph = run.graph
    n_rows = int(graph["row"].x.shape[0])
    n_tokens = int(graph["token"].x.shape[0])
    edge_index = graph["token", "in_row", "row"].edge_index.cpu().numpy()
    edge_col_idx = graph["token", "in_row", "row"].edge_col_idx.cpu().numpy()

    cols_a = [c for c in run.table_a.columns if c != "id"]
    cols_b = [c for c in run.table_b.columns if c != "id"]
    n_a = len(run.id_to_global_a)

    # Простой layout: A слева, B справа, токены в середине рандомно (детерминированно).
    rng = np.random.default_rng(42)
    rows_out = []
    global_to_id_a = {v: k for k, v in run.id_to_global_a.items()}
    global_to_id_b = {v: k for k, v in run.id_to_global_b.items()}
    for global_idx in range(n_rows):
        if global_idx in global_to_id_a:
            src = "A"
            row_id = global_to_id_a[global_idx]
            local_idx = sorted(run.id_to_global_a.values()).index(global_idx)
            x_init = 0.20
            y_init = 0.10 + 0.80 * (local_idx + 0.5) / max(1, n_a)
            row_df = run.table_a[run.table_a["id"].astype(str) == str(row_id)].iloc[0]
            cols = cols_a
        else:
            src = "B"
            row_id = global_to_id_b[global_idx]
            local_idx = sorted(run.id_to_global_b.values()).index(global_idx)
            x_init = 0.80
            y_init = 0.10 + 0.80 * (local_idx + 0.5) / max(
                1, len(run.id_to_global_b),
            )
            row_df = run.table_b[run.table_b["id"].astype(str) == str(row_id)].iloc[0]
            cols = cols_b
        label = " ".join(str(row_df.get(c, "")) for c in cols[:3] if row_df.get(c, "") )
        rows_out.append({
            "id": f"{src}{local_idx}",
            "global": global_idx,
            "source": src,
            "label": label[:60],
            "cols": {c: (None if (v is None or (isinstance(v, float) and math.isnan(v))) else v)
                     for c, v in row_df.to_dict().items()},
            "x_init": x_init, "y_init": y_init,
        })

    # Декодируем токен-IDs обратно в текст через cached tokenizer.
    from table_unifier.server.services.inference import _get_token_embedder
    token_ids = getattr(graph, "token_ids", None)
    token_texts: list[str] = []
    if token_ids is not None:
        try:
            emb = _get_token_embedder(run.model_tag)
            for tid in token_ids:
                txt = emb.tokenizer.decode([int(tid)], skip_special_tokens=True).strip()
                token_texts.append(txt if txt else f"#{tid}")
        except Exception as e:
            logger.warning("token decode failed: %s", e)
            token_texts = [f"#{int(t)}" for t in token_ids]
    else:
        token_texts = [f"#{ti}" for ti in range(n_tokens)]

    tokens_out = []
    for ti in range(n_tokens):
        x = 0.40 + 0.20 * float(rng.random())
        y = float(rng.random())
        tokens_out.append({
            "id": f"t_{ti}", "text": token_texts[ti] if ti < len(token_texts) else f"#{ti}",
            "df": 0, "x_init": x, "y_init": y,
        })

    # token document frequency
    df_counts = np.zeros(n_tokens, dtype=int)
    for t in edge_index[0]:
        df_counts[int(t)] += 1
    for i, tok in enumerate(tokens_out):
        tok["df"] = int(df_counts[i])

    edges_out = []
    # Имена столбцов: build_graph сохраняет их в graph.col_names в правильном
    # порядке (соответствует col_to_idx). Если поле отсутствует — fallback.
    n_cols = int(graph.col_embeddings.shape[0])
    col_names: list[str] = list(getattr(graph, "col_names", []) or [])
    while len(col_names) < n_cols:
        col_names.append(f"col_{len(col_names)}")

    src_arr = edge_index[0]
    dst_arr = edge_index[1]
    for k in range(edge_index.shape[1]):
        col_name = col_names[int(edge_col_idx[k])]
        src_token = int(src_arr[k])
        dst_row = int(dst_arr[k])
        edges_out.append({
            "row": rows_out[dst_row]["id"],
            "token": tokens_out[src_token]["id"],
            "col": col_name, "weight": 1.0,
        })

    return {"rows": rows_out, "tokens": tokens_out, "edges": edges_out,
            "stats": {"n_rows": n_rows, "n_tokens": n_tokens,
                       "n_edges": edge_index.shape[1],
                       "col_dim": int(graph.col_embeddings.shape[1])}}


@router.get("/runs/{run_id}/embeddings")
def get_embeddings(run_id: str, method: str = "umap") -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    if run.row_embeddings is None:
        raise HTTPException(409, detail="embeddings not ready")
    n_a = len(run.id_to_global_a)
    idx_a = sorted(run.id_to_global_a.values())
    idx_b = sorted(run.id_to_global_b.values())
    emb = run.row_embeddings.numpy()
    emb_ab = np.concatenate([emb[idx_a], emb[idx_b]], axis=0)
    proj = umap_project(emb_ab, n_components=2, method=method)
    # Нормализация в [0, 1] для удобства фронта.
    mn = proj.min(axis=0)
    mx = proj.max(axis=0)
    rng = (mx - mn).clip(min=1e-6)
    proj = (proj - mn) / rng

    candidate_cluster = {(c["a_idx"], c["b_idx"]): c.get("cluster_id")
                          for c in run.candidates}

    points = []
    for i in range(len(idx_a)):
        points.append({"row_id": f"A{i}", "source": "A",
                        "x": float(proj[i, 0]), "y": float(proj[i, 1]),
                        "cluster": next((v for (ai, _), v in candidate_cluster.items()
                                         if ai == i and v), None)})
    for j in range(len(idx_b)):
        points.append({"row_id": f"B{j}", "source": "B",
                        "x": float(proj[n_a + j, 0]),
                        "y": float(proj[n_a + j, 1]),
                        "cluster": next((v for (_, bj), v in candidate_cluster.items()
                                         if bj == j and v), None)})
    return {"method": method, "points": points}
