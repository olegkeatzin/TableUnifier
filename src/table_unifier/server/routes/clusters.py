"""Маршруты обзора кластеров и пользовательских решений."""

from __future__ import annotations

import logging

import numpy as np
from fastapi import APIRouter, HTTPException

from table_unifier.server.models import DecisionsRequest
from table_unifier.server.services.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["clusters"])


@router.get("/runs/{run_id}/clusters")
def get_clusters(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    if not run.candidates and run.status != "done":
        raise HTTPException(409, detail=f"inference not done (status={run.status})")

    # Histogram for similarity (20 bins, 0..1)
    sim = run.similarity_matrix
    if sim is not None:
        flat = np.clip(sim.flatten(), 0.0, 1.0)
        bins, _ = np.histogram(flat, bins=20, range=(0.0, 1.0))
        histogram = bins.astype(int).tolist()
    else:
        histogram = [0] * 20

    return {
        "candidates": run.candidates,
        "clusters": run.clusters,
        "metrics": run.metrics,
        "histogram": histogram,
        "table_a": _table_payload(run, "A"),
        "table_b": _table_payload(run, "B"),
        "model_tag": run.model_tag,
    }


def _table_payload(run, side: str) -> dict:
    df = run.table_a if side == "A" else run.table_b
    src_id = run.source_a_id if side == "A" else run.source_b_id
    session = store.get_session(run.session_id)
    name = session.sources[src_id].name if session and src_id in session.sources else src_id
    cols = [c for c in df.columns if c != "id"]
    sample = df.head(200).astype(object).where(df.head(200).notna(), None)
    return {"id": side, "name": name, "rows": len(df), "cols": cols,
            "data": sample[cols].values.tolist()}


@router.post("/runs/{run_id}/clusters/decisions")
def post_decisions(run_id: str, req: DecisionsRequest) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    for d in req.decisions:
        run.decisions[d.pair_id] = d.verdict
    return {"status": "ok", "n_decisions": len(run.decisions)}


@router.get("/runs/{run_id}/decisions")
def get_decisions(run_id: str) -> dict:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    return {"decisions": run.decisions}
