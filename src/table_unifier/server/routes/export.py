"""Маршруты экспорта унифицированной таблицы."""

from __future__ import annotations

import io
import json
import logging
from collections import Counter

import pandas as pd
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from table_unifier.server.services.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["export"])


def _build_unified(run) -> pd.DataFrame:
    """Собрать унифицированную таблицу из кластеров с учётом decisions."""
    decisions = run.decisions
    rejected_pairs = {pid for pid, v in decisions.items() if v == "reject"}

    cols_a = [c for c in run.table_a.columns if c != "id"]
    cols_b = [c for c in run.table_b.columns if c != "id"]
    all_cols = list(dict.fromkeys(cols_a + cols_b))

    rows_out = []

    # Если ребро rejected — разбиваем пару: их кластеры распадаются на синглтоны.
    rejected_globals = set()
    for pid in rejected_pairs:
        cand = next((c for c in run.candidates if c["id"] == pid), None)
        if cand is None:
            continue
        rejected_globals.add((cand["a_idx"], cand["b_idx"]))

    for c in run.clusters:
        members = c["members"]
        if len(members) > 1 and rejected_globals:
            # Если есть отклонённые пары внутри кластера — режем на синглтоны.
            has_reject = any(
                (ai["row"], bj["row"]) in rejected_globals
                for ai in members if ai["source"] == "A"
                for bj in members if bj["source"] == "B"
            )
        else:
            has_reject = False

        if has_reject:
            for m in members:
                rows_out.append(_singleton_row(run, m, c["id"]))
        else:
            rows_out.append(_canonical_row(run, c, all_cols))

    df = pd.DataFrame(rows_out)
    return df


def _singleton_row(run, member: dict, cluster_id: str) -> dict:
    df = run.table_a if member["source"] == "A" else run.table_b
    row = df[df["id"].astype(str) == str(member["id"])].iloc[0]
    out = {c: row.get(c, None) for c in df.columns if c != "id"}
    out["cluster_id"] = cluster_id
    out["source_ids"] = f"{member['source']}:{member['id']}"
    out["n_members"] = 1
    out["confidence"] = 1.0
    return out


def _canonical_row(run, cluster: dict, all_cols: list[str]) -> dict:
    """Простой merge: mode для категориальных, mean для числовых."""
    members = cluster["members"]
    rows: list[pd.Series] = []
    src_ids: list[str] = []
    for m in members:
        df = run.table_a if m["source"] == "A" else run.table_b
        r = df[df["id"].astype(str) == str(m["id"])].iloc[0]
        rows.append(r)
        src_ids.append(f"{m['source']}:{m['id']}")
    out: dict = {}
    for col in all_cols:
        values = [r.get(col) for r in rows if col in r.index and pd.notna(r.get(col))]
        if not values:
            out[col] = None
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in values):
            out[col] = float(sum(values)) / len(values)
        else:
            out[col] = Counter(map(str, values)).most_common(1)[0][0]
    out["cluster_id"] = cluster["id"]
    out["source_ids"] = ", ".join(src_ids)
    out["n_members"] = len(members)
    out["confidence"] = float(cluster.get("similarity", 1.0))
    return out


@router.get("/runs/{run_id}/unified.{fmt}")
def export_unified(run_id: str, fmt: str) -> Response:
    run = store.get_run(run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    if run.status != "done":
        raise HTTPException(409, detail=f"inference not done (status={run.status})")

    df = _build_unified(run)
    fmt = fmt.lower()
    name = f"unified_{run_id}.{fmt}"
    if fmt == "csv":
        body = df.to_csv(index=False).encode("utf-8")
        media = "text/csv"
    elif fmt == "json":
        body = json.dumps(df.to_dict(orient="records"),
                          ensure_ascii=False, default=str).encode("utf-8")
        media = "application/json"
    elif fmt == "xlsx":
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="openpyxl") as w:
            df.to_excel(w, index=False, sheet_name="unified")
        body = buf.getvalue()
        media = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    elif fmt == "parquet":
        buf = io.BytesIO()
        df.to_parquet(buf, index=False)
        body = buf.getvalue()
        media = "application/octet-stream"
    else:
        raise HTTPException(415, detail=f"unsupported format: {fmt}")

    return Response(
        content=body, media_type=media,
        headers={"Content-Disposition": f'attachment; filename="{name}"'},
    )
