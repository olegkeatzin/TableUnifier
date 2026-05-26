"""Маршруты инференса + WebSocket-стрим прогресса."""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect

from table_unifier.server.models import (
    InferRequest,
    RunStartedResponse,
    SinglePairRequest,
)
from table_unifier.server.services.inference import launch_inference
from table_unifier.server.services.progress import bus
from table_unifier.server.services.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["infer"])


@router.post("/infer/run", response_model=RunStartedResponse)
def run_infer(req: InferRequest) -> RunStartedResponse:
    run = store.get_run(req.run_id)
    if run is None:
        raise HTTPException(404, detail="run not found")
    if run.graph is None:
        raise HTTPException(409, detail=f"graph not ready (status={run.status})")
    launch_inference(
        req.run_id,
        checkpoint=req.checkpoint,
        similarity_threshold=req.similarity_threshold,
        use_ga_tuning=req.use_ga_tuning,
        top_k=req.top_k,
    )
    return RunStartedResponse(run_id=req.run_id, status="started")


@router.post("/infer/single_pair")
def single_pair(req: SinglePairRequest) -> dict:
    run = store.get_run(req.run_id)
    if run is None or run.similarity_matrix is None:
        raise HTTPException(409, detail="inference not done")
    sim = run.similarity_matrix
    # a/b are row labels like "A3" / "B7"
    try:
        ai = int(req.a[1:])
        bj = int(req.b[1:])
    except ValueError:
        raise HTTPException(400, detail="row ids must be like 'A3' / 'B7'")
    s = float(sim[ai, bj])
    return {"similarity": s, "match": s >= req.threshold}


@router.websocket("/ws/runs/{run_id}/stream")
async def ws_stream(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    bus.attach_loop(asyncio.get_event_loop())
    if store.get_run(run_id) is None:
        await websocket.send_json({"type": "error", "msg": "run not found"})
        await websocket.close()
        return
    queue = await bus.subscribe(run_id)
    try:
        while True:
            ev = await queue.get()
            await websocket.send_json(ev)
            if ev.get("type") in ("done", "error"):
                # Не закрываем сразу — клиент может ещё подтянуть оставшиеся.
                await asyncio.sleep(0.05)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.warning("ws_stream error: %s", e)
    finally:
        bus.unsubscribe(run_id, queue)
