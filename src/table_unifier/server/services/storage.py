"""In-memory state для сессий и runs.

Session = загруженные файлы.
Run = построенный граф + (опционально) результаты инференса.
"""

from __future__ import annotations

import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass
class SourceState:
    id: str
    name: str
    df: pd.DataFrame
    size_bytes: int

    @property
    def cols(self) -> list[str]:
        return [c for c in self.df.columns if c != "id"]


@dataclass
class SessionState:
    id: str
    sources: dict[str, SourceState] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)


@dataclass
class RunState:
    id: str
    session_id: str
    source_ids: list[str]
    model_tag: str = "bge-m3"

    # Артефакты построения графа.
    graph: Any = None  # HeteroData
    id_to_global_a: dict[str, int] = field(default_factory=dict)
    id_to_global_b: dict[str, int] = field(default_factory=dict)
    table_a: pd.DataFrame | None = None
    table_b: pd.DataFrame | None = None
    source_a_id: str = ""
    source_b_id: str = ""

    # Инференс.
    row_embeddings: Any = None  # torch.Tensor [N, D]
    similarity_matrix: Any = None  # [N_a, N_b]
    # Внимание GAT на рёбрах token→row, агрегированное по головам и слоям.
    # np.ndarray [n_edges], выровнено с graph["token","in_row","row"].edge_index.
    edge_attention: Any = None
    candidates: list[dict] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)

    # Пользовательские решения {pair_id: "approve"|"reject"}
    decisions: dict[str, str] = field(default_factory=dict)

    status: str = "created"
    error: str | None = None
    created_at: float = field(default_factory=time.time)


class _Store:
    def __init__(self) -> None:
        self.sessions: dict[str, SessionState] = {}
        self.runs: dict[str, RunState] = {}
        self.lock = threading.Lock()
        self.artifacts_root = Path("output") / "server_runs"

    def new_session(self) -> SessionState:
        sid = "sess_" + secrets.token_hex(4)
        s = SessionState(id=sid)
        with self.lock:
            self.sessions[sid] = s
        return s

    def get_session(self, sid: str) -> SessionState | None:
        return self.sessions.get(sid)

    def add_source(self, session: SessionState, name: str, df: pd.DataFrame,
                   size_bytes: int) -> SourceState:
        src_id = "src_" + secrets.token_hex(3)
        # гарантируем наличие колонки id (используется build_graph)
        if "id" not in df.columns:
            df = df.copy()
            df.insert(0, "id", [f"{src_id}_{i}" for i in range(len(df))])
        src = SourceState(id=src_id, name=name, df=df, size_bytes=size_bytes)
        with self.lock:
            session.sources[src_id] = src
        return src

    def new_run(self, session_id: str, source_ids: list[str],
                model_tag: str = "bge-m3") -> RunState:
        rid = "run_" + secrets.token_hex(5)
        run = RunState(
            id=rid, session_id=session_id, source_ids=source_ids, model_tag=model_tag,
        )
        with self.lock:
            self.runs[rid] = run
        return run

    def get_run(self, rid: str) -> RunState | None:
        return self.runs.get(rid)

    def artifacts_dir(self, run_id: str) -> Path:
        p = self.artifacts_root / run_id
        p.mkdir(parents=True, exist_ok=True)
        return p


store = _Store()
