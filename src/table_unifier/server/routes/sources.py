"""Маршруты загрузки источников (.xlsx / .csv / .parquet) и обзора схем."""

from __future__ import annotations

import io
import logging
from typing import Annotated

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from table_unifier.server.models import (
    SchemaDivergenceItem,
    SourceInfo,
    UploadResponse,
)
from table_unifier.server.services.inference import _name_only_divergence
from table_unifier.server.services.storage import store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/sources", tags=["sources"])


def _read_table(filename: str, raw: bytes) -> pd.DataFrame:
    lower = filename.lower()
    bio = io.BytesIO(raw)
    if lower.endswith(".xlsx") or lower.endswith(".xls"):
        return pd.read_excel(bio)
    if lower.endswith(".csv"):
        bio.seek(0)
        try:
            return pd.read_csv(bio)
        except UnicodeDecodeError:
            bio.seek(0)
            return pd.read_csv(bio, encoding="cp1251")
    if lower.endswith(".parquet"):
        return pd.read_parquet(bio)
    raise HTTPException(415, detail=f"Unsupported file format: {filename}")


def _source_info(src) -> SourceInfo:
    df = src.df
    sample = df.head(50).astype(object).where(df.head(50).notna(), None).values.tolist()
    return SourceInfo(
        id=src.id, name=src.name, rows=len(df), cols=src.cols,
        size_bytes=src.size_bytes, sample=sample,
        dtypes={c: str(df[c].dtype) for c in df.columns},
    )


@router.post("/upload", response_model=UploadResponse)
async def upload(
    files: Annotated[list[UploadFile], File(...)],
    session_id: Annotated[str | None, Form()] = None,
) -> UploadResponse:
    if not files:
        raise HTTPException(400, detail="No files uploaded")

    session = store.get_session(session_id) if session_id else None
    if session is None:
        session = store.new_session()

    for f in files:
        raw = await f.read()
        try:
            df = _read_table(f.filename or "upload", raw)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(400,
                                detail=f"failed to parse {f.filename}: {e}") from e
        store.add_source(session, f.filename or "upload", df, len(raw))

    sources = [_source_info(s) for s in session.sources.values()]

    divergence: list[SchemaDivergenceItem] = []
    src_list = list(session.sources.values())
    if len(src_list) >= 2:
        # Дешёвая name-based эвристика. Полноценное сравнение column embeddings
        # (qwen3) делается на шаге graph/build — там оно уже нужно для рёбер.
        cols_a = src_list[0].cols
        cols_b = src_list[1].cols
        try:
            items = _name_only_divergence(cols_a, cols_b, src_list[0].df, src_list[1].df)
            for it in items:
                divergence.append(SchemaDivergenceItem(**it))
        except Exception as e:
            logger.warning("schema divergence failed: %s", e)

    return UploadResponse(session_id=session.id, sources=sources, divergence=divergence)


@router.get("/{session_id}", response_model=UploadResponse)
def list_sources(session_id: str) -> UploadResponse:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(404, detail="session not found")
    return UploadResponse(
        session_id=session.id,
        sources=[_source_info(s) for s in session.sources.values()],
    )


@router.delete("/{session_id}/{source_id}")
def delete_source(session_id: str, source_id: str) -> dict[str, str]:
    session = store.get_session(session_id)
    if session is None:
        raise HTTPException(404, detail="session not found")
    session.sources.pop(source_id, None)
    return {"status": "deleted"}
