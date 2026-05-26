"""Pydantic-модели запросов и ответов для веб-API инференса."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ------------------------------ Sources ---------------------------------- #


class SourceInfo(BaseModel):
    id: str
    name: str
    rows: int
    cols: list[str]
    size_bytes: int
    sample: list[list[Any]] = Field(default_factory=list)
    dtypes: dict[str, str] = Field(default_factory=dict)


class SchemaDivergenceItem(BaseModel):
    a_col: str
    b_col: str
    similarity: float
    kind: Literal["renamed", "format", "exact"]
    note: str = ""


class UploadResponse(BaseModel):
    session_id: str
    sources: list[SourceInfo]
    divergence: list[SchemaDivergenceItem] = Field(default_factory=list)


# ------------------------------ Graph ------------------------------------ #


class BuildGraphRequest(BaseModel):
    session_id: str
    source_ids: list[str]
    model_tag: str = "bge-m3"
    idf_min_df: int = 2
    max_token_df: float = 0.3
    max_tokens_per_cell: int = 16
    target_col_dim: int = 1024


class RunStartedResponse(BaseModel):
    run_id: str
    status: str = "started"


# ------------------------------ Infer ------------------------------------ #


class InferRequest(BaseModel):
    run_id: str
    checkpoint: str = "output/bge-m3/v14_mrl_gat_model.pt"
    similarity_threshold: float = 0.831
    use_ga_tuning: bool = False
    top_k: int = 10


class SinglePairRequest(BaseModel):
    run_id: str
    a: str
    b: str
    threshold: float = 0.831


# --------------------------- Clusters / Review --------------------------- #


class Decision(BaseModel):
    pair_id: str
    verdict: Literal["approve", "reject"]


class DecisionsRequest(BaseModel):
    decisions: list[Decision]


# ----------------------------- Errors ------------------------------------ #


class ErrorResponse(BaseModel):
    detail: str
