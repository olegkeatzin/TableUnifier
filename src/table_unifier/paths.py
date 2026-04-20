"""Хелперы для namespaced-раскладки данных и артефактов.

Column embeddings (qwen3) не зависят от токен-модели → лежат в shared-каталоге.
Row embeddings и графы зависят от токен-модели → namespaced по model_tag.

Раскладка::

    data/embeddings/
        columns/<dataset>/
            column_embeddings_a.npz
            column_embeddings_b.npz
            columns_a.csv
            columns_b.csv
        rows/<model_tag>/<dataset>/
            row_embeddings_a.npy
            row_embeddings_b.npy
    data/graphs/<model_tag>/
        <dataset>/ ...           # per-dataset graphs (exp 02)
        v3_unified/ ...
        v14_mrl/ ...
        v14_mrl_cross/<dataset>/ ...
    output/<model_tag>/
        v14_mrl_gat_model.pt
        ...

Функции принимают ``Path`` и строку ``model_tag``.
"""

from __future__ import annotations

from pathlib import Path


def columns_dir(data_dir: Path, dataset: str | None = None) -> Path:
    """Shared каталог column embeddings (qwen3)."""
    p = data_dir / "embeddings" / "columns"
    return p / dataset if dataset else p


def rows_dir(data_dir: Path, model_tag: str, dataset: str | None = None) -> Path:
    """Per-model row embeddings."""
    p = data_dir / "embeddings" / "rows" / model_tag
    return p / dataset if dataset else p


def graphs_root(data_dir: Path, model_tag: str) -> Path:
    """Корень графов конкретной токен-модели."""
    return data_dir / "graphs" / model_tag


def graph_dir(data_dir: Path, model_tag: str, dataset: str) -> Path:
    """Per-dataset граф."""
    return graphs_root(data_dir, model_tag) / dataset


def unified_dir(data_dir: Path, model_tag: str, variant: str = "v14_mrl") -> Path:
    """Unified граф (v3_unified / v14_mrl / ...)."""
    return graphs_root(data_dir, model_tag) / variant


def output_dir_for(output_dir: Path, model_tag: str) -> Path:
    """Каталог моделей (output/) для model_tag."""
    return output_dir / model_tag
