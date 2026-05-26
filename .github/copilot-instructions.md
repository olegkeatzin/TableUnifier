# GitHub Copilot Instructions — TableUnifier

## Build & Test

```bash
uv sync                        # install dependencies (use uv, never pip)
uv run pytest                  # all unit tests
uv run pytest tests/test_losses.py          # single file
uv run pytest -k "test_triplet"             # single test
uv run python _integration_test.py         # end-to-end (requires Ollama)
uv run python -m table_unifier.server.main  # web app on :8000
```

Python **3.12** only. Package manager: **uv**. `pyg-lib` loaded from `wheels/` (local `.whl`).

## Architecture

Three-stage pipeline: **CSV → embeddings → hetero graph → GNN → cosine dedup**

- **Row embeddings**: `TokenEmbedder` wraps a HuggingFace model (default `cointegrated/rubert-tiny2`, best result: `BAAI/bge-m3`); pooling `cls` or `mean`.
- **Column embeddings**: Ollama `qwen3-embedding:8b` (4096-dim), shared across models, stored under `data/embeddings/columns/`.
- **Graph**: `HeteroData` with `row` nodes and `token` nodes; `token→row` edges carry column embeddings as attributes.
- **Models**: `EntityResolutionGNN` (EdgeMeanConv) or `EntityResolutionGAT` (GATv2Conv + GraphNorm). Both live in `src/table_unifier/models/entity_resolution.py`. Output: L2-normalised row embeddings.
- **Training**: `train_entity_resolution_minibatch` in `training/er_trainer.py` — unified graph, NT-Xent loss. This is the main path; round-robin (`train_entity_resolution_multidataset`) is legacy baseline.

## Path Layout

Use helpers from `src/table_unifier/paths.py` — never construct paths manually:

| Helper | Output |
|---|---|
| `columns_dir(data_dir, dataset?)` | `data/embeddings/columns/[dataset]` |
| `rows_dir(data_dir, model_tag, dataset?)` | `data/embeddings/rows/<model_tag>/[dataset]` |
| `unified_dir(data_dir, model_tag, variant)` | `data/graphs/<model_tag>/<variant>` |
| `output_dir_for(output_dir, model_tag)` | `output/<model_tag>` |

`model_tag` comes from `EntityResolutionConfig.token_model_tag` (default `rubert-tiny2`, current best `bge-m3`).

## Config System

All parameters in dataclasses in `src/table_unifier/config.py`:
- `OllamaConfig` — `host`, `llm_model` (`qwen3.5:9b`), `embedding_model` (`qwen3-embedding:8b`)
- `SchemaMatchingConfig` — dims, lr, epochs for `SchemaProjector`
- `EntityResolutionConfig` — `row_dim/token_dim/col_dim`, `hidden_dim=128`, `output_dim=128`, `num_heads=4`, `temperature=0.1`

Use `@dataclass` with `field(default_factory=...)` for mutable defaults. Never hardcode dimension values — read them from the config.

## Code Style

- Type hints everywhere; `from __future__ import annotations` for forward refs.
- Docstrings in **Russian** with `Args:` / `Returns:` blocks.
- `Path` objects throughout (not raw strings).
- `np.random.default_rng(42)` — new-style NumPy RNG only.
- Section comments in Russian: `# ---- 1. Проекционные слои ---- #`.

## Testing Conventions

Fixtures in `tests/conftest.py`:
- `small_hetero_data` — 6 rows, 10 tokens, 20 edges; dims **32/32/64** (row/token/col). **Always use these dims in new model tests**, not production defaults (312/312/4096).
- `table_a`, `table_b` — 3-row DataFrames with `id, title, brand, price`.
- `column_embeddings` — random 64-dim float32 dict.

New tests for models: pass `row_dim=32, token_dim=32, col_dim=64` explicitly to constructors.

## Key Modules

| Path | Purpose |
|---|---|
| `src/table_unifier/config.py` | All configs |
| `src/table_unifier/paths.py` | Namespaced path helpers |
| `src/table_unifier/models/entity_resolution.py` | `EntityResolutionGNN`, `EntityResolutionGAT` |
| `src/table_unifier/models/gnn_layer.py` | `GNNLayer`, `EdgeMeanConv` |
| `src/table_unifier/models/gat_layer.py` | `GATLayer` (GATv2 + edge features) |
| `src/table_unifier/models/losses.py` | `TripletLoss`, `nt_xent_loss`, `mine_semi_hard` |
| `src/table_unifier/training/er_trainer.py` | `train_entity_resolution_minibatch`, `get_row_embeddings`, `find_duplicates` |
| `src/table_unifier/dataset/graph_builder.py` | `HeteroData` construction with IDF token filtering |
| `src/table_unifier/server/main.py` | FastAPI inference server |

## External Services

- **Ollama**: must be running at `OllamaConfig.host` for column embedding generation. In this project Ollama/GPU is on a remote nvidia-server — do not start it locally.
- **MLflow**: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db` → http://localhost:5000
- **Data sync**: `rclone` ↔ Яндекс Диск. `data/`, `output/`, `mlflow.db` are excluded from git.
