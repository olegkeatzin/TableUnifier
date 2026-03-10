"""Temporary integration test script."""
from pathlib import Path

# 1. Dataset download
print("=== 1. DATASET DOWNLOAD ===")
from table_unifier.dataset.download import download_dataset, load_dataset

data_dir = Path("data")
path = download_dataset("beer", data_dir)
print(f"Downloaded to: {path}")
ds = load_dataset(path, name="beer")
for k, df in ds.items():
    print(f"  {k}: shape={df.shape}, cols={list(df.columns)}")

tableA = ds["tableA"]
tableB = ds["tableB"]
print(f"\ntableA sample:\n{tableA.head(2)}")
print(f"\ntableB sample:\n{tableB.head(2)}")

# 2. Ollama: column description + embedding
print("\n=== 2. OLLAMA: COLUMN EMBEDDINGS ===")
from table_unifier.dataset.embedding_generation import generate_column_embeddings
from table_unifier.ollama_client import OllamaClient

client = OllamaClient()
col_emb = generate_column_embeddings(client, tableA)
for name, vec in col_emb.items():
    print(f"  col '{name}': shape={vec.shape}, dtype={vec.dtype}")

# 3. ruBERT
print("\n=== 3. ruBERT: TokenEmbedder ===")
from table_unifier.dataset.embedding_generation import TokenEmbedder, serialize_row

embedder = TokenEmbedder()
print(f"  hidden_dim={embedder.hidden_dim}")

texts = [serialize_row(tableA.iloc[0]), serialize_row(tableA.iloc[1])]
print(f"  Serialized row 0: '{texts[0][:80]}...'")
cls_embs = embedder.embed_sentences(texts)
print(f"  CLS embeddings shape: {cls_embs.shape}")

tokens = embedder.get_token_ids("iPhone 13 Pro")
print(f"  Token IDs for 'iPhone 13 Pro': {tokens}")
vocab_emb = embedder.get_vocab_embedding(tokens[0])
print(f"  Vocab embedding shape: {vocab_emb.shape}")

# 4. Graph building
print("\n=== 4. GRAPH BUILDING ===")
from table_unifier.dataset.graph_builder import build_graph

data, id2a, id2b = build_graph(tableA, tableB, col_emb, embedder)
print(f"  row.x: {data['row'].x.shape}")
print(f"  token.x: {data['token'].x.shape}")
print(f"  edge (t->r): {data['token', 'in_row', 'row'].edge_index.shape}")
print(f"  edge_attr: {data['token', 'in_row', 'row'].edge_attr.shape}")
print(f"  edge (r->t): {data['row', 'has_token', 'token'].edge_index.shape}")
print(f"  id_to_global_a: {id2a}")
print(f"  id_to_global_b: {id2b}")

# 5. Dimensions check
print("\n=== 5. DIMENSION CHECK ===")
row_dim = data["row"].x.shape[1]
token_dim = data["token"].x.shape[1]
col_dim = data["token", "in_row", "row"].edge_attr.shape[1]
print(f"  row_dim={row_dim} (expected 312)")
print(f"  token_dim={token_dim} (expected 312)")
print(f"  col_dim={col_dim} (expected 4096)")
assert row_dim == 312, f"row_dim mismatch: {row_dim}"
assert token_dim == 312, f"token_dim mismatch: {token_dim}"
assert col_dim == 4096, f"col_dim mismatch: {col_dim}"

# 6. Model forward pass with real dims
print("\n=== 6. MODEL FORWARD PASS ===")
from table_unifier.models.entity_resolution import EntityResolutionGNN
from table_unifier.config import EntityResolutionConfig

er_cfg = EntityResolutionConfig()
print(f"  Config: row_dim={er_cfg.row_dim}, token_dim={er_cfg.token_dim}, col_dim={er_cfg.col_dim}")
print(f"  hidden={er_cfg.hidden_dim}, edge={er_cfg.edge_dim}, output={er_cfg.output_dim}")
print(f"  layers={er_cfg.num_gnn_layers}, heads={er_cfg.num_heads}")

model = EntityResolutionGNN(
    row_dim=er_cfg.row_dim, token_dim=er_cfg.token_dim,
    col_dim=er_cfg.col_dim, hidden_dim=er_cfg.hidden_dim,
    edge_dim=er_cfg.edge_dim, output_dim=er_cfg.output_dim,
    num_gnn_layers=er_cfg.num_gnn_layers, num_heads=er_cfg.num_heads,
)
import torch
model.eval()
with torch.no_grad():
    out = model(data)
print(f"  Output shape: {out.shape}")
norms = torch.norm(out, p=2, dim=-1)
print(f"  L2 norms (first 5): {norms[:5].tolist()}")
assert out.shape[0] == data['row'].x.shape[0]
assert out.shape[1] == er_cfg.output_dim

# 7. Schema Matching model
print("\n=== 7. SCHEMA MATCHING MODEL ===")
from table_unifier.models.schema_matching import SchemaProjector
from table_unifier.config import SchemaMatchingConfig
import numpy as np

sm_cfg = SchemaMatchingConfig()
sm_model = SchemaProjector(
    input_dim=sm_cfg.embedding_dim, hidden_dim=sm_cfg.hidden_dim,
    output_dim=sm_cfg.projection_dim,
)
sm_model.eval()
test_emb = torch.tensor(np.stack(list(col_emb.values())), dtype=torch.float32)
with torch.no_grad():
    proj = sm_model(test_emb)
print(f"  Input: {test_emb.shape} -> Output: {proj.shape}")
print(f"  Expected: ({len(col_emb)}, {sm_cfg.projection_dim})")

print("\n=== ALL CHECKS PASSED ===")
