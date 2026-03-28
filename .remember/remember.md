# Handoff

## State
I implemented the full v3 plan (`docs/superpowers/plans/2026-03-27-v3-gat-minibatch-hdbscan.md`): GATv2 layer, EntityResolutionGAT model, stratified split, unified graph builder, mini-batch training, HDBSCAN eval, experiments 08/09/10, and a Jupyter notebook (`experiments/v3_results_analysis.ipynb`). All 54 tests pass. Experiment 08 (unified graph build) completed successfully. Just fixed a CUDA crash in `src/table_unifier/models/gat_layer.py` — added `add_self_loops=False` to GATv2Conv (bipartite graph can't have self-loops).

## Next
1. User needs to rerun experiment 09: `uv run python -m experiments.09_train_gat --max-epochs 500 --patience 30`
2. Then run experiment 10: `uv run python -m experiments.10_evaluate`
3. Then populate the Jupyter notebook with actual results

## Context
- Branch: `v2`. pyg-lib installed from PyG wheels (`+pt210cu130`). torch-sparse/torch-scatter not installed (no MSVC) but not needed.
- IDF threshold changed to 5% (`--max-token-df 0.05`) after user noticed 30% was too weak.
- Unified graph: ~341K rows, ~8M edges. NeighborLoader keeps graph on CPU, batches to GPU.
