"""Connected-Components кластеризация + DEAP GA для подбора параметров.

Альтернатива HDBSCAN, ближе к природе ER-задачи:

  1. Считаем top-K cos-NN для каждой строки (K = top_k_max).
  2. Оставляем только рёбра (i, j) c sim ≥ τ и j ∈ top_k(i).
  3. Связные компоненты полученного графа = кластеры дубликатов.

Правило на парах: ``match(a, b) ⇔ component[a] == component[b]``.
Шумом считаются только синглтоны (компоненты размера 1) — но они автоматически
не дают пересечений, поэтому ``noise_label=None`` в фитнес-функции.

Гены ГА:
  * ``tau``   ∈ [tau_lo, tau_hi]   — порог cos-схожести.
  * ``top_k`` ∈ [1, top_k_max]    — макс. соседей на строку выше τ.

Эмбеддинги ожидаются L2-нормализованными (это так для выхода EntityResolutionGAT).
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from table_unifier.evaluation.ga_common import (
    pair_fitness_from_labels,
    pair_metrics_from_labels,
)

logger = logging.getLogger(__name__)


@dataclass
class GACCConfig:
    pop_size: int = 30
    n_gen: int = 20
    cxpb: float = 0.6
    mutpb: float = 0.4
    tournament_size: int = 3
    tau_bounds: tuple[float, float] = (0.5, 0.999)
    top_k_max: int = 10
    seed: int = 42
    fbeta: float = 1.0
    giant_cluster_threshold: float = 0.5
    giant_cluster_penalty: float = 0.5
    chunk_size: int = 1024  # для топ-K матмула


@dataclass
class GACCResult:
    best_params: dict[str, Any]
    best_fitness: float
    history: list[dict[str, float]] = field(default_factory=list)
    n_evaluated: int = 0


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.clip(norms, 1e-12, None)


def precompute_topk_cosine(
    embeddings: np.ndarray,
    top_k_max: int,
    chunk_size: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Top-K соседей по cos-сходству, исключая self.

    Возвращает (top_sims, top_inds) формы (n, top_k_max), отсортированные
    по убыванию sim.
    """
    emb = _l2_normalize(embeddings.astype(np.float32))
    n = len(emb)
    k = int(min(top_k_max, max(1, n - 1)))
    top_sims = np.empty((n, k), dtype=np.float32)
    top_inds = np.empty((n, k), dtype=np.int64)

    for start in range(0, n, chunk_size):
        end = min(start + chunk_size, n)
        sims = emb[start:end] @ emb.T  # (chunk, n)
        # маскируем self
        for i, ri in enumerate(range(start, end)):
            sims[i, ri] = -np.inf
        idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(end - start)[:, None]
        s = sims[rows, idx]
        order = np.argsort(-s, axis=1)
        top_sims[start:end] = np.take_along_axis(s, order, axis=1)
        top_inds[start:end] = np.take_along_axis(idx, order, axis=1)
    return top_sims, top_inds


def cluster_cc_from_topk(
    top_sims: np.ndarray,
    top_inds: np.ndarray,
    tau: float,
    top_k: int,
) -> np.ndarray:
    """Connected components от top_k-NN графа с порогом τ. Метки [0, n_comp)."""
    n = top_sims.shape[0]
    k = int(np.clip(top_k, 1, top_sims.shape[1]))
    sims = top_sims[:, :k]
    inds = top_inds[:, :k]

    mask = sims >= tau
    if not mask.any():
        return np.arange(n, dtype=np.int64)  # все синглтоны

    rows = np.broadcast_to(np.arange(n, dtype=np.int64)[:, None], inds.shape)[mask]
    cols = inds[mask]
    data = np.ones(rows.shape[0], dtype=np.int8)
    adj = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    _, labels = connected_components(adj, directed=False)
    return labels.astype(np.int64)


def cluster_embeddings_cc(
    embeddings: np.ndarray | torch.Tensor,
    params: dict[str, Any],
    chunk_size: int = 1024,
) -> np.ndarray:
    """Кластеры (CC) по эмбеддингам с заданными tau/top_k."""
    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    top_sims, top_inds = precompute_topk_cosine(
        embeddings, top_k_max=int(params["top_k"]), chunk_size=chunk_size,
    )
    return cluster_cc_from_topk(top_sims, top_inds,
                                tau=float(params["tau"]),
                                top_k=int(params["top_k"]))


def evaluate_params_on_pairs_cc(
    embeddings: np.ndarray | torch.Tensor,
    pairs: torch.Tensor | np.ndarray,
    params: dict[str, Any],
    chunk_size: int = 1024,
) -> dict[str, float]:
    """F1/Precision/Recall + статистика для CC-кластеризации."""
    labels = cluster_embeddings_cc(embeddings, params, chunk_size=chunk_size)
    return pair_metrics_from_labels(labels, pairs, noise_label=None)


def run_ga_cc(
    embeddings: np.ndarray | torch.Tensor,
    val_pairs: torch.Tensor,
    config: GACCConfig | None = None,
) -> GACCResult:
    """ГА над (tau, top_k) для CC-кластеризации.

    Top-K матрица соседей строится один раз; индивиды переиспользуют её
    через срез по top_k. Это делает один eval ≈ миллисекунды.
    """
    from deap import base, creator, tools

    cfg = config or GACCConfig()

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    logger.info("CC: precompute top-%d cosine neighbors...", cfg.top_k_max)
    top_sims, top_inds = precompute_topk_cosine(
        embeddings, top_k_max=cfg.top_k_max, chunk_size=cfg.chunk_size,
    )
    top_k_max_eff = top_sims.shape[1]
    logger.info("CC: top-K матрица %s готова", top_sims.shape)

    if not hasattr(creator, "FitnessMaxCC"):
        creator.create("FitnessMaxCC", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualCC"):
        creator.create("IndividualCC", list, fitness=creator.FitnessMaxCC)

    tau_lo, tau_hi = cfg.tau_bounds
    k_step = max(1, top_k_max_eff // 4)

    toolbox = base.Toolbox()
    toolbox.register("attr_tau", random.uniform, tau_lo, tau_hi)
    toolbox.register("attr_k", random.randint, 1, top_k_max_eff)

    def make_individual():
        return creator.IndividualCC([toolbox.attr_tau(), toolbox.attr_k()])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    cache: dict[tuple, float] = {}

    def _decode(ind: list) -> dict[str, Any]:
        tau = float(np.clip(ind[0], tau_lo, tau_hi))
        k = int(np.clip(ind[1], 1, top_k_max_eff))
        return {"tau": tau, "top_k": k}

    def evaluate(ind: list) -> tuple[float]:
        params = _decode(ind)
        key = (round(params["tau"], 4), params["top_k"])
        if key in cache:
            return (cache[key],)
        try:
            labels = cluster_cc_from_topk(top_sims, top_inds,
                                          tau=params["tau"],
                                          top_k=params["top_k"])
            score = pair_fitness_from_labels(
                labels, val_pairs,
                fbeta=cfg.fbeta,
                giant_cluster_threshold=cfg.giant_cluster_threshold,
                giant_cluster_penalty=cfg.giant_cluster_penalty,
                noise_label=None,
            )
        except Exception as e:
            logger.debug("CC fail for %s: %s", params, e)
            score = 0.0
        cache[key] = score
        return (score,)

    def mutate(ind: list) -> tuple[list]:
        if random.random() < 0.5:
            ind[0] = float(np.clip(ind[0] + random.gauss(0.0, 0.05), tau_lo, tau_hi))
        if random.random() < 0.5:
            ind[1] = int(np.clip(ind[1] + random.randint(-k_step, k_step),
                                 1, top_k_max_eff))
        return (ind,)

    toolbox.register("evaluate", evaluate)
    toolbox.register("mate", tools.cxTwoPoint)
    toolbox.register("mutate", mutate)
    toolbox.register("select", tools.selTournament, tournsize=cfg.tournament_size)

    pop = toolbox.population(n=cfg.pop_size)
    for ind, fit in zip(pop, map(toolbox.evaluate, pop)):
        ind.fitness.values = fit

    history: list[dict[str, float]] = []
    best_all = tools.selBest(pop, 1)[0]

    for gen in range(1, cfg.n_gen + 1):
        offspring = toolbox.select(pop, len(pop))
        offspring = [creator.IndividualCC(list(ind)) for ind in offspring]

        for c1, c2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cfg.cxpb:
                toolbox.mate(c1, c2)
                del c1.fitness.values
                del c2.fitness.values

        for m in offspring:
            if random.random() < cfg.mutpb:
                toolbox.mutate(m)
                del m.fitness.values

        invalid = [ind for ind in offspring if not ind.fitness.valid]
        for ind, fit in zip(invalid, map(toolbox.evaluate, invalid)):
            ind.fitness.values = fit

        pop = tools.selBest(pop, 1) + tools.selBest(offspring, len(offspring) - 1)

        best_gen = tools.selBest(pop, 1)[0]
        if best_gen.fitness.values[0] > best_all.fitness.values[0]:
            best_all = creator.IndividualCC(list(best_gen))
            best_all.fitness.values = best_gen.fitness.values

        fits = [ind.fitness.values[0] for ind in pop]
        history.append({
            "gen": gen,
            "best_f1": float(max(fits)),
            "mean_f1": float(np.mean(fits)),
            "worst_f1": float(min(fits)),
            "n_cached": len(cache),
        })
        logger.info("GA-CC gen %2d/%d | best=%.4f mean=%.4f cache=%d",
                    gen, cfg.n_gen, history[-1]["best_f1"],
                    history[-1]["mean_f1"], len(cache))

    return GACCResult(
        best_params=_decode(best_all),
        best_fitness=float(best_all.fitness.values[0]),
        history=history,
        n_evaluated=len(cache),
    )
