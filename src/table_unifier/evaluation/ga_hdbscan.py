"""HDBSCAN + Genetic Algorithm (DEAP) для подбора параметров кластеризации.

Идея: вместо подбора единого порога θ на косинусной близости (exp 10) — кластеризуем
row-эмбеддинги через HDBSCAN и решаем задачу бинарной классификации пар так:

    match(a, b)  ⇔  cluster[a] == cluster[b]  и  cluster[a] != -1  (не шум)

Параметры HDBSCAN подбираем генетическим алгоритмом (DEAP) по F1 на val pairs.

Backends:
    * "cpu"  — пакет ``hdbscan`` (CPU, поддерживает euclidean / cosine).
    * "gpu"  — ``cuml.cluster.HDBSCAN`` из RAPIDS (GPU, только euclidean, но
      на L2-нормализованных эмбеддингах это эквивалент cosine с точностью
      до монотонного пересчёта epsilon).
    * "auto" — "gpu" если доступен cuml, иначе "cpu".

Пространство поиска:
    * min_cluster_size           ∈ [2 .. 100]           (int)
    * min_samples                ∈ [1 .. 50]            (int, <= min_cluster_size)
    * cluster_selection_epsilon  ∈ [0.0 .. 1.0]         (float)
    * cluster_selection_method   ∈ {"eom", "leaf"}
    * metric                     ∈ {"euclidean", "cosine"}   (только CPU-backend;
      на GPU-backend всегда euclidean и ген игнорируется).

Возвращает лучший индивид + историю поколений.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from table_unifier.evaluation.ga_common import (
    pair_fitness_from_labels,
    pair_metrics_from_labels,
)

logger = logging.getLogger(__name__)

METRICS = ("euclidean", "cosine")
SELECTION_METHODS = ("eom", "leaf")


def _cuml_available() -> bool:
    try:
        import cuml  # noqa: F401
        return True
    except Exception:
        return False


def resolve_backend(backend: str) -> str:
    """auto → gpu если есть cuml, иначе cpu."""
    if backend == "auto":
        return "gpu" if _cuml_available() else "cpu"
    if backend not in ("cpu", "gpu"):
        raise ValueError(f"backend должен быть cpu|gpu|auto, got {backend!r}")
    return backend


@dataclass
class GAHDBSCANConfig:
    pop_size: int = 40
    n_gen: int = 25
    cxpb: float = 0.6
    mutpb: float = 0.3
    tournament_size: int = 3
    # ER-приор: дубликаты группируются по 2-3, а не по сотням, поэтому
    # min_cluster_size держим в узком диапазоне. Большие значения превращают
    # все настоящие пары в шум (-1) и обрушивают recall.
    min_cluster_size_bounds: tuple[int, int] = (2, 5)
    min_samples_bounds: tuple[int, int] = (1, 5)
    epsilon_bounds: tuple[float, float] = (0.0, 1.0)
    seed: int = 42
    n_jobs: int = 1  # core_dist_n_jobs для CPU-бэкенда
    backend: str = "auto"  # cpu / gpu / auto
    # Фитнес: F-beta + penalty за гигантский кластер (см. ga_common.py).
    fbeta: float = 1.0
    giant_cluster_threshold: float = 0.5
    giant_cluster_penalty: float = 0.5


@dataclass
class GAResult:
    best_params: dict[str, Any]
    best_fitness: float
    backend: str = "cpu"
    history: list[dict[str, float]] = field(default_factory=list)
    n_evaluated: int = 0


def _decode(ind: list) -> dict[str, Any]:
    """Перевести DEAP-индивида в kwargs для hdbscan.HDBSCAN."""
    mcs, ms, eps, sel_idx, metric_idx = ind
    mcs = int(max(2, mcs))
    ms = int(max(1, min(ms, mcs)))  # min_samples не должен превышать min_cluster_size
    eps = float(max(0.0, eps))
    return {
        "min_cluster_size": mcs,
        "min_samples": ms,
        "cluster_selection_epsilon": eps,
        "cluster_selection_method": SELECTION_METHODS[int(sel_idx) % len(SELECTION_METHODS)],
        "metric": METRICS[int(metric_idx) % len(METRICS)],
    }


def cluster_embeddings(
    embeddings: np.ndarray,
    params: dict[str, Any],
    n_jobs: int = 1,
    backend: str = "cpu",
) -> np.ndarray:
    """Применить HDBSCAN с заданными параметрами. Возвращает метки кластеров.

    backend="cpu"  — пакет hdbscan.
    backend="gpu"  — cuml.cluster.HDBSCAN (metric всегда euclidean).
    """
    if backend == "gpu":
        from cuml.cluster import HDBSCAN as CuHDBSCAN

        gpu_params = {
            "min_cluster_size": params["min_cluster_size"],
            "min_samples": params["min_samples"],
            "cluster_selection_epsilon": params["cluster_selection_epsilon"],
            "cluster_selection_method": params["cluster_selection_method"],
            "metric": "euclidean",  # cuml поддерживает только euclidean
        }
        clusterer = CuHDBSCAN(**gpu_params)
        data = np.ascontiguousarray(embeddings.astype(np.float32))
        labels = clusterer.fit_predict(data)
        # cuml возвращает cupy/numpy — приводим к numpy
        arr = getattr(labels, "get", None)
        return np.asarray(arr() if callable(arr) else labels)

    import hdbscan

    clusterer = hdbscan.HDBSCAN(
        core_dist_n_jobs=n_jobs,
        **params,
    )
    return clusterer.fit_predict(embeddings)


# Pair-предикторы и метрики переехали в ga_common.py (общие для HDBSCAN/CC).
# Реэкспортируем для обратной совместимости.
from table_unifier.evaluation.ga_common import (  # noqa: E402
    cluster_labels_to_pair_preds,
)


def evaluate_params_on_pairs(
    embeddings: np.ndarray,
    pairs: torch.Tensor | np.ndarray,
    params: dict[str, Any],
    n_jobs: int = 1,
    backend: str = "cpu",
) -> dict[str, float]:
    """F1 / Precision / Recall пар при заданных параметрах HDBSCAN."""
    cluster_labels = cluster_embeddings(embeddings, params, n_jobs=n_jobs, backend=backend)
    return pair_metrics_from_labels(cluster_labels, pairs, noise_label=-1)


def run_ga_hdbscan(
    embeddings: np.ndarray | torch.Tensor,
    val_pairs: torch.Tensor,
    config: GAHDBSCANConfig | None = None,
) -> GAResult:
    """Запустить DEAP-ГА для подбора параметров HDBSCAN по F1 на val_pairs.

    ВАЖНО: embeddings должны быть L2-нормализованы (это уже так для выхода
    EntityResolutionGAT). Работают и euclidean, и cosine — ГА сам выберет.
    """
    from deap import base, creator, tools

    cfg = config or GAHDBSCANConfig()
    backend = resolve_backend(cfg.backend)
    logger.info("GA-HDBSCAN backend: %s", backend)

    if isinstance(embeddings, torch.Tensor):
        embeddings = embeddings.detach().cpu().numpy()
    # cuml ожидает float32, hdbscan лучше работает с float64
    dtype = np.float32 if backend == "gpu" else np.float64
    embeddings = np.ascontiguousarray(embeddings.astype(dtype))

    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # DEAP creator классы — глобальные, защищаемся от повторного запуска
    if not hasattr(creator, "FitnessMaxGA"):
        creator.create("FitnessMaxGA", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "IndividualGA"):
        creator.create("IndividualGA", list, fitness=creator.FitnessMaxGA)

    mcs_lo, mcs_hi = cfg.min_cluster_size_bounds
    ms_lo, ms_hi = cfg.min_samples_bounds
    eps_lo, eps_hi = cfg.epsilon_bounds

    toolbox = base.Toolbox()
    toolbox.register("attr_mcs", random.randint, mcs_lo, mcs_hi)
    toolbox.register("attr_ms", random.randint, ms_lo, ms_hi)
    toolbox.register("attr_eps", random.uniform, eps_lo, eps_hi)
    toolbox.register("attr_sel", random.randint, 0, len(SELECTION_METHODS) - 1)
    toolbox.register("attr_metric", random.randint, 0, len(METRICS) - 1)

    def make_individual():
        return creator.IndividualGA([
            toolbox.attr_mcs(),
            toolbox.attr_ms(),
            toolbox.attr_eps(),
            toolbox.attr_sel(),
            toolbox.attr_metric(),
        ])

    toolbox.register("individual", make_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Кэш: одни и те же параметры часто всплывают в поколениях
    cache: dict[tuple, float] = {}

    def evaluate(ind: list) -> tuple[float]:
        params = _decode(ind)
        key = (params["min_cluster_size"], params["min_samples"],
               round(params["cluster_selection_epsilon"], 4),
               params["cluster_selection_method"], params["metric"])
        if key in cache:
            return (cache[key],)
        try:
            labels = cluster_embeddings(
                embeddings, params, n_jobs=cfg.n_jobs, backend=backend,
            )
            score = pair_fitness_from_labels(
                labels, val_pairs,
                fbeta=cfg.fbeta,
                giant_cluster_threshold=cfg.giant_cluster_threshold,
                giant_cluster_penalty=cfg.giant_cluster_penalty,
                noise_label=-1,
            )
        except Exception as e:  # HDBSCAN иногда падает на патологических параметрах
            logger.debug("HDBSCAN fail for %s: %s", params, e)
            score = 0.0
        cache[key] = score
        return (score,)

    # Шаг мутации int-генов масштабируем к ширине bounds (после сужения
    # дефолтных диапазонов до ER-приора фиксированный ±5 был бы вырожденным).
    mcs_step = max(1, (mcs_hi - mcs_lo) // 4)
    ms_step = max(1, (ms_hi - ms_lo) // 4)

    def mutate(ind: list) -> tuple[list]:
        # По одному гену мутируем с вероятностью 0.3
        if random.random() < 0.3:
            ind[0] = int(np.clip(ind[0] + random.randint(-mcs_step, mcs_step), mcs_lo, mcs_hi))
        if random.random() < 0.3:
            ind[1] = int(np.clip(ind[1] + random.randint(-ms_step, ms_step), ms_lo, ms_hi))
        if random.random() < 0.3:
            ind[2] = float(np.clip(ind[2] + random.gauss(0.0, 0.1), eps_lo, eps_hi))
        if random.random() < 0.2:
            ind[3] = random.randint(0, len(SELECTION_METHODS) - 1)
        if random.random() < 0.2:
            ind[4] = random.randint(0, len(METRICS) - 1)
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
        offspring = [creator.IndividualGA(list(ind)) for ind in offspring]

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

        # Элитизм: сохраняем лучшего
        pop = tools.selBest(pop, 1) + tools.selBest(offspring, len(offspring) - 1)

        best_gen = tools.selBest(pop, 1)[0]
        if best_gen.fitness.values[0] > best_all.fitness.values[0]:
            best_all = creator.IndividualGA(list(best_gen))
            best_all.fitness.values = best_gen.fitness.values

        fits = [ind.fitness.values[0] for ind in pop]
        history.append({
            "gen": gen,
            # Имена оставлены best_f1/mean_f1/worst_f1 для совместимости с
            # ноутбуком; реально это F-beta с possible penalty (см. GAHDBSCANConfig).
            "best_f1": float(max(fits)),
            "mean_f1": float(np.mean(fits)),
            "worst_f1": float(min(fits)),
            "n_cached": len(cache),
        })
        logger.info("GA gen %2d/%d | best=%.4f mean=%.4f cache=%d",
                    gen, cfg.n_gen, history[-1]["best_f1"],
                    history[-1]["mean_f1"], len(cache))

    return GAResult(
        best_params=_decode(best_all),
        best_fitness=float(best_all.fitness.values[0]),
        backend=backend,
        history=history,
        n_evaluated=len(cache),
    )
