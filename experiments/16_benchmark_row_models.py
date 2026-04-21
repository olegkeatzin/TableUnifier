"""Эксперимент 16 — Бенчмарк разных токен-моделей на MRL + NT-Xent пайплайне.

Сравнивает rubert-tiny2 (baseline) с 7 альтернативными моделями:
  * sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (384-dim)
  * intfloat/multilingual-e5-base (768-dim, prefix "query: ")
  * cointegrated/LaBSE-en-ru (768-dim)
  * ai-forever/sbert_large_nlu_ru (1024-dim)
  * Alibaba-NLP/gte-multilingual-base (768-dim, trust_remote_code)
  * intfloat/multilingual-e5-large (1024-dim, prefix "query: ")
  * BAAI/bge-m3 (1024-dim)

Для каждой модели:
  1. ``01_data_exploration.py --all --embeddings --skip-columns`` — row embeddings
  2. ``14_build_unified_graph_mrl.py`` — MRL unified + cross графы (target_dim = hidden)
  3. ``14_train_gat_mrl.py --loss ntxent`` — обучение

Артефакты:
  data/embeddings/rows/<tag>/<ds>/  — row embeddings
  data/graphs/<tag>/v14_mrl/        — MRL unified
  output/<tag>/v14_mrl_gat_model.pt — обученная модель

Запуск::

    # Все модели (долго: row-emb × 5 + train × 5):
    uv run python experiments/16_benchmark_row_models.py

    # Только одну из моделей:
    uv run python experiments/16_benchmark_row_models.py --only e5-base

    # Пропуск существующих row-emb/графов:
    uv run python experiments/16_benchmark_row_models.py --skip-existing
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger("benchmark")


@dataclass
class ModelSpec:
    tag: str
    hf_name: str
    hidden_dim: int
    pooling: str = "cls"
    row_prefix: str = ""
    trust_remote_code: bool = False
    notes: str = ""
    # num_heads: int = 4  # 4 делит 312/384/768/1024 — достаточно

    def emb_cli(self) -> list[str]:
        args = [
            "--model-tag", self.tag,
            "--row-model-name", self.hf_name,
            "--pooling", self.pooling,
            "--skip-columns",
        ]
        if self.row_prefix:
            args += ["--row-prefix", self.row_prefix]
        if self.trust_remote_code:
            args += ["--trust-remote-code"]
        return args

    def build_cli(self) -> list[str]:
        args = [
            "--model-tag", self.tag,
            "--row-model-name", self.hf_name,
        ]
        if self.trust_remote_code:
            args += ["--trust-remote-code"]
        return args

    def train_cli(self) -> list[str]:
        return ["--model-tag", self.tag]


REGISTRY: list[ModelSpec] = [
    ModelSpec(
        tag="rubert-tiny2",
        hf_name="cointegrated/rubert-tiny2",
        hidden_dim=312,
        pooling="cls",
        notes="baseline (текущий)",
    ),
    ModelSpec(
        tag="minilm-l12",
        hf_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        hidden_dim=384,
        pooling="mean",
    ),
    ModelSpec(
        tag="e5-base",
        hf_name="intfloat/multilingual-e5-base",
        hidden_dim=768,
        pooling="mean",
        row_prefix="query: ",
    ),
    ModelSpec(
        tag="labse-en-ru",
        hf_name="cointegrated/LaBSE-en-ru",
        hidden_dim=768,
        pooling="cls",
    ),
    ModelSpec(
        tag="sbert-ru-large",
        hf_name="ai-forever/sbert_large_nlu_ru",
        hidden_dim=1024,
        pooling="mean",
    ),
    ModelSpec(
        tag="gte-multilingual",
        hf_name="Alibaba-NLP/gte-multilingual-base",
        hidden_dim=768,
        pooling="cls",
        trust_remote_code=True,
    ),
    ModelSpec(
        tag="e5-large",
        hf_name="intfloat/multilingual-e5-large",
        hidden_dim=1024,
        pooling="mean",
        row_prefix="query: ",
        notes="scaling к e5-base",
    ),
    ModelSpec(
        tag="bge-m3",
        hf_name="BAAI/bge-m3",
        hidden_dim=1024,
        pooling="cls",
        notes="топ-retrieval multilingual baseline",
    ),
]


def run_step(title: str, cmd: list[str], dry_run: bool) -> tuple[bool, float]:
    logger.info("+ %s", " ".join(cmd))
    if dry_run:
        return True, 0.0
    t0 = time.time()
    proc = subprocess.run(cmd, check=False)
    dt = time.time() - t0
    ok = proc.returncode == 0
    logger.info("[%s] %s за %.1f мин (rc=%d)",
                title, "OK" if ok else "FAIL", dt / 60, proc.returncode)
    return ok, dt


def main() -> None:
    parser = argparse.ArgumentParser(description="Бенчмарк row-моделей (MRL + NT-Xent)")
    parser.add_argument("--only", nargs="+", default=None,
                        help="Запустить только указанные модели по tag (можно несколько)")
    parser.add_argument("--exclude", nargs="*", default=[],
                        help="Пропустить модели по tag")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Пропустить шаги, артефакты которых уже есть")
    parser.add_argument("--dry-run", action="store_true",
                        help="Показать команды без запуска")
    parser.add_argument("--steps", nargs="+",
                        choices=["embeddings", "build", "train"],
                        default=["embeddings", "build", "train"],
                        help="Какие шаги выполнить")
    parser.add_argument("--max-epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    selected = REGISTRY
    if args.only:
        only_set = set(args.only)
        selected = [m for m in REGISTRY if m.tag in only_set]
        unknown = only_set - {m.tag for m in REGISTRY}
        if unknown:
            parser.error(f"Неизвестные --only tags: {sorted(unknown)}. "
                         f"Доступно: {[m.tag for m in REGISTRY]}")
    if args.exclude:
        selected = [m for m in selected if m.tag not in args.exclude]

    logger.info("Модели для прогона: %s", [m.tag for m in selected])

    summary: list[dict] = []

    for spec in selected:
        logger.info("=" * 70)
        logger.info("  %s  (%s, %d-dim, pooling=%s)",
                    spec.tag, spec.hf_name, spec.hidden_dim, spec.pooling)
        logger.info("=" * 70)

        rec: dict = {"tag": spec.tag, "hf_name": spec.hf_name,
                     "hidden_dim": spec.hidden_dim, "steps": {}}

        # 1. Row embeddings
        if "embeddings" in args.steps:
            rows_root = data_dir / "embeddings" / "rows" / spec.tag
            existing = rows_root.exists() and any(rows_root.iterdir())
            if args.skip_existing and existing:
                logger.info("[%s] row embeddings уже есть в %s — пропуск", spec.tag, rows_root)
                rec["steps"]["embeddings"] = {"status": "skipped", "path": str(rows_root)}
            else:
                cmd = [args.python, "-m", "experiments.01_data_exploration",
                       "--all", "--embeddings",
                       "--data-dir", str(data_dir)] + spec.emb_cli()
                ok, dt = run_step("embeddings", cmd, args.dry_run)
                rec["steps"]["embeddings"] = {"status": "ok" if ok else "fail",
                                              "duration_sec": dt}
                if not ok:
                    summary.append(rec)
                    continue

        # 2. MRL unified graph
        if "build" in args.steps:
            g_dir = data_dir / "graphs" / spec.tag / "v14_mrl"
            if args.skip_existing and (g_dir / "graph.pt").exists():
                logger.info("[%s] v14_mrl граф уже есть в %s — пропуск", spec.tag, g_dir)
                rec["steps"]["build"] = {"status": "skipped", "path": str(g_dir)}
            else:
                cmd = [args.python, "-m", "experiments.14_build_unified_graph_mrl",
                       "--data-dir", str(data_dir)] + spec.build_cli()
                ok, dt = run_step("build", cmd, args.dry_run)
                rec["steps"]["build"] = {"status": "ok" if ok else "fail",
                                         "duration_sec": dt}
                if not ok:
                    summary.append(rec)
                    continue

        # 3. Train GAT NT-Xent
        if "train" in args.steps:
            out_ckpt = output_dir / spec.tag / "v14_mrl_gat_model.pt"
            if args.skip_existing and out_ckpt.exists():
                logger.info("[%s] модель уже обучена: %s — пропуск", spec.tag, out_ckpt)
                rec["steps"]["train"] = {"status": "skipped", "path": str(out_ckpt)}
            else:
                cmd = [args.python, "-m", "experiments.14_train_gat_mrl",
                       "--loss", "ntxent",
                       "--max-epochs", str(args.max_epochs),
                       "--patience", str(args.patience),
                       "--data-dir", str(data_dir),
                       "--output-dir", str(output_dir)] + spec.train_cli()
                ok, dt = run_step("train", cmd, args.dry_run)
                rec["steps"]["train"] = {"status": "ok" if ok else "fail",
                                         "duration_sec": dt}

        summary.append(rec)

    # Сводка: мержим с существующей, чтобы не терять записи других моделей
    summary_path = output_dir / "benchmark_row_models.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[str, dict] = {}
    if summary_path.exists():
        try:
            with open(summary_path, encoding="utf-8") as f:
                existing = json.load(f)
            for rec in existing.get("models", []):
                if "tag" in rec:
                    merged[rec["tag"]] = rec
        except json.JSONDecodeError:
            logger.warning("Существующая сводка повреждена — перезапишу целиком")

    for rec in summary:
        merged[rec["tag"]] = rec

    # Порядок = REGISTRY + хвост из неизвестных тегов (на всякий случай)
    order = [m.tag for m in REGISTRY]
    models_out = [merged[t] for t in order if t in merged]
    models_out += [merged[t] for t in merged if t not in order]

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({"models": models_out}, f, ensure_ascii=False, indent=2)
    logger.info("Сводка: %s (всего %d моделей)", summary_path, len(models_out))
    for rec in models_out:
        stats = " | ".join(
            f"{step}={info.get('status', '?')}"
            for step, info in rec.get("steps", {}).items()
        )
        logger.info("  %-18s %s", rec["tag"], stats)


if __name__ == "__main__":
    main()
