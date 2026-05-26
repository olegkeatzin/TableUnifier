"""FastAPI entrypoint для TableUnifier-инференса.

Запуск:
    uv run python -m table_unifier.server.main --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from table_unifier.server.routes import (
    clusters as clusters_router,
)
from table_unifier.server.routes import (
    export as export_router,
)
from table_unifier.server.routes import (
    graph as graph_router,
)
from table_unifier.server.routes import (
    infer as infer_router,
)
from table_unifier.server.routes import (
    sources as sources_router,
)
from table_unifier.server.services.progress import bus

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="TableUnifier", version="0.1.0")

    @app.on_event("startup")
    async def _attach_loop() -> None:
        bus.attach_loop(asyncio.get_event_loop())

    @app.get("/api/health")
    def health() -> dict:
        return {"status": "ok"}

    app.include_router(sources_router.router)
    app.include_router(graph_router.router)
    app.include_router(infer_router.router)
    app.include_router(clusters_router.router)
    app.include_router(export_router.router)

    static_dir = Path(__file__).parent / "static"
    if static_dir.exists():
        app.mount("/", StaticFiles(directory=static_dir, html=True), name="static")
    else:
        logger.warning("static dir not found at %s", static_dir)

    return app


app = create_app()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()
    uvicorn.run(
        "table_unifier.server.main:app",
        host=args.host, port=args.port, reload=args.reload,
    )


if __name__ == "__main__":
    main()
