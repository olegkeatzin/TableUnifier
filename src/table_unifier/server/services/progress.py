"""WebSocket-broadcaster + накопление событий run-а.

Каждый run имеет очередь событий и список подписчиков. События пушатся в очередь
(в т.ч. из фонового потока инференса), затем broadcaster доставляет их подписчикам.
Если на момент пуша подписчиков нет — событие копится в буфере (last_events),
чтобы переподключившийся клиент мог получить полный лог фазы.
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from typing import Any

logger = logging.getLogger(__name__)


class ProgressBus:
    def __init__(self) -> None:
        self._buffers: dict[str, list[dict]] = defaultdict(list)
        self._subscribers: dict[str, set[asyncio.Queue]] = defaultdict(set)
        self._loop: asyncio.AbstractEventLoop | None = None

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    # --- subscriber side --- #

    async def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self._subscribers[run_id].add(q)
        # отдадим уже накопленные события
        for ev in self._buffers[run_id]:
            await q.put(ev)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        self._subscribers[run_id].discard(q)

    # --- publisher side (sync — вызывается из фоновых потоков) --- #

    def publish(self, run_id: str, event: dict[str, Any]) -> None:
        self._buffers[run_id].append(event)
        loop = self._loop
        if loop is None:
            return
        for q in list(self._subscribers[run_id]):
            try:
                loop.call_soon_threadsafe(q.put_nowait, event)
            except Exception as e:
                logger.warning("publish failed: %s", e)

    def buffered(self, run_id: str) -> list[dict]:
        return list(self._buffers[run_id])


bus = ProgressBus()
