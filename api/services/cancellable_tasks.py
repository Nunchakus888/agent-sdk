"""
Cancellable task service with automatic resource cleanup.

Manages one-task-per-tag semantics: starting a new task for a tag
automatically cancels the old one. Token usage accumulates across
cancellations and is cleaned up via periodic garbage collection.

Usage:
    async with CancellableTaskService() as tasks:
        result = await tasks.restart(my_coro(), tag="session:123")
        # result.was_cancelled, result.task

        # Inside the coroutine:
        tasks.set_tokens("session:123", 100)
        total = tasks.total_tokens("session:123")
        # No manual finish() needed — collect() handles cleanup.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Coroutine, Optional
from typing_extensions import Self

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class RestartResult:
    task: asyncio.Task[None]
    was_cancelled: bool


class CancellableTaskService:
    """One-task-per-tag service with token tracking and auto-cleanup."""

    def __init__(self, *, gc_interval: float = 5.0) -> None:
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._tokens: dict[str, int] = {}
        self._cancelled_tokens: dict[str, int] = {}
        self._lock = asyncio.Lock()
        self._gc_interval = gc_interval
        self._last_gc = 0.0

    # -- Lifecycle --

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_value: Optional[BaseException],
        tb: Optional[object],
    ) -> bool:
        if exc_value:
            await self.cancel_all(reason="Shutting down")
        await self.collect(force=True)
        return False

    # -- Core API --

    async def restart(
        self, coro: Coroutine[Any, Any, None], /, *, tag: str,
    ) -> RestartResult:
        """Cancel existing task for tag (if any), start new one."""
        await self.collect()

        async with self._lock:
            was_cancelled = False

            if old := self._tasks.get(tag):
                if not old.done():
                    was_cancelled = True
                    self._cancelled_tokens[tag] = (
                        self._cancelled_tokens.get(tag, 0)
                        + self._tokens.get(tag, 0)
                    )
                    old.cancel(f"Restarting task '{tag}'")
                    await self._await_task(old)

            task = asyncio.create_task(coro)
            self._tasks[tag] = task
            self._tokens[tag] = 0
            return RestartResult(task=task, was_cancelled=was_cancelled)

    async def cancel(self, *, tag: str, reason: str = "(not given)") -> None:
        """Cancel a specific task."""
        async with self._lock:
            if task := self._tasks.get(tag):
                if not task.done():
                    task.cancel(f"Cancelled [reason: {reason}]")
        await self.collect()

    async def cancel_all(self, *, reason: str = "(not given)") -> None:
        """Cancel all tasks."""
        async with self._lock:
            for task in self._tasks.values():
                if not task.done():
                    task.cancel(f"Cancelled [reason: {reason}]")
        await self.collect()

    # -- Token tracking --

    def set_tokens(self, tag: str, tokens: int) -> None:
        """Record current task's token count."""
        self._tokens[tag] = tokens

    def total_tokens(self, tag: str) -> int:
        """Current task tokens + accumulated cancelled tokens."""
        return (
            self._tokens.get(tag, 0)
            + self._cancelled_tokens.get(tag, 0)
        )

    # -- GC --

    async def collect(self, *, force: bool = False) -> None:
        """Sweep done tasks and their token data."""
        now = asyncio.get_event_loop().time()
        if not force and (now - self._last_gc) < self._gc_interval:
            return

        async with self._lock:
            alive: dict[str, asyncio.Task[None]] = {}
            for tag, task in self._tasks.items():
                if task.done() or force:
                    if not task.done():
                        logger.info(f"Waiting for task '{tag}' to finish")
                    await self._await_task(task)
                    self._tokens.pop(tag, None)
                    self._cancelled_tokens.pop(tag, None)
                else:
                    alive[tag] = task
            self._tasks = alive

        self._last_gc = now

    # -- Introspection --

    def get_active_tasks(self) -> list[str]:
        """Get tags of currently running tasks."""
        return [tag for tag, task in self._tasks.items() if not task.done()]

    # -- Internal --

    async def _await_task(self, task: asyncio.Task[None]) -> None:
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(f"Task raised exception: {exc}", exc_info=True)
