"""
任务管理器

按 Parlant BackgroundTaskService.restart() 原理实现
支持同 session 新请求取消旧请求
"""

import asyncio
import logging
from typing import Any, Coroutine

logger = logging.getLogger(__name__)


class TaskManager:
    """
    任务管理器 - 协程取消机制

    核心思想：
    - 每个任务有唯一 tag（如 "session:{session_id}"）
    - 新请求到来时，取消同 tag 的旧任务
    - 旧任务收到 CancelledError，优雅退出

    实现要点：
    1. 维护 tasks: dict[str, asyncio.Task] 映射
    2. restart(coro, tag) 方法：取消同 tag 旧任务，创建新任务
    3. 定期垃圾回收已完成的任务
    """

    def __init__(self):
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()
        self._last_gc = 0.0
        self._gc_interval = 5.0  # 垃圾回收间隔（秒）

    async def restart(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        tag: str,
    ) -> asyncio.Task:
        """
        重启任务 - 取消同 tag 旧任务

        Args:
            coro: 要执行的协程
            tag: 任务标签（如 "session:{session_id}"）

        Returns:
            新创建的 asyncio.Task
        """
        await self._collect()

        async with self._lock:
            # 取消同 tag 旧任务
            if old := self._tasks.get(tag):
                if not old.done():
                    old.cancel(f"Restarting task '{tag}'")
                    try:
                        await old
                    except asyncio.CancelledError:
                        logger.info(f"Task '{tag}' cancelled")

            # 创建新任务
            task = asyncio.create_task(coro)
            self._tasks[tag] = task
            return task

    async def start(
        self,
        coro: Coroutine[Any, Any, Any],
        *,
        tag: str,
    ) -> asyncio.Task:
        """
        启动任务（不取消旧任务）

        如果同 tag 任务已存在且未完成，抛出异常

        Args:
            coro: 要执行的协程
            tag: 任务标签

        Returns:
            新创建的 asyncio.Task

        Raises:
            RuntimeError: 如果同 tag 任务已存在且未完成
        """
        await self._collect()

        async with self._lock:
            if existing := self._tasks.get(tag):
                if not existing.done():
                    raise RuntimeError(
                        f"Task '{tag}' is already running; "
                        "consider calling restart() instead"
                    )

            task = asyncio.create_task(coro)
            self._tasks[tag] = task
            return task

    async def cancel(self, *, tag: str, reason: str = "(not given)") -> None:
        """
        取消指定任务

        Args:
            tag: 任务标签
            reason: 取消原因
        """
        async with self._lock:
            if task := self._tasks.get(tag):
                if not task.done():
                    task.cancel(f"Forced cancellation [reason: {reason}]")
                    logger.info(f"Task '{tag}' cancelled: {reason}")

        await self._collect()

    async def cancel_all(self, *, reason: str = "(not given)") -> None:
        """
        取消所有任务

        Args:
            reason: 取消原因
        """
        async with self._lock:
            count = len(self._tasks)
            logger.info(f"Cancelling all tasks ({count}): {reason}")

            for tag, task in self._tasks.items():
                if not task.done():
                    task.cancel(f"Forced cancellation [reason: {reason}]")

        await self._collect()

    async def shutdown(self) -> None:
        """关闭任务管理器，取消所有任务"""
        await self.cancel_all(reason="Shutting down")
        await self._collect(force=True)
        logger.info("TaskManager shutdown complete")

    def get_active_tasks(self) -> list[str]:
        """
        获取当前活跃的任务标签列表

        Returns:
            活跃任务的 tag 列表
        """
        return [tag for tag, task in self._tasks.items() if not task.done()]

    async def _collect(self, *, force: bool = False) -> None:
        """
        垃圾回收已完成的任务

        Args:
            force: 是否强制回收（忽略时间间隔）
        """
        now = asyncio.get_event_loop().time()

        if not force:
            if (now - self._last_gc) < self._gc_interval:
                return

        async with self._lock:
            new_tasks = {}

            for tag, task in self._tasks.items():
                if task.done() or force:
                    if not task.done():
                        logger.info(f"Waiting for task '{tag}' to finish")
                    await self._await_task(task)
                else:
                    new_tasks[tag] = task

            self._tasks = new_tasks

        self._last_gc = now

    async def _await_task(self, task: asyncio.Task) -> None:
        """等待任务完成，处理异常"""
        try:
            await task
        except asyncio.CancelledError:
            pass
        except Exception as exc:
            logger.warning(f"Task raised exception: {exc}", exc_info=True)
