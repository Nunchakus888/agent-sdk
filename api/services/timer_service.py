"""
会话级 Timer 服务 v2

职责：
- 后台扫描超时 Timer
- 统一 Tool 调用模型执行触发动作
- Timer 生命周期管理
"""

import asyncio
import logging
from datetime import timedelta
from typing import Callable, Awaitable, Any

from api.utils.datetime import utc_now

logger = logging.getLogger(__name__)


# =============================================================================
# Timer 配置常量
# =============================================================================

MAX_TIMERS_PER_SESSION = 10       # 每个 Session 最大 Timer 数
MAX_CONCURRENT_TRIGGERS = 50      # 单次扫描最大触发数
DEFAULT_SCAN_INTERVAL = 30        # 默认扫描间隔（秒）
BATCH_SIZE = 100                  # 批量查询大小
TIMER_TTL_DAYS = 7                # Timer 过期天数


# =============================================================================
# Timer Service
# =============================================================================


class TimerService:
    """
    会话级 Timer 服务 - 统一 Tool 调用模型。

    Usage:
        ```python
        timer_service = TimerService(
            repos=repos,
            tool_executor=tool_executor,
            send_message=send_message_callback,
        )
        await timer_service.start()
        ```
    """

    def __init__(
        self,
        repos: Any,
        tool_executor: Any = None,
        send_message: Callable[[str, str], Awaitable[None]] = None,
        scan_interval: int = DEFAULT_SCAN_INTERVAL,
    ):
        """
        Args:
            repos: RepositoryManager
            tool_executor: ToolExecutor 实例
            send_message: 发送消息回调 (session_id, message) -> None
            scan_interval: 扫描间隔（秒）
        """
        self._repos = repos
        self._tool_executor = tool_executor
        self._send_message = send_message
        self._scan_interval = scan_interval
        self._task: asyncio.Task | None = None
        self._running = False

    # -------------------------------------------------------------------------
    # 生命周期
    # -------------------------------------------------------------------------

    async def start(self):
        """启动后台扫描。"""
        if self._running:
            return
        self._running = True
        self._task = asyncio.create_task(self._scan_loop())
        logger.info(f"TimerService started (interval={self._scan_interval}s)")

    async def stop(self):
        """停止后台扫描。"""
        self._running = False
        if self._task and not self._task.done():
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("TimerService stopped")

    # -------------------------------------------------------------------------
    # 扫描循环
    # -------------------------------------------------------------------------

    async def _scan_loop(self):
        """扫描循环。"""
        while self._running:
            try:
                await asyncio.sleep(self._scan_interval)
                await self._check_timeouts()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Timer scan error: {e}")

    async def _check_timeouts(self):
        """检查并触发超时 Timer。"""
        try:
            # 查询到期的 Timer（限制数量）
            timers = await self._find_due_timers()
            if not timers:
                return

            logger.debug(f"Found {len(timers)} due timers")

            # 并发执行（限制并发数）
            semaphore = asyncio.Semaphore(MAX_CONCURRENT_TRIGGERS)

            async def execute_with_limit(timer):
                async with semaphore:
                    await self._execute_timer(timer)

            await asyncio.gather(
                *[execute_with_limit(t) for t in timers],
                return_exceptions=True,
            )

        except Exception as e:
            logger.error(f"Check timeouts failed: {e}")

    async def _find_due_timers(self) -> list:
        """查询到期的 Timer。"""
        # 使用 sessions 表的 timer 字段（v1 兼容）
        if hasattr(self._repos, 'sessions'):
            return await self._repos.sessions.find_timeout_sessions()
        return []

    # -------------------------------------------------------------------------
    # Timer 执行
    # -------------------------------------------------------------------------

    async def _execute_timer(self, session):
        """执行单个 Timer - 统一 Tool 调用。"""
        session_id = session.session_id
        timer_config = session.timer_config or {}

        # 检查触发次数限制
        max_triggers = timer_config.get("max_triggers", 3)
        if session.timer_trigger_count >= max_triggers:
            await self._disable_timer(session_id)
            logger.info(f"Timer disabled (max triggers): {session_id}")
            return

        try:
            tool_name = timer_config.get("tool_name", "generate_response")
            tool_params = timer_config.get("tool_params", {})
            message = timer_config.get("message", "您好，请问还在吗？")

            if tool_name == "generate_response":
                # 内置：生成响应消息
                await self._generate_response(session, message)
            elif self._tool_executor:
                # 调用配置中的 Tool
                await self._tool_executor.execute_single(tool_name, tool_params)
            else:
                logger.warning(f"No tool executor for: {tool_name}")

            # 更新状态
            await self._update_timer_status(session)
            logger.info(f"Timer triggered: {session_id}")

        except Exception as e:
            logger.error(f"Timer trigger failed: {session_id}, {e}")

    async def _generate_response(self, session, message: str):
        """生成响应消息。"""
        session_id = session.session_id

        # 存储消息
        if hasattr(self._repos, 'messages'):
            from api.models import MessageRole
            await self._repos.messages.create(
                session_id=session_id,
                role=MessageRole.ASSISTANT,
                content=message,
            )

        # 发送消息（通过回调）
        if self._send_message:
            await self._send_message(session_id, message)

    async def _update_timer_status(self, session):
        """更新 Timer 状态。"""
        session_id = session.session_id
        new_count = session.timer_trigger_count + 1
        timer_config = session.timer_config or {}
        max_triggers = timer_config.get("max_triggers", 3)

        if max_triggers > 0 and new_count >= max_triggers:
            # 达到最大次数，禁用
            await self._repos.sessions.update(
                session_id,
                timer_status="disabled",
                timer_trigger_count=new_count,
            )
        else:
            # 更新触发次数，状态改为 triggered
            await self._repos.sessions.update(
                session_id,
                timer_status="triggered",
                timer_trigger_count=new_count,
            )

    async def _disable_timer(self, session_id: str):
        """禁用 Timer。"""
        await self._repos.sessions.update(session_id, timer_status="disabled")

    # -------------------------------------------------------------------------
    # 清理
    # -------------------------------------------------------------------------

    async def cleanup_expired(self):
        """清理过期 Timer。"""
        cutoff = utc_now() - timedelta(days=TIMER_TTL_DAYS)
        # 实现依赖具体的 Repository
        logger.info(f"Cleanup timers before {cutoff}")

    async def cancel_session_timers(self, session_id: str):
        """取消 Session 的所有 Timer。"""
        await self._repos.sessions.update(session_id, timer_status="cancelled")
        logger.debug(f"Cancelled timers for session: {session_id}")


# =============================================================================
# Factory
# =============================================================================


def create_timer_service(
    repos: Any,
    tool_executor: Any = None,
    send_message: Callable[[str, str], Awaitable[None]] = None,
    scan_interval: int = DEFAULT_SCAN_INTERVAL,
) -> TimerService:
    """
    创建 TimerService 实例。

    Args:
        repos: RepositoryManager
        tool_executor: ToolExecutor 实例（可选）
        send_message: 发送消息回调（可选）
        scan_interval: 扫描间隔

    Returns:
        TimerService 实例
    """
    return TimerService(
        repos=repos,
        tool_executor=tool_executor,
        send_message=send_message,
        scan_interval=scan_interval,
    )
