"""
会话管理器

职责：
1. 会话生命周期管理（创建、获取、销毁）
2. Agent 实例管理（会话级）
3. Timer 管理（会话级）
4. 空闲会话回收
"""

import asyncio
import logging
from typing import Optional, Callable, Awaitable

from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from bu_agent_sdk.tools.actions import WorkflowConfigSchema
from bu_agent_sdk.llm.messages import UserMessage, AssistantMessage

from api.services.v2.session_context import SessionContext, SessionTimer
from api.services.repositories import RepositoryManager, MessageRepository
from api.services.llm_service import LLMService

logger = logging.getLogger(__name__)


class SessionManager:
    """
    会话管理器

    管理所有会话的生命周期，包括：
    - 会话创建和销毁
    - Agent 实例管理（会话级）
    - Timer 管理（会话级）
    - 空闲会话自动回收

    Usage:
        ```python
        session_mgr = SessionManager(repos=repos, llm_provider=llm_provider)
        await session_mgr.start()

        # 获取或创建会话
        ctx = await session_mgr.get_or_create(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config=config,
        )

        # 执行查询
        result = await ctx.agent.query(message, session_id)

        # 重置 Timer
        session_mgr.reset_timer("sess_123")
        ```
    """

    def __init__(
        self,
        repos: RepositoryManager,
        llm_provider: LLMService | None = None,
        idle_timeout: int = 1800,
        cleanup_interval: int = 60,
        max_sessions: int = 10000,
    ):
        """
        Args:
            repos: RepositoryManager
            llm_provider: LLM 提供者
            idle_timeout: 空闲超时（秒），默认 30 分钟
            cleanup_interval: 清理间隔（秒）
            max_sessions: 最大会话数
        """
        self._repos = repos
        self._llm_provider = llm_provider
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval
        self._max_sessions = max_sessions

        # 会话池：session_id -> SessionContext
        self._sessions: dict[str, SessionContext] = {}

        # 消息发送回调
        self._send_message: Optional[Callable[[str, str], Awaitable[None]]] = None

        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        logger.info(
            f"SessionManager initialized: "
            f"idle_timeout={idle_timeout}s, max_sessions={max_sessions}"
        )

    # -------------------------------------------------------------------------
    # 生命周期
    # -------------------------------------------------------------------------

    async def start(self):
        """启动会话管理器"""
        if self._running:
            return
        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        logger.info("SessionManager started")

    async def stop(self):
        """停止会话管理器"""
        self._running = False
        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        # 清理所有会话
        for ctx in list(self._sessions.values()):
            ctx.cleanup()
        self._sessions.clear()

        logger.info("SessionManager stopped")

    def set_send_message_callback(
        self,
        callback: Callable[[str, str], Awaitable[None]]
    ):
        """设置消息发送回调"""
        self._send_message = callback

    # -------------------------------------------------------------------------
    # 会话管理
    # -------------------------------------------------------------------------

    async def get_or_create(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
        config: WorkflowConfigSchema,
    ) -> SessionContext:
        """
        获取或创建会话上下文

        Args:
            session_id: 会话 ID
            tenant_id: 租户 ID
            chatbot_id: Chatbot ID
            config: 解析后的配置（由 ConfigLoader 提供）

        Returns:
            SessionContext 实例

        Note:
            DB 操作（创建会话记录、加载历史）与 Agent 创建并行执行，
            不阻塞主流程。
        """
        # 1. 检查现有会话（快速路径）
        if session_id in self._sessions:
            ctx = self._sessions[session_id]
            ctx.touch()
            return ctx

        # 2. 检查容量
        if len(self._sessions) >= self._max_sessions:
            await self._evict_oldest()

        # 3. 并行执行：Agent 创建 + DB 操作
        #    - Agent 创建是 CPU 密集型，不依赖 DB
        #    - DB 记录创建和历史加载可以并行
        agent_task = self._create_agent(config)

        if self._repos:
            # 并行：创建 Agent + 创建 DB 记录 + 加载历史
            # 注意：历史加载需要 agent 实例，所以先等 agent 创建完成
            agent = await agent_task

            # 并行：DB 记录创建 + 历史加载
            await asyncio.gather(
                self._ensure_session_record(session_id, tenant_id, chatbot_id),
                self._load_history(agent, session_id),
                return_exceptions=True,
            )
        else:
            agent = await agent_task

        # 4. 创建会话上下文
        ctx = SessionContext(
            session_id=session_id,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            agent=agent,
        )

        # 5. 初始化 Timer
        self._init_timer(ctx, config)

        self._sessions[session_id] = ctx
        logger.info(f"Session created: {session_id}")

        return ctx

    async def _ensure_session_record(
        self,
        session_id: str,
        tenant_id: str,
        chatbot_id: str,
    ) -> None:
        """确保 DB 中存在会话记录（后台执行，不阻塞主流程）"""
        try:
            await self._repos.sessions.get_or_create(
                session_id=session_id,
                tenant_id=tenant_id,
                chatbot_id=chatbot_id,
            )
        except Exception as e:
            # DB 记录创建失败不应阻塞主流程
            logger.warning(f"Failed to ensure session record: {session_id}, {e}")

    def get(self, session_id: str) -> Optional[SessionContext]:
        """获取会话上下文（不创建）"""
        return self._sessions.get(session_id)

    async def destroy(self, session_id: str):
        """销毁会话"""
        ctx = self._sessions.pop(session_id, None)
        if ctx:
            ctx.cleanup()
            logger.info(f"Session destroyed: {session_id}")

    def exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        return session_id in self._sessions

    # -------------------------------------------------------------------------
    # Agent 创建
    # -------------------------------------------------------------------------

    async def _create_agent(
        self,
        config: WorkflowConfigSchema
    ) -> WorkflowAgentV2:
        """创建 Agent 实例"""
        llm = self._get_llm()
        return WorkflowAgentV2(config=config, llm=llm)

    def _get_llm(self):
        """获取 LLM 实例"""
        if self._llm_provider:
            if hasattr(self._llm_provider, 'get_decision_llm'):
                return self._llm_provider.get_decision_llm()
            return self._llm_provider
        # 延迟导入，避免循环依赖
        from api.services.llm_service import LLMService
        return LLMService.get_instance().get_decision_llm()

    async def _load_history(
        self,
        agent: WorkflowAgentV2,
        session_id: str,
        limit: int = 50
    ):
        """加载历史消息到 Agent"""
        if not self._repos:
            return

        try:
            history = await self._repos.messages.list_by_session(
                session_id=session_id,
                limit=limit,
                order="asc",
            )
            if not history:
                return

            # 转换为 Agent 消息格式
            messages = []
            for msg in history:
                role = msg.role.value if hasattr(msg.role, 'value') else msg.role
                if role == "user":
                    messages.append(UserMessage(content=msg.content))
                elif role == "assistant":
                    messages.append(AssistantMessage(content=msg.content))

            # 加载到 Agent
            if messages:
                agent._agent.load_history(messages)
                logger.debug(f"Loaded {len(messages)} history messages: {session_id}")

        except Exception as e:
            logger.warning(f"Failed to load history: {session_id}, {e}")

    # -------------------------------------------------------------------------
    # Timer 管理
    # -------------------------------------------------------------------------

    def _init_timer(self, ctx: SessionContext, config: WorkflowConfigSchema):
        """初始化会话 Timer"""
        if not config.timers:
            return

        # 使用第一个 Timer 配置
        timer_config = config.timers[0]
        if isinstance(timer_config, dict):
            ctx.timer = SessionTimer(
                session_id=ctx.session_id,
                timeout_seconds=timer_config.get("delay_seconds", 300),
                message=timer_config.get("message", "您好，请问还在吗？"),
                max_triggers=timer_config.get("max_triggers", 3),
                action=timer_config.get("action", "send_message"),
                next_timer_name=timer_config.get("next_timer"),
            )
            self._start_timer(ctx)
            logger.debug(f"Timer initialized: {ctx.session_id}")

    def _start_timer(self, ctx: SessionContext):
        """启动会话 Timer"""
        if not ctx.timer:
            return

        # 取消现有 Timer
        ctx.timer.cancel()

        async def timer_callback():
            try:
                await asyncio.sleep(ctx.timer.timeout_seconds)
                await self._trigger_timer(ctx)
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.error(f"Timer callback error: {ctx.session_id}, {e}")

        ctx.timer.task = asyncio.create_task(timer_callback())

    def reset_timer(self, session_id: str):
        """
        重置会话 Timer

        用户活动时调用，重新开始计时
        """
        ctx = self._sessions.get(session_id)
        if ctx:
            ctx.touch()
            if ctx.timer:
                ctx.timer.reset()
                self._start_timer(ctx)
                logger.debug(f"Timer reset: {session_id}")

    async def _trigger_timer(self, ctx: SessionContext):
        """触发 Timer"""
        if not ctx.timer or ctx.timer.is_exhausted():
            return

        ctx.timer.increment()
        message = ctx.timer.message
        action = ctx.timer.action

        logger.info(
            f"Timer triggered: {ctx.session_id} "
            f"({ctx.timer.trigger_count}/{ctx.timer.max_triggers})"
        )

        # 执行动作
        if action == "send_message":
            await self._send_timer_message(ctx, message)
        elif action == "close_conversation":
            await self._close_conversation(ctx, message)

        # 继续下一轮 Timer（如果未耗尽）
        if not ctx.timer.is_exhausted():
            self._start_timer(ctx)

    async def _send_timer_message(self, ctx: SessionContext, message: str):
        """发送 Timer 消息"""
        # 存储消息
        if self._repos:
            try:
                from api.models import MessageRole
                await self._repos.messages.create(
                    session_id=ctx.session_id,
                    role=MessageRole.ASSISTANT,
                    content=message,
                )
            except Exception as e:
                logger.error(f"Failed to store timer message: {e}")

        # 发送消息
        if self._send_message:
            try:
                await self._send_message(ctx.session_id, message)
            except Exception as e:
                logger.error(f"Failed to send timer message: {e}")

    async def _close_conversation(self, ctx: SessionContext, message: str):
        """关闭会话"""
        # 发送关闭消息
        await self._send_timer_message(ctx, message)

        # 更新会话状态
        if self._repos:
            try:
                await self._repos.sessions.update(
                    ctx.session_id,
                    closed_at=ctx.last_active_at,
                )
            except Exception as e:
                logger.error(f"Failed to close session: {e}")

        # 销毁会话上下文
        await self.destroy(ctx.session_id)

    # -------------------------------------------------------------------------
    # 清理
    # -------------------------------------------------------------------------

    async def _cleanup_loop(self):
        """清理循环"""
        while self._running:
            try:
                await asyncio.sleep(self._cleanup_interval)
                await self._evict_idle()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Cleanup error: {e}")

    async def _evict_idle(self):
        """回收空闲会话"""
        to_remove = [
            sid for sid, ctx in self._sessions.items()
            if ctx.idle_seconds > self._idle_timeout
        ]
        for sid in to_remove:
            await self.destroy(sid)
        if to_remove:
            logger.info(f"Evicted {len(to_remove)} idle sessions")

    async def _evict_oldest(self):
        """淘汰最旧的会话（容量满时）"""
        if not self._sessions:
            return

        oldest_sid = min(
            self._sessions,
            key=lambda k: self._sessions[k].last_active_at
        )
        await self.destroy(oldest_sid)
        logger.info(f"Evicted oldest session: {oldest_sid}")

    # -------------------------------------------------------------------------
    # 统计
    # -------------------------------------------------------------------------

    @property
    def session_count(self) -> int:
        """当前会话数"""
        return len(self._sessions)

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "session_count": len(self._sessions),
            "max_sessions": self._max_sessions,
            "idle_timeout": self._idle_timeout,
            "running": self._running,
        }

    def list_sessions(self) -> list[dict]:
        """列出所有会话"""
        return [ctx.to_dict() for ctx in self._sessions.values()]

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """获取会话信息"""
        ctx = self._sessions.get(session_id)
        return ctx.to_dict() if ctx else None
