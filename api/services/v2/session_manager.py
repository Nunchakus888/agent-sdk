"""
会话管理器

职责：
1. 会话生命周期管理（创建、获取、销毁）
2. Agent 实例管理（会话级）
3. Timer 管理（会话级）
4. 空闲会话回收
5. 配置加载（按需，仅在创建 session 时）

设计原则：
- Session 存在 → 直接返回（快速路径）
- Session 不存在 → 从 DB/HTTP 加载配置 → 创建 Agent → 创建 Session
"""

import asyncio
import logging
from typing import Optional, Callable, Awaitable, TYPE_CHECKING

from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from bu_agent_sdk.schemas import WorkflowConfigSchema
from bu_agent_sdk.llm.messages import UserMessage, AssistantMessage

from api.services.v2.session_context import SessionContext, SessionTimer
from api.services.repositories import RepositoryManager, MessageRepository
from api.services.llm_service import LLMService

if TYPE_CHECKING:
    from api.services.database import Database

logger = logging.getLogger(__name__)


class SessionManager:
    """
    会话管理器

    管理所有会话的生命周期，包括：
    - 会话创建和销毁
    - Agent 实例管理（会话级）
    - Timer 管理（会话级）
    - 空闲会话自动回收
    - 配置加载（按需，仅在创建 session 时从 DB/HTTP 加载）

    设计原则：
    - Session 存在 → 直接返回（快速路径，不加载配置）
    - Session 不存在 → 加载配置 → 创建 Agent → 创建 Session
    - 配置更新（config_hash 变化）→ 销毁旧 session → 创建新 session

    Usage:
        ```python
        session_mgr = SessionManager(repos=repos, llm_provider=llm_provider)
        await session_mgr.start()

        # 获取或创建会话（内部处理配置加载）
        ctx = await session_mgr.get_or_create(
            session_id="sess_123",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="abc123",
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
        database: "Database | None" = None,
        llm_provider: LLMService | None = None,
        idle_timeout: int = 1800,
        cleanup_interval: int = 60,
        max_sessions: int = 10000,
        enable_llm_parsing: bool = True,
    ):
        """
        Args:
            repos: RepositoryManager
            database: Database 实例（用于配置持久化）
            llm_provider: LLM 提供者
            idle_timeout: 空闲超时（秒），默认 30 分钟
            cleanup_interval: 清理间隔（秒）
            max_sessions: 最大会话数
            enable_llm_parsing: 是否启用 LLM 增强解析
        """
        self._repos = repos
        self._db = database
        self._llm_provider = llm_provider
        self._idle_timeout = idle_timeout
        self._cleanup_interval = cleanup_interval
        self._max_sessions = max_sessions
        self._enable_llm_parsing = enable_llm_parsing

        # 会话池：session_id -> SessionContext
        self._sessions: dict[str, SessionContext] = {}

        # 消息发送回调
        self._send_message: Optional[Callable[[str, str], Awaitable[None]]] = None

        # 清理任务
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

        # HTTP 配置加载器（延迟初始化）
        self._http_loader = None

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
        config_hash: str,
    ) -> SessionContext:
        """
        获取或创建会话上下文

        Args:
            session_id: 会话 ID
            tenant_id: 租户 ID
            chatbot_id: Chatbot ID
            config_hash: 配置哈希（用于检测配置更新）

        Returns:
            SessionContext 实例

        设计：
        - Session 存在且配置未变 → 直接返回（快速路径）
        - Session 存在但配置变了 → 销毁旧 session → 创建新 session
        - Session 不存在 → 从 DB/HTTP 加载配置 → 创建 Agent → 创建 Session
        """
        # 1. 检查现有会话（快速路径）
        if session_id in self._sessions:
            ctx = self._sessions[session_id]
            # 检查配置是否更新
            if ctx.config_hash == config_hash:
                ctx.touch()
                return ctx
            # 配置更新，销毁旧 session
            logger.info(f"Config changed for session {session_id}, recreating")
            await self.destroy(session_id)

        # 2. 检查容量
        if len(self._sessions) >= self._max_sessions:
            await self._evict_oldest()

        # 3. 加载配置
        config = await self._load_config(config_hash, tenant_id, chatbot_id)

        # 4. 并行执行：Agent 创建 + DB 操作
        agent_task = self._create_agent(config)

        if self._repos:
            agent = await agent_task
            await asyncio.gather(
                self._ensure_session_record(session_id, tenant_id, chatbot_id),
                self._load_history(agent, session_id),
                return_exceptions=True,
            )
        else:
            agent = await agent_task

        # 5. 创建会话上下文
        ctx = SessionContext(
            session_id=session_id,
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            agent=agent,
            config_hash=config_hash,
        )

        # 6. 初始化 Timer
        self._init_timer(ctx, config)

        self._sessions[session_id] = ctx
        logger.info(f"Session created: {session_id}")

        return ctx

    async def _load_config(
        self,
        config_hash: str,
        tenant_id: str,
        chatbot_id: str,
    ) -> WorkflowConfigSchema:
        """
        加载配置（DB → HTTP）

        流程：
        1. DB 命中且 hash 匹配 → 直接返回（access_count 自动 +1）
        2. DB 未命中或 hash 不匹配 → HTTP 加载 → 解析 → 存储 DB → 返回

        设计原则：
        - 按 chatbot_id 索引，每个 chatbot 一条记录
        - config_hash 用于缓存失效检测
        - 访问统计自动递增
        """
        # 从 DB 加载（自动递增 access_count）
        if self._repos:
            doc = await self._repos.configs.get(chatbot_id, tenant_id, expected_hash=config_hash)
            if doc:
                logger.debug(f"Config DB HIT: {chatbot_id}, hash={config_hash[:12]}")
                return WorkflowConfigSchema(**doc.parsed_config)

        # DB 未命中或 hash 不匹配，从 HTTP 加载
        logger.info(f"Config DB MISS: chatbot={chatbot_id}, hash={config_hash[:12]}")
        return await self._load_config_from_http(config_hash, tenant_id, chatbot_id)

    async def _load_config_from_http(
        self,
        config_hash: str,
        tenant_id: str,
        chatbot_id: str,
    ) -> WorkflowConfigSchema:
        """从 HTTP 加载配置并存储到 DB"""
        from config.http_config import HttpConfigLoader, AgentConfigRequest

        if self._http_loader is None:
            self._http_loader = HttpConfigLoader(logger)

        raw_config = await self._http_loader.load_config_from_http(
            AgentConfigRequest(tenant_id=tenant_id, chatbot_id=chatbot_id)
        )

        config = await self._parse_config(raw_config, config_hash)

        # 存储到 DB（解耦的独立操作）
        await self._save_config_to_db(
            tenant_id=tenant_id,
            chatbot_id=chatbot_id,
            config_hash=config_hash,
            raw_config=raw_config,
            parsed_config=config.model_dump(),
        )

        return config

    async def _save_config_to_db(
        self,
        tenant_id: str,
        chatbot_id: str,
        config_hash: str,
        raw_config: dict,
        parsed_config: dict,
    ) -> None:
        """
        存储配置到 DB（独立方法，便于复用和测试）
        
        按 chatbot_id 更新：
        - 每个 chatbot 一条记录
        - 更新时保留 created_at, 累加 access_count
        """
        if not self._repos:
            return

        try:
            await self._repos.configs.upsert(
                chatbot_id=chatbot_id,
                tenant_id=tenant_id,
                config_hash=config_hash,
                raw_config=raw_config,
                parsed_config=parsed_config,
            )
            logger.debug(f"Config stored: {chatbot_id}, hash={config_hash[:12]}")
        except Exception as e:
            # DB 存储失败不应阻塞主流程
            logger.warning(f"Failed to store config to DB: {e}")

    async def _parse_config(
        self, raw_config: dict, config_hash: str
    ) -> WorkflowConfigSchema:
        """解析配置"""
        import os

        # 环境变量覆盖
        max_iterations = int(
            os.getenv("MAX_ITERATIONS", raw_config.get("max_iterations", 5))
        )
        iteration_strategy = os.getenv(
            "ITERATION_STRATEGY", raw_config.get("iteration_strategy", "sop_driven")
        )

        # LLM 增强解析
        llm_parsed = await self._llm_enhance(raw_config)

        # 合并配置
        final_config = {
            "retrieve_knowledge_url": raw_config.get("retrieve_knowledge_url"),
            "max_iterations": max_iterations,
            "iteration_strategy": iteration_strategy,
            **llm_parsed,
            "system_actions": raw_config.get("system_actions"),
            "agent_actions": raw_config.get("agent_actions"),
        }

        logger.info(
            f"Config parsed: hash={config_hash[:12]}, "
            f"llm={self._enable_llm_parsing}"
        )

        return WorkflowConfigSchema(**final_config)

    async def _llm_enhance(self, raw_config: dict) -> dict:
        """
        LLM 增强配置

        增强字段：
        - instructions: 从 basic_settings 生成/增强
        - need_greeting: 推断或增强
        - timers: 从 instruction 文本推断
        - constraints: 安全边界

        不处理（其他地方处理）：
        - tools: 固定配置
        - max_iterations: 环境变量
        """
        if not self._enable_llm_parsing:
            return raw_config

        try:
            from api.services.v2.config_enhancer import ConfigEnhancer

            llm = self._get_llm()
            enhancer = ConfigEnhancer(llm=llm)

            logger.info("Starting LLM config enhancement...")
            enhanced = await enhancer.enhance(raw_config)

            # 合并：增强字段覆盖原始配置
            result = {**raw_config, **enhanced}

            logger.info("LLM config enhancement completed")
            return result

        except Exception as e:
            logger.error(f"LLM enhancement failed: {e}, using original config")
            return raw_config

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
