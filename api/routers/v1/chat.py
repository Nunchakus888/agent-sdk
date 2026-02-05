"""
V1 Chat 异步回调 API

设计：
- POST /chat: 异步聊天接口，立即返回 correlation_id
- 后台处理完成后通过 HTTP POST 回调通知结果
- 支持同 session 新请求自动取消旧请求
- 首次会话自动发送问候消息（如果配置了 need_greeting）

回调地址：CHAT_CALLBACK_HOST + /api/callback/agent/receive
"""

import asyncio
import time
from typing import Any, Optional, TYPE_CHECKING
from dataclasses import dataclass, field
from enum import Enum
from fastapi import APIRouter, status
from pydantic import BaseModel, Field, ConfigDict, field_validator

if TYPE_CHECKING:
    from api.utils.config.http_config import AgentConfigRequest

from api.core.correlation import get_correlation_id
from api.core.logging import get_logger, LogContext
from api.services.v2 import EventCollector, QueryRecorder
from api.services.callback import (
    CallbackService,
    CallbackCode,
    CallbackPayload,
    CallbackData,
    get_callback_service,
)

logger = get_logger(__name__)


# 兼容旧代码的别名
AsyncChatCode = CallbackCode
ChatCallbackResponse = CallbackPayload
ChatCallbackData = CallbackData


# =============================================================================
# 请求/响应模型
# =============================================================================

class EventSourceDTO(Enum):
    """
    Source of an event in the session.

    Identifies who or what generated the event.
    """
    AI_AGENT = "ai_agent"
    SYSTEM = "system"
    CUSTOMER = "customer"
    BACK_UI = "back_ui"
    PREVIEW_UI = "preview_ui"
    DEVELOPMENT = "development"


class AgentConfigMixin(BaseModel):
    """Mixin with common agent config fields for request DTOs."""

    tenant_id: str = Field(
        default="", description="Tenant ID", examples=["tenant_123xyz"]
    )
    chatbot_id: str = Field(
        default="", description="Chatbot ID", examples=["chatbot_123xyz"]
    )
    session_id: str = Field(description="Session ID", examples=["sess_123xyz", "12345"])

    @field_validator("session_id", mode="before")
    @classmethod
    def convert_session_id_to_str(cls, v: Any) -> str:
        """Convert session_id to string if it's a number."""
        if isinstance(v, (int, float)):
            return str(v)
        return v


class ChatRequest(AgentConfigMixin):
    """Chat request"""

    message: str = Field(
        ..., description="message to send to the AI agent", min_length=1
    )
    customer_id: Optional[str] = Field(
        default=None, description="customer ID, auto-created if not provided"
    )
    md5_checksum: Optional[str] = Field(
        default=None, description="MD5 checksum for config change detection"
    )
    timeout: int = Field(
        default=300, description="timeout in seconds for AI response", ge=1, le=600
    )
    source: EventSourceDTO = Field(
        default=EventSourceDTO.CUSTOMER, description="source of the event"
    )
    is_preview: bool = Field(
        default=False, description="whether to preview actionbooks"
    )
    preview_action_book_ids: list[str] = Field(
        default=[], description="actionbook IDs to preview"
    )
    autofill_params: dict = Field(default={}, description="auto-fill params for data-connector")
    session_title: str = Field(default="", description="title for new sessions")
    timeout: int = Field(default=300, description="timeout in seconds for AI response", ge=1, le=600)

    model_config = ConfigDict(
        json_schema_extra={
            "message": "Hello, I need help with my order",
            "customer_id": "cust_123xy",
            "session_id": "68d510aedff9455e5b019b3e",  # Required for async chat
            "tenant_id": "dev-test",
            "chatbot_id": "68d510aedff9455e5b019b3e",
            "md5_checksum": "1234567890",
            "source": EventSourceDTO.BACK_UI,
            "is_preview": False,
            "preview_action_book_ids": [],
            "autofill_params": {},
            "session_title": "",
            "timeout": 60,
        }
    )

    def to_config_request(self) -> "AgentConfigRequest":
        """转换为 AgentConfigRequest"""
        from api.utils.config.http_config import AgentConfigRequest

        return AgentConfigRequest(
            session_id=self.session_id,
            tenant_id=self.tenant_id,
            chatbot_id=self.chatbot_id,
            md5_checksum=self.md5_checksum,
            preview=self.is_preview,
            # action_book_id is a list of action book ids
            action_book_id=self.preview_action_book_ids or None,
            extra_param=self.autofill_params or None,
        )


class ChatAsyncResponse(BaseModel):
    """异步聊天立即响应"""

    status: int = Field(default=202, description="HTTP状态码")
    code: int = Field(default=0, description="业务状态码")
    message: str = Field(default="PROCESSING", description="状态消息")
    correlation_id: str = Field(..., description="关联ID")
    session_id: str = Field(..., description="会话ID")


# =============================================================================
# 任务管理器（会话级取消）
# =============================================================================


@dataclass
class PendingTask:
    """待处理任务"""
    task: asyncio.Task
    correlation_id: str
    session_id: str
    created_at: float = field(default_factory=time.time)


class ChatTaskManager:
    """
    Chat 任务管理器

    功能：
    - 管理每个 session 的待处理任务
    - 新请求自动取消同 session 的旧请求
    - 线程安全
    """

    def __init__(self):
        self._tasks: dict[str, PendingTask] = {}
        self._lock = asyncio.Lock()

    async def register(
        self, session_id: str, correlation_id: str, task: asyncio.Task
    ) -> Optional[PendingTask]:
        """注册新任务，返回被取消的旧任务（如果有）"""
        async with self._lock:
            old_task = self._tasks.get(session_id)
            self._tasks[session_id] = PendingTask(
                task=task,
                correlation_id=correlation_id,
                session_id=session_id,
            )
            if old_task and not old_task.task.done():
                old_task.task.cancel()
                return old_task
            return None

    async def unregister(self, session_id: str) -> None:
        """注销任务"""
        async with self._lock:
            self._tasks.pop(session_id, None)

    def get(self, session_id: str) -> Optional[PendingTask]:
        """获取任务"""
        return self._tasks.get(session_id)


# 全局任务管理器
_task_manager = ChatTaskManager()


# =============================================================================
# 便捷函数（兼容旧代码）
# =============================================================================


async def send_callback(payload: CallbackPayload) -> bool:
    """send callback (convenient function, compatible with old code)"""
    return await get_callback_service().send(payload)


# =============================================================================
# 路由创建
# =============================================================================


def create_router() -> APIRouter:
    """创建 V1 Chat 路由"""
    router = APIRouter()

    def get_deps():
        from api.container import get_session_manager, get_repository_manager
        return get_session_manager, get_repository_manager

    @router.post(
        "/chat",
        response_model=ChatAsyncResponse,
        status_code=status.HTTP_202_ACCEPTED,
        summary="async chat interface",
        description="return correlation_id immediately, and notify the result via callback after processing",
    )
    async def chat_async(request: ChatRequest):
        start_time = time.time()
        correlation_id = get_correlation_id()

        get_session_manager, get_repository_manager = get_deps()
        session_manager = get_session_manager()
        repos = get_repository_manager()
        callback_service = get_callback_service()

        with LogContext(
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            tenant_id=request.tenant_id,
        ):
            async def process_chat():
                """background processing task"""
                nonlocal start_time
                try:
                    # 获取或创建会话（返回是否为新会话）
                    is_new_session = not session_manager.exists(request.session_id)

                    ctx = await session_manager.get_or_create(request.to_config_request())

                    # 新会话且配置了问候语，先发送问候消息
                    if is_new_session and ctx.agent.config.need_greeting:
                        greeting_duration = time.time() - start_time
                        await callback_service.send_greeting(
                            correlation_id=correlation_id,
                            session_id=request.session_id,
                            greeting_message=ctx.agent.config.need_greeting,
                            duration=greeting_duration,
                        )
                        logger.info(
                            f"✅ Greeting sent: {request.session_id}, "
                            f"duration={greeting_duration:.3f}s"
                        )

                    # 创建事件收集器
                    collector = EventCollector(
                        correlation_id=correlation_id,
                        session_id=request.session_id,
                        user_message=request.message,
                    )

                    # 执行查询（带超时）
                    try:
                        async with asyncio.timeout(request.timeout):
                            async for event in ctx.agent.query_stream(request.message):
                                collector.collect(event)
                    except asyncio.TimeoutError:
                        duration = time.time() - start_time
                        await callback_service.send_timeout(correlation_id, duration)
                        return

                    # 获取 usage
                    usage = await ctx.agent.get_usage()
                    ctx.increment_query()
                    session_manager.reset_timer(request.session_id)

                    # 检查是否有响应
                    if not collector.final_response:
                        duration = time.time() - start_time
                        await callback_service.send_error(
                            correlation_id, duration, "NO_EVENTS_FOUND"
                        )
                        return

                    # 发送成功回调
                    duration = time.time() - start_time
                    await callback_service.send_success(
                        correlation_id=correlation_id,
                        session_id=request.session_id,
                        message=collector.final_response,
                        duration=duration,
                        total_tokens=usage.total_tokens if usage else 0,
                    )

                    logger.info(
                        f"Chat completed: {request.session_id}, "
                        f"tokens={usage.total_tokens if usage else 0}, "
                        f"duration={duration:.2f}s"
                    )

                    # 后台记录 messages / usages（与 V2 保持一致）
                    if repos:
                        recorder = QueryRecorder(repos)
                        recorder.record_async(collector, usage)

                except asyncio.CancelledError:
                    duration = time.time() - start_time
                    await callback_service.send_cancelled(correlation_id, duration)
                    logger.info(f"Chat cancelled: {request.session_id}")

                except Exception as e:
                    duration = time.time() - start_time
                    await callback_service.send_error(
                        correlation_id, duration, f"PROCESSING_ERROR: {str(e)}"
                    )
                    logger.error(f"Chat error: {e}", exc_info=True)

                finally:
                    await _task_manager.unregister(request.session_id)

            # 创建后台任务
            task = asyncio.create_task(process_chat())

            # 注册任务（自动取消旧任务）
            cancelled_task = await _task_manager.register(
                session_id=request.session_id,
                correlation_id=correlation_id,
                task=task,
            )

            if cancelled_task:
                logger.info(
                    f"❌ Cancelled old task: {cancelled_task.correlation_id} "
                    f"for session {request.session_id}"
                )

            return ChatAsyncResponse(
                status=202,
                code=0,
                message="PROCESSING",
                correlation_id=correlation_id,
                session_id=request.session_id,
            )

    return router
