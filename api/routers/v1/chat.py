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
from api.services.cancellable_tasks import CancellableTaskService
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
            "example": {
                "message": "Hello, I need help with my order",
                "customer_id": "100000",
                "session_id": "100000",
                "tenant_id": "6336b6724011d05a0edbbe1a",
                "chatbot_id": "698314232b6a6c6eca434c01",
                "md5_checksum": "1234567890",
                "source": "customer",
                "is_preview": False,
                "preview_action_book_ids": [],
                "autofill_params": {},
                "session_title": "",
                "timeout": 60,
            }
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


# 全局任务服务
_tasks = CancellableTaskService()


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
                """
                后台处理任务

                分为两个阶段：
                1. Session 初始化阶段（不可取消）：创建 session、发送 greeting
                2. AI 交互阶段（可取消）：执行查询、发送回调
                """
                nonlocal start_time
                greeting_tokens = 0

                try:
                    # ========================================
                    # 阶段 1: Session 初始化（不可取消）
                    # ========================================
                    is_new_session = not session_manager.exists(request.session_id)

                    # 获取或创建会话（可能触发 LLM 配置解析）
                    ctx = await session_manager.get_or_create(request.to_config_request())

                    # 获取配置解析消耗的 tokens（首次解析时有值，缓存命中时为 0）
                    greeting_tokens = ctx.config_parse_tokens

                    ctx.set_request_context(correlation_id=correlation_id)

                    # 新会话且配置了问候语，发送问候消息
                    if is_new_session and ctx.agent.config.need_greeting:
                        greeting_duration = time.time() - start_time
                        greeting_msg = ctx.agent.config.need_greeting
                        await callback_service.send_greeting(
                            correlation_id=correlation_id,
                            session_id=request.session_id,
                            greeting_message=greeting_msg,
                            duration=greeting_duration,
                            total_tokens=greeting_tokens,
                        )

                        # 复用 QueryRecorder 写入 greeting 到 messages 表
                        if repos:
                            greeting_collector = EventCollector(
                                correlation_id=correlation_id,
                                session_id=request.session_id,
                                final_response=greeting_msg,
                            )
                            QueryRecorder(repos).record_async(greeting_collector)

                        logger.info(
                            f"✅ Greeting sent: {request.session_id}, "
                            f"tokens={greeting_tokens}, duration={greeting_duration:.3f}s"
                        )

                    # ========================================
                    # 阶段 2: AI 交互（可取消）
                    # ========================================
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
                    current_tokens = usage.total_tokens if usage else 0

                    # 记录 tokens 并获取总数（含被取消任务的累计）
                    _tasks.set_tokens(request.session_id, current_tokens)
                    total_tokens = _tasks.total_tokens(request.session_id)

                    ctx.increment_query()
                    session_manager.reset_timer(request.session_id)

                    # 检查是否有响应
                    if not collector.final_response:
                        duration = time.time() - start_time
                        await callback_service.send_error(
                            correlation_id, duration, "NO_EVENTS_FOUND"
                        )
                        return

                    # 记录 messages / usages
                    if repos:
                        recorder = QueryRecorder(repos)
                        await recorder.record(collector, usage)

                    # 发送成功回调
                    duration = time.time() - start_time
                    await callback_service.send_success(
                        correlation_id=correlation_id,
                        session_id=request.session_id,
                        message=collector.final_response,
                        duration=duration,
                        total_tokens=total_tokens,
                    )

                    logger.info(
                        f"📊 Chat completed: {request.session_id}, "
                        f"tokens={total_tokens}, duration={duration:.2f}s"
                    )

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

            # 启动任务（自动取消旧任务）
            result = await _tasks.restart(process_chat(), tag=request.session_id)

            if result.was_cancelled:
                logger.info(f"❌ Cancelled old task for session {request.session_id}")

            return ChatAsyncResponse(
                status=202,
                code=0,
                message="PROCESSING",
                correlation_id=correlation_id,
                session_id=request.session_id,
            )

    return router
