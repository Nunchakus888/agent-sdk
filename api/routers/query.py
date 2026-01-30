"""
查询路由模块

提供同步和异步查询接口
"""

import time
from fastapi import APIRouter, HTTPException, status

from api.models import (
    QueryRequest,
    QueryResponse,
    ChatAsyncResponse,
    ErrorResponse,
    MessageRole,
    EventType,
    EventStatus,
)
from api.container import AgentManagerDep, TaskManagerDep, RepositoryManagerDep
from api.core.correlation import get_correlation_id
from api.core.logging import get_logger, LogContext

logger = get_logger(__name__)


def create_router() -> APIRouter:
    """
    创建查询路由器（工厂函数）

    Returns:
        配置完成的 APIRouter 实例
    """
    router = APIRouter()

    @router.post(
        "/query",
        response_model=QueryResponse,
        responses={
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
            400: {"model": ErrorResponse, "description": "Bad Request"},
        },
        summary="查询接口",
        description="发送消息到 Workflow Agent 并获取响应（会话维度）",
    )
    async def query(
        request: QueryRequest,
        manager: AgentManagerDep,
        repos: RepositoryManagerDep,
    ):
        """同步查询接口"""
        start_time = time.time()
        correlation_id = get_correlation_id()

        with LogContext(
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            tenant_id=request.tenant_id,
        ):
            try:
                logger.info("Query request received")
                logger.debug(f"msg={request.message[:80]}...")

                # 1. 获取或创建会话
                with LogContext.scope("session"):
                    session_metadata = {
                        **(request.autofill_params or {}),
                        "title": request.session_title,
                        "source": request.source or "api",
                        "is_preview": request.is_preview,
                        "config_hash": request.md5_checksum,
                    }
                    session, session_created = await repos.sessions.get_or_create(
                        session_id=request.session_id,
                        tenant_id=request.tenant_id,
                        chatbot_id=request.chatbot_id,
                        customer_id=request.customer_id,
                        metadata=session_metadata,
                    )

                    logger.info("Session created" if session_created else "Session reused")

                # 2. 获取或创建 Agent
                with LogContext.scope("agent"):
                    agent = await manager.get_or_create_agent(
                        chatbot_id=request.chatbot_id,
                        tenant_id=request.tenant_id,
                        session_id=request.session_id,
                        config_hash=request.md5_checksum or "default",
                    )

                    agent_info = manager.get_agent_info(request.chatbot_id, request.tenant_id)
                    agent_id = agent_info["agent_id"] if agent_info else None
                    config_hash = agent_info["config_hash"] if agent_info else None

                    await repos.sessions.update(request.session_id, agent_id=agent_id)

                # 3. 存储用户消息
                user_message = await repos.messages.create(
                    session_id=request.session_id,
                    role=MessageRole.USER,
                    content=request.message,
                    correlation_id=correlation_id,
                )

                # 4. 分配事件 offset 并记录请求事件
                event_offset = await repos.sessions.allocate_event_offset(request.session_id)
                await repos.events.create(
                    session_id=request.session_id,
                    correlation_id=correlation_id,
                    event_type=EventType.LLM_DECISION,
                    offset=event_offset,
                    message_id=user_message.message_id,
                    action="query",
                    status=EventStatus.STARTED,
                    input_data={"message": request.message[:500]},
                )

                # 5. 调用 WorkflowAgent
                with LogContext.scope("query", agent_id=agent_id):
                    query_start = time.time()
                    result = await agent.query(
                        message=request.message,
                        session_id=request.session_id,
                    )
                    query_latency_ms = int((time.time() - query_start) * 1000)
                    logger.info(f"Query completed ({query_latency_ms}ms, {len(result)} chars)")
                    logger.debug(f"response={result[:150]}...")

                # 6. 存储 Agent 响应消息
                assistant_message = await repos.messages.create(
                    session_id=request.session_id,
                    role=MessageRole.ASSISTANT,
                    content=result,
                    correlation_id=correlation_id,
                )

                # 7. 记录完成事件
                total_duration_ms = int((time.time() - start_time) * 1000)
                event_offset = await repos.sessions.allocate_event_offset(request.session_id)
                await repos.events.create(
                    session_id=request.session_id,
                    correlation_id=correlation_id,
                    event_type=EventType.RESPONSE_GENERATE,
                    offset=event_offset,
                    message_id=assistant_message.message_id,
                    action="query_complete",
                    status=EventStatus.COMPLETED,
                    duration_ms=query_latency_ms,
                    output_data={"response_length": len(result)},
                )

                logger.info(f"Request done ({total_duration_ms}ms total, {query_latency_ms}ms query)")

                return QueryResponse(
                    session_id=request.session_id,
                    message=result,
                    status="success",
                    agent_id=agent_id,
                    config_hash=config_hash,
                )

            except FileNotFoundError as e:
                logger.error(f"Configuration not found: {e}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={
                        "error": "ConfigurationNotFound",
                        "message": str(e),
                        "session_id": request.session_id,
                    },
                )

            except Exception as e:
                logger.error(f"Query failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": type(e).__name__,
                        "message": str(e),
                        "session_id": request.session_id,
                    },
                )

    @router.post(
        "/chat_async",
        response_model=ChatAsyncResponse,
        status_code=status.HTTP_202_ACCEPTED,
        responses={
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
            503: {"description": "Service Unavailable - Request Cancelled"},
        },
        summary="异步聊天接口",
        description="发送消息到 Workflow Agent，同 session 新请求会取消旧请求",
    )
    async def chat_async(
        request: QueryRequest,
        manager: AgentManagerDep,
        task_manager: TaskManagerDep,
        repos: RepositoryManagerDep,
    ):
        """异步聊天接口 - 同 session 新请求取消旧请求"""
        correlation_id = get_correlation_id()

        with LogContext(
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            tenant_id=request.tenant_id,
        ):
            logger.info("chat_async request received")

            # 预先获取或创建会话
            session_metadata = {
                **(request.autofill_params or {}),
                "title": request.session_title,
                "source": request.source or "api",
                "is_preview": request.is_preview,
                "config_hash": request.md5_checksum,
            }
            session, session_created = await repos.sessions.get_or_create(
                session_id=request.session_id,
                tenant_id=request.tenant_id,
                chatbot_id=request.chatbot_id,
                customer_id=request.customer_id,
                metadata=session_metadata,
            )

            if session_created:
                logger.info("Session created for chat_async")

        # 定义异步处理函数
        async def _process_chat():
            """后台处理聊天请求"""
            start_time = time.time()
            with LogContext(
                session_id=request.session_id,
                chatbot_id=request.chatbot_id,
                tenant_id=request.tenant_id,
            ):
                try:
                    # 1. 获取或创建 Agent
                    agent = await manager.get_or_create_agent(
                        chatbot_id=request.chatbot_id,
                        tenant_id=request.tenant_id,
                        session_id=request.session_id,
                        config_hash=request.md5_checksum or "default",
                    )

                    agent_info = manager.get_agent_info(request.chatbot_id, request.tenant_id)
                    agent_id = agent_info["agent_id"] if agent_info else None

                    await repos.sessions.update(request.session_id, agent_id=agent_id)

                    # 2. 存储用户消息
                    user_message = await repos.messages.create(
                        session_id=request.session_id,
                        role=MessageRole.USER,
                        content=request.message,
                        correlation_id=correlation_id,
                    )

                    # 3. 记录开始事件
                    event_offset = await repos.sessions.allocate_event_offset(request.session_id)
                    await repos.events.create(
                        session_id=request.session_id,
                        correlation_id=correlation_id,
                        event_type=EventType.LLM_DECISION,
                        offset=event_offset,
                        message_id=user_message.message_id,
                        action="chat_async",
                        status=EventStatus.STARTED,
                        input_data={"message": request.message[:500]},
                    )

                    # 4. 调用 WorkflowAgent
                    with LogContext.scope("query", agent_id=agent_id):
                        query_start = time.time()
                        result = await agent.query(
                            message=request.message,
                            session_id=request.session_id,
                        )
                        query_latency_ms = int((time.time() - query_start) * 1000)

                    # 5. 存储 Agent 响应消息
                    assistant_message = await repos.messages.create(
                        session_id=request.session_id,
                        role=MessageRole.ASSISTANT,
                        content=result,
                        correlation_id=correlation_id,
                    )

                    # 6. 记录完成事件
                    total_duration_ms = int((time.time() - start_time) * 1000)
                    event_offset = await repos.sessions.allocate_event_offset(request.session_id)
                    await repos.events.create(
                        session_id=request.session_id,
                        correlation_id=correlation_id,
                        event_type=EventType.RESPONSE_GENERATE,
                        offset=event_offset,
                        message_id=assistant_message.message_id,
                        action="chat_async_complete",
                        status=EventStatus.COMPLETED,
                        duration_ms=query_latency_ms,
                        output_data={"response_length": len(result)},
                    )

                    logger.info(f"chat_async completed, duration={total_duration_ms}ms")

                except Exception as e:
                    duration_ms = int((time.time() - start_time) * 1000)
                    # 记录错误事件
                    event_offset = await repos.sessions.allocate_event_offset(request.session_id)
                    await repos.events.create(
                        session_id=request.session_id,
                        correlation_id=correlation_id,
                        event_type=EventType.ERROR,
                        offset=event_offset,
                        action="chat_async_error",
                        status=EventStatus.FAILED,
                        duration_ms=duration_ms,
                        error=str(e),
                    )
                    logger.error(f"chat_async failed: {e}", exc_info=True)

        # 核心：使用 restart 取消同 session 旧任务
        await task_manager.restart(
            _process_chat(),
            tag=f"session:{request.session_id}",
        )

        return ChatAsyncResponse(
            status=202,
            code=0,
            message="processing",
            correlation_id=correlation_id,
            session_id=request.session_id,
        )

    return router
