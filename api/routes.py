"""
API 路由模块

定义所有 API 端点
"""

import logging
import time
from fastapi import APIRouter, HTTPException, status

from api.models import (
    QueryRequest,
    QueryResponse,
    SessionInfo,
    HealthResponse,
    ErrorResponse,
    AgentStats,
    ChatAsyncResponse,
    MessageRole,
    AuditAction,
    AgentStatus,
)
from api.container import AgentManagerDep, TaskManagerDep, RepositoryManagerDep
from api.core.correlation import generate_request_id

logger = logging.getLogger(__name__)

router = APIRouter()


# =============================================================================
# 查询接口
# =============================================================================


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
    """
    同步查询接口

    **工作流程**：
    1. 根据 chatbot_id + tenant_id 获取或创建 Agent
    2. 如果配置文件变更（md5_checksum），自动重新加载
    3. 执行查询并返回响应
    4. Agent 保持活跃，直到空闲超时自动回收

    **请求示例**：
    ```json
    {
        "message": "Hello, I need help with my order",
        "customer_id": "cust_123xy",
        "session_id": "68d510aedff9455e5b019b3e",
        "tenant_id": "dev-test",
        "chatbot_id": "68d510aedff9455e5b019b3e",
        "md5_checksum": "1234567890",
        "source": "bacmk_ui",
        "is_preview": false,
        "autofill_params": {},
        "session_title": "Order Inquiry"
    }
    ```

    **响应示例**：
    ```json
    {
        "session_id": "68d510aedff9455e5b019b3e",
        "message": "您的订单正在处理中，预计明天送达。",
        "status": "success",
        "agent_id": "dev-test:68d510aedff9455e5b019b3e",
        "config_hash": "abc123def456"
    }
    ```
    """
    start_time = time.time()
    correlation_id = f"R{generate_request_id()}"

    try:
        logger.info(
            f"Query request - session_id: {request.session_id}, "
            f"chatbot_id: {request.chatbot_id}, tenant_id: {request.tenant_id}, "
            f"customer_id: {request.customer_id}, message: {request.message[:50]}..."
        )

        # 1. 获取或创建会话
        session, session_created = await repos.sessions.get_or_create(
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            chatbot_id=request.chatbot_id,
            customer_id=request.customer_id,
            config_hash=request.md5_checksum,
            title=request.session_title,
            source=request.source or "api",
            is_preview=request.is_preview,
            metadata=request.autofill_params,
        )

        if session_created:
            # 记录会话创建审计日志
            await repos.audit_logs.log(
                tenant_id=request.tenant_id,
                action=AuditAction.SESSION_CREATED,
                session_id=request.session_id,
                chatbot_id=request.chatbot_id,
                correlation_id=correlation_id,
                details={"customer_id": request.customer_id, "source": request.source},
            )

        # 2. 获取或创建 Agent
        agent = await manager.get_or_create_agent(
            chatbot_id=request.chatbot_id,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            config_hash=request.md5_checksum or "default",
        )

        # 获取 Agent 信息
        agent_info = manager.get_agent_info(request.chatbot_id, request.tenant_id)
        agent_id = agent_info["agent_id"] if agent_info else None
        config_hash = agent_info["config_hash"] if agent_info else None

        # 更新会话的 agent_id 和 config_hash
        await repos.sessions.update(
            request.session_id,
            agent_id=agent_id,
            config_hash=config_hash,
        )

        # 3. 更新 Agent 状态
        await repos.agent_states.create_or_update(
            tenant_id=request.tenant_id,
            chatbot_id=request.chatbot_id,
            status=AgentStatus.PROCESSING,
            config_hash=config_hash or "",
        )
        await repos.agent_states.add_session(agent_id, request.session_id)

        # 4. 存储用户消息
        user_message = await repos.messages.create(
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            role=MessageRole.USER,
            content=request.message,
            correlation_id=correlation_id,
        )

        # 记录消息接收审计日志
        await repos.audit_logs.log(
            tenant_id=request.tenant_id,
            action=AuditAction.MESSAGE_RECEIVED,
            session_id=request.session_id,
            agent_id=agent_id,
            chatbot_id=request.chatbot_id,
            message_id=user_message.message_id,
            correlation_id=correlation_id,
            details={"content_length": len(request.message)},
        )

        # 5. 调用 WorkflowAgent
        query_start = time.time()
        result = await agent.query(
            message=request.message,
            session_id=request.session_id,
        )
        query_latency_ms = int((time.time() - query_start) * 1000)

        # 6. 存储 Agent 响应消息
        assistant_message = await repos.messages.create(
            session_id=request.session_id,
            tenant_id=request.tenant_id,
            role=MessageRole.ASSISTANT,
            content=result,
            correlation_id=correlation_id,
            parent_message_id=user_message.message_id,
            latency_ms=query_latency_ms,
        )

        # 7. 更新会话消息计数
        await repos.sessions.increment_message_count(request.session_id, increment=2)

        # 8. 更新 Agent 状态统计
        await repos.agent_states.increment_message_count(agent_id, increment=2)

        # 记录消息发送审计日志
        total_duration_ms = int((time.time() - start_time) * 1000)
        await repos.audit_logs.log(
            tenant_id=request.tenant_id,
            action=AuditAction.MESSAGE_SENT,
            session_id=request.session_id,
            agent_id=agent_id,
            chatbot_id=request.chatbot_id,
            message_id=assistant_message.message_id,
            correlation_id=correlation_id,
            details={
                "content_length": len(result),
                "query_latency_ms": query_latency_ms,
            },
            duration_ms=total_duration_ms,
        )

        logger.info(
            f"Query success - session_id: {request.session_id}, "
            f"agent_id: {agent_id}, latency: {query_latency_ms}ms, "
            f"response: {result[:50]}..."
        )

        return QueryResponse(
            session_id=request.session_id,
            message=result,
            status="success",
            agent_id=agent_id,
            config_hash=config_hash,
        )

    except FileNotFoundError as e:
        # 记录错误审计日志
        await repos.audit_logs.log(
            tenant_id=request.tenant_id,
            action=AuditAction.ERROR_OCCURRED,
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            correlation_id=correlation_id,
            success=False,
            error_message=str(e),
            details={"error_type": "ConfigurationNotFound"},
        )

        logger.error(
            f"Configuration not found - chatbot_id: {request.chatbot_id}, "
            f"tenant_id: {request.tenant_id}, error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": "ConfigurationNotFound",
                "message": str(e),
                "session_id": request.session_id,
            },
        )

    except Exception as e:
        # 记录错误审计日志
        await repos.audit_logs.log(
            tenant_id=request.tenant_id,
            action=AuditAction.ERROR_OCCURRED,
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            correlation_id=correlation_id,
            success=False,
            error_message=str(e),
            details={"error_type": type(e).__name__},
            duration_ms=int((time.time() - start_time) * 1000),
        )

        logger.error(
            f"Query failed - session_id: {request.session_id}, error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": type(e).__name__,
                "message": str(e),
                "session_id": request.session_id,
            },
        )


# =============================================================================
# 异步聊天接口（从 Parlant chat_async 移植）
# =============================================================================


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
    """
    异步聊天接口 - 同 session 新请求取消旧请求

    **核心特性**：
    - 同一 session_id 的新请求会自动取消正在处理的旧请求
    - 旧请求收到 CancelledError，返回 503 状态码
    - 新请求立即返回 202 Accepted，后台异步处理

    **使用场景**：
    - 用户快速连续发送消息
    - 用户中途修改问题
    - 需要"打断"当前对话

    **工作流程**：
    1. 生成 correlation_id 用于追踪
    2. 使用 TaskManager.restart() 取消同 session 旧任务
    3. 立即返回 202 Accepted
    4. 后台异步执行查询
    """
    correlation_id = f"R{generate_request_id()}"

    logger.info(
        f"chat_async request - session_id: {request.session_id}, "
        f"correlation_id: {correlation_id}, "
        f"chatbot_id: {request.chatbot_id}, tenant_id: {request.tenant_id}"
    )

    # 预先获取或创建会话
    session, session_created = await repos.sessions.get_or_create(
        session_id=request.session_id,
        tenant_id=request.tenant_id,
        chatbot_id=request.chatbot_id,
        customer_id=request.customer_id,
        config_hash=request.md5_checksum,
        title=request.session_title,
        source=request.source or "api",
        is_preview=request.is_preview,
        metadata=request.autofill_params,
    )

    if session_created:
        await repos.audit_logs.log(
            tenant_id=request.tenant_id,
            action=AuditAction.SESSION_CREATED,
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            correlation_id=correlation_id,
            details={"customer_id": request.customer_id, "source": request.source},
        )

    # 定义异步处理函数
    async def _process_chat():
        """后台处理聊天请求"""
        start_time = time.time()
        try:
            # 1. 获取或创建 Agent
            agent = await manager.get_or_create_agent(
                chatbot_id=request.chatbot_id,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                config_hash=request.md5_checksum or "default",
            )

            # 获取 Agent 信息
            agent_info = manager.get_agent_info(request.chatbot_id, request.tenant_id)
            agent_id = agent_info["agent_id"] if agent_info else None
            config_hash = agent_info["config_hash"] if agent_info else None

            # 更新会话
            await repos.sessions.update(
                request.session_id,
                agent_id=agent_id,
                config_hash=config_hash,
            )

            # 2. 更新 Agent 状态
            await repos.agent_states.create_or_update(
                tenant_id=request.tenant_id,
                chatbot_id=request.chatbot_id,
                status=AgentStatus.PROCESSING,
                config_hash=config_hash or "",
            )
            await repos.agent_states.add_session(agent_id, request.session_id)

            # 3. 存储用户消息
            user_message = await repos.messages.create(
                session_id=request.session_id,
                tenant_id=request.tenant_id,
                role=MessageRole.USER,
                content=request.message,
                correlation_id=correlation_id,
            )

            await repos.audit_logs.log(
                tenant_id=request.tenant_id,
                action=AuditAction.MESSAGE_RECEIVED,
                session_id=request.session_id,
                agent_id=agent_id,
                chatbot_id=request.chatbot_id,
                message_id=user_message.message_id,
                correlation_id=correlation_id,
                details={"content_length": len(request.message)},
            )

            # 4. 调用 WorkflowAgent
            query_start = time.time()
            result = await agent.query(
                message=request.message,
                session_id=request.session_id,
            )
            query_latency_ms = int((time.time() - query_start) * 1000)

            # 5. 存储 Agent 响应消息
            assistant_message = await repos.messages.create(
                session_id=request.session_id,
                tenant_id=request.tenant_id,
                role=MessageRole.ASSISTANT,
                content=result,
                correlation_id=correlation_id,
                parent_message_id=user_message.message_id,
                latency_ms=query_latency_ms,
            )

            # 6. 更新统计
            await repos.sessions.increment_message_count(request.session_id, increment=2)
            await repos.agent_states.increment_message_count(agent_id, increment=2)

            # 记录审计日志
            total_duration_ms = int((time.time() - start_time) * 1000)
            await repos.audit_logs.log(
                tenant_id=request.tenant_id,
                action=AuditAction.MESSAGE_SENT,
                session_id=request.session_id,
                agent_id=agent_id,
                chatbot_id=request.chatbot_id,
                message_id=assistant_message.message_id,
                correlation_id=correlation_id,
                details={
                    "content_length": len(result),
                    "query_latency_ms": query_latency_ms,
                },
                duration_ms=total_duration_ms,
            )

            logger.info(
                f"chat_async completed - session_id: {request.session_id}, "
                f"correlation_id: {correlation_id}, "
                f"duration: {total_duration_ms}ms, "
                f"response: {result[:50]}..."
            )

        except Exception as e:
            duration_ms = int((time.time() - start_time) * 1000)

            # 记录错误审计日志
            await repos.audit_logs.log(
                tenant_id=request.tenant_id,
                action=AuditAction.ERROR_OCCURRED,
                session_id=request.session_id,
                chatbot_id=request.chatbot_id,
                correlation_id=correlation_id,
                success=False,
                error_message=str(e),
                details={"error_type": type(e).__name__},
                duration_ms=duration_ms,
            )

            logger.error(
                f"chat_async failed - session_id: {request.session_id}, "
                f"correlation_id: {correlation_id}, "
                f"duration: {duration_ms}ms, "
                f"error: {str(e)}",
                exc_info=True,
            )

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


# =============================================================================
# 会话管理接口
# =============================================================================


@router.delete(
    "/session/{session_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Session Not Found"},
    },
    summary="释放会话",
    description="释放会话，减少 Agent 的会话计数（Agent 会在空闲超时后自动回收）",
)
async def release_session(
    session_id: str,
    chatbot_id: str,
    tenant_id: str,
    manager: AgentManagerDep,
    repos: RepositoryManagerDep,
):
    """
    释放会话

    **使用场景**：
    - 用户登出
    - 会话超时
    - 主动结束对话

    **注意**：
    - 不会立即删除 Agent，只是减少会话计数
    - Agent 会在空闲超时后自动回收
    """
    try:
        # 1. 关闭会话
        session = await repos.sessions.close(session_id)

        # 2. 释放 Agent 会话
        await manager.release_session(chatbot_id, tenant_id, session_id)

        # 3. 更新 Agent 状态
        agent_id = f"{tenant_id}:{chatbot_id}"
        await repos.agent_states.remove_session(agent_id, session_id)

        # 4. 记录审计日志
        await repos.audit_logs.log(
            tenant_id=tenant_id,
            action=AuditAction.SESSION_CLOSED,
            session_id=session_id,
            agent_id=agent_id,
            chatbot_id=chatbot_id,
            details={"message_count": session.message_count if session else 0},
        )

        logger.info(
            f"Session released - session_id: {session_id}, "
            f"chatbot_id: {chatbot_id}, tenant_id: {tenant_id}"
        )

        return {
            "status": "released",
            "session_id": session_id,
            "message": "Session released successfully",
        }

    except Exception as e:
        logger.error(
            f"Release session failed - session_id: {session_id}, error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": type(e).__name__,
                "message": str(e),
                "session_id": session_id,
            },
        )


# =============================================================================
# Agent 管理接口
# =============================================================================


@router.get(
    "/agent/{chatbot_id}",
    response_model=AgentStats,
    responses={
        404: {"model": ErrorResponse, "description": "Agent Not Found"},
    },
    summary="获取 Agent 信息",
    description="获取指定 Agent 的统计信息",
)
async def get_agent_info(
    chatbot_id: str,
    tenant_id: str,
    manager: AgentManagerDep,
):
    """
    获取 Agent 信息

    **返回**：
    - Agent ID
    - 配置哈希
    - 会话数量
    - 创建时间
    - 最后活跃时间
    """
    try:
        agent_info = manager.get_agent_info(chatbot_id, tenant_id)

        if not agent_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "AgentNotFound",
                    "message": f"Agent not found for chatbot_id={chatbot_id}, tenant_id={tenant_id}",
                },
            )

        return AgentStats(**agent_info)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Get agent info failed - chatbot_id: {chatbot_id}, "
            f"tenant_id: {tenant_id}, error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": type(e).__name__, "message": str(e)},
        )


@router.delete(
    "/agent/{chatbot_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Agent Not Found"},
    },
    summary="删除 Agent",
    description="强制删除指定 Agent（即使有活跃会话）",
)
async def delete_agent(
    chatbot_id: str,
    tenant_id: str,
    manager: AgentManagerDep,
    repos: RepositoryManagerDep,
):
    """
    删除 Agent

    **使用场景**：
    - 配置文件更新后强制重新加载
    - 清理异常 Agent
    - 手动释放资源

    **注意**：
    - 会强制删除 Agent，即使有活跃会话
    - 下次请求会自动创建新 Agent
    """
    try:
        agent_info = manager.get_agent_info(chatbot_id, tenant_id)

        if not agent_info:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={
                    "error": "AgentNotFound",
                    "message": f"Agent not found for chatbot_id={chatbot_id}, tenant_id={tenant_id}",
                },
            )

        agent_id = agent_info["agent_id"]

        # 1. 删除 Agent
        await manager.remove_agent(chatbot_id, tenant_id)

        # 2. 更新 Agent 状态为已终止
        await repos.agent_states.update(
            agent_id,
            status=AgentStatus.TERMINATED.value,
        )

        # 3. 记录审计日志
        await repos.audit_logs.log(
            tenant_id=tenant_id,
            action=AuditAction.AGENT_DESTROYED,
            agent_id=agent_id,
            chatbot_id=chatbot_id,
            details={
                "session_count": agent_info.get("session_count", 0),
                "config_hash": agent_info.get("config_hash"),
            },
        )

        logger.info(
            f"Agent deleted - chatbot_id: {chatbot_id}, tenant_id: {tenant_id}"
        )

        return {
            "status": "deleted",
            "chatbot_id": chatbot_id,
            "tenant_id": tenant_id,
            "message": "Agent deleted successfully",
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Delete agent failed - chatbot_id: {chatbot_id}, "
            f"tenant_id: {tenant_id}, error: {str(e)}"
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": type(e).__name__, "message": str(e)},
        )


# =============================================================================
# 健康检查接口
# =============================================================================


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="健康检查",
    description="检查 API 服务和 AgentManager 的健康状态",
)
async def health_check(manager: AgentManagerDep):
    """
    健康检查

    **返回**：
    - 服务状态
    - 活跃会话数
    - 活跃 Agent 数
    - API 版本
    - 运行时间
    """
    from api import __version__

    stats = manager.get_stats()

    return HealthResponse(
        status="healthy",
        active_sessions=stats["active_sessions"],
        active_agents=stats["active_agents"],
        version=__version__,
        uptime=stats["uptime"],
    )
