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
)
from api.dependencies import AgentManagerDep, TaskManagerDep
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
async def query(request: QueryRequest, manager: AgentManagerDep):
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
    try:
        logger.info(
            f"Query request - session_id: {request.session_id}, "
            f"chatbot_id: {request.chatbot_id}, tenant_id: {request.tenant_id}, "
            f"customer_id: {request.customer_id}, message: {request.message[:50]}..."
        )

        # 获取或创建 Agent
        # config_hash 由客户端提供，用于配置变更检测
        agent = await manager.get_or_create_agent(
            chatbot_id=request.chatbot_id,
            tenant_id=request.tenant_id,
            session_id=request.session_id,
            config_hash=request.md5_checksum or "default",
        )

        # 调用 WorkflowAgent
        result = await agent.query(
            message=request.message,
            session_id=request.session_id,
        )

        # 获取 Agent 信息
        agent_info = manager.get_agent_info(request.chatbot_id, request.tenant_id)
        agent_id = agent_info["agent_id"] if agent_info else None
        config_hash = agent_info["config_hash"] if agent_info else None

        logger.info(
            f"Query success - session_id: {request.session_id}, "
            f"agent_id: {agent_id}, response: {result[:50]}..."
        )

        return QueryResponse(
            session_id=request.session_id,
            message=result,
            status="success",
            agent_id=agent_id,
            config_hash=config_hash,
        )

    except FileNotFoundError as e:
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

    # 定义异步处理函数
    async def _process_chat():
        """后台处理聊天请求"""
        start_time = time.time()
        try:
            # 获取或创建 Agent
            agent = await manager.get_or_create_agent(
                chatbot_id=request.chatbot_id,
                tenant_id=request.tenant_id,
                session_id=request.session_id,
                config_hash=request.md5_checksum or "default",
            )

            # 调用 WorkflowAgent
            result = await agent.query(
                message=request.message,
                session_id=request.session_id,
            )

            duration = time.time() - start_time
            logger.info(
                f"chat_async completed - session_id: {request.session_id}, "
                f"correlation_id: {correlation_id}, "
                f"duration: {duration:.2f}s, "
                f"response: {result[:50]}..."
            )

            # TODO: 可以在这里添加回调通知机制
            # await notify_callback(correlation_id, result, duration)

        except Exception as e:
            duration = time.time() - start_time
            logger.error(
                f"chat_async failed - session_id: {request.session_id}, "
                f"correlation_id: {correlation_id}, "
                f"duration: {duration:.2f}s, "
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
        await manager.release_session(chatbot_id, tenant_id, session_id)

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

        await manager.remove_agent(chatbot_id, tenant_id)

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
