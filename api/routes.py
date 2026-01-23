"""
API 路由模块

定义所有 API 端点
"""

import logging
from fastapi import APIRouter, HTTPException, status

from api.models import (
    QueryRequest,
    QueryResponse,
    SessionInfo,
    HealthResponse,
    ErrorResponse,
)
from api.dependencies import WorkflowAgentDep

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
    description="发送消息到 Workflow Agent 并获取响应",
)
async def query(request: QueryRequest, agent: WorkflowAgentDep):
    """
    同步查询接口

    **使用场景**：
    - 简单的请求-响应模式
    - 不需要实时反馈

    **请求示例**：
    ```json
    {
        "message": "你好，帮我查询订单状态",
        "session_id": "user_123_session_001",
        "user_id": "user_123"
    }
    ```

    **响应示例**：
    ```json
    {
        "session_id": "user_123_session_001",
        "message": "您的订单正在处理中，预计明天送达。",
        "status": "success"
    }
    ```
    """
    try:
        logger.info(
            f"Query request - session_id: {request.session_id}, "
            f"user_id: {request.user_id}, message: {request.message[:50]}..."
        )

        # 调用 WorkflowAgent
        result = await agent.query(
            message=request.message,
            session_id=request.session_id,
        )

        logger.info(
            f"Query success - session_id: {request.session_id}, "
            f"response: {result[:50]}..."
        )

        return QueryResponse(
            session_id=request.session_id, message=result, status="success"
        )

    except Exception as e:
        logger.error(
            f"Query failed - session_id: {request.session_id}, error: {str(e)}",
            exc_info=True,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": type(e).__name__, "message": str(e)},
        )


# =============================================================================
# 会话管理接口
# =============================================================================


@router.get(
    "/session/{session_id}",
    response_model=SessionInfo,
    responses={
        404: {"model": ErrorResponse, "description": "Session Not Found"},
    },
    summary="获取会话信息",
    description="获取指定会话的状态和元数据",
)
async def get_session(session_id: str, agent: WorkflowAgentDep):
    """
    获取会话信息

    **返回**：
    - 会话历史
    - 当前状态
    - 元数据
    """
    try:
        session = agent._sessions.get(session_id)

        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "SessionNotFound", "message": f"Session {session_id} not found"},
            )

        return SessionInfo(
            session_id=session.session_id,
            agent_id=session.agent_id,
            config_hash=session.workflow_state.config_hash,
            need_greeting=session.workflow_state.need_greeting,
            status=session.workflow_state.status,
            message_count=len(session.messages),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get session failed - session_id: {session_id}, error: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={"error": type(e).__name__, "message": str(e)},
        )


@router.delete(
    "/session/{session_id}",
    responses={
        404: {"model": ErrorResponse, "description": "Session Not Found"},
    },
    summary="删除会话",
    description="清除指定会话的所有数据",
)
async def delete_session(session_id: str, agent: WorkflowAgentDep):
    """
    删除会话

    **使用场景**：
    - 用户登出
    - 清理过期会话
    - 重置对话状态
    """
    try:
        if session_id in agent._sessions:
            del agent._sessions[session_id]
            logger.info(f"Session deleted - session_id: {session_id}")
            return {"status": "deleted", "session_id": session_id}

        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": "SessionNotFound", "message": f"Session {session_id} not found"},
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Delete session failed - session_id: {session_id}, error: {str(e)}"
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
    description="检查 API 服务和 WorkflowAgent 的健康状态",
)
async def health_check(agent: WorkflowAgentDep):
    """
    健康检查

    **返回**：
    - 服务状态
    - 配置哈希
    - 活跃会话数
    - API 版本
    """
    from api import __version__

    return HealthResponse(
        status="healthy",
        config_hash=agent.config_hash,
        sessions_count=len(agent._sessions),
        version=__version__,
    )
