"""
会话管理路由模块

提供会话释放等管理接口
"""

from fastapi import APIRouter, HTTPException, status

from api.models import ErrorResponse
from api.container import AgentManagerDep, RepositoryManagerDep
from api.core.logging import get_logger, LogContext

logger = get_logger(__name__)


def create_router() -> APIRouter:
    """
    创建会话管理路由器（工厂函数）

    Returns:
        配置完成的 APIRouter 实例
    """
    router = APIRouter()

    @router.delete(
        "/{session_id}",
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
        """释放会话"""
        with LogContext(session_id=session_id, chatbot_id=chatbot_id, tenant_id=tenant_id):
            try:
                # 关闭会话
                session = await repos.sessions.close(session_id)

                # 释放 Agent 的会话引用
                await manager.release_session(chatbot_id, tenant_id, session_id)

                logger.info("Session released")
                return {
                    "status": "released",
                    "session_id": session_id,
                    "message": "Session released successfully",
                }

            except Exception as e:
                logger.error(f"Release session failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={
                        "error": type(e).__name__,
                        "message": str(e),
                        "session_id": session_id,
                    },
                )

    return router
