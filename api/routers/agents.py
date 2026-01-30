"""
Agent 管理路由模块

提供 Agent 信息查询和删除接口
"""

from fastapi import APIRouter, HTTPException, status

from api.models import ErrorResponse, AgentStats, AuditAction, AgentStatus
from api.container import AgentManagerDep, RepositoryManagerDep
from api.core.logging import get_logger, LogContext

logger = get_logger(__name__)


def create_router() -> APIRouter:
    """
    创建 Agent 管理路由器（工厂函数）

    Returns:
        配置完成的 APIRouter 实例
    """
    router = APIRouter()

    @router.get(
        "/{chatbot_id}",
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
        """获取 Agent 信息"""
        with LogContext(chatbot_id=chatbot_id, tenant_id=tenant_id):
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
                logger.error(f"Get agent info failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": type(e).__name__, "message": str(e)},
                )

    @router.delete(
        "/{chatbot_id}",
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
        """删除 Agent"""
        with LogContext(chatbot_id=chatbot_id, tenant_id=tenant_id):
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
                await manager.remove_agent(chatbot_id, tenant_id)

                await repos.agent_states.update(
                    agent_id,
                    status=AgentStatus.TERMINATED.value,
                )

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

                logger.info("Agent deleted")
                return {
                    "status": "deleted",
                    "chatbot_id": chatbot_id,
                    "tenant_id": tenant_id,
                    "message": "Agent deleted successfully",
                }

            except HTTPException:
                raise
            except Exception as e:
                logger.error(f"Delete agent failed: {e}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": type(e).__name__, "message": str(e)},
                )

    return router
