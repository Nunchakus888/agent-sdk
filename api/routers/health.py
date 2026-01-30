"""
健康检查路由模块

提供服务健康状态检查接口
"""

from fastapi import APIRouter

from api.models import HealthResponse
from api.container import AgentManagerDep


def create_router() -> APIRouter:
    """
    创建健康检查路由器（工厂函数）

    Returns:
        配置完成的 APIRouter 实例
    """
    router = APIRouter()

    @router.get(
        "/health",
        response_model=HealthResponse,
        summary="健康检查",
        description="检查 API 服务和 AgentManager 的健康状态",
    )
    async def health_check(manager: AgentManagerDep):
        """健康检查"""
        from api import __version__

        stats = manager.get_stats()

        return HealthResponse(
            status="healthy",
            active_sessions=stats["active_sessions"],
            active_agents=stats["active_agents"],
            version=__version__,
            uptime=stats["uptime"],
        )

    return router
