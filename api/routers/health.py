"""
健康检查路由模块 (V2)

提供服务健康状态检查接口
"""

from fastapi import APIRouter

from api.models import HealthResponse
from api.container import SessionManagerDep


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
        description="检查 API 服务和 SessionManager 的健康状态",
    )
    async def health_check(session_manager: SessionManagerDep):
        """健康检查"""
        from api import __version__

        stats = session_manager.get_stats()

        return HealthResponse(
            status="healthy",
            active_sessions=stats.get("active_sessions", 0),
            active_agents=stats.get("active_agents", 0),
            version=__version__,
            uptime=stats.get("uptime", 0),
        )

    return router
