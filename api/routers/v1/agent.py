"""
V1 Agent 路由

提供 Agent 列表 API，供前端 Chat UI 使用。
从 SessionManager 的活跃会话中提取唯一的 agent 信息。

端点：
- GET /agent — Agent 列表
"""

from fastapi import APIRouter

from api.core.logging import get_logger

logger = get_logger(__name__)


def create_router() -> APIRouter:
    """创建 V1 Agent 路由"""
    router = APIRouter(prefix="/agent", tags=["Agent"])

    def get_deps():
        from api.container import get_session_manager
        return get_session_manager

    @router.get(
        "",
        summary="Agent 列表",
        description="获取当前所有活跃的 Agent 信息",
    )
    async def list_agents():
        get_session_manager = get_deps()
        session_manager = get_session_manager()

        sessions = session_manager.list_sessions()
        seen = set()
        agents = []
        for s in sessions:
            agent_id = f"{s.get('tenant_id', '')}:{s.get('chatbot_id', '')}"
            if agent_id not in seen:
                seen.add(agent_id)
                agents.append({
                    "id": agent_id,
                    "name": s.get("chatbot_id", agent_id),
                })

        return agents

    return router
