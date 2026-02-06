"""V1 API 路由模块"""

from api.routers.v1.chat import create_router as create_chat_router
from api.routers.v1.session import create_router as create_session_router
from api.routers.v1.agent import create_router as create_agent_router
from api.routers.v1.config import create_router as create_config_router

__all__ = [
    "create_chat_router",
    "create_session_router",
    "create_agent_router",
    "create_config_router",
]
