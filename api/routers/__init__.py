"""
路由模块

采用模块化设计，每个路由模块提供 create_router() 工厂函数
"""

from api.routers import query, sessions, agents, health

__all__ = [
    "query",
    "sessions",
    "agents",
    "health",
]
