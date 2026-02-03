"""
V2 Routers - 基于 WorkflowAgentV2 的 API 路由

完全隔离的 v2 版本路由，与旧版本互不影响
"""

from api.routers.v2.query import create_router as create_query_router

__all__ = [
    "create_query_router",
]
