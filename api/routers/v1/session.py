"""
V1 Session 路由

提供会话管理的 RESTful API，供前端 Chat UI 使用。
复用 RepositoryManager 中的数据访问方法，遵循 DRY 原则。

端点：
- GET    /session                        — 分页会话列表
- GET    /session/{session_id}           — 会话详情
- GET    /session/{session_id}/events    — 会话事件（消息）
- PATCH  /session/{session_id}           — 更新会话
- DELETE /session/{session_id}           — 删除会话
- DELETE /session/{session_id}/events    — 删除指定offset后的消息
"""

import math
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, Field

from api.core.logging import get_logger

logger = get_logger(__name__)


class SessionUpdateRequest(BaseModel):
    """会话更新请求"""
    title: Optional[str] = Field(default=None, description="会话标题")


def create_router() -> APIRouter:
    """创建 V1 Session 路由"""
    router = APIRouter(prefix="/session", tags=["Session"])

    def get_deps():
        from api.container import get_session_manager, get_repository_manager
        return get_session_manager, get_repository_manager

    @router.get(
        "",
        summary="分页会话列表",
        description="获取会话列表，支持分页和筛选",
    )
    async def list_sessions(
        page: int = Query(1, ge=1, description="页码"),
        page_size: int = Query(10, ge=1, le=100, description="每页数量"),
        tenant_id: Optional[str] = Query(None, description="租户ID筛选"),
        chatbot_id: Optional[str] = Query(None, description="Chatbot ID筛选"),
    ):
        _, get_repository_manager = get_deps()
        repos = get_repository_manager()

        total = await repos.sessions.count(tenant_id=tenant_id, chatbot_id=chatbot_id)
        offset = (page - 1) * page_size
        sessions = await repos.sessions.list_by_tenant(
            tenant_id=tenant_id or "",
            chatbot_id=chatbot_id,
            limit=page_size,
            offset=offset,
        )

        items = []
        for s in sessions:
            items.append({
                "id": s.session_id,
                "title": s.title or s.session_id,
                "customer_id": s.customer_id or "",
                "creation_utc": s.created_at.isoformat() if s.created_at else "",
                "agent_id": f"{s.tenant_id}:{s.chatbot_id}",
                "tenant_id": s.tenant_id,
                "chatbot_id": s.chatbot_id,
                "md5_checksum": s.config_hash,
                "source": s.source,
            })

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": math.ceil(total / page_size) if total > 0 else 1,
        }

    @router.get(
        "/{session_id}",
        summary="会话详情",
        description="获取指定会话的详细信息",
    )
    async def get_session(session_id: str):
        _, get_repository_manager = get_deps()
        repos = get_repository_manager()

        session = await repos.sessions.get(session_id)
        if not session:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "SessionNotFound", "message": f"Session not found: {session_id}"},
            )

        usage_stats = await repos.usages.get_session_stats(session_id)
        msg_count = await repos.messages.count_by_session(session_id)

        return {
            "id": session.session_id,
            "title": session.title or session.session_id,
            "customer_id": session.customer_id or "",
            "creation_utc": session.created_at.isoformat() if session.created_at else "",
            "agent_id": f"{session.tenant_id}:{session.chatbot_id}",
            "tenant_id": session.tenant_id,
            "chatbot_id": session.chatbot_id,
            "md5_checksum": session.config_hash,
            "source": session.source,
            "message_count": msg_count,
            "usage": usage_stats,
        }

    @router.get(
        "/{session_id}/events",
        summary="会话事件",
        description="获取会话的消息事件，支持 offset 轮询",
    )
    async def get_session_events(
        session_id: str,
        min_offset: int = Query(0, ge=0, description="最小 offset"),
        wait_for_data: int = Query(0, ge=0, description="等待数据（暂不支持长轮询）"),
    ):
        _, get_repository_manager = get_deps()
        repos = get_repository_manager()

        messages = await repos.messages.list_by_session(
            session_id=session_id,
            limit=1000,
            order="asc",
        )

        events = []
        for i, msg in enumerate(messages):
            if i < min_offset:
                continue
            source = "ai_agent" if msg.role == "assistant" else "customer"
            events.append({
                "id": msg.message_id,
                "source": source,
                "kind": "message",
                "correlation_id": msg.correlation_id or msg.message_id,
                "serverStatus": "ready",
                "offset": i,
                "creation_utc": msg.created_at.isoformat() if msg.created_at else "",
                "data": {
                    "message": msg.content,
                    "status": "ready",
                },
            })

        return events

    @router.patch(
        "/{session_id}",
        summary="更新会话",
        description="更新会话信息（如标题）",
    )
    async def update_session(session_id: str, body: SessionUpdateRequest):
        _, get_repository_manager = get_deps()
        repos = get_repository_manager()

        updates = {}
        if body.title is not None:
            updates["title"] = body.title

        if not updates:
            return {"status": "ok", "session_id": session_id}

        result = await repos.sessions.update(session_id, **updates)
        if not result:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "SessionNotFound", "message": f"Session not found: {session_id}"},
            )
        return {"status": "ok", "session_id": session_id}

    @router.delete(
        "/{session_id}",
        summary="删除会话",
        description="删除指定会话及其关联数据",
    )
    async def delete_session(session_id: str):
        get_session_manager, get_repository_manager = get_deps()
        session_manager = get_session_manager()
        repos = get_repository_manager()

        # 从 SessionManager 中销毁（如果存在内存中的活跃会话）
        if session_manager.exists(session_id):
            await session_manager.destroy(session_id)

        # 从数据库中删除
        await repos.sessions.delete(session_id)

        return {"status": "deleted", "session_id": session_id}

    @router.delete(
        "/{session_id}/events",
        summary="删除事件",
        description="删除指定 offset 之后的消息",
    )
    async def delete_session_events(
        session_id: str,
        min_offset: int = Query(0, ge=0, description="从此 offset 开始删除"),
    ):
        _, get_repository_manager = get_deps()
        repos = get_repository_manager()

        deleted = await repos.messages.delete_from_offset(session_id, min_offset)
        return {"status": "ok", "deleted_count": deleted}

    return router
