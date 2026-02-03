"""
V2 查询路由

基于 WorkflowAgentV2 + SessionManager 的查询接口

设计：
- Phase 1: 配置准备（服务级缓存）
- Phase 2: 会话准备（会话级 Agent）
- Phase 3: 核心执行（统一事件收集）
- Phase 4: 后台记录（统一 QueryRecorder）
"""

import asyncio
import time
from typing import Annotated

from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import StreamingResponse

from api.models import (
    QueryRequest,
    QueryResponse,
    ErrorResponse,
    MessageRole,
    EventType,
    EventStatus,
)
from api.core.correlation import get_correlation_id
from api.core.logging import get_logger, LogContext
from api.services.v2 import SessionManager, EventCollector, QueryRecorder

logger = get_logger(__name__)


def create_router() -> APIRouter:
    """
    创建 V2 查询路由

    使用依赖注入获取服务实例

    Returns:
        FastAPI Router
    """
    router = APIRouter(prefix="/v2", tags=["v2"])

    # 延迟导入，避免循环依赖
    def get_deps():
        from api.container import (
            get_config_loader,
            get_session_manager,
            get_repository_manager,
        )
        return get_config_loader, get_session_manager, get_repository_manager

    @router.post(
        "/query",
        response_model=QueryResponse,
        responses={
            400: {"model": ErrorResponse, "description": "Bad Request"},
            404: {"model": ErrorResponse, "description": "Config Not Found"},
            500: {"model": ErrorResponse, "description": "Internal Server Error"},
        },
        summary="V2 查询接口",
        description="基于 WorkflowAgentV2 的查询接口，会话级 Agent 实例",
    )
    async def query(request: QueryRequest):
        start_time = time.time()
        correlation_id = get_correlation_id()

        # 获取依赖
        get_config_loader, get_session_manager, get_repository_manager = get_deps()
        config_loader = get_config_loader()
        session_manager = get_session_manager()
        repos = get_repository_manager()

        with LogContext(
            session_id=request.session_id,
            chatbot_id=request.chatbot_id,
            tenant_id=request.tenant_id,
        ):
            try:
                # ─────────────────────────────────────────────────────────
                # Phase 1: 配置准备（两级缓存）
                # ─────────────────────────────────────────────────────────
                config = await config_loader.load(
                    config_hash=request.md5_checksum,
                    tenant_id=request.tenant_id,
                    chatbot_id=request.chatbot_id,
                )

                # ─────────────────────────────────────────────────────────
                # Phase 2: 会话准备（会话级 Agent）
                # ─────────────────────────────────────────────────────────
                ctx = await session_manager.get_or_create(
                    session_id=request.session_id,
                    tenant_id=request.tenant_id,
                    chatbot_id=request.chatbot_id,
                    config=config,
                )

                # ─────────────────────────────────────────────────────────
                # Phase 3: 核心执行（统一事件收集）
                # ─────────────────────────────────────────────────────────
                query_start = time.time()

                # 创建统一事件收集器
                collector = EventCollector(
                    correlation_id=correlation_id,
                    session_id=request.session_id,
                    user_message=request.message,
                )

                # 使用 query_stream 收集所有事件
                async for event in ctx.agent.query_stream(request.message):
                    collector.collect(event)  # 仅收集，不透传

                query_latency_ms = int((time.time() - query_start) * 1000)
                total_duration_ms = int((time.time() - start_time) * 1000)

                # 更新会话统计
                ctx.increment_query()

                logger.info(
                    f"V2 Query done: {query_latency_ms}ms query, "
                    f"{total_duration_ms}ms total, "
                    f"{len(collector.tool_calls)} tool calls"
                )

                # ─────────────────────────────────────────────────────────
                # Phase 4: 后处理
                # ─────────────────────────────────────────────────────────

                # 4.1 重置 Timer
                session_manager.reset_timer(request.session_id)

                # 4.2 后台记录（统一 QueryRecorder）
                if repos:
                    usage = await ctx.agent.get_usage()
                    recorder = QueryRecorder(repos)
                    recorder.record_async(collector, usage)

                return QueryResponse(
                    session_id=request.session_id,
                    message=collector.final_response,
                    status="success",
                    agent_id=f"{request.tenant_id}:{request.chatbot_id}",
                )

            except ValueError as e:
                logger.error(f"Config not found: {e}")
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail={"error": "ConfigNotFound", "message": str(e)},
                )
            except Exception as e:
                logger.error(f"V2 Query failed: {e}", exc_info=True)
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail={"error": type(e).__name__, "message": str(e)},
                )

    @router.get(
        "/sessions",
        summary="列出所有会话",
        description="获取当前所有活跃会话的信息",
    )
    async def list_sessions():
        _, get_session_manager, _ = get_deps()
        session_manager = get_session_manager()
        return {
            "sessions": session_manager.list_sessions(),
            "stats": session_manager.get_stats(),
        }

    @router.get(
        "/sessions/{session_id}",
        summary="获取会话信息",
        description="获取指定会话的详细信息",
    )
    async def get_session(session_id: str):
        _, get_session_manager, _ = get_deps()
        session_manager = get_session_manager()
        info = session_manager.get_session_info(session_id)
        if info is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "SessionNotFound", "message": f"Session not found: {session_id}"},
            )
        return info

    @router.delete(
        "/sessions/{session_id}",
        summary="销毁会话",
        description="销毁指定会话，释放资源",
    )
    async def destroy_session(session_id: str):
        _, get_session_manager, _ = get_deps()
        session_manager = get_session_manager()
        if not session_manager.exists(session_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail={"error": "SessionNotFound", "message": f"Session not found: {session_id}"},
            )
        await session_manager.destroy(session_id)
        return {"status": "destroyed", "session_id": session_id}

    @router.get(
        "/config-cache/stats",
        summary="配置缓存统计",
        description="获取配置缓存的统计信息",
    )
    async def get_config_cache_stats():
        get_config_loader, _, _ = get_deps()
        config_loader = get_config_loader()
        return config_loader.get_stats()

    return router
