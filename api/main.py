"""
FastAPI Web API for Workflow Agent

主应用入口 - V2 架构
"""

import asyncio
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import RedirectResponse
from starlette.types import Receive, Scope, Send

from api import __version__
from api.core import setup_logging, get_logger, setup_middlewares, setup_exception_handlers
from api.routers import health
from api.routers.v2 import query as query_v2
from api.routers.v1 import chat as chat_v1
from api.routers.v1 import session as session_v1
from api.routers.v1 import agent as agent_v1
from api.routers.v1 import config as config_v1
from api.container import AppContext

# 初始化日志
log_file_path = setup_logging()
logger = get_logger(__name__)
logger.info(f"Log file: {log_file_path}")


# =============================================================================
# 生命周期管理
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时：
    1. 加载 Apollo 配置（如果配置了）
    2. 通过 AppContext.create() 初始化所有服务

    关闭时：通过 AppContext.shutdown() 清理资源
    """
    logger.info("Starting Workflow Agent API...")

    try:
        # Step 1: 加载 Apollo 配置（设置环境变量）
        from api.utils.apollo_config.loader import load_config_from_env
        await load_config_from_env()

        # Step 2: 初始化所有服务（使用已设置的环境变量）
        ctx = await AppContext.create()

        # 启动 SessionManager 后台任务
        if ctx.session_manager:
            await ctx.session_manager.start()

        logger.info(
            f"AppContext initialized - "
            f"db={'mongodb' if ctx.database else 'memory'}, "
            f"persistent={ctx.repository_manager.is_persistent if ctx.repository_manager else False}"
        )

    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        raise

    yield

    # 关闭
    logger.info("Shutting down Workflow Agent API...")
    await AppContext.get_instance().shutdown()
    logger.info("Shutdown complete")


# =============================================================================
# AppWrapper - ASGI 异常处理
# =============================================================================


class AppWrapper:
    """ASGI 应用包装器，处理 asyncio.CancelledError"""

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            return await self.app(scope, receive, send)
        except asyncio.CancelledError:
            pass


# =============================================================================
# 应用创建
# =============================================================================


def create_api_app(
    title: str = "Workflow Agent API",
    description: str = "RESTful API for Workflow Agent (V2)",
    version: str = __version__,
    docs_url: str = "/docs",
    redoc_url: str = "/redoc",
    openapi_url: str = "/openapi.json",
) -> FastAPI:
    """创建 FastAPI 应用实例"""
    fastapi_app = FastAPI(
        title=title,
        description=description,
        version=version,
        lifespan=lifespan,
        docs_url=docs_url,
        redoc_url=redoc_url,
        openapi_url=openapi_url,
    )

    # 配置中间件和异常处理
    setup_middlewares(fastapi_app)
    setup_exception_handlers(fastapi_app)

    # 注册路由
    # V1 Chat
    fastapi_app.include_router(
        chat_v1.create_router(),
        prefix="/api/v1",
        tags=["Chat"],
    )
    # V1 Session
    fastapi_app.include_router(
        session_v1.create_router(),
        prefix="/api/v1",
        tags=["Session"],
    )
    # V1 Agent
    fastapi_app.include_router(
        agent_v1.create_router(),
        prefix="/api/v1",
        tags=["Agent"],
    )
    # V1 Config
    fastapi_app.include_router(
        config_v1.create_router(),
        prefix="/api/v1",
        tags=["Config"],
    )
    fastapi_app.include_router(
        query_v2.create_router(),
        prefix="/api",
        tags=["Query"],
    )
    fastapi_app.include_router(
        health.create_router(),
        prefix="/api/v1",
        tags=["Health"],
    )

    # 挂载 Chat Web UI
    chat_dist_path = Path(__file__).parent / "chat" / "dist"
    if chat_dist_path.exists():
        fastapi_app.mount(
            "/chat",
            StaticFiles(directory=str(chat_dist_path), html=True),
            name="chat-ui",
        )
        logger.info(f"Chat UI mounted at /chat")
    else:
        logger.warning(f"Chat UI dist not found. Run 'cd api/chat && npm run build'")

    # 根路径重定向
    @fastapi_app.get("/", tags=["Root"])
    async def root():
        return RedirectResponse(url="/chat/")

    # API 信息
    @fastapi_app.get("/api", tags=["Root"])
    async def api_info():
        return {
            "name": title,
            "version": version,
            "docs": docs_url,
            "health": "/api/v1/health",
        }

    return fastapi_app


# 创建应用实例
_fastapi_app = create_api_app()
app = AppWrapper(_fastapi_app)


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    debug = os.getenv("DEBUG", "false").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))
    reload_enabled = os.getenv("RELOAD", str(debug)).lower() == "true"

    logger.info(f"Starting server - host: {host}, port: {port}, debug: {debug}")

    if reload_enabled:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=["api", "bu_agent_sdk"],
            log_level="debug" if debug else "info",
        )
    else:
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
        )
