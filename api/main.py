"""
FastAPI Web API for Workflow Agent

主应用入口
"""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Awaitable, Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from starlette.types import Receive, Scope, Send

from api import __version__
from api.core import setup_logging, get_logger, setup_middlewares
from api.routes import router
from api.container import (
    initialize_workflow_engine,
    shutdown_workflow_engine,
    initialize_agent_manager,
    shutdown_agent_manager,
    initialize_task_manager,
    shutdown_task_manager,
    initialize_database,
    initialize_repository_manager,
)

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

    启动时：初始化 WorkflowEngine
    关闭时：清理资源
    """
    # 启动
    logger.info("Starting Workflow Agent API...")

    try:
        # 从环境变量读取配置
        config_dir = os.getenv("CONFIG_DIR", "config")
        idle_timeout = int(os.getenv("AGENT_IDLE_TIMEOUT", "300"))
        cleanup_interval = int(os.getenv("AGENT_CLEANUP_INTERVAL", "60"))
        max_agents = int(os.getenv("MAX_AGENTS", "100"))
        agent_ttl = int(os.getenv("AGENT_TTL", "3600"))

        # MongoDB 配置（可选）
        # 如果设置了 MONGODB_URI，自动启用 MongoDB
        mongodb_uri = os.getenv("MONGODB_URI")
        enable_mongodb = os.getenv("ENABLE_MONGODB", "false").lower() == "true"
        
        # 如果 ENABLE_MONGODB=true 但 MONGODB_URI 未设置，给出警告
        if enable_mongodb and not mongodb_uri:
            logger.warning(
                "ENABLE_MONGODB=true but MONGODB_URI not set. "
                "Falling back to memory mode."
            )
            mongodb_uri = None
        
        mongodb_db = os.getenv("MONGODB_DB", "workflow_agent")

        # 初始化 WorkflowEngine
        engine = await initialize_workflow_engine(
            mongodb_uri=mongodb_uri,
            db_name=mongodb_db,
            config_dir=config_dir,
            max_agents=max_agents,
            agent_ttl=agent_ttl,
            idle_timeout=idle_timeout,
            cleanup_interval=cleanup_interval,
        )

        logger.info(
            f"WorkflowEngine initialized - "
            f"mode={'mongodb' if mongodb_uri else 'memory'}, "
            f"config_dir={config_dir}, "
            f"max_agents={max_agents}"
        )

        # 初始化 AgentManager（复用 MongoDB 连接）
        enable_llm_parsing = os.getenv("ENABLE_LLM_PARSING", "false").lower() == "true"
        manager = initialize_agent_manager(
            config_dir=config_dir,
            idle_timeout=idle_timeout,
            cleanup_interval=cleanup_interval,
            enable_llm_parsing=enable_llm_parsing,
            db_name=mongodb_db,
        )
        manager.start_cleanup()

        # 获取配置存储类型
        store_type = manager._config_store.store_type
        logger.info(
            f"AgentManager initialized - "
            f"config_dir={config_dir}, "
            f"idle_timeout={idle_timeout}s, "
            f"enable_llm_parsing={enable_llm_parsing}, "
            f"config_store={store_type}"
        )

        # 初始化 TaskManager（协程任务取消机制）
        task_manager = initialize_task_manager()
        logger.info("TaskManager initialized")

        # 初始化 Database 和 RepositoryManager
        if mongodb_uri:
            logger.info(f"Initializing database: {mongodb_db}")
            db = await initialize_database(db_name=mongodb_db)
            if db:
                logger.info(f"✓ Database initialized successfully: {mongodb_db}")
            else:
                logger.error(
                    f"✗ Database initialization failed: {mongodb_db}. "
                    "Check MongoDB connection and configuration."
                )
        else:
            logger.info("Database not enabled (memory mode)")
            db = None

        repo_manager = initialize_repository_manager()
        logger.info(f"RepositoryManager initialized: persistent={repo_manager.is_persistent}")

    except Exception as e:
        logger.error(f"Failed to initialize: {e}", exc_info=True)
        raise

    yield

    # 关闭
    logger.info("Shutting down Workflow Agent API...")
    await shutdown_task_manager()
    await shutdown_agent_manager()
    await shutdown_workflow_engine()
    logger.info("Shutdown complete")


# =============================================================================
# AppWrapper - ASGI 异常处理（从 Parlant 移植）
# =============================================================================


class AppWrapper:
    """
    ASGI 应用包装器

    FastAPI 内置的异常处理不会捕获 BaseException（如 asyncio.CancelledError）
    这会导致服务进程以丑陋的 traceback 终止
    此包装器专门处理 asyncio.CancelledError，使其优雅退出
    """

    def __init__(self, app: FastAPI) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        try:
            return await self.app(scope, receive, send)
        except asyncio.CancelledError:
            pass  # 优雅退出，不抛出异常


# =============================================================================
# 应用创建
# =============================================================================

_fastapi_app = FastAPI(
    title="Workflow Agent API",
    description="RESTful API for BU Agent SDK Workflow Agent (Multi-tenant, Session-based)",
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)


# =============================================================================
# 中间件配置
# =============================================================================

setup_middlewares(_fastapi_app)


# =============================================================================
# 异常处理
# =============================================================================


@_fastapi_app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """全局异常处理"""
    _ = request  # 避免未使用警告
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": type(exc).__name__,
            "message": str(exc),
            "detail": "Internal server error",
        },
    )


# =============================================================================
# 路由注册
# =============================================================================

# 注册 API 路由
_fastapi_app.include_router(router, prefix="/api/v1", tags=["Workflow Agent"])


# 根路径
@_fastapi_app.get("/", tags=["Root"])
async def root():
    """API 根路径"""
    return {
        "name": "Workflow Agent API",
        "version": __version__,
        "description": "Multi-tenant, Session-based Workflow Agent API",
        "docs": "/docs",
        "health": "/api/v1/health",
    }


# =============================================================================
# 导出 ASGI 应用（使用 AppWrapper 包装）
# =============================================================================

app = AppWrapper(_fastapi_app)


# =============================================================================
# 启动入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    # 从环境变量读取运行配置
    debug = os.getenv("DEBUG", "false").lower() == "true"
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    workers = int(os.getenv("WORKERS", "1"))

    # 热更新配置
    # 注意：使用 reload 时必须用字符串形式的应用路径
    reload_enabled = os.getenv("RELOAD", str(debug)).lower() == "true"
    reload_dirs = ["api", "bu_agent_sdk"] if reload_enabled else None

    logger.info(
        f"Starting server - host: {host}, port: {port}, "
        f"debug: {debug}, reload: {reload_enabled}"
    )

    if reload_enabled:
        # 开发模式：启用热更新
        # 必须使用字符串形式 "module:app"，不能直接传入 app 对象
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            reload=True,
            reload_dirs=reload_dirs,
            reload_delay=0.25,  # 文件变化后延迟重载（秒）
            log_level="debug" if debug else "info",
        )
    else:
        # 生产模式：多 worker 支持
        uvicorn.run(
            "api.main:app",
            host=host,
            port=port,
            workers=workers,
            log_level="info",
        )
