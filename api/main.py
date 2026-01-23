"""
FastAPI Web API for Workflow Agent

主应用入口
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.routes import router
from api.dependencies import initialize_agent_manager, shutdown_agent_manager

# =============================================================================
# 日志配置
# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)

logger = logging.getLogger(__name__)


# =============================================================================
# 生命周期管理
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    应用生命周期管理

    启动时：初始化 AgentManager
    关闭时：清理资源
    """
    # 启动
    logger.info("Starting Workflow Agent API...")

    try:
        # 从环境变量读取配置
        config_dir = os.getenv("CONFIG_DIR", "config")
        idle_timeout = int(os.getenv("AGENT_IDLE_TIMEOUT", "300"))  # 5分钟
        cleanup_interval = int(os.getenv("AGENT_CLEANUP_INTERVAL", "60"))  # 1分钟

        # 初始化 AgentManager
        manager = await initialize_agent_manager(
            config_dir=config_dir,
            idle_timeout=idle_timeout,
            cleanup_interval=cleanup_interval,
        )

        logger.info(
            f"AgentManager initialized successfully - "
            f"config_dir: {config_dir}, "
            f"idle_timeout: {idle_timeout}s, "
            f"cleanup_interval: {cleanup_interval}s"
        )

    except Exception as e:
        logger.error(f"Failed to initialize AgentManager: {e}", exc_info=True)
        raise

    yield

    # 关闭
    logger.info("Shutting down Workflow Agent API...")
    await shutdown_agent_manager()
    logger.info("AgentManager shutdown complete")


# =============================================================================
# 应用创建
# =============================================================================

app = FastAPI(
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

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# 异常处理
# =============================================================================


@app.exception_handler(Exception)
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
app.include_router(router, prefix="/api/v1", tags=["Workflow Agent"])


# 根路径
@app.get("/", tags=["Root"])
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
# 启动入口
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
    )
