"""
FastAPI Web API for Workflow Agent

主应用入口
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api import __version__
from api.routes import router
from api.dependencies import initialize_agent

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

    启动时：初始化 WorkflowAgent
    关闭时：清理资源
    """
    # 启动
    logger.info("Starting Workflow Agent API...")

    try:
        # 初始化 WorkflowAgent
        agent = await initialize_agent()
        logger.info(f"WorkflowAgent initialized successfully - config_hash: {agent.config_hash}")

    except Exception as e:
        logger.error(f"Failed to initialize WorkflowAgent: {e}", exc_info=True)
        raise

    yield

    # 关闭
    logger.info("Shutting down Workflow Agent API...")


# =============================================================================
# 应用创建
# =============================================================================

app = FastAPI(
    title="Workflow Agent API",
    description="RESTful API for BU Agent SDK Workflow Agent",
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
