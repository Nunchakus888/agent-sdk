"""
中间件模块

提供 FastAPI 应用的中间件配置
"""

import asyncio
import logging
from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware

from api.core.correlation import correlator, generate_request_id

logger = logging.getLogger(__name__)


def setup_middlewares(app: FastAPI) -> None:
    """
    配置所有中间件

    Args:
        app: FastAPI 应用实例
    """
    _setup_cors(app)
    _setup_cancellation_handler(app)
    _setup_correlation_id(app)


def _setup_cors(app: FastAPI) -> None:
    """配置 CORS 中间件"""
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # 生产环境应限制具体域名
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def _setup_cancellation_handler(app: FastAPI) -> None:
    """
    配置取消处理中间件（从 Parlant 移植）

    处理 asyncio.CancelledError，返回 503 状态码
    """

    @app.middleware("http")
    async def cancellation_middleware(request: Request, call_next):
        """
        处理请求取消

        当请求被取消时（如同 session 新请求取消旧请求），
        返回 503 Service Unavailable
        """
        try:
            return await call_next(request)
        except asyncio.CancelledError:
            logger.info(f"Request cancelled: {request.method} {request.url.path}")
            return Response(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content="Service Unavailable - Request Cancelled",
            )


def _setup_correlation_id(app: FastAPI) -> None:
    """配置 Correlation ID 中间件"""
    
    @app.middleware("http")
    async def correlation_id_middleware(request: Request, call_next):
        """
        为每个请求生成唯一的 correlation_id
        
        - 从请求头 X-Correlation-ID 获取，或自动生成
        - 在响应头中返回 X-Correlation-ID
        - 所有日志自动包含 correlation_id
        """
        # 优先从请求头获取，否则生成新的
        correlation_id = request.headers.get("X-Correlation-ID")
        if not correlation_id:
            correlation_id = f"R{generate_request_id()}"
        
        # 进入 correlation 作用域
        with correlator.scope(
            correlation_id,
            properties={
                "request_id": correlation_id,
                "method": request.method,
                "path": request.url.path,
            }
        ):
            logger.info(f"→ {request.method} {request.url.path}")
            
            response = await call_next(request)
            
            # 在响应头中返回 correlation_id
            response.headers["X-Correlation-ID"] = correlation_id
            
            logger.info(f"← {request.method} {request.url.path} [{response.status_code}]")
            
            return response
