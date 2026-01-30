"""
统一异常处理模块

提供：
1. 自定义异常类
2. 统一错误响应格式
3. 异常处理器注册函数
"""

from typing import Any, Optional
from fastapi import FastAPI, Request, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel

from api.core.logging import get_logger

logger = get_logger(__name__)


# =============================================================================
# 自定义异常类
# =============================================================================


class APIException(Exception):
    """API 基础异常"""

    def __init__(
        self,
        message: str,
        status_code: int = 500,
        error_code: str = "INTERNAL_ERROR",
        detail: Any = None,
    ):
        self.message = message
        self.status_code = status_code
        self.error_code = error_code
        self.detail = detail
        super().__init__(message)


class ItemNotFoundError(APIException):
    """资源未找到"""

    def __init__(self, message: str, detail: Any = None):
        super().__init__(
            message=message,
            status_code=404,
            error_code="NOT_FOUND",
            detail=detail,
        )


class AuthorizationError(APIException):
    """授权错误"""

    def __init__(self, message: str = "Unauthorized", detail: Any = None):
        super().__init__(
            message=message,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            detail=detail,
        )


class RateLimitExceededError(APIException):
    """速率限制超出"""

    def __init__(self, message: str = "Rate limit exceeded", detail: Any = None):
        super().__init__(
            message=message,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            detail=detail,
        )


class ConfigurationError(APIException):
    """配置错误"""

    def __init__(self, message: str, detail: Any = None):
        super().__init__(
            message=message,
            status_code=400,
            error_code="CONFIGURATION_ERROR",
            detail=detail,
        )


# =============================================================================
# 统一错误响应模型
# =============================================================================


class ErrorResponseModel(BaseModel):
    """统一错误响应格式"""
    status: int
    code: str
    message: str
    detail: Optional[Any] = None


# =============================================================================
# 错误响应工厂函数
# =============================================================================


def is_restful_api_request(request: Request) -> bool:
    """
    判断是否为 RESTful API 请求

    RESTful API 返回统一的 JSON 格式
    传统 API 抛出 HTTPException
    """
    path = request.url.path
    # RESTful 路径模式
    restful_patterns = ["/api/v1/query", "/api/v1/chat_async"]
    restful_endings = ["/chat", "/query"]

    return path in restful_patterns or any(path.endswith(e) for e in restful_endings)


def create_error_response(
    request: Request,
    status_code: int,
    error_code: str,
    message: str,
    detail: Any = None,
) -> JSONResponse:
    """
    创建统一格式的错误响应

    根据请求类型返回不同格式：
    - RESTful API: 统一的 JSON 格式
    - 传统 API: HTTPException
    """
    content = ErrorResponseModel(
        status=status_code,
        code=error_code,
        message=message,
        detail=detail,
    ).model_dump()

    return JSONResponse(status_code=status_code, content=content)


# =============================================================================
# 异常处理器注册
# =============================================================================


def setup_exception_handlers(app: FastAPI) -> None:
    """
    注册所有异常处理器

    按优先级从高到低：
    1. 自定义 API 异常
    2. 请求验证错误 (422)
    3. HTTP 异常
    4. 通用异常 (500)
    """

    @app.exception_handler(APIException)
    async def api_exception_handler(request: Request, exc: APIException):
        """处理自定义 API 异常"""
        logger.warning(f"API Exception: {exc.error_code} - {exc.message}")
        return create_error_response(
            request=request,
            status_code=exc.status_code,
            error_code=exc.error_code,
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(ItemNotFoundError)
    async def item_not_found_handler(request: Request, exc: ItemNotFoundError):
        """处理资源未找到异常"""
        logger.warning(f"Item not found: {exc.message}")
        return create_error_response(
            request=request,
            status_code=404,
            error_code="NOT_FOUND",
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(AuthorizationError)
    async def authorization_error_handler(request: Request, exc: AuthorizationError):
        """处理授权错误"""
        logger.warning(f"Authorization error: {exc.message}")
        return create_error_response(
            request=request,
            status_code=403,
            error_code="AUTHORIZATION_ERROR",
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(RateLimitExceededError)
    async def rate_limit_handler(request: Request, exc: RateLimitExceededError):
        """处理速率限制异常"""
        logger.warning(f"Rate limit exceeded: {exc.message}")
        return create_error_response(
            request=request,
            status_code=429,
            error_code="RATE_LIMIT_EXCEEDED",
            message=exc.message,
            detail=exc.detail,
        )

    @app.exception_handler(RequestValidationError)
    async def validation_error_handler(request: Request, exc: RequestValidationError):
        """处理请求验证错误"""
        logger.warning(f"Validation error: {exc.errors()}")
        return create_error_response(
            request=request,
            status_code=422,
            error_code="VALIDATION_ERROR",
            message="Request validation failed",
            detail=exc.errors(),
        )

    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        """处理 HTTP 异常"""
        return create_error_response(
            request=request,
            status_code=exc.status_code,
            error_code=f"HTTP_{exc.status_code}",
            message=str(exc.detail) if exc.detail else "HTTP Error",
            detail=exc.detail if isinstance(exc.detail, dict) else None,
        )

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(request: Request, exc: FileNotFoundError):
        """处理文件未找到异常（配置文件等）"""
        logger.error(f"File not found: {exc}")
        return create_error_response(
            request=request,
            status_code=404,
            error_code="CONFIGURATION_NOT_FOUND",
            message=str(exc),
        )

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """处理所有未捕获的异常"""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)
        return create_error_response(
            request=request,
            status_code=500,
            error_code="INTERNAL_SERVER_ERROR",
            message="An unexpected error occurred",
            detail=str(exc) if logger.isEnabledFor(10) else None,  # DEBUG level
        )
