"""
核心基础设施模块

提供日志、追踪、中间件、异常处理等基础功能
"""

from api.core.correlation import (
    correlator,
    generate_id,
    generate_request_id,
    get_correlation_id,
    get_request_id,
    get_session_id,
    ContextualCorrelator,
    UniqueId,
    ID_GENERATION_ALPHABET,
    ID_SIZE,
)
from api.core.logging import (
    setup_logging,
    get_logger,
    LogContext,
    LogLevel,
    log,
)
from api.core.middleware import setup_middlewares
from api.core.exceptions import (
    APIException,
    ItemNotFoundError,
    AuthorizationError,
    RateLimitExceededError,
    ConfigurationError,
    ErrorResponseModel,
    create_error_response,
    setup_exception_handlers,
)

__all__ = [
    # Correlation
    "correlator",
    "generate_id",
    "generate_request_id",
    "get_correlation_id",
    "get_request_id",
    "get_session_id",
    "ContextualCorrelator",
    "UniqueId",
    "ID_GENERATION_ALPHABET",
    "ID_SIZE",
    # Logging
    "setup_logging",
    "get_logger",
    "LogContext",
    "LogLevel",
    "log",
    # Middleware
    "setup_middlewares",
    # Exceptions
    "APIException",
    "ItemNotFoundError",
    "AuthorizationError",
    "RateLimitExceededError",
    "ConfigurationError",
    "ErrorResponseModel",
    "create_error_response",
    "setup_exception_handlers",
]
