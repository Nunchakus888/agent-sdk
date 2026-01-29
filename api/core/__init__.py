"""
核心基础设施模块

提供日志、追踪、中间件等基础功能
"""

from api.core.correlation import (
    correlator,
    generate_request_id,
    get_correlation_id,
    get_request_id,
    get_session_id,
    ContextualCorrelator,
)
from api.core.logging import setup_logging, get_logger
from api.core.middleware import setup_middlewares

__all__ = [
    # Correlation
    "correlator",
    "generate_request_id",
    "get_correlation_id",
    "get_request_id",
    "get_session_id",
    "ContextualCorrelator",
    # Logging
    "setup_logging",
    "get_logger",
    # Middleware
    "setup_middlewares",
]
