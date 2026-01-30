"""
统一日志模块

提供基于上下文作用域的结构化日志：
- 自动注入 correlation_id、session_id、agent_id 等上下文信息
- 支持作用域嵌套，自动追踪调用链
- 支持操作计时和异常捕获
- 简洁的 API，减少日志代码冗余

使用示例:
    from api.core.logging import get_logger, LogContext

    logger = get_logger(__name__)

    # 简单日志
    logger.info("Processing request")

    # 带上下文的日志
    with LogContext(session_id="sess_123", agent_id="agent_456"):
        logger.info("Handling session")  # 自动包含 session_id, agent_id

        with LogContext.operation("query", chatbot_id="bot_789"):
            logger.info("Executing query")  # 自动计时
"""

from __future__ import annotations
import asyncio
import contextvars
import logging
import os
import time
import traceback
from contextlib import contextmanager
from datetime import datetime
from enum import Enum, auto
from pathlib import Path
from typing import Any, Iterator

from api.core.correlation import correlator, generate_request_id


# =============================================================================
# 日志级别
# =============================================================================


class LogLevel(Enum):
    """日志级别枚举"""

    TRACE = auto()
    DEBUG = auto()
    INFO = auto()
    WARNING = auto()
    ERROR = auto()
    CRITICAL = auto()

    def to_logging_level(self) -> int:
        """转换为 logging 模块级别"""
        return {
            LogLevel.TRACE: logging.DEBUG,
            LogLevel.DEBUG: logging.DEBUG,
            LogLevel.INFO: logging.INFO,
            LogLevel.WARNING: logging.WARNING,
            LogLevel.ERROR: logging.ERROR,
            LogLevel.CRITICAL: logging.CRITICAL,
        }[self]


# =============================================================================
# 上下文变量
# =============================================================================

# 日志上下文属性（协程安全）
_log_context: contextvars.ContextVar[dict[str, Any]] = contextvars.ContextVar(
    "log_context", default={}
)

# 作用域栈
_scope_stack: contextvars.ContextVar[list[str]] = contextvars.ContextVar(
    "scope_stack", default=[]
)


# =============================================================================
# LogContext - 上下文管理器
# =============================================================================


class LogContext:
    """
    日志上下文管理器

    自动将上下文信息注入到日志中，支持嵌套作用域。

    使用示例:
        # 设置上下文
        with LogContext(session_id="sess_123", chatbot_id="bot_456"):
            logger.info("Processing")  # 自动包含 session_id, chatbot_id

        # 操作计时
        with LogContext.operation("query"):
            do_something()  # 自动记录开始/结束时间
    """

    def __init__(self, **kwargs: Any):
        """
        初始化日志上下文

        Args:
            **kwargs: 上下文属性（如 session_id, agent_id, chatbot_id 等）
        """
        self._props = kwargs
        self._token: contextvars.Token | None = None

    def __enter__(self) -> "LogContext":
        # 合并当前上下文
        current = _log_context.get().copy()
        current.update(self._props)
        self._token = _log_context.set(current)
        return self

    def __exit__(self, *args: Any) -> None:
        if self._token:
            _log_context.reset(self._token)

    @classmethod
    def get(cls, key: str, default: Any = None) -> Any:
        """获取上下文属性"""
        return _log_context.get().get(key, default)

    @classmethod
    def get_all(cls) -> dict[str, Any]:
        """获取所有上下文属性"""
        return _log_context.get().copy()

    @classmethod
    @contextmanager
    def scope(cls, name: str, **kwargs: Any) -> Iterator[None]:
        """
        进入命名作用域

        Args:
            name: 作用域名称
            **kwargs: 附加上下文属性
        """
        # 更新作用域栈
        current_stack = _scope_stack.get().copy()
        current_stack.append(name)
        stack_token = _scope_stack.set(current_stack)

        # 更新上下文属性
        current_ctx = _log_context.get().copy()
        current_ctx.update(kwargs)
        ctx_token = _log_context.set(current_ctx)

        try:
            yield
        finally:
            _scope_stack.reset(stack_token)
            _log_context.reset(ctx_token)

    @classmethod
    @contextmanager
    def operation(
        cls,
        name: str,
        level: LogLevel = LogLevel.INFO,
        log_start: bool = False,
        **kwargs: Any,
    ) -> Iterator[None]:
        """
        操作计时上下文

        自动记录操作开始/结束时间，捕获异常。

        Args:
            name: 操作名称
            level: 日志级别
            log_start: 是否记录开始日志
            **kwargs: 附加上下文属性
        """
        logger = get_logger("operation")
        t_start = time.time()

        with cls.scope(name, **kwargs):
            try:
                if log_start:
                    logger.debug(f"{name} started")

                yield

                elapsed = time.time() - t_start
                _log_at_level(logger, level, f"{name} completed in {elapsed:.3f}s")

            except asyncio.CancelledError:
                elapsed = time.time() - t_start
                logger.warning(f"{name} cancelled after {elapsed:.3f}s")
                raise
            except Exception as e:
                elapsed = time.time() - t_start
                logger.error(f"{name} failed after {elapsed:.3f}s: {e}")
                logger.debug(traceback.format_exc())
                raise


def _log_at_level(logger: logging.Logger, level: LogLevel, message: str) -> None:
    """按指定级别记录日志"""
    log_func = {
        LogLevel.TRACE: logger.debug,
        LogLevel.DEBUG: logger.debug,
        LogLevel.INFO: logger.info,
        LogLevel.WARNING: logger.warning,
        LogLevel.ERROR: logger.error,
        LogLevel.CRITICAL: logger.critical,
    }[level]
    log_func(message)


# =============================================================================
# 日志格式化器
# =============================================================================


class ContextFormatter(logging.Formatter):
    """
    上下文感知的日志格式化器

    自动将上下文信息添加到日志消息中。
    """

    def formatTime(self, record: logging.LogRecord, datefmt: str | None = None) -> str:
        """格式化时间戳（支持微秒）"""
        ct = datetime.fromtimestamp(record.created)
        if datefmt:
            # 手动处理 %f（微秒）
            s = ct.strftime(datefmt.replace("%f", f"{ct.microsecond:06d}"))
        else:
            s = ct.strftime("%Y-%m-%dT%H:%M:%S.%fZ".replace("%f", f"{ct.microsecond:06d}"))
        return s

    def format(self, record: logging.LogRecord) -> str:
        # 获取 correlation_id（短格式）
        correlation_id = correlator.correlation_id
        if not correlation_id or correlation_id == "-":
            cid = "*"
        else:
            cid = correlation_id[:11]  # 保留11位，足够唯一识别
        
        # 获取上下文属性
        ctx = _log_context.get()
        
        # 获取 session_id（简化显示）
        session_id = ctx.get("session_id")
        sid = self._truncate(session_id, 20) if session_id else None
        
        # 获取作用域
        scopes = _scope_stack.get()
        scope = scopes[-1] if scopes else None  # 只显示当前作用域
        
        # 构建上下文字符串（紧凑格式）
        if sid and scope:
            ctx_str = f"[{cid}] [{sid}:{scope}]"
        elif sid:
            ctx_str = f"[{cid}] [{sid}]"
        else:
            ctx_str = f"[{cid}]"
        
        # 设置到 record
        record.ctx = ctx_str

        return super().format(record)
    
    @staticmethod
    def _truncate(value: str | None, max_len: int = 20) -> str:
        """截断过长的值"""
        if not value:
            return ""
        if len(value) <= max_len:
            return value
        return value[:max_len - 3] + "..."


# =============================================================================
# 日志配置
# =============================================================================


_initialized = False


def setup_logging(
    log_dir: str | None = None,
    log_level: str = "INFO",
    console: bool = True,
    file: bool = True,
) -> str | None:
    """
    配置日志系统

    Args:
        log_dir: 日志目录，默认从 LOG_DIR 环境变量读取或使用 "logs"
        log_level: 日志级别，默认从 LOG_LEVEL 环境变量读取或使用 "INFO"
        console: 是否输出到控制台
        file: 是否输出到文件

    Returns:
        日志文件路径（如果启用文件输出）
    """
    global _initialized

    if _initialized:
        return None

    # 获取配置
    log_base_dir = log_dir or os.getenv("LOG_DIR", "logs")
    level_name = os.getenv("LOG_LEVEL", log_level).upper()
    level = getattr(logging, level_name, logging.INFO)

    # 日志格式：带时间戳的简洁格式
    # timestamp [level] [cid] [sid:scope] message
    log_format = "%(asctime)s [%(levelname)-8s] %(ctx)s %(message)s"
    date_format = "%Y-%m-%dT%H:%M:%S.%fZ"
    formatter = ContextFormatter(log_format, datefmt=date_format)

    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    root_logger.handlers.clear()

    log_file_path = None

    # 控制台处理器
    if console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # 文件处理器
    if file:
        now = datetime.now()
        date_dir = now.strftime("%Y-%m-%d")
        log_path = Path(log_base_dir) / date_dir
        log_path.mkdir(parents=True, exist_ok=True)

        time_filename = now.strftime("%H-%M-%S") + ".log"
        log_file = log_path / time_filename
        log_file_path = str(log_file)

        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    _initialized = True
    return log_file_path


def get_logger(name: str) -> logging.Logger:
    """
    获取日志器

    Args:
        name: 日志器名称，通常使用 __name__

    Returns:
        Logger 实例
    """
    return logging.getLogger(name)


# =============================================================================
# 便捷日志函数
# =============================================================================


class _QuickLogger:
    """
    快捷日志器

    提供模块级别的便捷日志函数，自动包含上下文信息。

    使用示例:
        from api.core.logging import log

        log.info("Processing request")
        log.error("Failed to process", exc_info=True)
    """

    def __init__(self):
        self._logger = logging.getLogger("api")

    def debug(self, msg: str, **kwargs: Any) -> None:
        self._logger.debug(msg, **kwargs)

    def info(self, msg: str, **kwargs: Any) -> None:
        self._logger.info(msg, **kwargs)

    def warning(self, msg: str, **kwargs: Any) -> None:
        self._logger.warning(msg, **kwargs)

    def error(self, msg: str, **kwargs: Any) -> None:
        self._logger.error(msg, **kwargs)

    def critical(self, msg: str, **kwargs: Any) -> None:
        self._logger.critical(msg, **kwargs)

    def exception(self, msg: str, **kwargs: Any) -> None:
        self._logger.exception(msg, **kwargs)


# 全局快捷日志器
log = _QuickLogger()


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    "LogLevel",
    "LogContext",
    "setup_logging",
    "get_logger",
    "log",
]
