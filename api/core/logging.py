"""
日志配置模块

支持：
- 控制台输出
- 文件存档（按日期目录 + 时间戳文件名）
- Correlation ID 追踪
- 可配置日志级别和目录
"""

import logging
import os
from datetime import datetime
from pathlib import Path


class CorrelationIdFilter(logging.Filter):
    """
    日志过滤器：自动注入 correlation_id
    
    从 contextvars 中获取当前的 correlation_id 并添加到日志记录中。
    """
    
    def filter(self, record: logging.LogRecord) -> bool:
        """为日志记录添加 correlation_id 字段"""
        # 延迟导入避免循环依赖
        from api.core.correlation import get_correlation_id
        record.correlation_id = get_correlation_id()
        return True


def setup_logging(
    log_dir: str | None = None,
    log_level: str = "INFO",
) -> str:
    """
    配置日志系统，支持控制台和文件输出
    
    日志文件路径: logs/yyyy-mm-dd/hh-mm-ss.log
    日志格式包含 correlation_id 用于请求追踪
    
    Args:
        log_dir: 日志根目录，默认从环境变量 LOG_DIR 读取或使用 "logs"
        log_level: 日志级别，默认从环境变量 LOG_LEVEL 读取或使用 "INFO"
    
    Returns:
        日志文件完整路径
    """
    # 获取日志目录
    log_base_dir = log_dir or os.getenv("LOG_DIR", "logs")
    
    # 获取日志级别
    level_name = os.getenv("LOG_LEVEL", log_level).upper()
    level = getattr(logging, level_name, logging.INFO)
    
    # 创建日期目录: logs/yyyy-mm-dd
    now = datetime.now()
    date_dir = now.strftime("%Y-%m-%d")
    log_path = Path(log_base_dir) / date_dir
    log_path.mkdir(parents=True, exist_ok=True)
    
    # 日志文件名: hh-mm-ss.log
    time_filename = now.strftime("%H-%M-%S") + ".log"
    log_file = log_path / time_filename
    
    # 日志格式（包含 correlation_id）
    log_format = "%(asctime)s - [%(correlation_id)s] - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(log_format)
    
    # Correlation ID 过滤器
    correlation_filter = CorrelationIdFilter()
    
    # 获取根日志器
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # 清除已有的处理器（避免重复）
    root_logger.handlers.clear()
    
    # 控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(correlation_filter)
    root_logger.addHandler(console_handler)
    
    # 文件处理器
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(correlation_filter)
    root_logger.addHandler(file_handler)
    
    return str(log_file)


def get_logger(name: str) -> logging.Logger:
    """
    获取指定名称的日志器
    
    Args:
        name: 日志器名称，通常使用 __name__
    
    Returns:
        Logger 实例
    """
    return logging.getLogger(name)
