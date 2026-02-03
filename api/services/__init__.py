"""
业务服务模块 V2

提供 Agent 管理、配置存储、数据库管理等功能
6表设计：configs, sessions, messages, tool_calls, usages, timers
"""

from api.services.database import (
    DB_NAME,
    COLLECTIONS,
    Database,
    get_database,
)
from api.models import (
    # V2 模型
    ConfigDocumentV2,
    SessionDocumentV2,
    MessageDocumentV2,
    ToolCallDocumentV2,
    UsageDocumentV2,
    TimerDocumentV2,
    # 枚举
    MessageRole,
)
from api.services.repositories import (
    RepositoryManager,
    ConfigRepository,
    SessionRepository,
    MessageRepository,
    ToolCallRepository,
    UsageRepository,
    TimerRepository,
    create_repository_manager,
)

__all__ = [
    # 数据库
    "DB_NAME",
    "COLLECTIONS",
    "Database",
    "get_database",
    # V2 模型
    "ConfigDocumentV2",
    "SessionDocumentV2",
    "MessageDocumentV2",
    "ToolCallDocumentV2",
    "UsageDocumentV2",
    "TimerDocumentV2",
    "MessageRole",
    # V2 Repository
    "RepositoryManager",
    "ConfigRepository",
    "SessionRepository",
    "MessageRepository",
    "ToolCallRepository",
    "UsageRepository",
    "TimerRepository",
    "create_repository_manager",
]
