"""
业务服务模块

提供 Agent 管理、配置存储、数据库管理、任务管理等功能
"""

from api.services.agent_manager import AgentManager, AgentInfo, ParsedConfig
from api.services.config_store import (
    ConfigStore,
    StoredConfig,
    MemoryConfigStore,
    MongoConfigStore,
    create_config_store,
)
from api.services.database import (
    DB_NAME,
    COLLECTIONS,
    Database,
    get_database,
)
from api.models import (
    SessionDocument,
    SessionStatus,
    MessageDocument,
    MessageRole,
    AgentStateDocument,
    AgentStatus,
    AuditLogDocument,
    AuditAction,
)
from api.services.task_manager import TaskManager
from api.services.repositories import (
    RepositoryManager,
    SessionRepository,
    MessageRepository,
    AgentStateRepository,
    AuditLogRepository,
    create_repository_manager,
)

__all__ = [
    # Agent 管理
    "AgentManager",
    "AgentInfo",
    "ParsedConfig",
    # 配置存储
    "ConfigStore",
    "StoredConfig",
    "MemoryConfigStore",
    "MongoConfigStore",
    "create_config_store",
    # 数据库
    "DB_NAME",
    "COLLECTIONS",
    "Database",
    "get_database",
    # 数据模型
    "SessionDocument",
    "SessionStatus",
    "MessageDocument",
    "MessageRole",
    "AgentStateDocument",
    "AgentStatus",
    "AuditLogDocument",
    "AuditAction",
    # Repository
    "RepositoryManager",
    "SessionRepository",
    "MessageRepository",
    "AgentStateRepository",
    "AuditLogRepository",
    "create_repository_manager",
    # 任务管理（协程取消机制）
    "TaskManager",
]
