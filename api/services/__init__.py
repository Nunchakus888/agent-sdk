"""
业务服务模块 v3

提供 Agent 管理、配置存储、数据库管理、任务管理、LLM 服务等功能
5表设计：configs, sessions, messages, events, usages
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
    MessageDocument,
    MessageRole,
    EventDocument,
    EventType,
    EventStatus,
    TokenDocument,
    TokenDetail,
    TokenSummary,
)
from api.services.task_manager import TaskManager
from api.services.repositories import (
    RepositoryManager,
    SessionRepository,
    MessageRepository,
    EventRepository,
    UsageRepository,
    create_repository_manager,
)
from api.services.llm_service import LLMService, LLMConfig, ModelTask

__all__ = [
    # LLM 服务
    "LLMService",
    "LLMConfig",
    "ModelTask",
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
    # v3 核心模型
    "SessionDocument",
    "MessageDocument",
    "MessageRole",
    "EventDocument",
    "EventType",
    "EventStatus",
    "TokenDocument",
    "TokenDetail",
    "TokenSummary",
    # v3 Repository
    "RepositoryManager",
    "SessionRepository",
    "MessageRepository",
    "EventRepository",
    "UsageRepository",
    "create_repository_manager",
    # 任务管理
    "TaskManager",
]
