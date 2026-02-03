"""
数据模型模块

v2: 6表设计 (configs, sessions, messages, tool_calls, usages, timers) - 简化版
"""

from api.models.enums import (
    SessionStatus,
    MessageRole,
    AgentPhase,
    EventType,
    EventStatus,
    AuditAction,
    AgentStatus,
)
from api.models.documents_v2 import (
    # v2 优化模型 (独立文件)
    ConfigDocumentV2,
    SessionDocumentV2,
    MessageDocumentV2,
    ToolCallDocumentV2,
    UsageDocumentV2,
    TimerDocumentV2,
)
from api.models.schemas import (
    QueryRequest,
    QueryResponse,
    SessionInfo,
    HealthResponse,
    ErrorResponse,
    AgentStats,
    ChatResponseDTO,
    ChatAsyncResponse,
)

__all__ = [
    # 枚举
    "SessionStatus",
    "MessageRole",
    "AgentPhase",
    "EventType",
    "EventStatus",
    "AuditAction",
    "AgentStatus",

    # v2 优化模型
    "ConfigDocumentV2",
    "SessionDocumentV2",
    "MessageDocumentV2",
    "ToolCallDocumentV2",
    "UsageDocumentV2",
    "TimerDocumentV2",
    # API 模型
    "QueryRequest",
    "QueryResponse",
    "SessionInfo",
    "HealthResponse",
    "ErrorResponse",
    "AgentStats",
    "ChatResponseDTO",
    "ChatAsyncResponse",
]
