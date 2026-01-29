"""
数据模型枚举

定义所有枚举类型
"""

from enum import Enum


class SessionStatus(str, Enum):
    """会话状态"""
    ACTIVE = "active"
    IDLE = "idle"
    CLOSED = "closed"
    EXPIRED = "expired"


class MessageRole(str, Enum):
    """消息角色"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class AgentStatus(str, Enum):
    """Agent 状态"""
    INITIALIZING = "initializing"
    READY = "ready"
    PROCESSING = "processing"
    IDLE = "idle"
    ERROR = "error"
    TERMINATED = "terminated"


class AuditAction(str, Enum):
    """审计动作类型"""
    # 会话相关
    SESSION_CREATED = "session_created"
    SESSION_CLOSED = "session_closed"
    SESSION_EXPIRED = "session_expired"

    # 消息相关
    MESSAGE_RECEIVED = "message_received"
    MESSAGE_SENT = "message_sent"

    # Agent 相关
    AGENT_CREATED = "agent_created"
    AGENT_DESTROYED = "agent_destroyed"
    AGENT_CONFIG_CHANGED = "agent_config_changed"

    # 配置相关
    CONFIG_PARSED = "config_parsed"
    CONFIG_LOADED = "config_loaded"

    # 工具调用
    TOOL_INVOKED = "tool_invoked"
    TOOL_COMPLETED = "tool_completed"
    TOOL_FAILED = "tool_failed"

    # 错误
    ERROR_OCCURRED = "error_occurred"
