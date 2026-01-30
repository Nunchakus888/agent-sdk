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


class AgentPhase(str, Enum):
    """Agent 处理阶段"""
    PLANNING = "planning"
    EXECUTING = "executing"
    RESPONDING = "responding"
    IDLE = "idle"


class EventType(str, Enum):
    """事件类型"""
    CONFIG_LOAD = "config_load"
    CONFIG_PARSE = "config_parse"
    CACHE_HIT = "cache_hit"
    CACHE_MISS = "cache_miss"
    LLM_DECISION = "llm_decision"
    TOOL_CALL = "tool_call"
    KB_RETRIEVE = "kb_retrieve"
    RESPONSE_GENERATE = "response_gen"
    ERROR = "error"


class EventStatus(str, Enum):
    """事件状态"""
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


class InspectionPhase(str, Enum):
    """Token 消耗阶段"""
    CONFIG_PARSE = "config_parse"
    DECISION = "decision"
    TOOL_CALL = "tool_call"
    KB_RETRIEVE = "kb_retrieve"
    RESPONSE = "response"
    OTHER = "other"


class AuditAction(str, Enum):
    """审计日志动作类型"""
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
    AGENT_CONFIG_RELOADED = "agent_config_reloaded"
    # 错误相关
    ERROR_OCCURRED = "error_occurred"
    # 系统相关
    SYSTEM_STARTUP = "system_startup"
    SYSTEM_SHUTDOWN = "system_shutdown"


class AgentStatus(str, Enum):
    """Agent 状态"""
    IDLE = "idle"
    PROCESSING = "processing"
    ERROR = "error"
    TERMINATED = "terminated"
