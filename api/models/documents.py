"""
MongoDB 文档模型

定义所有 MongoDB 文档的数据模型
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from api.models.enums import (
    SessionStatus,
    MessageRole,
    AgentStatus,
    AuditAction,
)


@dataclass
class SessionDocument:
    """
    会话文档模型

    存储会话的元数据和状态
    """
    session_id: str                          # 主键
    tenant_id: str                           # 租户ID
    chatbot_id: str                          # Chatbot ID
    customer_id: Optional[str] = None        # 客户ID

    # 状态
    status: SessionStatus = SessionStatus.ACTIVE

    # 关联
    agent_id: Optional[str] = None           # 关联的 Agent ID
    config_hash: Optional[str] = None        # 配置哈希

    # 统计
    message_count: int = 0                   # 消息数量

    # 元数据
    title: Optional[str] = None              # 会话标题
    source: str = "api"                      # 来源
    is_preview: bool = False                 # 是否预览模式
    metadata: dict = field(default_factory=dict)  # 扩展元数据

    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_message_at: Optional[datetime] = None
    closed_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "_id": self.session_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "customer_id": self.customer_id,
            "status": self.status.value if isinstance(self.status, SessionStatus) else self.status,
            "agent_id": self.agent_id,
            "config_hash": self.config_hash,
            "message_count": self.message_count,
            "title": self.title,
            "source": self.source,
            "is_preview": self.is_preview,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_message_at": self.last_message_at,
            "closed_at": self.closed_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionDocument":
        """从字典创建"""
        status = data.get("status", SessionStatus.ACTIVE)
        if isinstance(status, str):
            status = SessionStatus(status)

        return cls(
            session_id=data.get("_id") or data.get("session_id"),
            tenant_id=data["tenant_id"],
            chatbot_id=data["chatbot_id"],
            customer_id=data.get("customer_id"),
            status=status,
            agent_id=data.get("agent_id"),
            config_hash=data.get("config_hash"),
            message_count=data.get("message_count", 0),
            title=data.get("title"),
            source=data.get("source", "api"),
            is_preview=data.get("is_preview", False),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
            last_message_at=data.get("last_message_at"),
            closed_at=data.get("closed_at"),
        )


@dataclass
class MessageDocument:
    """
    消息文档模型

    存储会话中的每条消息
    """
    message_id: str                          # 主键（自动生成）
    session_id: str                          # 会话ID（外键）
    tenant_id: str                           # 租户ID

    # 消息内容
    role: MessageRole                        # 角色
    content: str                             # 消息内容

    # 元数据
    correlation_id: Optional[str] = None     # 关联ID（用于追踪）
    parent_message_id: Optional[str] = None  # 父消息ID（用于对话树）

    # 工具调用相关
    tool_calls: Optional[List[dict]] = None  # 工具调用列表
    tool_call_id: Optional[str] = None       # 工具调用ID（tool 角色时）

    # 统计
    token_count: Optional[int] = None        # Token 数量
    latency_ms: Optional[int] = None         # 响应延迟（毫秒）

    # 扩展
    metadata: dict = field(default_factory=dict)

    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "_id": self.message_id,
            "message_id": self.message_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "parent_message_id": self.parent_message_id,
            "tool_calls": self.tool_calls,
            "tool_call_id": self.tool_call_id,
            "token_count": self.token_count,
            "latency_ms": self.latency_ms,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageDocument":
        """从字典创建"""
        role = data.get("role", MessageRole.USER)
        if isinstance(role, str):
            role = MessageRole(role)

        return cls(
            message_id=data.get("_id") or data.get("message_id"),
            session_id=data["session_id"],
            tenant_id=data["tenant_id"],
            role=role,
            content=data["content"],
            correlation_id=data.get("correlation_id"),
            parent_message_id=data.get("parent_message_id"),
            tool_calls=data.get("tool_calls"),
            tool_call_id=data.get("tool_call_id"),
            token_count=data.get("token_count"),
            latency_ms=data.get("latency_ms"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow()),
        )


@dataclass
class AgentStateDocument:
    """
    Agent 状态文档模型

    存储 Agent 的运行时状态快照
    """
    agent_id: str                            # 主键（tenant_id:chatbot_id）
    tenant_id: str                           # 租户ID
    chatbot_id: str                          # Chatbot ID

    # 状态
    status: AgentStatus = AgentStatus.READY

    # 配置
    config_hash: str = ""                    # 当前配置哈希

    # 会话统计
    active_sessions: List[str] = field(default_factory=list)  # 活跃会话列表
    total_sessions: int = 0                  # 总会话数
    total_messages: int = 0                  # 总消息数

    # 性能统计
    avg_latency_ms: Optional[float] = None   # 平均响应延迟
    error_count: int = 0                     # 错误计数

    # 元数据
    metadata: dict = field(default_factory=dict)

    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    last_active_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "_id": self.agent_id,
            "agent_id": self.agent_id,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "status": self.status.value if isinstance(self.status, AgentStatus) else self.status,
            "config_hash": self.config_hash,
            "active_sessions": self.active_sessions,
            "total_sessions": self.total_sessions,
            "total_messages": self.total_messages,
            "avg_latency_ms": self.avg_latency_ms,
            "error_count": self.error_count,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "last_active_at": self.last_active_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AgentStateDocument":
        """从字典创建"""
        status = data.get("status", AgentStatus.READY)
        if isinstance(status, str):
            status = AgentStatus(status)

        return cls(
            agent_id=data.get("_id") or data.get("agent_id"),
            tenant_id=data["tenant_id"],
            chatbot_id=data["chatbot_id"],
            status=status,
            config_hash=data.get("config_hash", ""),
            active_sessions=data.get("active_sessions", []),
            total_sessions=data.get("total_sessions", 0),
            total_messages=data.get("total_messages", 0),
            avg_latency_ms=data.get("avg_latency_ms"),
            error_count=data.get("error_count", 0),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
            last_active_at=data.get("last_active_at", datetime.utcnow()),
        )


@dataclass
class AuditLogDocument:
    """
    审计日志文档模型

    记录所有重要操作的审计轨迹
    """
    log_id: str                              # 主键（自动生成）
    tenant_id: str                           # 租户ID

    # 动作
    action: AuditAction                      # 动作类型

    # 关联实体
    session_id: Optional[str] = None         # 会话ID
    agent_id: Optional[str] = None           # Agent ID
    chatbot_id: Optional[str] = None         # Chatbot ID
    message_id: Optional[str] = None         # 消息ID

    # 请求上下文
    correlation_id: Optional[str] = None     # 关联ID
    request_id: Optional[str] = None         # 请求ID

    # 详情
    details: dict = field(default_factory=dict)  # 详细信息

    # 结果
    success: bool = True                     # 是否成功
    error_message: Optional[str] = None      # 错误信息

    # 性能
    duration_ms: Optional[int] = None        # 耗时（毫秒）

    # 来源
    source: str = "api"                      # 来源
    ip_address: Optional[str] = None         # IP 地址
    user_agent: Optional[str] = None         # User Agent

    # 时间戳
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """转换为字典"""
        return {
            "_id": self.log_id,
            "log_id": self.log_id,
            "tenant_id": self.tenant_id,
            "action": self.action.value if isinstance(self.action, AuditAction) else self.action,
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "chatbot_id": self.chatbot_id,
            "message_id": self.message_id,
            "correlation_id": self.correlation_id,
            "request_id": self.request_id,
            "details": self.details,
            "success": self.success,
            "error_message": self.error_message,
            "duration_ms": self.duration_ms,
            "source": self.source,
            "ip_address": self.ip_address,
            "user_agent": self.user_agent,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "AuditLogDocument":
        """从字典创建"""
        action = data.get("action", AuditAction.MESSAGE_RECEIVED)
        if isinstance(action, str):
            action = AuditAction(action)

        return cls(
            log_id=data.get("_id") or data.get("log_id"),
            tenant_id=data["tenant_id"],
            action=action,
            session_id=data.get("session_id"),
            agent_id=data.get("agent_id"),
            chatbot_id=data.get("chatbot_id"),
            message_id=data.get("message_id"),
            correlation_id=data.get("correlation_id"),
            request_id=data.get("request_id"),
            details=data.get("details", {}),
            success=data.get("success", True),
            error_message=data.get("error_message"),
            duration_ms=data.get("duration_ms"),
            source=data.get("source", "api"),
            ip_address=data.get("ip_address"),
            user_agent=data.get("user_agent"),
            created_at=data.get("created_at", datetime.utcnow()),
        )
