"""
V2 数据模型

基于 data_model_design_v2.md 设计
- 6表设计：configs, sessions, messages, tool_calls, usages, timers
- 简化字段，对齐 EventCollector/QueryRecorder 模式
- 扁平结构，避免嵌套

时间字段命名规范：
- created_at: 创建时间（UTC）
- updated_at: 更新时间（UTC）
- closed_at: 关闭时间（UTC）
- next_trigger_at: 下次触发时间（UTC）
- last_triggered_at: 上次触发时间（UTC）

"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

from api.utils.datetime import utc_now, ensure_utc
from bu_agent_sdk.schemas import WorkflowConfigSchema


# =============================================================================
# ConfigDocumentV2 - 配置缓存表
# =============================================================================


@dataclass
class ConfigDocumentV2:
    """
    配置文档模型 (V2)

    _id = chatbot_id（全局唯一），config_hash 用于缓存失效检测
    parsed_config 存储序列化的 WorkflowConfigSchema
    """
    chatbot_id: str                          # _id
    tenant_id: str
    config_hash: str                         # 缓存失效检测
    raw_config: dict
    parsed_config: dict                      # WorkflowConfigSchema.model_dump()
    version: Optional[str] = None
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    access_count: int = 0

    def to_dict(self) -> dict:
        return {
            "_id": self.chatbot_id,
            "tenant_id": self.tenant_id,
            "config_hash": self.config_hash,
            "raw_config": self.raw_config,
            "parsed_config": self.parsed_config,
            "version": self.version,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "access_count": self.access_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ConfigDocumentV2":
        return cls(
            chatbot_id=data["_id"],
            tenant_id=data["tenant_id"],
            config_hash=data.get("config_hash", ""),
            raw_config=data.get("raw_config", {}),
            parsed_config=data.get("parsed_config", {}),
            version=data.get("version"),
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
            updated_at=ensure_utc(data.get("updated_at")) or utc_now(),
            access_count=data.get("access_count", 0),
        )

    def to_workflow_config(self) -> "WorkflowConfigSchema":
        """反序列化为 WorkflowConfigSchema"""
        from bu_agent_sdk.schemas import WorkflowConfigSchema
        return WorkflowConfigSchema(**self.parsed_config)

    @classmethod
    def from_workflow_config(
        cls,
        chatbot_id: str,
        tenant_id: str,
        config_hash: str,
        raw_config: dict,
        config: "WorkflowConfigSchema",
    ) -> "ConfigDocumentV2":
        """从 WorkflowConfigSchema 创建"""
        return cls(
            chatbot_id=chatbot_id,
            tenant_id=tenant_id,
            config_hash=config_hash,
            raw_config=raw_config,
            parsed_config=config.model_dump(),
        )


# =============================================================================
# SessionDocumentV2 - 会话表
# =============================================================================


@dataclass
class SessionDocumentV2:
    """
    会话文档模型 (V2 优化版)

    简化设计：
    - 移除 timer 相关字段（独立 timers 表管理）
    - 移除统计字段（按需计算）
    - 保留核心标识和时间戳
    """
    session_id: str                          # _id
    tenant_id: str                           # 租户ID
    chatbot_id: str                          # Chatbot ID

    # 核心字段
    customer_id: Optional[str] = None        # 客户ID
    config_hash: Optional[str] = None        # 配置哈希（用于变更检测）
    title: Optional[str] = None              # 会话标题
    source: Optional[str] = None             # 来源渠道 (web | app | api)

    # 时间字段
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    closed_at: Optional[datetime] = None     # 会话关闭时间

    # 可选元数据
    metadata: dict = field(default_factory=dict)  # tags, language 等

    def to_dict(self) -> dict:
        return {
            "_id": self.session_id,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "customer_id": self.customer_id,
            "config_hash": self.config_hash,
            "title": self.title,
            "source": self.source,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "closed_at": self.closed_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionDocumentV2":
        return cls(
            session_id=data.get("_id") or data.get("session_id"),
            tenant_id=data["tenant_id"],
            chatbot_id=data["chatbot_id"],
            customer_id=data.get("customer_id"),
            config_hash=data.get("config_hash"),
            title=data.get("title"),
            source=data.get("source"),
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
            updated_at=ensure_utc(data.get("updated_at")) or utc_now(),
            closed_at=ensure_utc(data.get("closed_at")),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# MessageDocumentV2 - 消息表
# =============================================================================


@dataclass
class MessageDocumentV2:
    """
    消息文档模型 (V2 优化版)

    简化设计：
    - 移除 MessageState（V2 不使用状态追踪）
    - 移除 metadata（V2 不使用）
    - 只存储 user/assistant 消息
    """
    message_id: str                          # _id (UUID)
    session_id: str                          # 外键：会话ID
    role: str                                # "user" | "assistant"
    content: str                             # 消息内容
    correlation_id: Optional[str] = None     # 请求关联ID
    offset: int = 0                          # 会话内单调递增序号
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "offset": self.offset,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageDocumentV2":
        return cls(
            message_id=data.get("_id") or data.get("message_id"),
            session_id=data["session_id"],
            role=data["role"],
            content=data["content"],
            correlation_id=data.get("correlation_id"),
            offset=data.get("offset", 0),
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
        )


# =============================================================================
# ToolCallDocumentV2 - 工具调用表
# =============================================================================


@dataclass
class ToolCallDocumentV2:
    """
    工具调用文档模型 (V2 优化版)

    从 events 表重命名，简化设计：
    - 只记录工具调用（移除其他事件类型）
    - 字段对齐 EventCollector.ToolCallRecord
    - 移除冗余字段
    """
    tool_call_id: str                        # _id (来自 LLM)
    session_id: str                          # 外键：会话ID
    correlation_id: str                      # 请求关联ID

    # 排序
    offset: int = 0                          # 会话内序号

    # 工具调用数据
    tool_name: str = ""                      # 工具名称 (如 "get_weather")
    arguments: dict = field(default_factory=dict)  # 工具输入
    result: Optional[str] = None             # 工具输出（截断）
    is_error: bool = False                   # 是否错误

    # 性能
    duration_ms: int = 0                     # 耗时（毫秒）

    # 时间戳
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "_id": self.tool_call_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "offset": self.offset,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "result": self.result,
            "is_error": self.is_error,
            "duration_ms": self.duration_ms,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ToolCallDocumentV2":
        return cls(
            tool_call_id=data.get("_id") or data.get("tool_call_id"),
            session_id=data["session_id"],
            correlation_id=data["correlation_id"],
            offset=data.get("offset", 0),
            tool_name=data.get("tool_name", ""),
            arguments=data.get("arguments", {}),
            result=data.get("result"),
            is_error=data.get("is_error", False),
            duration_ms=data.get("duration_ms", 0),
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
        )


# =============================================================================
# UsageDocumentV2 - Token 消耗表
# =============================================================================


@dataclass
class UsageDocumentV2:
    """
    Token 消耗文档模型 (V2 优化版)

    简化设计：
    - 扁平化结构（移除 details 数组）
    - 移除 is_finalized（写入即完成）
    - 内嵌 by_model 字典
    - 支持 cached_tokens 统计
    """
    usage_id: str                            # _id (UUID)
    session_id: str                          # 外键：会话ID
    correlation_id: str                      # 请求关联ID（唯一）

    # 输入统计
    total_input_tokens: int = 0              # 总输入 tokens
    cached_input_tokens: int = 0             # 缓存命中的输入 tokens

    # 输出统计
    total_output_tokens: int = 0             # 总输出 tokens

    # 汇总统计
    total_tokens: int = 0                    # 总 tokens (input + output)
    total_cost: float = 0.0                  # 总费用 (USD)

    # 按模型统计
    by_model: dict = field(default_factory=dict)
    # 结构: {"gpt-4": {"input": 100, "cached": 20, "output": 50, "cost": 0.01}}

    # 时间戳
    created_at: datetime = field(default_factory=utc_now)

    def to_dict(self) -> dict:
        return {
            "_id": self.usage_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "total_input_tokens": self.total_input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "by_model": self.by_model,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "UsageDocumentV2":
        return cls(
            usage_id=data.get("_id") or data.get("usage_id"),
            session_id=data["session_id"],
            correlation_id=data["correlation_id"],
            total_input_tokens=data.get("total_input_tokens", 0),
            cached_input_tokens=data.get("cached_input_tokens", 0),
            total_output_tokens=data.get("total_output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            by_model=data.get("by_model", {}),
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
        )


# =============================================================================
# TimerDocumentV2 - 定时器表
# =============================================================================


@dataclass
class TimerDocumentV2:
    """
    定时器文档模型 (V2)

    独立表设计：
    - 支持每个 Session 多个 Timer
    - 跟随 Session 生命周期，不跟随 Agent
    - 持久化存储，服务重启后可恢复
    - 配置快照确保 Agent 配置更新不影响已有 Timer
    """
    timer_instance_id: str                   # _id (UUID)
    session_id: str                          # 外键：会话ID
    timer_id: str                            # 关联 TimerConfig.timer_id

    # 状态
    status: str = "pending"                  # pending | triggered | disabled | cancelled
    trigger_count: int = 0                   # 已触发次数

    # 时间
    created_at: datetime = field(default_factory=utc_now)
    next_trigger_at: Optional[datetime] = None  # 下次触发时间（索引字段）
    last_triggered_at: Optional[datetime] = None

    # 配置快照（创建时复制）
    delay_seconds: int = 300                 # 延迟秒数
    max_triggers: int = 1                    # 最大触发次数，0=无限
    tool_name: str = ""                      # 工具名称
    tool_params: dict = field(default_factory=dict)  # 工具参数
    message: Optional[str] = None            # 消息内容（generate_response 专用）

    def to_dict(self) -> dict:
        return {
            "_id": self.timer_instance_id,
            "session_id": self.session_id,
            "timer_id": self.timer_id,
            "status": self.status,
            "trigger_count": self.trigger_count,
            "created_at": self.created_at,
            "next_trigger_at": self.next_trigger_at,
            "last_triggered_at": self.last_triggered_at,
            "delay_seconds": self.delay_seconds,
            "max_triggers": self.max_triggers,
            "tool_name": self.tool_name,
            "tool_params": self.tool_params,
            "message": self.message,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TimerDocumentV2":
        return cls(
            timer_instance_id=data.get("_id") or data.get("timer_instance_id"),
            session_id=data["session_id"],
            timer_id=data["timer_id"],
            status=data.get("status", "pending"),
            trigger_count=data.get("trigger_count", 0),
            created_at=ensure_utc(data.get("created_at")) or utc_now(),
            next_trigger_at=ensure_utc(data.get("next_trigger_at")),
            last_triggered_at=ensure_utc(data.get("last_triggered_at")),
            delay_seconds=data.get("delay_seconds", 300),
            max_triggers=data.get("max_triggers", 1),
            tool_name=data.get("tool_name", ""),
            tool_params=data.get("tool_params", {}),
            message=data.get("message"),
        )


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    "ConfigDocumentV2",
    "SessionDocumentV2",
    "MessageDocumentV2",
    "ToolCallDocumentV2",
    "UsageDocumentV2",
    "TimerDocumentV2",
]
