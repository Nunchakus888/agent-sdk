"""
MongoDB 文档模型 v3

基于 data_model_design.md v3 设计
- 5表设计：configs, sessions, messages, events, usages
- sessions 字段平铺，常用字段直接访问
- events 增加 offset 字段，支持顺序追溯
- usages 独立表，明细 + 汇总
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from api.models.enums import (
    SessionStatus,
    MessageRole,
    AgentPhase,
    EventType,
    EventStatus,
)


# =============================================================================
# Session Document (v3: 字段平铺设计)
# =============================================================================


@dataclass
class SessionDocument:
    """
    会话文档模型 (v3)

    唯一存储 tenant_id 和 chatbot_id 的表
    常用字段平铺，不常用字段放 metadata
    """
    session_id: str                          # 主键
    tenant_id: str                           # 租户ID (唯一存储位置)
    chatbot_id: str                          # Chatbot ID (唯一存储位置)

    # === 核心字段 (平铺) ===
    customer_id: Optional[str] = None        # 客户ID
    config_hash: Optional[str] = None        # 关联配置
    title: Optional[str] = None              # 会话标题
    source: Optional[str] = None             # 来源渠道 (web | app | api)

    # === 时间字段 ===
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    # === 扩展字段 (不常用) ===
    metadata: dict = field(default_factory=dict)  # tags, language, user_agent, ip_address, extra

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
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionDocument":
        return cls(
            session_id=data.get("_id") or data.get("session_id"),
            tenant_id=data["tenant_id"],
            chatbot_id=data["chatbot_id"],
            customer_id=data.get("customer_id"),
            config_hash=data.get("config_hash"),
            title=data.get("title"),
            source=data.get("source"),
            created_at=data.get("created_at", datetime.utcnow()),
            updated_at=data.get("updated_at", datetime.utcnow()),
            metadata=data.get("metadata", {}),
        )


# =============================================================================
# MessageState (嵌入式文档，用于 MessageDocument)
# =============================================================================


@dataclass
class MessageState:
    """
    消息状态 (嵌入式)

    4表设计：将 StateDocument 嵌入到 MessageDocument
    仅 assistant 消息有值
    """
    phase: AgentPhase = AgentPhase.IDLE      # 处理阶段
    sop_step: Optional[str] = None           # 当前 SOP 步骤
    context: dict = field(default_factory=dict)  # 上下文快照 (可选，按需存储)
    decision: Optional[dict] = None          # {action, confidence, reasoning}

    def to_dict(self) -> dict:
        return {
            "phase": self.phase.value if isinstance(self.phase, AgentPhase) else self.phase,
            "sop_step": self.sop_step,
            "context": self.context,
            "decision": self.decision,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageState":
        phase = data.get("phase", AgentPhase.IDLE)
        if isinstance(phase, str):
            phase = AgentPhase(phase)

        return cls(
            phase=phase,
            sop_step=data.get("sop_step"),
            context=data.get("context", {}),
            decision=data.get("decision"),
        )


# =============================================================================
# Message Document (v3: 精简设计)
# =============================================================================


@dataclass
class MessageDocument:
    """
    消息文档模型 (v3)

    不存储 tenant_id/chatbot_id，通过 session_id 关联查询

    v3 变更：
    - 移除 tool_calls/tool_call_id (工具调用在单次请求内完成，详情存储在 events 表)
    - 移除 parent_message_id (不必要，线性对话按 created_at 排序)
    - 移除 latency_ms (移至 events.duration_ms，便于性能分析)
    - role 只包含 user/assistant/system，不再包含 tool
    """
    message_id: str                          # 主键 (UUID)
    session_id: str                          # 外键：会话ID
    role: MessageRole                        # 角色 (user | assistant | system)
    content: str                             # 消息内容
    correlation_id: Optional[str] = None     # 请求关联ID
    # 嵌入状态 (仅 assistant 消息)
    state: Optional[MessageState] = None     # Agent 状态快照
    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "_id": self.message_id,
            "session_id": self.session_id,
            "role": self.role.value if isinstance(self.role, MessageRole) else self.role,
            "content": self.content,
            "correlation_id": self.correlation_id,
            "state": self.state.to_dict() if self.state else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MessageDocument":
        role = data.get("role", MessageRole.USER)
        if isinstance(role, str):
            role = MessageRole(role)

        # 解析嵌入的 state
        state_data = data.get("state")
        state = MessageState.from_dict(state_data) if state_data else None

        return cls(
            message_id=data.get("_id") or data.get("message_id"),
            session_id=data["session_id"],
            role=role,
            content=data["content"],
            correlation_id=data.get("correlation_id"),
            state=state,
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow()),
        )


# =============================================================================
# Token Usage (嵌入式文档，用于 TokenDocument.details)
# =============================================================================


@dataclass
class TokenUsage:
    """Token 消耗统计 (嵌入式)"""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenUsage":
        return cls(
            model=data.get("model", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cost_usd=data.get("cost_usd"),
        )


# =============================================================================
# Event Document (v3: 纯事件日志，增加 offset 和 action)
# =============================================================================


@dataclass
class EventDocument:
    """
    事件文档模型 (v3)

    记录请求处理过程中的所有事件
    v3 变更：
    - 移除 tokens 字段（独立到 tokens 表）
    - 增加 offset 字段（会话内事件序号，从 0 开始）
    - 增加 action 字段（具体动作，便于分析）
    """
    event_id: str                            # 主键 (UUID)
    session_id: str                          # 外键：会话ID
    correlation_id: str                      # 请求关联ID

    # === 事件顺序 ===
    offset: int = 0                          # 会话内事件序号 (从 0 开始自增)

    # === 事件信息 ===
    message_id: Optional[str] = None         # 外键：消息ID (可选)
    event_type: EventType = EventType.ERROR  # 事件类型
    action: Optional[str] = None             # 具体动作 (如 "get_weather", "to_human")
    status: EventStatus = EventStatus.STARTED

    # === 时间和性能 ===
    duration_ms: Optional[int] = None        # 耗时 (性能分析核心指标)
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None

    # === 输入输出 (调试追溯核心字段) ===
    input_data: Optional[dict] = None        # 输入数据
    output_data: Optional[dict] = None       # 输出数据
    error: Optional[str] = None              # 错误信息

    metadata: dict = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "_id": self.event_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "offset": self.offset,
            "message_id": self.message_id,
            "event_type": self.event_type.value if isinstance(self.event_type, EventType) else self.event_type,
            "action": self.action,
            "status": self.status.value if isinstance(self.status, EventStatus) else self.status,
            "duration_ms": self.duration_ms,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "input_data": self.input_data,
            "output_data": self.output_data,
            "error": self.error,
            "metadata": self.metadata,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "EventDocument":
        event_type = data.get("event_type", EventType.ERROR)
        if isinstance(event_type, str):
            event_type = EventType(event_type)

        status = data.get("status", EventStatus.STARTED)
        if isinstance(status, str):
            status = EventStatus(status)

        return cls(
            event_id=data.get("_id") or data.get("event_id"),
            session_id=data["session_id"],
            correlation_id=data["correlation_id"],
            offset=data.get("offset", 0),
            message_id=data.get("message_id"),
            event_type=event_type,
            action=data.get("action"),
            status=status,
            duration_ms=data.get("duration_ms"),
            start_time=data.get("start_time"),
            end_time=data.get("end_time"),
            input_data=data.get("input_data"),
            output_data=data.get("output_data"),
            error=data.get("error"),
            metadata=data.get("metadata", {}),
            created_at=data.get("created_at", datetime.utcnow()),
        )


# =============================================================================
# Token Detail (嵌入式文档，用于 TokenDocument.details)
# =============================================================================


@dataclass
class TokenDetail:
    """
    Token 消耗明细 (嵌入式) - 对齐 OpenAI API 规范

    支持 OpenAI prompt_tokens_details 和 completion_tokens_details 字段
    """
    phase: str                               # decision | tool_call | response
    event_id: str                            # 关联事件ID
    model: str                               # 模型名称

    # === 基础 Token 统计 ===
    input_tokens: int = 0                    # prompt_tokens
    output_tokens: int = 0                   # completion_tokens
    total_tokens: int = 0                    # 总计

    # === 输入 Token 明细 (OpenAI prompt_tokens_details) ===
    cached_tokens: int = 0                   # 缓存命中的 tokens
    audio_input_tokens: int = 0              # 音频输入 tokens

    # === 输出 Token 明细 (OpenAI completion_tokens_details) ===
    reasoning_tokens: int = 0                # 推理 tokens (o1/o3 模型)
    audio_output_tokens: int = 0             # 音频输出 tokens

    # === 费用 ===
    cost_usd: float = 0.0

    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        return {
            "phase": self.phase,
            "event_id": self.event_id,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cached_tokens": self.cached_tokens,
            "audio_input_tokens": self.audio_input_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "audio_output_tokens": self.audio_output_tokens,
            "cost_usd": self.cost_usd,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenDetail":
        return cls(
            phase=data.get("phase", ""),
            event_id=data.get("event_id", ""),
            model=data.get("model", ""),
            input_tokens=data.get("input_tokens", 0),
            output_tokens=data.get("output_tokens", 0),
            total_tokens=data.get("total_tokens", 0),
            cached_tokens=data.get("cached_tokens", 0),
            audio_input_tokens=data.get("audio_input_tokens", 0),
            reasoning_tokens=data.get("reasoning_tokens", 0),
            audio_output_tokens=data.get("audio_output_tokens", 0),
            cost_usd=data.get("cost_usd", 0.0),
            timestamp=data.get("timestamp", datetime.utcnow()),
        )


# =============================================================================
# Token Summary (嵌入式文档，用于 TokenDocument.summary)
# =============================================================================


@dataclass
class TokenSummary:
    """Token 消耗汇总 (嵌入式)"""
    # === 基础汇总 ===
    total_input: int = 0
    total_output: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0

    # === 明细汇总 ===
    total_cached: int = 0                    # 缓存命中总计
    total_reasoning: int = 0                 # 推理 tokens 总计

    # === 按模型统计 ===
    model_breakdown: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "total_input": self.total_input,
            "total_output": self.total_output,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "total_cached": self.total_cached,
            "total_reasoning": self.total_reasoning,
            "model_breakdown": self.model_breakdown,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenSummary":
        return cls(
            total_input=data.get("total_input", 0),
            total_output=data.get("total_output", 0),
            total_tokens=data.get("total_tokens", 0),
            total_cost=data.get("total_cost", 0.0),
            total_cached=data.get("total_cached", 0),
            total_reasoning=data.get("total_reasoning", 0),
            model_breakdown=data.get("model_breakdown", {}),
        )


# =============================================================================
# Token Document (v3: 独立 Token 消耗表)
# =============================================================================


@dataclass
class TokenDocument:
    """
    Token 消耗文档模型 (v3)

    独立存储 Token 消耗，支持明细和汇总
    一个 correlation_id 对应一条记录
    """
    token_id: str                            # 主键 (UUID)
    session_id: str                          # 外键：会话ID
    correlation_id: str                      # 请求关联ID (唯一索引)
    message_id: Optional[str] = None         # 关联消息ID (可选)

    # === Token 明细 ===
    details: List[TokenDetail] = field(default_factory=list)

    # === Token 汇总 (请求结束时计算) ===
    summary: Optional[TokenSummary] = None

    # === 状态 ===
    is_finalized: bool = False               # 是否已完成统计
    created_at: datetime = field(default_factory=datetime.utcnow)
    finalized_at: Optional[datetime] = None

    def to_dict(self) -> dict:
        return {
            "_id": self.token_id,
            "session_id": self.session_id,
            "correlation_id": self.correlation_id,
            "message_id": self.message_id,
            "details": [d.to_dict() for d in self.details],
            "summary": self.summary.to_dict() if self.summary else None,
            "is_finalized": self.is_finalized,
            "created_at": self.created_at,
            "finalized_at": self.finalized_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "TokenDocument":
        details = [TokenDetail.from_dict(d) for d in data.get("details", [])]
        summary = TokenSummary.from_dict(data["summary"]) if data.get("summary") else None

        return cls(
            token_id=data.get("_id") or data.get("token_id"),
            session_id=data["session_id"],
            correlation_id=data["correlation_id"],
            message_id=data.get("message_id"),
            details=details,
            summary=summary,
            is_finalized=data.get("is_finalized", False),
            created_at=data.get("created_at", datetime.utcnow()),
            finalized_at=data.get("finalized_at"),
        )

    def add_detail(self, detail: TokenDetail) -> None:
        """添加 Token 消耗明细"""
        self.details.append(detail)

    def finalize(self) -> None:
        """计算汇总并标记完成"""
        total_input = 0
        total_output = 0
        total_cost = 0.0
        total_cached = 0
        total_reasoning = 0
        model_breakdown: dict = {}

        for detail in self.details:
            total_input += detail.input_tokens
            total_output += detail.output_tokens
            total_cost += detail.cost_usd
            total_cached += detail.cached_tokens
            total_reasoning += detail.reasoning_tokens

            if detail.model not in model_breakdown:
                model_breakdown[detail.model] = {
                    "input": 0, "output": 0, "cached": 0, "reasoning": 0, "cost": 0.0
                }
            model_breakdown[detail.model]["input"] += detail.input_tokens
            model_breakdown[detail.model]["output"] += detail.output_tokens
            model_breakdown[detail.model]["cached"] += detail.cached_tokens
            model_breakdown[detail.model]["reasoning"] += detail.reasoning_tokens
            model_breakdown[detail.model]["cost"] += detail.cost_usd

        self.summary = TokenSummary(
            total_input=total_input,
            total_output=total_output,
            total_tokens=total_input + total_output,
            total_cost=total_cost,
            total_cached=total_cached,
            total_reasoning=total_reasoning,
            model_breakdown=model_breakdown,
        )
        self.is_finalized = True
        self.finalized_at = datetime.utcnow()
