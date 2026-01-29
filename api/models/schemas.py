"""
API 请求/响应模型

定义 FastAPI 的 Pydantic 模型（请求和响应）
"""

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求 - 完整的会话请求参数"""

    # 必填字段
    message: str = Field(..., description="用户消息", min_length=1)
    session_id: str = Field(..., description="会话ID")
    chatbot_id: str = Field(..., description="Chatbot ID")
    tenant_id: str = Field(..., description="租户ID")

    # 可选字段
    customer_id: Optional[str] = Field(default=None, description="客户ID")
    md5_checksum: Optional[str] = Field(default=None, description="配置文件MD5校验和")
    source: Optional[str] = Field(default="api", description="请求来源")
    is_preview: bool = Field(default=False, description="是否为预览模式")
    autofill_params: Dict[str, Any] = Field(default_factory=dict, description="自动填充参数")
    session_title: Optional[str] = Field(default=None, description="会话标题")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "Hello, I need help with my order",
                "customer_id": "cust_123xy",
                "session_id": "68d510aedff9455e5b019b3e",
                "tenant_id": "dev-test",
                "chatbot_id": "68d510aedff9455e5b019b3e",
                "md5_checksum": "1234567890",
                "source": "bacmk_ui",
                "is_preview": False,
                "autofill_params": {},
                "session_title": "Order Inquiry"
            }
        }


class QueryResponse(BaseModel):
    """查询响应"""

    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="Agent响应消息")
    status: str = Field(..., description="状态：success | error")
    agent_id: Optional[str] = Field(default=None, description="Agent ID")
    config_hash: Optional[str] = Field(default=None, description="配置哈希")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "68d510aedff9455e5b019b3e",
                "message": "您的订单正在处理中，预计明天送达。",
                "status": "success",
                "agent_id": "workflow_agent_68d510aedff9455e5b019b3e",
                "config_hash": "abc123def456"
            }
        }


class SessionInfo(BaseModel):
    """会话信息"""

    session_id: str = Field(..., description="会话ID")
    chatbot_id: str = Field(..., description="Chatbot ID")
    tenant_id: str = Field(..., description="租户ID")
    customer_id: Optional[str] = Field(default=None, description="客户ID")
    agent_id: str = Field(..., description="Agent ID")
    config_hash: str = Field(..., description="配置哈希")
    status: str = Field(..., description="会话状态")
    message_count: int = Field(..., description="消息数量")
    created_at: Optional[str] = Field(default=None, description="创建时间")
    last_active_at: Optional[str] = Field(default=None, description="最后活跃时间")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "68d510aedff9455e5b019b3e",
                "chatbot_id": "68d510aedff9455e5b019b3e",
                "tenant_id": "dev-test",
                "customer_id": "cust_123xy",
                "agent_id": "workflow_agent_68d510aedff9455e5b019b3e",
                "config_hash": "abc123def456",
                "status": "active",
                "message_count": 10,
                "created_at": "2026-01-23T10:00:00Z",
                "last_active_at": "2026-01-23T10:30:00Z"
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="健康状态：healthy | unhealthy")
    active_sessions: int = Field(..., description="活跃会话数量")
    active_agents: int = Field(..., description="活跃Agent数量")
    version: str = Field(..., description="API 版本")
    uptime: Optional[float] = Field(default=None, description="运行时间（秒）")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "active_sessions": 5,
                "active_agents": 3,
                "version": "1.0.0",
                "uptime": 3600.5
            }
        }


class ErrorResponse(BaseModel):
    """错误响应"""

    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    detail: Optional[str] = Field(default=None, description="详细信息")
    session_id: Optional[str] = Field(default=None, description="会话ID")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValueError",
                "message": "Invalid session_id",
                "detail": "Session ID must be non-empty",
                "session_id": "68d510aedff9455e5b019b3e"
            }
        }


class AgentStats(BaseModel):
    """Agent 统计信息"""

    agent_id: str = Field(..., description="Agent ID")
    chatbot_id: str = Field(..., description="Chatbot ID")
    tenant_id: str = Field(..., description="租户ID")
    config_hash: str = Field(..., description="配置哈希")
    session_count: int = Field(..., description="关联会话数")
    created_at: str = Field(..., description="创建时间")
    last_active_at: str = Field(..., description="最后活跃时间")

    class Config:
        json_schema_extra = {
            "example": {
                "agent_id": "workflow_agent_68d510aedff9455e5b019b3e",
                "chatbot_id": "68d510aedff9455e5b019b3e",
                "tenant_id": "dev-test",
                "config_hash": "abc123def456",
                "session_count": 5,
                "created_at": "2026-01-23T10:00:00Z",
                "last_active_at": "2026-01-23T10:30:00Z"
            }
        }


# =============================================================================
# 统一响应格式（从 Parlant ChatResponseDTO 移植）
# =============================================================================


class ChatResponseDTO(BaseModel):
    """
    统一响应格式 - 从 Parlant 移植

    用于 chat_async 等 RESTful API 的标准响应格式
    """

    status: int = Field(default=200, description="HTTP 状态码")
    code: int = Field(default=0, description="业务状态码（0=成功）")
    message: str = Field(default="success", description="状态消息")
    data: Optional[Any] = Field(default=None, description="响应数据")
    correlation_id: Optional[str] = Field(default=None, description="关联ID")
    duration: Optional[float] = Field(default=None, description="处理耗时（秒）")

    class Config:
        json_schema_extra = {
            "example": {
                "status": 200,
                "code": 0,
                "message": "success",
                "data": {
                    "session_id": "68d510aedff9455e5b019b3e",
                    "response": "您的订单正在处理中"
                },
                "correlation_id": "R1234567890",
                "duration": 1.25
            }
        }


class ChatAsyncResponse(BaseModel):
    """chat_async 异步响应"""

    status: int = Field(default=202, description="HTTP 状态码（202=已接受）")
    code: int = Field(default=0, description="业务状态码")
    message: str = Field(default="processing", description="状态消息")
    correlation_id: str = Field(..., description="关联ID，用于追踪请求")
    session_id: str = Field(..., description="会话ID")

    class Config:
        json_schema_extra = {
            "example": {
                "status": 202,
                "code": 0,
                "message": "processing",
                "correlation_id": "R1234567890",
                "session_id": "68d510aedff9455e5b019b3e"
            }
        }
