"""
API 数据模型

定义请求和响应的 Pydantic 模型
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """查询请求"""

    message: str = Field(..., description="用户消息", min_length=1)
    session_id: str = Field(..., description="会话ID")
    user_id: str | None = Field(default=None, description="用户ID（可选）")

    class Config:
        json_schema_extra = {
            "example": {
                "message": "你好，帮我查询订单状态",
                "session_id": "user_123_session_001",
                "user_id": "user_123",
            }
        }


class QueryResponse(BaseModel):
    """查询响应"""

    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="Agent响应消息")
    status: str = Field(..., description="状态：success | error")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123_session_001",
                "message": "您的订单正在处理中，预计明天送达。",
                "status": "success",
            }
        }


class SessionInfo(BaseModel):
    """会话信息"""

    session_id: str = Field(..., description="会话ID")
    agent_id: str = Field(..., description="Agent ID")
    config_hash: str = Field(..., description="配置哈希")
    need_greeting: bool = Field(..., description="是否需要问候")
    status: str = Field(..., description="会话状态")
    message_count: int = Field(..., description="消息数量")

    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "user_123_session_001",
                "agent_id": "workflow_agent_001",
                "config_hash": "abc123def456",
                "need_greeting": False,
                "status": "active",
                "message_count": 10,
            }
        }


class HealthResponse(BaseModel):
    """健康检查响应"""

    status: str = Field(..., description="健康状态：healthy | unhealthy")
    config_hash: str = Field(..., description="当前配置哈希")
    sessions_count: int = Field(..., description="活跃会话数量")
    version: str = Field(..., description="API 版本")

    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "config_hash": "abc123def456",
                "sessions_count": 5,
                "version": "1.0.0",
            }
        }


class ErrorResponse(BaseModel):
    """错误响应"""

    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    detail: str | None = Field(default=None, description="详细信息")

    class Config:
        json_schema_extra = {
            "example": {
                "error": "ValueError",
                "message": "Invalid session_id",
                "detail": "Session ID must be non-empty",
            }
        }
