"""
Callback 服务模块

提供统一的回调发送功能，供 chat API 和其他模块使用。
"""

from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional

import httpx
from pydantic import BaseModel, Field, field_validator

from api.core.logging import get_logger
from api.utils.config.api_config import API, get_callback_host

logger = get_logger(__name__)


# =============================================================================
# 业务状态码
# =============================================================================


class CallbackCode(IntEnum):
    """回调业务状态码"""
    TIMEOUT = -2
    PROCESSING_ERROR = -1
    SUCCESS = 0
    CANCELLED = 1


# =============================================================================
# 回调数据模型
# =============================================================================


class CallbackData(BaseModel):
    """回调数据"""
    # todo: source model
    source: str = Field(default="ai_agent", description="事件来源")
    kind: str = Field(default="message", description="事件类型")
    creation_utc: str = Field(..., description="创建时间 ISO8601")
    correlation_id: str = Field(..., description="关联ID")
    total_tokens: int = Field(default=0, description="消耗token数")
    session_id: str = Field(..., description="会话ID")
    message: str = Field(..., description="消息内容")


class CallbackPayload(BaseModel):
    """回调请求体"""
    status: int = Field(..., description="HTTP状态码")
    code: int = Field(..., description="业务状态码")
    message: str = Field(..., description="状态消息")
    duration: float = Field(..., description="处理耗时(秒)")
    correlation_id: str = Field(..., description="关联ID")
    data: Optional[CallbackData] = Field(default=None, description="响应数据")

    @field_validator("duration", mode="before")
    @classmethod
    def round_duration(cls, v: float) -> float:
        """Round duration to 3 decimal places."""
        return round(v, 3)


# =============================================================================
# Callback 服务
# =============================================================================


class CallbackService:
    """
    回调服务

    提供统一的回调发送功能：
    - 发送成功/失败/取消/超时回调
    - 发送问候消息回调
    - 支持自定义回调地址
    """

    def __init__(self, callback_host: str | None = None, timeout: float = 10.0):
        """
        Args:
            callback_host: 回调地址，默认从环境变量 CHAT_CALLBACK_HOST 获取
            timeout: HTTP 请求超时时间
        """
        self._callback_host = callback_host
        self._timeout = timeout

    @property
    def callback_host(self) -> str:
        """获取回调地址"""
        host = self._callback_host or get_callback_host() or ""
        return host.rstrip("/")

    @property
    def callback_url(self) -> str:
        """获取完整回调 URL"""
        return API.build_url(API.CALLBACK_AGENT_RECEIVE, base_url=self.callback_host)

    async def send(self, payload: CallbackPayload) -> bool:
        """
        发送回调

        Returns:
            True 发送成功，False 发送失败
        """
        if not self.callback_host:
            logger.warning("CHAT_CALLBACK_HOST not configured, skip callback")
            return False

        request_body = payload.model_dump(mode="json")
        url = self.callback_url

        # 请求日志：简洁格式
        data = payload.data
        req_summary = (
            f"[{payload.correlation_id[:11]}] "
            f"code={payload.code}, "
            f"msg={payload.message}, "
            f"dur={payload.duration:.2f}s"
        )
        if data:
            req_summary += f", kind={data.kind}, tokens={data.total_tokens}"
        logger.info(f"📤 Callback REQ: {req_summary}")
        logger.debug(f"Callback REQ body: {request_body}")

        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                resp = await client.post(
                    url,
                    json=request_body,
                    headers={"Content-Type": "application/json"},
                )
                # 响应日志
                if resp.status_code == 200:
                    logger.info(
                        f"✅ Callback RESP: [{payload.correlation_id[:11]}] "
                        f"status=200 OK"
                    )
                    return True
                else:
                    resp_text = resp.text[:200] if resp.text else ""
                    logger.warning(
                        f"⚠️ Callback RESP: [{payload.correlation_id[:11]}] "
                        f"status={resp.status_code}, body={resp_text}"
                    )
                    return False
        except Exception as e:
            logger.error(
                f"❌ Callback ERROR: [{payload.correlation_id[:11]}] {type(e).__name__}: {e}"
            )
            return False

    async def send_success(
        self,
        correlation_id: str,
        session_id: str,
        message: str,
        duration: float,
        total_tokens: int = 0,
        kind: str = "message",
    ) -> bool:
        """发送成功回调"""
        return await self.send(CallbackPayload(
            status=200,
            code=CallbackCode.SUCCESS,
            message="SUCCESS",
            duration=duration,
            correlation_id=correlation_id,
            data=CallbackData(
                source="ai_agent",
                kind=kind,
                creation_utc=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id,
                total_tokens=total_tokens,
                session_id=session_id,
                message=message,
            ),
        ))

    async def send_greeting(
        self,
        correlation_id: str,
        session_id: str,
        greeting_message: str,
        duration: float = 0.0,
        total_tokens: int = 0,
    ) -> bool:
        """
        发送问候消息回调

        Args:
            correlation_id: 关联ID（与请求相同）
            session_id: 会话ID
            greeting_message: 问候消息内容
            duration: 处理耗时
            total_tokens: 配置解析消耗的 tokens（首次解析时有值，缓存命中时为 0）
        """
        return await self.send(CallbackPayload(
            status=200,
            code=CallbackCode.SUCCESS,
            message="SUCCESS",
            duration=duration,
            correlation_id=correlation_id,
            data=CallbackData(
                source="ai_agent",
                kind="message",  # 使用 message 类型，后端 AiAgentEventKind 不支持 greeting
                creation_utc=datetime.now(timezone.utc).isoformat(),
                correlation_id=correlation_id,
                total_tokens=total_tokens,
                session_id=session_id,
                message=greeting_message,
            ),
        ))

    async def send_cancelled(
        self,
        correlation_id: str,
        duration: float,
    ) -> bool:
        """发送取消回调"""
        return await self.send(CallbackPayload(
            status=200,
            code=CallbackCode.CANCELLED,
            message="CANCELLED",
            duration=duration,
            correlation_id=correlation_id,
            data=None,
        ))

    async def send_timeout(
        self,
        correlation_id: str,
        duration: float,
    ) -> bool:
        """发送超时回调"""
        return await self.send(CallbackPayload(
            status=504,
            code=CallbackCode.TIMEOUT,
            message="TIMEOUT_ERROR",
            duration=duration,
            correlation_id=correlation_id,
            data=None,
        ))

    async def send_error(
        self,
        correlation_id: str,
        duration: float,
        error_message: str = "PROCESSING_ERROR",
    ) -> bool:
        """发送错误回调"""
        return await self.send(CallbackPayload(
            status=500,
            code=CallbackCode.PROCESSING_ERROR,
            message=error_message,
            duration=duration,
            correlation_id=correlation_id,
            data=None,
        ))


# 默认实例（使用环境变量配置）
_default_service: CallbackService | None = None


def get_callback_service() -> CallbackService:
    """获取默认回调服务实例"""
    global _default_service
    if _default_service is None:
        _default_service = CallbackService()
    return _default_service


# 便捷函数
async def send_callback(payload: CallbackPayload) -> bool:
    """发送回调（便捷函数）"""
    return await get_callback_service().send(payload)
