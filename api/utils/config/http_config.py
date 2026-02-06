"""
HTTP工具模块

提供通用的异步HTTP请求功能和Agent配置加载
"""

import json
import logging
from typing import Dict, Any, Optional
from dataclasses import dataclass
import aiohttp

from .api_config import API, get_chatbot_host


class HttpRequestError(Exception):
    """HTTP请求相关异常"""
    def __init__(self, message: str, code: Optional[int] = None, status_code: Optional[int] = None):
        super().__init__(message)
        self.message = message
        self.code = code
        self.status_code = status_code


class AgentConfigError(HttpRequestError):
    """Agent配置相关业务异常（向后兼容）"""
    pass


@dataclass
class AgentConfigRequest:
    """
    Agent配置请求参数结构
    
    用于获取/创建会话和加载配置的统一参数对象
    """
    # 核心必填字段
    session_id: str
    tenant_id: str
    chatbot_id: str
    
    # 可选配置字段
    md5_checksum: Optional[str] = None
    preview: bool = False
    action_book_id: Optional[str] = None
    extra_param: Optional[Dict[str, Any]] = None
    
    def to_context_vars(self) -> Dict[str, str]:
        """
        构建 context_vars（用于 HTTP 工具的占位符替换）

        同时提供 camelCase 和 snake_case 两种风格的 key，
        以兼容不同调用方的占位符约定。
        """
        return {
            "tenantId": self.tenant_id,
            "chatbotId": self.chatbot_id,
            "sessionId": self.session_id,
            "tenant_id": self.tenant_id,
            "chatbot_id": self.chatbot_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AgentConfigRequest':
        """从字典创建 AgentConfigRequest 实例"""
        return cls(
            session_id=data["sessionId"],
            tenant_id=data["tenantId"],
            chatbot_id=data["chatbotId"],
            md5_checksum=data.get("md5Checksum"),
            preview=data.get("preview", False),
            action_book_id=data.get("actionBookId"),
            extra_param=data.get("extraParam"),
        )

class AsyncHttpClient:
    """通用异步HTTP客户端"""
    
    def __init__(self, logger: logging.Logger, timeout: float = 10.0):
        """
        Args:
            logger: 日志记录器
            timeout: 默认超时时间（秒）
        """
        self.logger = logger
        self.timeout = timeout
    
    async def post_json(
        self,
        url: str,
        payload: Dict[str, Any],
        headers: Optional[Dict[str, str]] = None,
        timeout: Optional[float] = None,
    ) -> Dict[str, Any]:
        """
        发送POST请求（JSON格式）
        
        Args:
            url: 目标URL
            payload: 请求体数据
            headers: 自定义请求头
            timeout: 超时时间（覆盖默认值）
            
        Returns:
            响应JSON数据
            
        Raises:
            HttpRequestError: HTTP请求失败或响应错误
        """
        final_timeout = aiohttp.ClientTimeout(total=timeout or self.timeout)
        final_headers = {"Content-Type": "application/json"}
        if headers:
            final_headers.update(headers)
        
        try:
            async with aiohttp.ClientSession(timeout=final_timeout) as session:
                self.logger.debug(f"📤 POST {url}")
                self.logger.info(f"📦 Payload: {json.dumps(payload, indent=2, ensure_ascii=False)}")
                
                async with session.post(url, json=payload, headers=final_headers) as response:
                    status = response.status
                    
                    try:
                        data = await response.json()
                    except (aiohttp.ContentTypeError, json.JSONDecodeError):
                        text = await response.text()
                        raise HttpRequestError(
                            f"Invalid JSON response: {text[:200]}",
                            status_code=status
                        )
                    
                    if status >= 400:
                        error_msg = data.get("message", f"HTTP {status}")
                        raise HttpRequestError(error_msg, status_code=status)

                    self.logger.info(f"📥 Response: {json.dumps(data, indent=2, ensure_ascii=False)}")
                    return data
                    
        except aiohttp.ClientError as e:
            self.logger.error(f"❌ HTTP request failed: {e}")
            raise HttpRequestError(f"HTTP request failed: {e}")
        except Exception as e:
            if isinstance(e, HttpRequestError):
                raise
            self.logger.error(f"❌ Unexpected error: {e}")
            raise HttpRequestError(f"Unexpected error: {e}")


class HttpConfigLoader:
    """HTTP配置加载器"""
    
    def __init__(self, logger: logging.Logger):
        self.logger = logger
        self._http_client = AsyncHttpClient(logger, timeout=10.0)
    
    async def load_config_from_http(self, request: AgentConfigRequest) -> Dict[str, Any]:
        """
        从HTTP请求获取配置信息
        
        Args:
            request: 配置请求参数
            base_url: API基础URL
            
        Returns:
            配置字典，结构与本地配置文件一致
            
        Raises:
            httpx.HTTPError: HTTP请求失败
            AgentConfigError: 业务逻辑错误（如配置未找到、验证失败等）
            ValueError: 响应数据格式错误
        """
        
        # config_path = "docs/configs/sopv3.json"
        
        # with open(config_path, "r", encoding="utf-8") as f:
        #     return json.load(f)

        url = API.build_url(API.GET_AGENT_CONFIG, base_url=get_chatbot_host())
        
        request_data = {
            "tenantId": request.tenant_id,
            "chatbotId": request.chatbot_id,
            "preview": request.preview,
            "actionBookId": request.action_book_id,
            "extraParam": request.extra_param or {},
            "sessionId": request.session_id,
        }
        
        try:
            response = await self._http_client.post_json(url, request_data)
            
            # 检查业务响应码
            if response.get("code") != 0:
                error_code = response.get("code")
                error_message = response.get("message", "unknown business error")
                self.logger.error(f"❌ business request failed: code={error_code}, msg={error_message}")
                raise AgentConfigError(error_message, error_code)
            
            return response.get("data")
            
        except HttpRequestError:
            raise
        except Exception as e:
            self.logger.error(f"❌ AgentConfig loading failed: {e}")
            raise
