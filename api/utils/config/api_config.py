"""
API 路径配置管理

简单、直接、实用的 API 路径配置
"""

import os
from typing import Optional


# API 路径常量
class API:
    """API 路径配置"""
    
    # Chatbot 相关
    GET_AGENT_CONFIG = "/chatbot/ai-inner/get-ai-agent-config"
    RETRIEVE_KNOWLEDGE = "/chatbot/ai-inner/retrieve-knowledge"
    
    # 回调相关
    CALLBACK_AGENT_RECEIVE = "/api/callback/agent/receive"
    
    @staticmethod
    def build_url(path: str, base_url: Optional[str] = None, **params) -> str:
        """
        构建完整 URL，支持路径参数替换
        
        Args:
            path: API 路径
            base_url: 基础 URL（可选）
            **params: 路径参数，例如 user_id="123" 会替换 {user_id}
            
        Returns:
            完整的 URL
            
        Examples:
            >>> API.build_url(API.GET_AGENT_CONFIG, base_url="https://api.com")
            'https://api.com/chatbot/ai-inner/get-agent-config'
            
            >>> API.build_url("/users/{user_id}", base_url="https://api.com", user_id="123")
            'https://api.com/users/123'
        """
        # 替换路径参数
        for key, value in params.items():
            path = path.replace(f"{{{key}}}", str(value))
        
        # 添加基础 URL
        if base_url:
            return f"{base_url.rstrip('/')}{path}"
        return path


# 环境变量配置
def get_chatbot_host() -> str:
    """获取 Chatbot API Host"""
    return os.environ.get('AGENT_CONFIGS_HOST')


def get_callback_host() -> str:
    """获取回调 API Host"""
    return os.environ.get('CHAT_CALLBACK_HOST')

