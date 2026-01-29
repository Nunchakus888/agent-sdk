"""
Correlation ID 模块

提供分布式请求追踪的轻量级实现：
- ContextualCorrelator: 上下文管理器，支持作用域嵌套
- 中间件: 自动为每个请求生成唯一 correlation_id
- 日志集成: 自动注入 correlation_id 到日志

使用示例:
    with correlator.scope("process"):
        logger.info("Processing...")  # 日志自动包含 correlation_id
"""

import contextvars
import uuid
from contextlib import contextmanager
from typing import Any, Generator


# =============================================================================
# Correlation ID 上下文变量
# =============================================================================

# 全局上下文变量，协程安全
_correlation_id: contextvars.ContextVar[str] = contextvars.ContextVar(
    "correlation_id", default=""
)
_correlation_properties: contextvars.ContextVar[dict] = contextvars.ContextVar(
    "correlation_properties", default={}
)


def generate_request_id() -> str:
    """生成唯一请求 ID"""
    return uuid.uuid4().hex[:12]


# =============================================================================
# ContextualCorrelator
# =============================================================================


class ContextualCorrelator:
    """
    上下文关联器
    
    支持作用域嵌套，自动管理 correlation_id 的生命周期。
    
    使用示例:
        correlator = ContextualCorrelator()
        
        with correlator.scope("R1234xyz"):
            print(correlator.correlation_id)  # R1234xyz
            
            with correlator.scope("process"):
                print(correlator.correlation_id)  # R1234xyz::process
                
                with correlator.scope("tool"):
                    print(correlator.correlation_id)  # R1234xyz::process::tool
    """
    
    @contextmanager
    def scope(
        self,
        scope_id: str,
        properties: dict[str, Any] | None = None
    ) -> Generator[str, None, None]:
        """
        进入新的作用域
        
        Args:
            scope_id: 作用域标识符
            properties: 附加属性（如 request_id, session_id 等）
        
        Yields:
            当前完整的 correlation_id
        """
        current = _correlation_id.get()
        new_scope = f"{current}::{scope_id}" if current else scope_id
        
        # 合并属性
        current_props = _correlation_properties.get().copy()
        if properties:
            current_props.update(properties)
        
        # 设置新值
        token_id = _correlation_id.set(new_scope)
        token_props = _correlation_properties.set(current_props)
        
        try:
            yield new_scope
        finally:
            # 恢复原值
            _correlation_id.reset(token_id)
            _correlation_properties.reset(token_props)
    
    @property
    def correlation_id(self) -> str:
        """获取当前 correlation_id"""
        return _correlation_id.get() or "-"
    
    @property
    def properties(self) -> dict[str, Any]:
        """获取当前属性"""
        return _correlation_properties.get().copy()
    
    def get_property(self, key: str, default: Any = None) -> Any:
        """获取指定属性"""
        return _correlation_properties.get().get(key, default)


# 全局单例
correlator = ContextualCorrelator()


# =============================================================================
# 便捷函数
# =============================================================================


def get_correlation_id() -> str:
    """获取当前 correlation_id（便捷函数）"""
    return correlator.correlation_id


def get_request_id() -> str | None:
    """获取当前请求 ID"""
    return correlator.get_property("request_id")


def get_session_id() -> str | None:
    """获取当前会话 ID"""
    return correlator.get_property("session_id")
