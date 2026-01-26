"""
依赖注入模块

提供全局依赖项，支持：
- AgentManager 单例（管理多个 Agent）
- 配置管理
- Agent 生命周期管理
"""

from typing import Annotated

from fastapi import Depends

from api.agent_manager import AgentManager


# =============================================================================
# 全局状态（单例模式）
# =============================================================================

_agent_manager: AgentManager | None = None


async def initialize_agent_manager(
    config_dir: str = "config",
    idle_timeout: int = 300,  # 5分钟无会话自动回收
    cleanup_interval: int = 60,  # 每分钟检查一次
    enable_llm_parsing: bool = False,  # 是否启用 LLM 配置解析
) -> AgentManager:
    """
    初始化 AgentManager（单例）

    Args:
        config_dir: 配置文件目录
        idle_timeout: Agent 空闲超时时间（秒）
        cleanup_interval: 清理检查间隔（秒）
        enable_llm_parsing: 是否启用 LLM 配置解析（默认关闭）

    Returns:
        AgentManager 实例
    """
    global _agent_manager

    if _agent_manager is not None:
        return _agent_manager

    # 创建 AgentManager
    _agent_manager = AgentManager(
        config_dir=config_dir,
        idle_timeout=idle_timeout,
        cleanup_interval=cleanup_interval,
        enable_llm_parsing=enable_llm_parsing,
    )

    # 启动清理任务
    _agent_manager.start_cleanup()

    return _agent_manager


def get_agent_manager() -> AgentManager:
    """
    获取 AgentManager 实例（依赖注入）

    Returns:
        AgentManager 实例

    Raises:
        RuntimeError: 如果 AgentManager 未初始化
    """
    if _agent_manager is None:
        raise RuntimeError(
            "AgentManager not initialized. Call initialize_agent_manager() first."
        )
    return _agent_manager


async def shutdown_agent_manager():
    """关闭 AgentManager"""
    global _agent_manager

    if _agent_manager is not None:
        await _agent_manager.stop_cleanup()
        _agent_manager = None


# =============================================================================
# 类型别名（方便使用）
# =============================================================================

AgentManagerDep = Annotated[AgentManager, Depends(get_agent_manager)]
