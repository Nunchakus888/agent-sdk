"""
依赖注入模块

提供全局依赖项，支持：
- WorkflowEngine 单例（统一的工作流引擎）
- AgentManager 单例（Agent 生命周期管理）
- TaskManager 单例（协程任务取消机制）
- 配置管理
- Agent 生命周期管理
"""

from typing import Annotated, Any

from fastapi import Depends

from bu_agent_sdk.workflow.engine import WorkflowEngine
from api.services import AgentManager
from api.services.task_manager import TaskManager
from api.services.database import DB_NAME


# =============================================================================
# 全局状态（单例模式）
# =============================================================================

_workflow_engine: WorkflowEngine | None = None
_mongo_client: Any | None = None
_agent_manager: AgentManager | None = None
_task_manager: TaskManager | None = None


async def initialize_workflow_engine(
    mongodb_uri: str | None = None,
    db_name: str = DB_NAME,
    config_dir: str = "config",
    max_agents: int = 100,
    agent_ttl: int = 3600,
    idle_timeout: int = 300,
    cleanup_interval: int = 60,
) -> WorkflowEngine:
    """
    初始化 WorkflowEngine（单例）

    Args:
        mongodb_uri: MongoDB URI（可选，启用持久化模式）
        db_name: 数据库名称
        config_dir: 配置文件目录
        max_agents: 最大缓存 Agent 数量
        agent_ttl: Agent TTL（秒）
        idle_timeout: 空闲超时（秒）
        cleanup_interval: 清理间隔（秒）

    Returns:
        WorkflowEngine 实例
    """
    global _workflow_engine, _mongo_client

    if _workflow_engine is not None:
        return _workflow_engine

    # LLM 工厂
    from bu_agent_sdk.llm.anthropic.chat import ChatAnthropic
    llm_factory = lambda: ChatAnthropic()

    # MongoDB 客户端（可选）
    mongo_client = None
    if mongodb_uri:
        from motor.motor_asyncio import AsyncIOMotorClient
        _mongo_client = AsyncIOMotorClient(mongodb_uri)
        await _mongo_client.admin.command("ping")
        mongo_client = _mongo_client

    # 创建 WorkflowEngine
    _workflow_engine = WorkflowEngine(
        llm_factory=llm_factory,
        mongo_client=mongo_client,
        db_name=db_name,
        config_dir=config_dir,
        max_agents=max_agents,
        agent_ttl=agent_ttl,
        idle_timeout=idle_timeout,
        cleanup_interval=cleanup_interval,
    )

    await _workflow_engine.init()
    return _workflow_engine


def get_workflow_engine() -> WorkflowEngine:
    """
    获取 WorkflowEngine 实例（依赖注入）

    Returns:
        WorkflowEngine 实例

    Raises:
        RuntimeError: 如果 WorkflowEngine 未初始化
    """
    if _workflow_engine is None:
        raise RuntimeError(
            "WorkflowEngine not initialized. Call initialize_workflow_engine() first."
        )
    return _workflow_engine


async def shutdown_workflow_engine() -> None:
    """关闭 WorkflowEngine"""
    global _workflow_engine, _mongo_client

    if _workflow_engine is not None:
        await _workflow_engine.shutdown()
        _workflow_engine = None

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None


# =============================================================================
# AgentManager 依赖注入
# =============================================================================


def get_mongo_db(db_name: str = DB_NAME) -> Any | None:
    """
    获取 MongoDB 数据库实例（复用全局连接）
    
    Args:
        db_name: 数据库名称
    
    Returns:
        MongoDB 数据库实例，未启用时返回 None
    """
    if _mongo_client is not None:
        return _mongo_client[db_name]
    return None


def initialize_agent_manager(
    config_dir: str = "config",
    idle_timeout: int = 300,
    cleanup_interval: int = 60,
    enable_llm_parsing: bool = False,
    db_name: str = DB_NAME,
) -> AgentManager:
    """
    初始化 AgentManager（单例）
    
    自动复用 WorkflowEngine 的 MongoDB 连接（如已启用）
    
    Args:
        config_dir: 配置文件目录
        idle_timeout: 空闲超时（秒）
        cleanup_interval: 清理间隔（秒）
        enable_llm_parsing: 是否启用 LLM 配置解析
        db_name: MongoDB 数据库名称
    
    Returns:
        AgentManager 实例
    """
    global _agent_manager

    if _agent_manager is not None:
        return _agent_manager

    # 复用全局 MongoDB 连接
    mongo_db = get_mongo_db(db_name)

    _agent_manager = AgentManager(
        config_dir=config_dir,
        idle_timeout=idle_timeout,
        cleanup_interval=cleanup_interval,
        enable_llm_parsing=enable_llm_parsing,
        mongo_db=mongo_db,
    )

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


async def shutdown_agent_manager() -> None:
    """关闭 AgentManager"""
    global _agent_manager

    if _agent_manager is not None:
        await _agent_manager.stop_cleanup()
        _agent_manager = None


# =============================================================================
# 类型别名（方便使用）
# =============================================================================

WorkflowEngineDep = Annotated[WorkflowEngine, Depends(get_workflow_engine)]
AgentManagerDep = Annotated[AgentManager, Depends(get_agent_manager)]


# =============================================================================
# TaskManager 依赖注入（协程任务取消机制）
# =============================================================================


def initialize_task_manager() -> TaskManager:
    """
    初始化 TaskManager（单例）

    Returns:
        TaskManager 实例
    """
    global _task_manager

    if _task_manager is not None:
        return _task_manager

    _task_manager = TaskManager()
    return _task_manager


def get_task_manager() -> TaskManager:
    """
    获取 TaskManager 实例（依赖注入）

    Returns:
        TaskManager 实例

    Raises:
        RuntimeError: 如果 TaskManager 未初始化
    """
    if _task_manager is None:
        raise RuntimeError(
            "TaskManager not initialized. Call initialize_task_manager() first."
        )
    return _task_manager


async def shutdown_task_manager() -> None:
    """关闭 TaskManager"""
    global _task_manager

    if _task_manager is not None:
        await _task_manager.shutdown()
        _task_manager = None


TaskManagerDep = Annotated[TaskManager, Depends(get_task_manager)]
