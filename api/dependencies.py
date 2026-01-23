"""
依赖注入模块

提供全局依赖项，支持：
- WorkflowAgent 单例
- 配置管理
- 数据库连接（可选）
"""

import json
from pathlib import Path
from typing import Annotated

from fastapi import Depends

from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.config import (
    load_config,
    get_llm_decision_llm,
    get_session_store_from_config,
    get_plan_cache_from_config,
)
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema


# =============================================================================
# 全局状态（单例模式）
# =============================================================================

_workflow_agent: WorkflowAgent | None = None
_app_config = None


async def initialize_agent(
    workflow_config_path: str = "config/workflow_config.json",
) -> WorkflowAgent:
    """
    初始化 WorkflowAgent（单例）

    Args:
        workflow_config_path: workflow 配置文件路径

    Returns:
        WorkflowAgent 实例
    """
    global _workflow_agent, _app_config

    if _workflow_agent is not None:
        return _workflow_agent

    # 1. 加载应用配置
    _app_config = load_config()

    # 2. 加载 workflow 配置
    config_path = Path(workflow_config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Workflow config not found: {workflow_config_path}")

    with open(config_path, encoding="utf-8") as f:
        config_data = json.load(f)
    workflow_config = WorkflowConfigSchema(**config_data)

    # 3. 创建 LLM（使用任务特定模型）
    llm = get_llm_decision_llm(_app_config)

    # 4. 创建存储组件（可选）
    try:
        session_store = await get_session_store_from_config(_app_config)
        plan_cache = await get_plan_cache_from_config(_app_config)
    except Exception as e:
        # 如果数据库未配置，使用内存存储
        print(f"Warning: Database not configured, using in-memory storage: {e}")
        session_store = None
        plan_cache = None

    # 5. 创建 WorkflowAgent
    _workflow_agent = WorkflowAgent(
        config=workflow_config,
        llm=llm,
        session_store=session_store,
        plan_cache=plan_cache,
    )

    return _workflow_agent


def get_workflow_agent() -> WorkflowAgent:
    """
    获取 WorkflowAgent 实例（依赖注入）

    Returns:
        WorkflowAgent 实例

    Raises:
        RuntimeError: 如果 Agent 未初始化
    """
    if _workflow_agent is None:
        raise RuntimeError(
            "WorkflowAgent not initialized. Call initialize_agent() first."
        )
    return _workflow_agent


def get_app_config():
    """
    获取应用配置（依赖注入）

    Returns:
        AppConfig 实例
    """
    if _app_config is None:
        return load_config()
    return _app_config


# =============================================================================
# 类型别名（方便使用）
# =============================================================================

WorkflowAgentDep = Annotated[WorkflowAgent, Depends(get_workflow_agent)]
AppConfigDep = Annotated[object, Depends(get_app_config)]
