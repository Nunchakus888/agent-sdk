"""
配置管理模块

支持从环境变量、配置文件加载配置
"""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic import BaseModel, Field


class DatabaseConfig(BaseModel):
    """数据库配置"""

    # MongoDB
    mongodb_uri: str = Field(default="mongodb://localhost:27017")
    mongodb_db_name: str = Field(default="workflow_agent")

    # PostgreSQL
    postgresql_uri: str | None = Field(default=None)

    # Redis
    redis_url: str = Field(default="redis://localhost:6379")
    redis_ttl: int = Field(default=3600)  # 1小时


class LLMConfig(BaseModel):
    """LLM配置"""

    openai_api_key: str | None = Field(default=None)
    openai_base_url: str | None = Field(default=None)

    anthropic_api_key: str | None = Field(default=None)

    google_api_key: str | None = Field(default=None)

    default_model: str = Field(default="gpt-4o")

    # 任务特定模型配置
    intent_matching_model: str | None = Field(default=None, description="意图识别模型（快速、低成本）")
    content_generation_model: str | None = Field(default=None, description="内容生成模型（高质量）")
    workflow_planning_model: str | None = Field(default=None, description="工作流规划模型（复杂逻辑理解）")
    llm_decision_model: str | None = Field(default=None, description="LLM决策模型（SOP驱动迭代）")
    response_generation_model: str | None = Field(default=None, description="响应生成模型（最终用户响应）")


class AppConfig(BaseModel):
    """应用配置"""

    environment: str = Field(default="development")
    log_level: str = Field(default="INFO")

    # Workflow Agent Settings
    max_iterations: int = Field(default=5)
    iteration_strategy: str = Field(default="sop_driven")

    # Database
    database: DatabaseConfig = Field(default_factory=DatabaseConfig)

    # LLM
    llm: LLMConfig = Field(default_factory=LLMConfig)


def load_config(env_file: str | Path | None = None) -> AppConfig:
    """
    加载配置

    优先级：
    1. 环境变量
    2. .env 文件
    3. 默认值

    Args:
        env_file: .env 文件路径，默认为项目根目录的 .env

    Returns:
        AppConfig: 应用配置对象

    Example:
        ```python
        from bu_agent_sdk.config import load_config

        config = load_config()
        print(config.llm.openai_api_key)
        print(config.database.mongodb_uri)
        ```
    """
    # 加载 .env 文件
    if env_file is None:
        # 默认查找项目根目录的 .env
        env_file = Path.cwd() / ".env"

    if Path(env_file).exists():
        load_dotenv(env_file)

    # 从环境变量构建配置
    config = AppConfig(
        environment=os.getenv("ENVIRONMENT", "development"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        max_iterations=int(os.getenv("MAX_ITERATIONS", "5")),
        iteration_strategy=os.getenv("ITERATION_STRATEGY", "sop_driven"),

        database=DatabaseConfig(
            mongodb_uri=os.getenv("MONGODB_URI", "mongodb://localhost:27017"),
            mongodb_db_name=os.getenv("MONGODB_DB_NAME", "workflow_agent"),
            postgresql_uri=os.getenv("POSTGRESQL_URI"),
            redis_url=os.getenv("REDIS_URL", "redis://localhost:6379"),
            redis_ttl=int(os.getenv("REDIS_TTL", "3600")),
        ),

        llm=LLMConfig(
            openai_api_key=os.getenv("OPENAI_API_KEY"),
            openai_base_url=os.getenv("OPENAI_BASE_URL"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            default_model=os.getenv("DEFAULT_MODEL", "gpt-4o"),
            # 任务特定模型
            intent_matching_model=os.getenv("INTENT_MATCHING_MODEL"),
            content_generation_model=os.getenv("CONTENT_GENERATION_MODEL"),
            workflow_planning_model=os.getenv("WORKFLOW_PLANNING_MODEL"),
            llm_decision_model=os.getenv("LLM_DECISION_MODEL"),
            response_generation_model=os.getenv("RESPONSE_GENERATION_MODEL"),
        ),
    )

    return config


def get_llm_from_config(config: AppConfig, model_override: str | None = None):
    """
    根据配置创建LLM实例

    Args:
        config: 应用配置
        model_override: 可选的模型覆盖（用于任务特定模型）

    Returns:
        BaseChatModel: LLM实例

    Example:
        ```python
        from bu_agent_sdk.config import load_config, get_llm_from_config

        config = load_config()
        llm = get_llm_from_config(config)

        # 使用特定模型
        intent_llm = get_llm_from_config(config, config.llm.intent_matching_model)
        ```
    """
    from bu_agent_sdk.llm import ChatOpenAI, ChatAnthropic, ChatGoogleGenAI

    model = model_override or config.llm.default_model

    # 根据模型名称选择LLM
    if model.startswith("gpt"):
        if not config.llm.openai_api_key:
            raise ValueError("OPENAI_API_KEY not set")

        return ChatOpenAI(
            model=model,
            api_key=config.llm.openai_api_key,
            base_url=config.llm.openai_base_url,
        )

    elif model.startswith("claude"):
        if not config.llm.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        return ChatAnthropic(
            model=model,
            api_key=config.llm.anthropic_api_key,
        )

    elif model.startswith("gemini"):
        if not config.llm.google_api_key:
            raise ValueError("GOOGLE_API_KEY not set")

        return ChatGoogleGenAI(
            model=model,
            api_key=config.llm.google_api_key,
        )

    else:
        raise ValueError(f"Unsupported model: {model}")


async def get_session_store_from_config(config: AppConfig):
    """
    根据配置创建会话存储

    Args:
        config: 应用配置

    Returns:
        SessionStore: 会话存储实例

    Example:
        ```python
        from bu_agent_sdk.config import load_config, get_session_store_from_config

        config = load_config()
        session_store = await get_session_store_from_config(config)
        ```
    """
    # 优先使用PostgreSQL，其次MongoDB
    if config.database.postgresql_uri:
        import asyncpg
        from bu_agent_sdk.workflow.storage import PostgreSQLSessionStore

        pool = await asyncpg.create_pool(config.database.postgresql_uri)
        store = PostgreSQLSessionStore(pool)
        await store.init_schema()
        return store

    else:
        from motor.motor_asyncio import AsyncIOMotorClient
        from bu_agent_sdk.workflow.storage import MongoDBSessionStore

        client = AsyncIOMotorClient(config.database.mongodb_uri)
        return MongoDBSessionStore(client, db_name=config.database.mongodb_db_name)


async def get_plan_cache_from_config(config: AppConfig):
    """
    根据配置创建计划缓存

    Args:
        config: 应用配置

    Returns:
        PlanCache: 计划缓存实例

    Example:
        ```python
        from bu_agent_sdk.config import load_config, get_plan_cache_from_config

        config = load_config()
        plan_cache = await get_plan_cache_from_config(config)
        ```
    """
    import redis.asyncio as redis
    from bu_agent_sdk.workflow.storage import RedisPlanCache

    client = redis.from_url(config.database.redis_url)
    return RedisPlanCache(client, ttl=config.database.redis_ttl)


# =============================================================================
# 任务特定LLM辅助函数
# =============================================================================


def get_intent_matching_llm(config: AppConfig):
    """
    获取意图识别LLM（快速、低成本）

    Args:
        config: 应用配置

    Returns:
        BaseChatModel: 意图识别LLM实例

    Example:
        ```python
        config = load_config()
        intent_llm = get_intent_matching_llm(config)
        ```
    """
    model = config.llm.intent_matching_model or config.llm.default_model
    return get_llm_from_config(config, model)


def get_content_generation_llm(config: AppConfig):
    """
    获取内容生成LLM（高质量）

    Args:
        config: 应用配置

    Returns:
        BaseChatModel: 内容生成LLM实例

    Example:
        ```python
        config = load_config()
        content_llm = get_content_generation_llm(config)
        ```
    """
    model = config.llm.content_generation_model or config.llm.default_model
    return get_llm_from_config(config, model)


def get_workflow_planning_llm(config: AppConfig):
    """
    获取工作流规划LLM（复杂逻辑理解）

    Args:
        config: 应用配置

    Returns:
        BaseChatModel: 工作流规划LLM实例

    Example:
        ```python
        config = load_config()
        planning_llm = get_workflow_planning_llm(config)
        ```
    """
    model = config.llm.workflow_planning_model or config.llm.default_model
    return get_llm_from_config(config, model)


def get_llm_decision_llm(config: AppConfig):
    """
    获取LLM决策模型（SOP驱动迭代）

    Args:
        config: 应用配置

    Returns:
        BaseChatModel: LLM决策模型实例

    Example:
        ```python
        config = load_config()
        decision_llm = get_llm_decision_llm(config)
        ```
    """
    model = config.llm.llm_decision_model or config.llm.default_model
    return get_llm_from_config(config, model)


def get_response_generation_llm(config: AppConfig):
    """
    获取响应生成LLM（最终用户响应）

    Args:
        config: 应用配置

    Returns:
        BaseChatModel: 响应生成LLM实例

    Example:
        ```python
        config = load_config()
        response_llm = get_response_generation_llm(config)
        ```
    """
    model = config.llm.response_generation_model or config.llm.default_model
    return get_llm_from_config(config, model)

