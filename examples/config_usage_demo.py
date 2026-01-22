"""
Workflow Agent 配置使用示例

演示如何使用配置管理系统
"""

import asyncio
import json
from pathlib import Path

from bu_agent_sdk.config import (
    load_config,
    get_llm_from_config,
    get_session_store_from_config,
    get_plan_cache_from_config,
)
from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema


async def main():
    """主函数"""

    # =========================================================================
    # 方式1：使用配置管理系统（推荐）
    # =========================================================================

    print("=" * 60)
    print("方式1：使用配置管理系统")
    print("=" * 60)

    # 1. 加载配置（自动从 .env 文件读取）
    config = load_config()

    print(f"Environment: {config.environment}")
    print(f"Log Level: {config.log_level}")
    print(f"MongoDB URI: {config.database.mongodb_uri}")
    print(f"Redis URL: {config.database.redis_url}")
    print(f"Default Model: {config.llm.default_model}")

    # 2. 创建LLM
    llm = get_llm_from_config(config)
    print(f"LLM created: {llm.__class__.__name__}")

    # 3. 创建存储（可选）
    session_store = await get_session_store_from_config(config)
    plan_cache = await get_plan_cache_from_config(config)
    print(f"Session Store: {session_store.__class__.__name__}")
    print(f"Plan Cache: {plan_cache.__class__.__name__}")

    # 4. 加载Workflow配置
    workflow_config_path = Path("config/workflow_config.json")
    if workflow_config_path.exists():
        with open(workflow_config_path, encoding="utf-8") as f:
            workflow_config_data = json.load(f)

        workflow_config = WorkflowConfigSchema(**workflow_config_data)

        # 5. 创建WorkflowAgent
        agent = WorkflowAgent(
            config=workflow_config,
            llm=llm,
            session_store=session_store,
            plan_cache=plan_cache,
        )

        print("\n✅ WorkflowAgent 创建成功！")

        # 6. 测试查询
        response = await agent.query(
            message="你好",
            session_id="demo_session_001"
        )
        print(f"\n🤖 Agent Response: {response}")

    # =========================================================================
    # 方式2：手动配置（灵活但繁琐）
    # =========================================================================

    print("\n" + "=" * 60)
    print("方式2：手动配置")
    print("=" * 60)

    from bu_agent_sdk.llm import ChatOpenAI
    from motor.motor_asyncio import AsyncIOMotorClient
    from bu_agent_sdk.workflow.storage import MongoDBSessionStore
    import redis.asyncio as redis
    from bu_agent_sdk.workflow.storage import RedisPlanCache

    # 手动创建LLM
    llm_manual = ChatOpenAI(
        model="gpt-4o",
        api_key="sk-xxx",  # 不推荐硬编码
    )

    # 手动创建存储
    mongo_client = AsyncIOMotorClient("mongodb://localhost:27017")
    session_store_manual = MongoDBSessionStore(mongo_client)

    redis_client = redis.from_url("redis://localhost:6379")
    plan_cache_manual = RedisPlanCache(redis_client, ttl=3600)

    print("✅ 手动配置完成")

    # =========================================================================
    # 方式3：混合配置（部分使用配置系统）
    # =========================================================================

    print("\n" + "=" * 60)
    print("方式3：混合配置")
    print("=" * 60)

    # 从配置加载基础设置
    config = load_config()

    # 但使用自定义LLM
    from bu_agent_sdk.llm import ChatAnthropic

    custom_llm = ChatAnthropic(
        model="claude-3-5-sonnet-20241022",
        api_key=config.llm.anthropic_api_key,
    )

    # 使用配置系统的存储
    session_store = await get_session_store_from_config(config)

    print("✅ 混合配置完成")


async def production_example():
    """生产环境示例"""

    print("\n" + "=" * 60)
    print("生产环境配置示例")
    print("=" * 60)

    # 1. 加载配置
    config = load_config()

    # 2. 验证必要的配置
    if not config.llm.openai_api_key:
        raise ValueError("生产环境必须设置 OPENAI_API_KEY")

    if config.environment != "production":
        print("⚠️  警告：当前不是生产环境")

    # 3. 创建组件
    llm = get_llm_from_config(config)
    session_store = await get_session_store_from_config(config)
    plan_cache = await get_plan_cache_from_config(config)

    # 4. 加载workflow配置
    workflow_config_path = Path("config/workflow_config.json")
    with open(workflow_config_path, encoding="utf-8") as f:
        workflow_config_data = json.load(f)

    workflow_config = WorkflowConfigSchema(**workflow_config_data)

    # 5. 创建agent
    agent = WorkflowAgent(
        config=workflow_config,
        llm=llm,
        session_store=session_store,
        plan_cache=plan_cache,
    )

    print("✅ 生产环境 WorkflowAgent 已就绪")

    return agent


async def docker_example():
    """Docker环境示例"""

    print("\n" + "=" * 60)
    print("Docker环境配置示例")
    print("=" * 60)

    # Docker环境通常通过环境变量传递配置
    # 不需要 .env 文件

    import os

    # 验证环境变量
    required_vars = [
        "OPENAI_API_KEY",
        "MONGODB_URI",
        "REDIS_URL",
    ]

    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        raise ValueError(f"缺少必要的环境变量: {missing_vars}")

    # 加载配置（会自动读取环境变量）
    config = load_config()

    print(f"MongoDB URI: {config.database.mongodb_uri}")
    print(f"Redis URL: {config.database.redis_url}")

    # 创建组件
    llm = get_llm_from_config(config)
    session_store = await get_session_store_from_config(config)
    plan_cache = await get_plan_cache_from_config(config)

    print("✅ Docker环境配置完成")


if __name__ == "__main__":
    # 运行示例
    asyncio.run(main())

    # 生产环境示例（需要有效的配置）
    # asyncio.run(production_example())

    # Docker环境示例
    # asyncio.run(docker_example())
