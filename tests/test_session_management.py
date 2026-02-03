"""
V2 Query API Session 管理测试

测试场景：
1. 不同 session_id 创建独立会话
2. 相同 session_id 复用会话
3. 相同 session_id 但不同 config_hash 重建会话
4. 不同 chatbot_id 创建独立会话
"""

import asyncio
import os
import pytest
import pytest_asyncio
import logging
import httpx
from unittest.mock import AsyncMock, MagicMock, patch

from fastapi import FastAPI

from bu_agent_sdk.tools.actions import WorkflowConfigSchema
from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2

from api.services.v2 import SessionManager, SessionContext

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# =============================================================================
# Skip if no API key
# =============================================================================

SKIP_REASON = "No API key configured (set OPENAI_API_KEY or ANTHROPIC_API_KEY)"
HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_config() -> WorkflowConfigSchema:
    """创建示例配置"""
    return WorkflowConfigSchema(
        basic_settings={
            "name": "Test Assistant",
            "description": "A test assistant for session testing",
            "language": "English",
            "tone": "professional",
        },
        instructions="You are a helpful assistant. Answer questions concisely.",
        max_iterations=3,
    )


@pytest.fixture
def sample_config_v2() -> WorkflowConfigSchema:
    """创建第二个示例配置（用于测试配置更新）"""
    return WorkflowConfigSchema(
        basic_settings={
            "name": "Updated Assistant",
            "description": "An updated assistant",
            "language": "English",
            "tone": "friendly",
        },
        instructions="You are a friendly assistant. Be helpful and warm.",
        max_iterations=5,
    )


@pytest_asyncio.fixture
async def app_client(sample_config, sample_config_v2):
    """
    创建测试环境

    Mock 配置加载，使用真实 SessionManager
    """
    from api.container import AppContext, AppConfig
    from api.routers.v2.query import create_router

    # 重置 AppContext 单例
    AppContext.reset()

    # 使用内存模式配置
    config = AppConfig(
        mongodb_uri=None,  # 内存模式
        idle_timeout=1800,
        cleanup_interval=60,
        max_sessions=10000,
        enable_llm_parsing=False,
    )

    # 初始化 AppContext
    ctx = await AppContext.create(config)

    # Mock _load_config 方法，根据 config_hash 返回不同配置
    original_load_config = ctx.session_manager._load_config

    async def mock_load_config(config_hash: str, tenant_id: str, chatbot_id: str):
        if config_hash == "config_hash_v1":
            return sample_config
        elif config_hash == "config_hash_v2":
            return sample_config_v2
        else:
            # 默认返回 sample_config
            return sample_config

    ctx.session_manager._load_config = mock_load_config

    # 启动 SessionManager
    await ctx.session_manager.start()

    # 创建 FastAPI 应用
    app = FastAPI()
    router = create_router()
    app.include_router(router)

    # 使用 ASGITransport
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, ctx.session_manager

    # 清理
    await ctx.shutdown()


# =============================================================================
# Session 管理测试
# =============================================================================


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestSessionManagement:
    """Session 管理测试"""

    @pytest.mark.asyncio
    async def test_same_session_reuse(self, app_client):
        """测试相同 session_id 复用会话"""
        client, session_manager = app_client
        session_id = "sess_reuse_test"
        config_hash = "config_hash_v1"

        # 第一次请求
        response1 = await client.post(
            "/v2/query",
            json={
                "message": "Hello, my name is Alice",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": config_hash,
            },
        )
        assert response1.status_code == 200

        # 获取会话信息
        info1 = session_manager.get_session_info(session_id)
        assert info1 is not None
        assert info1["query_count"] == 1
        assert info1["config_hash"] == config_hash
        created_at_1 = info1["created_at"]

        print(f"\n[Request 1] Session created: {session_id}")
        print(f"  config_hash: {info1['config_hash']}")
        print(f"  query_count: {info1['query_count']}")

        # 第二次请求（相同 session_id 和 config_hash）
        response2 = await client.post(
            "/v2/query",
            json={
                "message": "What is my name?",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": config_hash,
            },
        )
        assert response2.status_code == 200

        # 验证会话被复用
        info2 = session_manager.get_session_info(session_id)
        assert info2 is not None
        assert info2["query_count"] == 2  # 查询计数增加
        assert info2["created_at"] == created_at_1  # 创建时间不变
        assert info2["config_hash"] == config_hash

        print(f"\n[Request 2] Session reused: {session_id}")
        print(f"  config_hash: {info2['config_hash']}")
        print(f"  query_count: {info2['query_count']}")
        print(f"  created_at unchanged: {info2['created_at'] == created_at_1}")

    @pytest.mark.asyncio
    async def test_different_session_independent(self, app_client):
        """测试不同 session_id 创建独立会话"""
        client, session_manager = app_client
        config_hash = "config_hash_v1"

        # 创建第一个会话
        response1 = await client.post(
            "/v2/query",
            json={
                "message": "I am session A",
                "session_id": "sess_A",
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": config_hash,
            },
        )
        assert response1.status_code == 200

        # 创建第二个会话
        response2 = await client.post(
            "/v2/query",
            json={
                "message": "I am session B",
                "session_id": "sess_B",
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": config_hash,
            },
        )
        assert response2.status_code == 200

        # 验证两个会话独立存在
        info_a = session_manager.get_session_info("sess_A")
        info_b = session_manager.get_session_info("sess_B")

        assert info_a is not None
        assert info_b is not None
        assert info_a["session_id"] != info_b["session_id"]
        assert info_a["query_count"] == 1
        assert info_b["query_count"] == 1

        print(f"\n[Independent Sessions]")
        print(f"  Session A: {info_a['session_id']}, query_count={info_a['query_count']}")
        print(f"  Session B: {info_b['session_id']}, query_count={info_b['query_count']}")
        print(f"  Total sessions: {session_manager.session_count}")

    @pytest.mark.asyncio
    async def test_config_hash_change_recreates_session(self, app_client):
        """测试 config_hash 变化时重建会话"""
        client, session_manager = app_client
        session_id = "sess_config_change"

        # 第一次请求（config_hash_v1）
        response1 = await client.post(
            "/v2/query",
            json={
                "message": "Hello with config v1",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "config_hash_v1",
            },
        )
        assert response1.status_code == 200

        info1 = session_manager.get_session_info(session_id)
        assert info1 is not None
        assert info1["config_hash"] == "config_hash_v1"
        created_at_1 = info1["created_at"]

        print(f"\n[Request 1] config_hash=config_hash_v1")
        print(f"  created_at: {created_at_1}")
        print(f"  query_count: {info1['query_count']}")

        # 等待一小段时间，确保时间戳不同
        await asyncio.sleep(0.1)

        # 第二次请求（config_hash_v2 - 配置变化）
        response2 = await client.post(
            "/v2/query",
            json={
                "message": "Hello with config v2",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "config_hash_v2",
            },
        )
        assert response2.status_code == 200

        info2 = session_manager.get_session_info(session_id)
        assert info2 is not None
        assert info2["config_hash"] == "config_hash_v2"  # 配置哈希已更新
        created_at_2 = info2["created_at"]

        print(f"\n[Request 2] config_hash=config_hash_v2")
        print(f"  created_at: {created_at_2}")
        print(f"  query_count: {info2['query_count']}")
        print(f"  Session recreated: {created_at_2 != created_at_1}")

        # 验证会话被重建（创建时间不同）
        assert created_at_2 != created_at_1, "Session should be recreated when config_hash changes"
        # 验证查询计数重置
        assert info2["query_count"] == 1, "Query count should reset after session recreation"

    @pytest.mark.asyncio
    async def test_different_chatbot_independent(self, app_client):
        """测试不同 chatbot_id 创建独立会话"""
        client, session_manager = app_client
        session_id = "sess_chatbot_test"
        config_hash = "config_hash_v1"

        # 第一个 chatbot
        response1 = await client.post(
            "/v2/query",
            json={
                "message": "Hello from bot_1",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": config_hash,
            },
        )
        assert response1.status_code == 200

        info1 = session_manager.get_session_info(session_id)
        assert info1["chatbot_id"] == "bot_1"

        print(f"\n[Chatbot 1] chatbot_id=bot_1")
        print(f"  session_id: {session_id}")

        # 注意：当前设计中，session_id 是唯一标识
        # 如果需要按 chatbot_id 隔离，需要在 session_id 中包含 chatbot_id
        # 这里验证当前行为

    @pytest.mark.asyncio
    async def test_session_stats(self, app_client):
        """测试会话统计"""
        client, session_manager = app_client
        config_hash = "config_hash_v1"

        # 创建多个会话
        for i in range(3):
            await client.post(
                "/v2/query",
                json={
                    "message": f"Hello from session {i}",
                    "session_id": f"sess_stats_{i}",
                    "tenant_id": "tenant_1",
                    "chatbot_id": "bot_1",
                    "md5_checksum": config_hash,
                },
            )

        # 获取统计
        stats = session_manager.get_stats()

        print(f"\n[Session Stats]")
        print(f"  session_count: {stats['session_count']}")
        print(f"  max_sessions: {stats['max_sessions']}")
        print(f"  idle_timeout: {stats['idle_timeout']}")

        assert stats["session_count"] >= 3

    @pytest.mark.asyncio
    async def test_list_sessions_endpoint(self, app_client):
        """测试列出会话端点"""
        client, session_manager = app_client
        config_hash = "config_hash_v1"

        # 创建会话
        await client.post(
            "/v2/query",
            json={
                "message": "Hello",
                "session_id": "sess_list_test",
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": config_hash,
            },
        )

        # 调用列出会话端点
        response = await client.get("/v2/sessions")
        assert response.status_code == 200

        data = response.json()
        print(f"\n[List Sessions Endpoint]")
        print(f"  sessions: {len(data['sessions'])}")
        print(f"  stats: {data['stats']}")

        # 验证返回的会话包含 config_hash
        for session in data["sessions"]:
            assert "config_hash" in session
            print(f"  - {session['session_id']}: config_hash={session['config_hash']}")


# =============================================================================
# SessionManager 单元测试（不需要 API Key）
# =============================================================================


class TestSessionManagerUnit:
    """SessionManager 单元测试"""

    @pytest.fixture
    def mock_repos(self):
        """创建 Mock RepositoryManager"""
        repos = MagicMock()
        repos.sessions = MagicMock()
        repos.sessions.get_or_create = AsyncMock()
        repos.messages = MagicMock()
        repos.messages.list_by_session = AsyncMock(return_value=[])
        return repos

    @pytest.fixture
    def mock_llm(self):
        """创建 Mock LLM"""
        llm = MagicMock()
        llm.ainvoke = AsyncMock(return_value=MagicMock(content="Hello"))
        return llm

    @pytest_asyncio.fixture
    async def session_manager(self, mock_repos, mock_llm, sample_config):
        """创建 SessionManager 实例"""
        manager = SessionManager(
            repos=mock_repos,
            database=None,
            llm_provider=mock_llm,
            idle_timeout=1800,
            cleanup_interval=60,
            max_sessions=100,
            enable_llm_parsing=False,
        )

        # Mock _load_config
        async def mock_load_config(config_hash, tenant_id, chatbot_id):
            return sample_config

        manager._load_config = mock_load_config

        await manager.start()
        yield manager
        await manager.stop()

    @pytest.mark.asyncio
    async def test_get_or_create_new_session(self, session_manager):
        """测试创建新会话"""
        ctx = await session_manager.get_or_create(
            session_id="new_sess_1",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash_1",
        )

        assert ctx is not None
        assert ctx.session_id == "new_sess_1"
        assert ctx.config_hash == "hash_1"
        assert session_manager.exists("new_sess_1")

        print(f"\n[New Session Created]")
        print(f"  session_id: {ctx.session_id}")
        print(f"  config_hash: {ctx.config_hash}")

    @pytest.mark.asyncio
    async def test_get_or_create_existing_session(self, session_manager):
        """测试获取现有会话"""
        # 创建会话
        ctx1 = await session_manager.get_or_create(
            session_id="existing_sess",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash_1",
        )
        created_at_1 = ctx1.created_at

        # 再次获取（相同 config_hash）
        ctx2 = await session_manager.get_or_create(
            session_id="existing_sess",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash_1",
        )

        # 应该是同一个会话
        assert ctx2.created_at == created_at_1
        assert ctx2.config_hash == "hash_1"

        print(f"\n[Existing Session Reused]")
        print(f"  same created_at: {ctx2.created_at == created_at_1}")

    @pytest.mark.asyncio
    async def test_get_or_create_config_change(self, session_manager, sample_config_v2):
        """测试配置变化时重建会话"""
        # 创建会话
        ctx1 = await session_manager.get_or_create(
            session_id="config_change_sess",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash_1",
        )
        created_at_1 = ctx1.created_at

        # 等待一小段时间
        await asyncio.sleep(0.01)

        # Mock 返回不同配置
        async def mock_load_config_v2(config_hash, tenant_id, chatbot_id):
            return sample_config_v2

        session_manager._load_config = mock_load_config_v2

        # 使用不同 config_hash 获取
        ctx2 = await session_manager.get_or_create(
            session_id="config_change_sess",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash_2",  # 不同的 hash
        )

        # 应该是新会话
        assert ctx2.config_hash == "hash_2"
        assert ctx2.created_at != created_at_1

        print(f"\n[Session Recreated on Config Change]")
        print(f"  old config_hash: hash_1")
        print(f"  new config_hash: {ctx2.config_hash}")
        print(f"  created_at changed: {ctx2.created_at != created_at_1}")

    @pytest.mark.asyncio
    async def test_destroy_session(self, session_manager):
        """测试销毁会话"""
        # 创建会话
        await session_manager.get_or_create(
            session_id="destroy_sess",
            tenant_id="tenant_1",
            chatbot_id="bot_1",
            config_hash="hash_1",
        )

        assert session_manager.exists("destroy_sess")

        # 销毁会话
        await session_manager.destroy("destroy_sess")

        assert not session_manager.exists("destroy_sess")

        print(f"\n[Session Destroyed]")
        print(f"  exists after destroy: {session_manager.exists('destroy_sess')}")

    @pytest.mark.asyncio
    async def test_session_count(self, session_manager):
        """测试会话计数"""
        initial_count = session_manager.session_count

        # 创建多个会话
        for i in range(5):
            await session_manager.get_or_create(
                session_id=f"count_sess_{i}",
                tenant_id="tenant_1",
                chatbot_id="bot_1",
                config_hash="hash_1",
            )

        assert session_manager.session_count == initial_count + 5

        print(f"\n[Session Count]")
        print(f"  initial: {initial_count}")
        print(f"  after creating 5: {session_manager.session_count}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
