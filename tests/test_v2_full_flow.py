"""
V2 Query API 全流程集成测试

测试覆盖：
1. 缓存命中 - 不同 hash 验证配置缓存
2. 会话隔离 - 不同会话独立上下文
3. 数据统计 - 查询计数
4. 会话生命周期 - 创建、销毁

使用真实 LLM 请求
"""

import os
import pytest
import pytest_asyncio
import httpx
from dataclasses import dataclass
from datetime import datetime, timezone
from fastapi import FastAPI
from unittest.mock import AsyncMock, MagicMock

from bu_agent_sdk.schemas import WorkflowConfigSchema


# =============================================================================
# Skip Conditions
# =============================================================================

SKIP_REASON = "No API key (set OPENAI_API_KEY or ANTHROPIC_API_KEY)"
HAS_API_KEY = bool(os.getenv("OPENAI_API_KEY") or os.getenv("ANTHROPIC_API_KEY"))


# =============================================================================
# Test Configurations
# =============================================================================

CONFIG_A = {
    "basic_settings": {
        "name": "Assistant A",
        "description": "Math helper",
        "language": "English",
    },
    "instructions": "You are a math assistant. Answer with numbers only.",
    "max_iterations": 2,
}

CONFIG_B = {
    "basic_settings": {
        "name": "Assistant B",
        "description": "Greeting bot",
        "language": "English",
    },
    "instructions": "You are a greeting bot. Always say hello first.",
    "max_iterations": 2,
}


@dataclass
class MockConfigDoc:
    """Mock config document"""
    chatbot_id: str
    tenant_id: str
    config_hash: str
    raw_config: dict
    parsed_config: dict
    created_at: datetime = None
    updated_at: datetime = None
    access_count: int = 1

    def __post_init__(self):
        now = datetime.now(timezone.utc)
        self.created_at = self.created_at or now
        self.updated_at = self.updated_at or now


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def test_env():
    """
    创建测试环境

    使用 mock repository 注入测试配置
    """
    from api.container import AppContext, AppConfig
    from api.routers.v2.query import create_router
    from api.services.llm_service import LLMService

    AppContext.reset()

    # 创建 mock repository
    mock_repos = MagicMock()
    mock_configs = MagicMock()

    # 配置 mock 返回值
    async def mock_get(chatbot_id, tenant_id, expected_hash=None):
        configs = {
            "hash_a": MockConfigDoc(
                chatbot_id=chatbot_id,
                tenant_id=tenant_id,
                config_hash="hash_a",
                raw_config=CONFIG_A,
                parsed_config=CONFIG_A,
            ),
            "hash_b": MockConfigDoc(
                chatbot_id=chatbot_id,
                tenant_id=tenant_id,
                config_hash="hash_b",
                raw_config=CONFIG_B,
                parsed_config=CONFIG_B,
            ),
        }
        return configs.get(expected_hash)

    async def mock_upsert(*args, **kwargs):
        return None

    mock_configs.get = mock_get
    mock_configs.upsert = mock_upsert
    mock_repos.configs = mock_configs

    # Mock sessions repository
    mock_repos.sessions = MagicMock()
    mock_repos.sessions.get_or_create = AsyncMock(return_value=MagicMock())

    # Mock messages repository
    mock_repos.messages = MagicMock()
    mock_repos.messages.create = AsyncMock()
    mock_repos.messages.get_history = AsyncMock(return_value=[])

    # Mock tool_calls repository
    mock_repos.tool_calls = MagicMock()
    mock_repos.tool_calls.create = AsyncMock()

    # Mock usages repository
    mock_repos.usages = MagicMock()
    mock_repos.usages.create = AsyncMock()

    # 初始化 LLM
    LLMService.initialize()

    # 创建 SessionManager
    from api.services.v2 import SessionManager

    sm = SessionManager(
        repos=mock_repos,
        database=None,
        llm_provider=LLMService.get_instance(),
        idle_timeout=300,
        cleanup_interval=60,
        max_sessions=100,
        enable_llm_parsing=False,
    )
    await sm.start()

    # 注入到 AppContext
    ctx = AppContext()
    ctx.session_manager = sm
    ctx.repository_manager = mock_repos
    AppContext._instance = ctx

    # 创建 FastAPI 应用
    app = FastAPI()
    app.include_router(create_router())

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, sm

    await sm.stop()
    AppContext.reset()


# =============================================================================
# Full Flow Tests
# =============================================================================


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestCacheHit:
    """缓存命中测试"""

    @pytest.mark.asyncio
    async def test_same_hash_reuses_session(self, test_env):
        """相同 hash 复用会话"""
        client, sm = test_env

        # 第一次请求 - 创建会话
        r1 = await client.post("/v2/query", json={
            "message": "2+2",
            "session_id": "s1",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })
        assert r1.status_code == 200

        # 第二次请求 - 复用会话
        r2 = await client.post("/v2/query", json={
            "message": "3+3",
            "session_id": "s1",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })
        assert r2.status_code == 200

        # 验证会话复用
        info = sm.get_session_info("s1")
        assert info["query_count"] == 2
        print(f"\n[Cache Hit] session reused, query_count={info['query_count']}")

    @pytest.mark.asyncio
    async def test_different_hash_recreates_agent(self, test_env):
        """不同 hash 重建 Agent"""
        client, sm = test_env

        # 使用 hash_a 创建会话
        r1 = await client.post("/v2/query", json={
            "message": "hello",
            "session_id": "s2",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })
        assert r1.status_code == 200

        # 获取初始配置名称
        ctx1 = sm._sessions.get("s2")
        name1 = ctx1.agent.config.basic_settings.get("name")

        # 使用 hash_b 更新配置
        r2 = await client.post("/v2/query", json={
            "message": "hello",
            "session_id": "s2",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_b",
        })
        assert r2.status_code == 200

        # 验证 Agent 已重建
        ctx2 = sm._sessions.get("s2")
        name2 = ctx2.agent.config.basic_settings.get("name")

        assert name1 != name2
        print(f"\n[Config Change] {name1} -> {name2}")


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestSessionIsolation:
    """会话隔离测试"""

    @pytest.mark.asyncio
    async def test_sessions_have_independent_context(self, test_env):
        """不同会话有独立上下文"""
        client, sm = test_env

        # Session A: 记住数字 42
        await client.post("/v2/query", json={
            "message": "Remember: 42",
            "session_id": "iso_a",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })

        # Session B: 记住数字 99
        await client.post("/v2/query", json={
            "message": "Remember: 99",
            "session_id": "iso_b",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })

        # Session A: 询问数字
        r_a = await client.post("/v2/query", json={
            "message": "What number?",
            "session_id": "iso_a",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })

        # Session B: 询问数字
        r_b = await client.post("/v2/query", json={
            "message": "What number?",
            "session_id": "iso_b",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })

        # 验证会话隔离
        assert sm.exists("iso_a")
        assert sm.exists("iso_b")

        msg_a = r_a.json()["message"]
        msg_b = r_b.json()["message"]

        print(f"\n[Session A] {msg_a}")
        print(f"[Session B] {msg_b}")


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestStatistics:
    """数据统计测试"""

    @pytest.mark.asyncio
    async def test_query_count_increments(self, test_env):
        """查询计数递增"""
        client, sm = test_env

        for i in range(3):
            await client.post("/v2/query", json={
                "message": f"Query {i}",
                "session_id": "stat_s1",
                "tenant_id": "t1",
                "chatbot_id": "b1",
                "md5_checksum": "hash_a",
            })

        info = sm.get_session_info("stat_s1")
        assert info["query_count"] == 3
        print(f"\n[Stats] query_count={info['query_count']}")

    @pytest.mark.asyncio
    async def test_global_stats(self, test_env):
        """全局统计"""
        client, sm = test_env

        # 创建多个会话
        for i in range(2):
            await client.post("/v2/query", json={
                "message": "hi",
                "session_id": f"stat_g{i}",
                "tenant_id": "t1",
                "chatbot_id": "b1",
                "md5_checksum": "hash_a",
            })

        stats = sm.get_stats()
        assert stats["session_count"] >= 2
        print(f"\n[Global Stats] {stats}")


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestSessionLifecycle:
    """会话生命周期测试"""

    @pytest.mark.asyncio
    async def test_session_destroy(self, test_env):
        """会话销毁"""
        client, sm = test_env

        # 创建会话
        await client.post("/v2/query", json={
            "message": "hi",
            "session_id": "lc_s1",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })
        assert sm.exists("lc_s1")

        # 销毁会话
        r = await client.delete("/v2/sessions/lc_s1")
        assert r.status_code == 200
        assert not sm.exists("lc_s1")
        print("\n[Lifecycle] session destroyed")

    @pytest.mark.asyncio
    async def test_list_sessions(self, test_env):
        """列出会话"""
        client, sm = test_env

        # 创建会话
        await client.post("/v2/query", json={
            "message": "hi",
            "session_id": "lc_list",
            "tenant_id": "t1",
            "chatbot_id": "b1",
            "md5_checksum": "hash_a",
        })

        r = await client.get("/v2/sessions")
        assert r.status_code == 200

        data = r.json()
        assert "sessions" in data
        assert "stats" in data
        print(f"\n[List] {len(data['sessions'])} sessions")


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_session_not_found(self, test_env):
        """会话不存在"""
        client, _ = test_env

        r = await client.get("/v2/sessions/nonexistent_session")
        assert r.status_code == 404

        r = await client.delete("/v2/sessions/nonexistent_session")
        assert r.status_code == 404


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
