"""
V2 Query API 集成测试

使用真实请求测试 V2 Query API
需要设置环境变量：OPENAI_API_KEY 或 ANTHROPIC_API_KEY
"""

import asyncio
import os
import pytest
import pytest_asyncio
import logging
import httpx
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI

from bu_agent_sdk.tools.actions import WorkflowConfigSchema
from bu_agent_sdk.agent.workflow_agent_v2 import WorkflowAgentV2
from bu_agent_sdk.agent.events import (
    ToolCallEvent,
    ToolResultEvent,
    FinalResponseEvent,
    TextEvent,
)

from api.services.v2 import (
    SessionManager,
    SessionContext,
    EventCollector,
    QueryRecorder,
    QueryResult,
    ToolCallRecord,
)
# ConfigCache 不在 __all__ 中，需要直接导入
from api.services.v2.config_cache import ConfigCache

# 配置日志以查看 prompt 上下文
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("agent_sdk.workflow_agent")
logger.setLevel(logging.DEBUG)


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
            "description": "A test assistant for unit testing",
            "language": "English",
            "tone": "professional",
        },
        instructions="You are a helpful assistant. Answer questions concisely in one sentence.",
        max_iterations=3,
    )


@pytest.fixture
def real_llm():
    """获取真实 LLM（使用环境变量配置）"""
    from api.services.llm_service import LLMService, LLMConfig

    # 从环境变量加载配置
    config = LLMConfig.from_env()
    LLMService.initialize(config)
    return LLMService.get_instance().get_decision_llm()


@pytest_asyncio.fixture
async def app_client(sample_config):
    """
    创建真实测试环境
    
    使用真实服务（MongoDB 模式）进行集成测试
    
    返回 (AsyncClient, SessionManager, ConfigLoader)
    """
    from api.container import AppContext, AppConfig
    from api.routers.v2.query import create_router
    
    # 重置 AppContext 单例（确保测试隔离）
    AppContext.reset()
    
    # 使用 MongoDB 模式配置
    config = AppConfig(
        mongodb_uri=os.getenv("MONGODB_URI"),
        mongodb_db=os.getenv("MONGODB_DB"),
        cache_size=100,
        cache_ttl=3600,
        enable_llm_parsing=False,
        idle_timeout=1800,
        cleanup_interval=60,
        max_sessions=10000,
    )
    
    # 初始化真实 AppContext
    ctx = await AppContext.create(config)
    
    # 预缓存测试配置到 ConfigLoader
    ctx.config_loader._l1.set("test_config_hash_123", sample_config)
    
    # 启动 SessionManager
    await ctx.session_manager.start()
    
    # 创建 FastAPI 应用
    app = FastAPI()
    router = create_router()
    app.include_router(router)
    
    # 使用 ASGITransport（httpx 0.24+ 需要）
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client, ctx.session_manager, ctx.config_loader
    
    # 清理
    await ctx.shutdown()


# =============================================================================
# WorkflowAgentV2 Tests (Real LLM)
# =============================================================================


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestWorkflowAgentV2Real:
    """WorkflowAgentV2 真实 LLM 测试"""

    @pytest.mark.asyncio
    async def test_agent_creation(self, sample_config, real_llm):
        """测试 Agent 创建"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        assert agent.config == sample_config
        assert agent._system_prompt is not None
        assert "Test Assistant" in agent._system_prompt
        print(f"\n[System Prompt Preview]\n{agent._system_prompt[:500]}...")

    @pytest.mark.asyncio
    async def test_simple_query(self, sample_config, real_llm):
        """测试简单查询"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        response = await agent.query(message="What is 2 + 2?")

        assert response is not None
        assert isinstance(response, str)
        assert len(response) > 0
        print(f"\n[Query] What is 2 + 2?")
        print(f"[Response] {response}")

    @pytest.mark.asyncio
    async def test_query_with_context(self, sample_config, real_llm):
        """测试带上下文的查询"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        # 注入历史上下文
        context = [
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Nice to meet you, Alice!"},
        ]

        response = await agent.query(
            message="What is my name?",
            context=context,
        )

        assert response is not None
        print(f"\n[Context] {context}")
        print(f"[Query] What is my name?")
        print(f"[Response] {response}")

        # 验证历史已加载
        assert len(agent.messages) > 0

    @pytest.mark.asyncio
    async def test_multi_turn_conversation(self, sample_config, real_llm):
        """测试多轮对话"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        # 第一轮
        print("\n[Turn 1]")
        response1 = await agent.query(message="Remember this number: 42")
        print(f"  User: Remember this number: 42")
        print(f"  Assistant: {response1}")

        # 第二轮（应该记住数字）
        print("\n[Turn 2]")
        response2 = await agent.query(message="What number did I ask you to remember?")
        print(f"  User: What number did I ask you to remember?")
        print(f"  Assistant: {response2}")

        assert response2 is not None
        # 验证上下文保持
        assert len(agent.messages) >= 4  # 2 user + 2 assistant

    @pytest.mark.asyncio
    async def test_clear_history(self, sample_config, real_llm):
        """测试清除历史"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        await agent.query(message="Hello")
        assert len(agent.messages) > 0

        agent.clear_history()
        assert len(agent.messages) == 0
        print("\n[History cleared successfully]")

    @pytest.mark.asyncio
    async def test_token_usage(self, sample_config, real_llm):
        """测试 Token 使用统计"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        await agent.query(message="Say hello")

        usage = await agent.get_usage()
        print(f"\n[Token Usage]")
        print(f"  Total tokens: {usage.total_tokens}")
        print(f"  Total cost: ${usage.total_cost:.6f}")

        assert usage.total_tokens > 0



# =============================================================================
# ConfigCache Tests (No API needed)
# =============================================================================


class TestConfigCacheIntegration:
    """ConfigCache 集成测试（不需要 API）"""

    def test_cache_and_retrieve(self, sample_config):
        """测试缓存和检索"""
        cache = ConfigCache(max_size=5, ttl=3600)

        cache.set("hash_1", sample_config)
        retrieved = cache.get("hash_1")

        assert retrieved is not None
        assert retrieved.basic_settings["name"] == "Test Assistant"

    def test_cache_stats(self, sample_config):
        """测试缓存统计"""
        cache = ConfigCache(max_size=5, ttl=3600)

        cache.set("hash_1", sample_config)
        cache.set("hash_2", sample_config)
        cache.get("hash_1")  # 增加访问计数

        stats = cache.get_stats()
        assert stats["size"] == 2
        assert len(stats["entries"]) == 2

    def test_lru_eviction(self, sample_config):
        """测试 LRU 淘汰"""
        cache = ConfigCache(max_size=2, ttl=3600)

        cache.set("hash_1", sample_config)
        cache.set("hash_2", sample_config)
        cache.get("hash_1")  # 访问 hash_1

        # 添加 hash_3，应该淘汰 hash_2
        cache.set("hash_3", sample_config)

        assert cache.get("hash_1") is not None
        assert cache.get("hash_2") is None
        assert cache.get("hash_3") is not None


# =============================================================================
# EventCollector Tests (No API needed)
# =============================================================================


class TestEventCollector:
    """EventCollector 单元测试"""

    def test_collect_tool_call_event(self):
        """测试收集 ToolCallEvent"""
        collector = EventCollector(
            correlation_id="corr_1",
            session_id="sess_1",
            user_message="test message",
        )

        # 模拟 ToolCallEvent
        event = ToolCallEvent(
            tool="search",
            args={"query": "test"},
            tool_call_id="call_1",
        )
        collector.collect(event)

        assert len(collector._pending_calls) == 1
        assert "call_1" in collector._pending_calls

    def test_collect_tool_result_event(self):
        """测试收集 ToolResultEvent"""
        collector = EventCollector(
            correlation_id="corr_1",
            session_id="sess_1",
        )

        # 先收集 ToolCallEvent
        call_event = ToolCallEvent(
            tool="search",
            args={"query": "test"},
            tool_call_id="call_1",
        )
        collector.collect(call_event)

        # 再收集 ToolResultEvent
        result_event = ToolResultEvent(
            tool="search",
            result="search results",
            tool_call_id="call_1",
            is_error=False,
        )
        collector.collect(result_event)

        assert len(collector._pending_calls) == 0
        assert len(collector.tool_calls) == 1
        assert collector.tool_calls[0].tool_name == "search"
        assert collector.tool_calls[0].result == "search results"
        assert collector.tool_calls[0].is_error is False

    def test_collect_final_response_event(self):
        """测试收集 FinalResponseEvent"""
        collector = EventCollector(
            correlation_id="corr_1",
            session_id="sess_1",
        )

        event = FinalResponseEvent(content="Final answer")
        collector.collect(event)

        assert collector.final_response == "Final answer"

    def test_collect_text_event(self):
        """测试收集 TextEvent"""
        collector = EventCollector(
            correlation_id="corr_1",
            session_id="sess_1",
        )

        event1 = TextEvent(content="Hello ")
        event2 = TextEvent(content="World")
        collector.collect(event1)
        collector.collect(event2)

        assert len(collector.text_chunks) == 2
        assert collector.text_chunks == ["Hello ", "World"]

    def test_to_result(self):
        """测试转换为 QueryResult"""
        collector = EventCollector(
            correlation_id="corr_1",
            session_id="sess_1",
            user_message="test",
        )

        # 收集一些事件
        collector.collect(ToolCallEvent(tool="t1", args={}, tool_call_id="c1"))
        collector.collect(ToolResultEvent(tool="t1", result="r1", tool_call_id="c1"))
        collector.collect(FinalResponseEvent(content="Done"))

        result = collector.to_result(usage=None)

        assert isinstance(result, QueryResult)
        assert result.response == "Done"
        assert len(result.tool_calls) == 1
        assert result.total_duration_ms > 0

    def test_get_event_records(self):
        """测试获取事件记录"""
        collector = EventCollector(
            correlation_id="corr_1",
            session_id="sess_1",
        )

        # 收集工具调用
        collector.collect(ToolCallEvent(tool="search", args={"q": "test"}, tool_call_id="c1"))
        collector.collect(ToolResultEvent(tool="search", result="found", tool_call_id="c1"))

        records = collector.get_event_records()

        assert len(records) == 1
        assert records[0]["correlation_id"] == "corr_1"
        assert records[0]["session_id"] == "sess_1"
        assert records[0]["event_type"] == "tool_call"
        assert records[0]["tool_name"] == "search"
        assert records[0]["arguments"] == {"q": "test"}
        assert records[0]["result"] == "found"


# =============================================================================
# EventCollector + WorkflowAgentV2 Integration Tests (Real LLM)
# =============================================================================


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestEventCollectorIntegration:
    """EventCollector 与 WorkflowAgentV2 集成测试"""

    @pytest.mark.asyncio
    async def test_collect_from_query_stream(self, sample_config, real_llm):
        """测试从 query_stream 收集事件"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        collector = EventCollector(
            correlation_id="corr_test_1",
            session_id="sess_test_1",
            user_message="What is 2 + 2?",
        )

        # 使用 query_stream 收集事件
        async for event in agent.query_stream("What is 2 + 2?"):
            collector.collect(event)
            print(f"  Collected: {type(event).__name__}")

        # 验证收集结果
        assert collector.final_response != ""
        print(f"\n[Final Response] {collector.final_response}")

        # 获取 usage
        usage = await agent.get_usage()
        result = collector.to_result(usage)

        print(f"[Total Tokens] {result.usage.total_tokens if result.usage else 0}")
        print(f"[Tool Calls] {len(result.tool_calls)}")
        print(f"[Duration] {result.total_duration_ms:.2f}ms")

        assert result.response == collector.final_response

    @pytest.mark.asyncio
    async def test_unified_collection_pattern(self, sample_config, real_llm):
        """测试统一收集模式（非流式场景）"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        # 模拟非流式场景：仅收集，不透传
        collector = EventCollector(
            correlation_id="corr_unified_1",
            session_id="sess_unified_1",
            user_message="Say hello",
        )

        async for event in agent.query_stream("Say hello"):
            collector.collect(event)  # 仅收集

        usage = await agent.get_usage()
        result = collector.to_result(usage)

        print(f"\n[Unified Collection Pattern]")
        print(f"  Response: {result.response}")
        print(f"  Tool Calls: {len(result.tool_calls)}")
        print(f"  Total Tokens: {result.usage.total_tokens if result.usage else 0}")

        assert result.response != ""

    @pytest.mark.asyncio
    async def test_streaming_collection_pattern(self, sample_config, real_llm):
        """测试流式收集模式"""
        agent = WorkflowAgentV2(config=sample_config, llm=real_llm)

        collector = EventCollector(
            correlation_id="corr_stream_1",
            session_id="sess_stream_1",
            user_message="Count from 1 to 3",
        )

        # 模拟流式场景：收集 + 透传
        events_yielded = []
        async for event in agent.query_stream("Count from 1 to 3"):
            collector.collect(event)
            events_yielded.append(event)  # 模拟 yield

        print(f"\n[Streaming Collection Pattern]")
        print(f"  Events Yielded: {len(events_yielded)}")
        print(f"  Final Response: {collector.final_response}")

        assert len(events_yielded) > 0
        assert collector.final_response != ""


# =============================================================================
# V2 Query API Unit Tests (Real Endpoint)
# =============================================================================


@pytest.mark.skipif(not HAS_API_KEY, reason=SKIP_REASON)
class TestV2QueryAPIEndpoint:
    """V2 Query API 端点测试（真实请求）"""

    @pytest.fixture
    def sample_config(self) -> WorkflowConfigSchema:
        """创建示例配置"""
        return WorkflowConfigSchema(
            basic_settings={
                "name": "Test Assistant",
                "description": "A test assistant for endpoint testing",
                "language": "English",
                "tone": "professional",
            },
            instructions="You are a helpful assistant. Answer questions concisely in one sentence.",
            max_iterations=3,
        )

    
    # -------------------------------------------------------------------------
    # POST /v2/query Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_query_simple(self, app_client):
        """测试简单查询"""
        client, session_manager, _ = app_client

        response = await client.post(
            "/v2/query",
            json={
                "message": "What is 2 + 2?",
                "session_id": "sess_simple_1",
                "tenant_id": "test_tenant",
                "chatbot_id": "test_chatbot",
                "md5_checksum": "test_config_hash_123",  # 使用预缓存的配置
            },
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n[Query] What is 2 + 2?")
        print(f"[Response] {data['message']}")

        assert data["session_id"] == "sess_simple_1"
        assert data["status"] == "success"
        assert len(data["message"]) > 0
        assert data["agent_id"] == "test_tenant:test_chatbot"

    @pytest.mark.asyncio
    async def test_query_with_config_hash(self, app_client):
        """测试带配置哈希的查询（使用预缓存配置）"""
        client, _, _ = app_client

        response = await client.post(
            "/v2/query",
            json={
                "message": "Say hello",
                "session_id": "sess_hash_1",
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "test_config_hash_123",  # 使用预缓存的配置
            },
        )

        assert response.status_code == 200
        data = response.json()

        print(f"\n[Query with config hash]")
        print(f"[Response] {data['message']}")

        # 验证响应正常
        assert data["status"] == "success"
        assert len(data["message"]) > 0

    @pytest.mark.asyncio
    async def test_query_multi_turn(self, app_client):
        """测试多轮对话"""
        client, session_manager, _ = app_client
        session_id = "sess_multi_turn_1"

        # 第一轮
        response1 = await client.post(
            "/v2/query",
            json={
                "message": "Remember this number: 42",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "test_config_hash_123",
            },
        )
        assert response1.status_code == 200
        data1 = response1.json()
        print(f"\n[Turn 1] Remember this number: 42")
        print(f"[Response 1] {data1['message']}")

        # 第二轮
        response2 = await client.post(
            "/v2/query",
            json={
                "message": "What number did I ask you to remember?",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "test_config_hash_123",
            },
        )
        assert response2.status_code == 200
        data2 = response2.json()
        print(f"\n[Turn 2] What number did I ask you to remember?")
        print(f"[Response 2] {data2['message']}")

        # 验证会话存在且有两次查询记录
        session_info = session_manager.get_session_info(session_id)
        assert session_info is not None
        assert session_info["query_count"] >= 2
        print(f"\n[Session Info] query_count={session_info['query_count']}")

    @pytest.mark.asyncio
    async def test_query_validation_error(self, app_client):
        """测试请求验证错误"""
        client, _, _ = app_client

        # 缺少必填字段
        response = await client.post(
            "/v2/query",
            json={
                "message": "Hello",
                # 缺少 session_id, tenant_id, chatbot_id
            },
        )

        assert response.status_code == 422  # Validation Error
        print(f"\n[Validation Error] {response.json()}")

    # -------------------------------------------------------------------------
    # GET /v2/sessions Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_list_sessions(self, app_client):
        """测试列出会话"""
        client, _, _ = app_client

        # 先创建一个会话
        await client.post(
            "/v2/query",
            json={
                "message": "Hello",
                "session_id": "sess_list_1",
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "test_config_hash_123",
            },
        )

        # 列出会话
        response = await client.get("/v2/sessions")

        assert response.status_code == 200
        data = response.json()

        print(f"\n[List Sessions]")
        print(f"  Sessions: {len(data['sessions'])}")
        print(f"  Stats: {data['stats']}")

        assert "sessions" in data
        assert "stats" in data
        assert data["stats"]["session_count"] >= 1

    @pytest.mark.asyncio
    async def test_get_session_info(self, app_client):
        """测试获取会话信息"""
        client, _, _ = app_client
        session_id = "sess_info_1"

        # 先创建会话
        await client.post(
            "/v2/query",
            json={
                "message": "Hello",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "test_config_hash_123",
            },
        )

        # 获取会话信息
        response = await client.get(f"/v2/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()

        print(f"\n[Session Info] {session_id}")
        print(f"  Data: {data}")

        assert data["session_id"] == session_id

    @pytest.mark.asyncio
    async def test_get_session_not_found(self, app_client):
        """测试获取不存在的会话"""
        client, _, _ = app_client

        response = await client.get("/v2/sessions/nonexistent_session_xyz")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "SessionNotFound"

    # -------------------------------------------------------------------------
    # DELETE /v2/sessions/{session_id} Tests
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_destroy_session(self, app_client):
        """测试销毁会话"""
        client, session_manager, _ = app_client
        session_id = "sess_destroy_1"

        # 先创建会话
        await client.post(
            "/v2/query",
            json={
                "message": "Hello",
                "session_id": session_id,
                "tenant_id": "tenant_1",
                "chatbot_id": "bot_1",
                "md5_checksum": "test_config_hash_123",
            },
        )

        # 确认会话存在
        assert session_manager.exists(session_id)

        # 销毁会话
        response = await client.delete(f"/v2/sessions/{session_id}")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "destroyed"
        assert data["session_id"] == session_id

        # 确认会话已销毁
        assert not session_manager.exists(session_id)

        print(f"\n[Session Destroyed] {session_id}")

    @pytest.mark.asyncio
    async def test_destroy_session_not_found(self, app_client):
        """测试销毁不存在的会话"""
        client, _, _ = app_client

        response = await client.delete("/v2/sessions/nonexistent_session_abc")

        assert response.status_code == 404
        data = response.json()
        assert data["detail"]["error"] == "SessionNotFound"

# =============================================================================
# V2 Query API Router Tests (Direct Router Testing)
# =============================================================================


class TestV2QueryRouter:
    """V2 Query Router 直接测试"""

    def test_router_creation(self):
        """测试路由创建"""
        from api.routers.v2.query import create_router

        router = create_router()

        assert router is not None
        assert router.prefix == "/v2"
        assert "v2" in router.tags

        # 验证路由端点
        routes = {r.path for r in router.routes}
        assert "/query" in routes or "/v2/query" in routes


# =============================================================================
# QueryRecorder Unit Tests
# =============================================================================


class TestQueryRecorderUnit:
    """QueryRecorder 单元测试"""

    @pytest.fixture
    def mock_repos(self):
        """创建 Mock RepositoryManager"""
        repos = MagicMock()
        repos.messages = MagicMock()
        repos.messages.create = AsyncMock()
        repos.tool_calls = MagicMock()
        repos.tool_calls.create = AsyncMock()
        repos.usages = MagicMock()
        repos.usages.create = AsyncMock()
        return repos

    @pytest.fixture
    def sample_collector(self):
        """创建示例 EventCollector"""
        collector = EventCollector(
            correlation_id="corr_test_1",
            session_id="sess_test_1",
            user_message="Hello",
        )
        # 模拟收集一些事件
        collector.collect(
            ToolCallEvent(tool="search", args={"query": "test"}, tool_call_id="call_1")
        )
        collector.collect(
            ToolResultEvent(
                tool="search", result="found", tool_call_id="call_1", is_error=False
            )
        )
        collector.collect(FinalResponseEvent(content="Here is the answer"))
        return collector

    def test_recorder_initialization(self, mock_repos):
        """测试 Recorder 初始化"""
        recorder = QueryRecorder(mock_repos)
        assert recorder._repos == mock_repos

    @pytest.mark.asyncio
    async def test_recorder_record_async(self, mock_repos, sample_collector):
        """测试异步记录"""
        from bu_agent_sdk.tokens import UsageSummary, ModelUsageStats

        recorder = QueryRecorder(mock_repos)

        # 创建完整的 UsageSummary mock
        mock_usage = MagicMock(spec=UsageSummary)
        mock_usage.total_tokens = 150
        mock_usage.total_cost = 0.002
        mock_usage.total_prompt_tokens = 100
        mock_usage.total_completion_tokens = 50
        mock_usage.by_model = {
            "gpt-4": MagicMock(prompt_tokens=100, completion_tokens=50, cost=0.002)
        }

        # 调用 record_async（启动后台任务）
        task = recorder.record_async(sample_collector, mock_usage)

        # 等待任务完成
        await task

        # 验证消息创建被调用
        assert mock_repos.messages.create.call_count >= 2  # user + assistant
        # 验证工具调用创建被调用
        assert mock_repos.tool_calls.create.call_count >= 1
        # 验证 usage 创建被调用
        assert mock_repos.usages.create.called


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
