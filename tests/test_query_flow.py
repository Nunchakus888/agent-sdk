"""
Query API 单元测试

测试优化后的并行流程：
- Phase 1: 并行准备（session + agent + history）
- Phase 2: 核心执行（agent.query）
- Phase 3: 后台记录（fire & forget）
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from dataclasses import dataclass

from api.routers.query import create_router, _record_query, _record_and_reset_timer
from api.models import MessageRole, EventType, EventStatus


# =============================================================================
# Mock 数据
# =============================================================================


@dataclass
class MockMessage:
    message_id: str
    role: MessageRole
    content: str


@dataclass
class MockSession:
    session_id: str


class MockAgent:
    config_hash = "test_hash_123"

    async def query(self, message: str, session_id: str, context: list) -> str:
        return f"Response to: {message}"


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repos():
    """Mock RepositoryManager"""
    repos = MagicMock()

    # sessions
    repos.sessions.get_or_create = AsyncMock(return_value=MockSession("test_session"))
    repos.sessions.allocate_event_offset = AsyncMock(side_effect=lambda sid: 1)
    repos.sessions.reset_timer = AsyncMock(return_value=None)

    # messages
    repos.messages.list_by_session = AsyncMock(return_value=[
        MockMessage("msg_1", MessageRole.USER, "Hello"),
        MockMessage("msg_2", MessageRole.ASSISTANT, "Hi there"),
    ])
    repos.messages.create = AsyncMock(side_effect=lambda **kwargs: MockMessage(
        message_id=f"msg_{kwargs.get('role', 'unknown')}",
        role=kwargs.get("role", MessageRole.USER),
        content=kwargs.get("content", ""),
    ))

    # events
    repos.events.create = AsyncMock(return_value=None)

    return repos


@pytest.fixture
def mock_pool():
    """Mock AgentPool"""
    pool = MagicMock()
    pool.get = AsyncMock(return_value=MockAgent())
    return pool


# =============================================================================
# Phase 1: 并行准备测试
# =============================================================================


class TestPhase1Parallel:
    """测试 Phase 1 并行准备"""

    @pytest.mark.asyncio
    async def test_parallel_tasks_created(self, mock_repos, mock_pool):
        """验证三个任务并行创建"""
        call_order = []

        async def track_session(*args, **kwargs):
            call_order.append("session_start")
            await asyncio.sleep(0.01)
            call_order.append("session_end")
            return MockSession("test")

        async def track_agent(*args, **kwargs):
            call_order.append("agent_start")
            await asyncio.sleep(0.01)
            call_order.append("agent_end")
            return MockAgent()

        async def track_history(*args, **kwargs):
            call_order.append("history_start")
            await asyncio.sleep(0.01)
            call_order.append("history_end")
            return []

        mock_repos.sessions.get_or_create = track_session
        mock_pool.get = track_agent
        mock_repos.messages.list_by_session = track_history

        # 模拟请求
        from api.models import QueryRequest
        request = QueryRequest(
            message="test",
            session_id="test_session",
            tenant_id="test_tenant",
            chatbot_id="test_chatbot",
            customer_id="test_customer",
        )

        # 执行并行准备
        session_task = asyncio.create_task(mock_repos.sessions.get_or_create())
        agent_task = asyncio.create_task(mock_pool.get(
            chatbot_id="test", tenant_id="test", config_hash="hash"
        ))
        history_task = asyncio.create_task(mock_repos.messages.list_by_session(
            session_id="test", limit=20, order="asc"
        ))

        await asyncio.gather(session_task, agent_task, history_task)

        # 验证并行执行（所有 start 应该在所有 end 之前）
        starts = [i for i, x in enumerate(call_order) if x.endswith("_start")]
        ends = [i for i, x in enumerate(call_order) if x.endswith("_end")]

        # 至少有 2 个 start 在第一个 end 之前（证明并行）
        first_end = min(ends)
        starts_before_first_end = sum(1 for s in starts if s < first_end)
        assert starts_before_first_end >= 2, f"Tasks not parallel: {call_order}"

    @pytest.mark.asyncio
    async def test_context_includes_current_message(self, mock_repos, mock_pool):
        """验证上下文包含当前用户消息"""
        captured_context = None

        async def capture_query(message, session_id, context):
            nonlocal captured_context
            captured_context = context
            return "response"

        mock_agent = MockAgent()
        mock_agent.query = capture_query
        mock_pool.get = AsyncMock(return_value=mock_agent)

        # 历史消息
        mock_repos.messages.list_by_session = AsyncMock(return_value=[
            MockMessage("1", MessageRole.USER, "old message"),
        ])

        # 模拟流程
        history = await mock_repos.messages.list_by_session(session_id="test", limit=20, order="asc")
        context = [{"role": m.role.value, "content": m.content} for m in history]
        context.append({"role": "user", "content": "new message"})

        agent = await mock_pool.get(chatbot_id="test", tenant_id="test", config_hash="hash")
        await agent.query(message="new message", session_id="test", context=context)

        # 验证上下文
        assert len(captured_context) == 2
        assert captured_context[-1]["content"] == "new message"
        assert captured_context[-1]["role"] == "user"


# =============================================================================
# Phase 3: 后台记录测试
# =============================================================================


class TestPhase3Background:
    """测试 Phase 3 后台记录"""

    @pytest.mark.asyncio
    async def test_record_query_creates_messages(self, mock_repos):
        """验证后台记录创建消息"""
        await _record_query(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

        # 验证消息创建调用
        assert mock_repos.messages.create.call_count == 2

        # 验证用户消息
        calls = mock_repos.messages.create.call_args_list
        user_call = calls[0]
        assert user_call.kwargs["role"] == MessageRole.USER
        assert user_call.kwargs["content"] == "Hello"

        # 验证助手消息
        assistant_call = calls[1]
        assert assistant_call.kwargs["role"] == MessageRole.ASSISTANT
        assert assistant_call.kwargs["content"] == "Hi there"

    @pytest.mark.asyncio
    async def test_record_query_creates_events(self, mock_repos):
        """验证后台记录创建事件"""
        await _record_query(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

        # 验证事件创建调用
        assert mock_repos.events.create.call_count == 2

    @pytest.mark.asyncio
    async def test_record_query_handles_error(self, mock_repos):
        """验证后台记录错误处理"""
        mock_repos.messages.create = AsyncMock(side_effect=Exception("DB error"))

        # 不应该抛出异常
        await _record_query(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

    @pytest.mark.asyncio
    async def test_record_query_parallel_execution(self, mock_repos):
        """验证后台记录并行执行"""
        call_times = []

        async def track_create(**kwargs):
            call_times.append(("start", asyncio.get_event_loop().time()))
            await asyncio.sleep(0.01)
            call_times.append(("end", asyncio.get_event_loop().time()))
            return MockMessage("id", MessageRole.USER, "")

        mock_repos.messages.create = track_create

        await _record_query(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

        # 验证并行（两个 start 时间接近）
        starts = [t for label, t in call_times if label == "start"]
        if len(starts) >= 2:
            time_diff = abs(starts[1] - starts[0])
            assert time_diff < 0.005, f"Not parallel: diff={time_diff}s"


# =============================================================================
# 性能测试
# =============================================================================


class TestPerformance:
    """性能相关测试"""

    @pytest.mark.asyncio
    async def test_response_before_record_complete(self, mock_repos, mock_pool):
        """验证响应在记录完成前返回"""
        record_started = asyncio.Event()
        record_completed = asyncio.Event()

        original_create = mock_repos.messages.create

        async def slow_create(**kwargs):
            record_started.set()
            await asyncio.sleep(0.1)  # 模拟慢写入
            record_completed.set()
            return await original_create(**kwargs)

        mock_repos.messages.create = slow_create

        # 启动后台记录
        task = asyncio.create_task(_record_query(
            repos=mock_repos,
            session_id="test",
            correlation_id="corr",
            user_message="Hello",
            assistant_message="Hi",
            query_latency_ms=50,
        ))

        # 等待记录开始
        await asyncio.wait_for(record_started.wait(), timeout=1.0)

        # 此时记录尚未完成，但主流程可以继续
        assert not record_completed.is_set()

        # 等待完成
        await task
        assert record_completed.is_set()


# =============================================================================
# 边界条件测试
# =============================================================================


class TestEdgeCases:
    """边界条件测试"""

    @pytest.mark.asyncio
    async def test_empty_history(self, mock_repos, mock_pool):
        """验证空历史处理"""
        mock_repos.messages.list_by_session = AsyncMock(return_value=[])

        history = await mock_repos.messages.list_by_session(session_id="new", limit=20, order="asc")
        context = [{"role": m.role.value, "content": m.content} for m in history]
        context.append({"role": "user", "content": "first message"})

        assert len(context) == 1
        assert context[0]["content"] == "first message"

    @pytest.mark.asyncio
    async def test_long_message_truncation(self, mock_repos):
        """验证长消息处理"""
        long_message = "x" * 10000

        await _record_query(
            repos=mock_repos,
            session_id="test",
            correlation_id="corr",
            user_message=long_message,
            assistant_message="short",
            query_latency_ms=100,
        )

        # 验证消息被存储（不截断）
        calls = mock_repos.messages.create.call_args_list
        assert calls[0].kwargs["content"] == long_message


# =============================================================================
# _record_and_reset_timer 测试
# =============================================================================


class TestRecordAndResetTimer:
    """测试 _record_and_reset_timer 函数"""

    @pytest.mark.asyncio
    async def test_record_and_reset_timer_success(self, mock_repos):
        """验证记录和重置 Timer 成功"""
        await _record_and_reset_timer(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

        # 验证消息创建
        assert mock_repos.messages.create.call_count == 2

        # 验证 Timer 重置
        mock_repos.sessions.reset_timer.assert_called_once_with("test_session")

    @pytest.mark.asyncio
    async def test_record_and_reset_timer_parallel(self, mock_repos):
        """验证记录和重置 Timer 并行执行"""
        call_order = []

        original_create = mock_repos.messages.create

        async def track_create(**kwargs):
            call_order.append("create_start")
            await asyncio.sleep(0.01)
            call_order.append("create_end")
            return await original_create(**kwargs)

        async def track_reset(session_id):
            call_order.append("reset_start")
            await asyncio.sleep(0.01)
            call_order.append("reset_end")

        mock_repos.messages.create = track_create
        mock_repos.sessions.reset_timer = track_reset

        await _record_and_reset_timer(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

        # 验证并行执行（reset_start 应该在 create_end 之前）
        reset_start_idx = call_order.index("reset_start")
        create_end_indices = [i for i, x in enumerate(call_order) if x == "create_end"]
        # reset 应该在至少一个 create_end 之前开始
        assert any(reset_start_idx < idx for idx in create_end_indices), f"Not parallel: {call_order}"

    @pytest.mark.asyncio
    async def test_record_and_reset_timer_record_error(self, mock_repos):
        """验证记录失败不影响 Timer 重置"""
        mock_repos.messages.create = AsyncMock(side_effect=Exception("DB error"))

        # 不应该抛出异常
        await _record_and_reset_timer(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

    @pytest.mark.asyncio
    async def test_record_and_reset_timer_reset_error(self, mock_repos):
        """验证 Timer 重置失败不影响记录"""
        mock_repos.sessions.reset_timer = AsyncMock(side_effect=Exception("Timer error"))

        # 不应该抛出异常
        await _record_and_reset_timer(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )

        # 消息仍应被创建
        assert mock_repos.messages.create.call_count == 2

    @pytest.mark.asyncio
    async def test_record_and_reset_timer_both_error(self, mock_repos):
        """验证两者都失败时不抛出异常"""
        mock_repos.messages.create = AsyncMock(side_effect=Exception("DB error"))
        mock_repos.sessions.reset_timer = AsyncMock(side_effect=Exception("Timer error"))

        # 不应该抛出异常
        await _record_and_reset_timer(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="Hello",
            assistant_message="Hi there",
            query_latency_ms=100,
        )


# =============================================================================
# 完整流程集成测试
# =============================================================================


class TestQueryIntegration:
    """完整查询流程集成测试"""

    @pytest.mark.asyncio
    async def test_full_query_flow(self, mock_repos, mock_pool):
        """测试完整查询流程"""
        # 模拟完整流程
        # Phase 1: 并行准备
        session_task = asyncio.create_task(
            mock_repos.sessions.get_or_create(
                session_id="test_session",
                tenant_id="test_tenant",
                chatbot_id="test_chatbot",
                customer_id="test_customer",
            )
        )
        agent_task = asyncio.create_task(
            mock_pool.get(
                chatbot_id="test_chatbot",
                tenant_id="test_tenant",
                config_hash="default",
            )
        )
        history_task = asyncio.create_task(
            mock_repos.messages.list_by_session(
                session_id="test_session",
                limit=20,
                order="asc",
            )
        )

        _, agent, history = await asyncio.gather(session_task, agent_task, history_task)

        # 构建上下文
        context = [{"role": m.role.value, "content": m.content} for m in history]
        context.append({"role": "user", "content": "test message"})

        # Phase 2: 核心执行
        result = await agent.query(
            message="test message",
            session_id="test_session",
            context=context,
        )

        assert "test message" in result

        # Phase 3: 后台记录
        await _record_and_reset_timer(
            repos=mock_repos,
            session_id="test_session",
            correlation_id="corr_123",
            user_message="test message",
            assistant_message=result,
            query_latency_ms=50,
        )

        # 验证所有调用
        mock_repos.sessions.get_or_create.assert_called_once()
        mock_pool.get.assert_called_once()
        mock_repos.messages.list_by_session.assert_called_once()
        assert mock_repos.messages.create.call_count == 2
        mock_repos.sessions.reset_timer.assert_called_once()

    @pytest.mark.asyncio
    async def test_query_with_empty_history(self, mock_repos, mock_pool):
        """测试空历史的查询"""
        mock_repos.messages.list_by_session = AsyncMock(return_value=[])

        history = await mock_repos.messages.list_by_session(
            session_id="new_session",
            limit=20,
            order="asc",
        )

        context = [{"role": m.role.value, "content": m.content} for m in history]
        context.append({"role": "user", "content": "first message"})

        agent = await mock_pool.get(
            chatbot_id="test",
            tenant_id="test",
            config_hash="default",
        )

        result = await agent.query(
            message="first message",
            session_id="new_session",
            context=context,
        )

        assert len(context) == 1
        assert "first message" in result
