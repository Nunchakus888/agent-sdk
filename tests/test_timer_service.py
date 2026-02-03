"""
TimerService 单元测试

测试内容：
- 生命周期管理（start/stop）
- 扫描循环
- Timer 执行（统一 Tool 模型）
- 状态更新
- 资源控制
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

from api.services.timer_service import (
    TimerService,
    create_timer_service,
    MAX_TIMERS_PER_SESSION,
    MAX_CONCURRENT_TRIGGERS,
    DEFAULT_SCAN_INTERVAL,
    TIMER_TTL_DAYS,
)


# =============================================================================
# Mock Objects
# =============================================================================


class MockSession:
    """模拟 Session 对象。"""

    def __init__(
        self,
        session_id: str = "test_session",
        timer_status: str = "pending",
        timer_trigger_count: int = 0,
        timer_config: dict = None,
    ):
        self.session_id = session_id
        self.timer_status = timer_status
        self.timer_trigger_count = timer_trigger_count
        self.timer_config = timer_config or {
            "tool_name": "generate_response",
            "message": "您好，请问还在吗？",
            "max_triggers": 3,
        }


class MockSessionRepository:
    """模拟 Session Repository。"""

    def __init__(self):
        self.sessions = {}
        self.update_calls = []

    async def find_timeout_sessions(self):
        """返回超时的 sessions。"""
        return [s for s in self.sessions.values() if s.timer_status == "pending"]

    async def update(self, session_id: str, **kwargs):
        """更新 session。"""
        self.update_calls.append({"session_id": session_id, **kwargs})
        if session_id in self.sessions:
            for key, value in kwargs.items():
                setattr(self.sessions[session_id], key, value)


class MockMessageRepository:
    """模拟 Message Repository。"""

    def __init__(self):
        self.messages = []

    async def create(self, **kwargs):
        """创建消息。"""
        self.messages.append(kwargs)


class MockRepos:
    """模拟 RepositoryManager。"""

    def __init__(self):
        self.sessions = MockSessionRepository()
        self.messages = MockMessageRepository()


class MockToolExecutor:
    """模拟 ToolExecutor。"""

    def __init__(self):
        self.executed_tools = []

    async def execute_single(self, tool_name: str, params: dict = None):
        """执行单个工具。"""
        self.executed_tools.append({"tool_name": tool_name, "params": params})
        return {"status": "ok"}


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_repos():
    """创建模拟 repos。"""
    return MockRepos()


@pytest.fixture
def mock_tool_executor():
    """创建模拟 tool executor。"""
    return MockToolExecutor()


@pytest.fixture
def timer_service(mock_repos, mock_tool_executor):
    """创建 TimerService 实例。"""
    return TimerService(
        repos=mock_repos,
        tool_executor=mock_tool_executor,
        send_message=AsyncMock(),
        scan_interval=1,  # 快速测试
    )


# =============================================================================
# 常量测试
# =============================================================================


class TestConstants:
    """常量配置测试。"""

    def test_default_constants(self):
        """测试默认常量值。"""
        assert MAX_TIMERS_PER_SESSION == 10
        assert MAX_CONCURRENT_TRIGGERS == 50
        assert DEFAULT_SCAN_INTERVAL == 30
        assert TIMER_TTL_DAYS == 7


# =============================================================================
# 生命周期测试
# =============================================================================


class TestLifecycle:
    """生命周期测试。"""

    @pytest.mark.asyncio
    async def test_start(self, timer_service):
        """测试启动。"""
        assert timer_service._running is False
        assert timer_service._task is None

        await timer_service.start()

        assert timer_service._running is True
        assert timer_service._task is not None

        await timer_service.stop()

    @pytest.mark.asyncio
    async def test_start_idempotent(self, timer_service):
        """测试重复启动幂等。"""
        await timer_service.start()
        task1 = timer_service._task

        await timer_service.start()
        task2 = timer_service._task

        assert task1 is task2  # 同一个任务

        await timer_service.stop()

    @pytest.mark.asyncio
    async def test_stop(self, timer_service):
        """测试停止。"""
        await timer_service.start()
        assert timer_service._running is True

        await timer_service.stop()

        assert timer_service._running is False

    @pytest.mark.asyncio
    async def test_stop_without_start(self, timer_service):
        """测试未启动时停止。"""
        # 不应抛出异常
        await timer_service.stop()
        assert timer_service._running is False


# =============================================================================
# 扫描循环测试
# =============================================================================


class TestScanLoop:
    """扫描循环测试。"""

    @pytest.mark.asyncio
    async def test_scan_loop_calls_check_timeouts(self, timer_service):
        """测试扫描循环调用 _check_timeouts。"""
        check_count = {"count": 0}

        async def mock_check_timeouts():
            check_count["count"] += 1
            if check_count["count"] >= 2:
                timer_service._running = False

        timer_service._check_timeouts = mock_check_timeouts
        timer_service._scan_interval = 0.01

        await timer_service.start()
        await asyncio.sleep(0.05)
        await timer_service.stop()

        assert check_count["count"] >= 1

    @pytest.mark.asyncio
    async def test_scan_loop_handles_errors(self, timer_service):
        """测试扫描循环错误处理。"""
        call_count = {"count": 0}

        async def mock_check_timeouts():
            call_count["count"] += 1
            if call_count["count"] == 1:
                raise Exception("Test error")
            if call_count["count"] >= 3:
                timer_service._running = False

        timer_service._check_timeouts = mock_check_timeouts
        timer_service._scan_interval = 0.01

        await timer_service.start()
        await asyncio.sleep(0.1)
        await timer_service.stop()

        # 错误后继续执行
        assert call_count["count"] >= 2


# =============================================================================
# Timer 执行测试
# =============================================================================


class TestTimerExecution:
    """Timer 执行测试。"""

    @pytest.mark.asyncio
    async def test_execute_timer_generate_response(self, timer_service, mock_repos):
        """测试执行 generate_response 类型 Timer。"""
        session = MockSession(
            session_id="sess_001",
            timer_config={
                "tool_name": "generate_response",
                "message": "测试消息",
                "max_triggers": 3,
            },
        )
        mock_repos.sessions.sessions["sess_001"] = session

        await timer_service._execute_timer(session)

        # 验证消息被创建
        assert len(mock_repos.messages.messages) == 1
        assert mock_repos.messages.messages[0]["content"] == "测试消息"

        # 验证 send_message 被调用
        timer_service._send_message.assert_called_once_with("sess_001", "测试消息")

    @pytest.mark.asyncio
    async def test_execute_timer_tool_call(self, timer_service, mock_repos, mock_tool_executor):
        """测试执行 Tool 调用类型 Timer。"""
        session = MockSession(
            session_id="sess_002",
            timer_config={
                "tool_name": "send_notification",
                "tool_params": {"user_id": "123", "message": "提醒"},
                "max_triggers": 3,
            },
        )
        mock_repos.sessions.sessions["sess_002"] = session

        await timer_service._execute_timer(session)

        # 验证 tool 被执行
        assert len(mock_tool_executor.executed_tools) == 1
        assert mock_tool_executor.executed_tools[0]["tool_name"] == "send_notification"
        assert mock_tool_executor.executed_tools[0]["params"]["user_id"] == "123"

    @pytest.mark.asyncio
    async def test_execute_timer_max_triggers_reached(self, timer_service, mock_repos):
        """测试达到最大触发次数。"""
        session = MockSession(
            session_id="sess_003",
            timer_trigger_count=3,
            timer_config={"max_triggers": 3},
        )
        mock_repos.sessions.sessions["sess_003"] = session

        await timer_service._execute_timer(session)

        # 验证 Timer 被禁用
        assert any(
            call.get("timer_status") == "disabled"
            for call in mock_repos.sessions.update_calls
        )

    @pytest.mark.asyncio
    async def test_execute_timer_updates_status(self, timer_service, mock_repos):
        """测试 Timer 状态更新。"""
        session = MockSession(
            session_id="sess_004",
            timer_trigger_count=0,
            timer_config={
                "tool_name": "generate_response",
                "message": "测试",
                "max_triggers": 3,
            },
        )
        mock_repos.sessions.sessions["sess_004"] = session

        await timer_service._execute_timer(session)

        # 验证触发次数更新
        update_call = mock_repos.sessions.update_calls[-1]
        assert update_call["timer_trigger_count"] == 1
        assert update_call["timer_status"] == "triggered"


# =============================================================================
# 状态更新测试
# =============================================================================


class TestStatusUpdate:
    """状态更新测试。"""

    @pytest.mark.asyncio
    async def test_update_timer_status_triggered(self, timer_service, mock_repos):
        """测试更新为 triggered 状态。"""
        session = MockSession(
            session_id="sess_005",
            timer_trigger_count=0,
            timer_config={"max_triggers": 3},
        )
        mock_repos.sessions.sessions["sess_005"] = session

        await timer_service._update_timer_status(session)

        update_call = mock_repos.sessions.update_calls[-1]
        assert update_call["timer_status"] == "triggered"
        assert update_call["timer_trigger_count"] == 1

    @pytest.mark.asyncio
    async def test_update_timer_status_disabled(self, timer_service, mock_repos):
        """测试更新为 disabled 状态（达到最大次数）。"""
        session = MockSession(
            session_id="sess_006",
            timer_trigger_count=2,
            timer_config={"max_triggers": 3},
        )
        mock_repos.sessions.sessions["sess_006"] = session

        await timer_service._update_timer_status(session)

        update_call = mock_repos.sessions.update_calls[-1]
        assert update_call["timer_status"] == "disabled"
        assert update_call["timer_trigger_count"] == 3

    @pytest.mark.asyncio
    async def test_disable_timer(self, timer_service, mock_repos):
        """测试禁用 Timer。"""
        await timer_service._disable_timer("sess_007")

        assert len(mock_repos.sessions.update_calls) == 1
        assert mock_repos.sessions.update_calls[0]["timer_status"] == "disabled"

    @pytest.mark.asyncio
    async def test_cancel_session_timers(self, timer_service, mock_repos):
        """测试取消 Session 的所有 Timer。"""
        await timer_service.cancel_session_timers("sess_008")

        assert len(mock_repos.sessions.update_calls) == 1
        assert mock_repos.sessions.update_calls[0]["timer_status"] == "cancelled"


# =============================================================================
# 并发控制测试
# =============================================================================


class TestConcurrencyControl:
    """并发控制测试。"""

    @pytest.mark.asyncio
    async def test_concurrent_triggers_limited(self, timer_service, mock_repos):
        """测试并发触发数限制。"""
        # 创建多个待触发的 session
        for i in range(60):
            session = MockSession(
                session_id=f"sess_{i:03d}",
                timer_config={
                    "tool_name": "generate_response",
                    "message": f"消息 {i}",
                    "max_triggers": 3,
                },
            )
            mock_repos.sessions.sessions[f"sess_{i:03d}"] = session

        concurrent_count = {"max": 0, "current": 0}
        original_execute = timer_service._execute_timer

        async def tracked_execute(session):
            concurrent_count["current"] += 1
            concurrent_count["max"] = max(concurrent_count["max"], concurrent_count["current"])
            await asyncio.sleep(0.01)
            concurrent_count["current"] -= 1

        timer_service._execute_timer = tracked_execute

        await timer_service._check_timeouts()

        # 验证并发数不超过限制
        assert concurrent_count["max"] <= MAX_CONCURRENT_TRIGGERS


# =============================================================================
# 工厂函数测试
# =============================================================================


class TestFactory:
    """工厂函数测试。"""

    def test_create_timer_service(self, mock_repos):
        """测试创建 TimerService。"""
        service = create_timer_service(
            repos=mock_repos,
            scan_interval=60,
        )

        assert isinstance(service, TimerService)
        assert service._scan_interval == 60

    def test_create_timer_service_with_all_params(self, mock_repos, mock_tool_executor):
        """测试带所有参数创建 TimerService。"""
        send_message = AsyncMock()

        service = create_timer_service(
            repos=mock_repos,
            tool_executor=mock_tool_executor,
            send_message=send_message,
            scan_interval=120,
        )

        assert service._tool_executor is mock_tool_executor
        assert service._send_message is send_message
        assert service._scan_interval == 120


# =============================================================================
# 边界情况测试
# =============================================================================


class TestEdgeCases:
    """边界情况测试。"""

    @pytest.mark.asyncio
    async def test_no_due_timers(self, timer_service, mock_repos):
        """测试无到期 Timer。"""
        # 不添加任何 session
        await timer_service._check_timeouts()
        # 不应抛出异常

    @pytest.mark.asyncio
    async def test_execute_timer_no_tool_executor(self, mock_repos):
        """测试无 ToolExecutor 时执行非 generate_response 工具。"""
        service = TimerService(
            repos=mock_repos,
            tool_executor=None,  # 无 executor
            send_message=AsyncMock(),
        )

        session = MockSession(
            session_id="sess_009",
            timer_config={
                "tool_name": "custom_tool",
                "tool_params": {},
                "max_triggers": 3,
            },
        )

        # 不应抛出异常，只记录警告
        await service._execute_timer(session)

    @pytest.mark.asyncio
    async def test_execute_timer_error_handling(self, timer_service, mock_repos):
        """测试 Timer 执行错误处理。"""
        session = MockSession(
            session_id="sess_010",
            timer_config={
                "tool_name": "generate_response",
                "message": "测试",
                "max_triggers": 3,
            },
        )

        # 模拟消息创建失败
        mock_repos.messages.create = AsyncMock(side_effect=Exception("DB error"))

        # 不应抛出异常
        await timer_service._execute_timer(session)

    @pytest.mark.asyncio
    async def test_find_due_timers_no_sessions_repo(self):
        """测试无 sessions repo 时查找到期 Timer。"""
        repos = MagicMock()
        del repos.sessions  # 移除 sessions 属性

        service = TimerService(repos=repos)
        timers = await service._find_due_timers()

        assert timers == []

    @pytest.mark.asyncio
    async def test_generate_response_no_message_repo(self, timer_service):
        """测试无 messages repo 时生成响应。"""
        # 移除 messages 属性
        delattr(timer_service._repos, 'messages')

        session = MockSession(session_id="sess_011")

        # 不应抛出异常
        await timer_service._generate_response(session, "测试消息")

        # send_message 仍应被调用
        timer_service._send_message.assert_called_with("sess_011", "测试消息")

    @pytest.mark.asyncio
    async def test_generate_response_no_send_callback(self, mock_repos):
        """测试无 send_message 回调时生成响应。"""
        service = TimerService(
            repos=mock_repos,
            send_message=None,  # 无回调
        )

        session = MockSession(session_id="sess_012")

        # 不应抛出异常
        await service._generate_response(session, "测试消息")

        # 消息仍应被存储
        assert len(mock_repos.messages.messages) == 1
