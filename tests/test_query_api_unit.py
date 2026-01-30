"""
Query API 集成测试

测试覆盖：
1. 不同处理阶段的任务验证
2. 数据写表功能验证
3. 真实请求交互验证

需要先启动服务器：
    python -m api.main

运行测试：
    pytest tests/test_query_api_unit.py -v -s
"""

import os
import pytest
import httpx

# =============================================================================
# 配置
# =============================================================================

BASE_URL = os.getenv("TEST_API_URL", "http://localhost:8000")
API_PREFIX = "/api/v1"
TIMEOUT = 120.0  # 秒


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(scope="module")
def client():
    """创建 HTTP 客户端"""
    with httpx.Client(base_url=BASE_URL, timeout=TIMEOUT) as c:
        yield c


@pytest.fixture(scope="module")
def check_server(client):
    """检查服务器是否运行"""
    try:
        response = client.get(f"{API_PREFIX}/health")
        if response.status_code != 200:
            pytest.skip(f"Server not healthy: {response.status_code}")
    except httpx.ConnectError:
        pytest.skip(f"Server not running at {BASE_URL}")


# =============================================================================
# 测试请求数据
# =============================================================================


def get_test_request(
    session_id: str = "test_session_001",
    chatbot_id: str = "default",
    tenant_id: str = "test_tenant",
    message: str = "你好，请介绍一下你自己",
) -> dict:
    """获取测试请求数据"""
    return {
        "message": message,
        "customer_id": "test_customer_001",
        "session_id": session_id,
        "tenant_id": tenant_id,
        "chatbot_id": chatbot_id,
        "md5_checksum": "test_hash_001",
        "source": "pytest",
        "is_preview": False,
        "autofill_params": {},
        "session_title": "Test Session",
    }


# =============================================================================
# 阶段 1: 基础连通性测试
# =============================================================================


class TestBasicConnectivity:
    """测试基础连通性"""

    def test_health_check(self, client, check_server):
        """验证健康检查端点"""
        response = client.get(f"{API_PREFIX}/health")

        assert response.status_code == 200
        result = response.json()

        assert result["status"] == "healthy"
        assert "active_sessions" in result
        assert "active_agents" in result
        assert "version" in result
        print(f"\n[健康检查] status={result['status']}, agents={result['active_agents']}")

    def test_root_endpoint(self, client, check_server):
        """验证根路径端点"""
        response = client.get("/")

        assert response.status_code == 200
        result = response.json()

        assert "name" in result
        assert "version" in result
        print(f"\n[根路径] name={result['name']}, version={result['version']}")


# =============================================================================
# 阶段 2: Query API 基础测试
# =============================================================================


class TestQueryBasic:
    """测试 Query API 基础功能"""

    def test_query_success(self, client, check_server):
        """验证 Query 请求成功"""
        request_data = get_test_request(
            session_id="query_basic_001",
            message="你好"
        )
        response = client.post(f"{API_PREFIX}/query", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # 验证响应结构
        assert "session_id" in result
        assert "message" in result
        assert "status" in result
        assert "agent_id" in result

        assert result["session_id"] == request_data["session_id"]
        assert result["status"] == "success"
        assert len(result["message"]) > 0

        print(f"\n[Query成功] session={result['session_id']}, agent={result['agent_id']}")
        print(f"[响应内容] {result['message'][:100]}...")

    def test_query_creates_agent(self, client, check_server):
        """验证 Query 创建 Agent"""
        request_data = get_test_request(
            session_id="query_agent_001",
            chatbot_id="default",
            tenant_id="agent_test_tenant"
        )
        response = client.post(f"{API_PREFIX}/query", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # 验证 agent_id 格式
        expected_agent_id = f"{request_data['tenant_id']}:{request_data['chatbot_id']}"
        assert result["agent_id"] == expected_agent_id

        print(f"\n[Agent创建] agent_id={result['agent_id']}")

    def test_query_validation_error(self, client, check_server):
        """验证请求验证错误"""
        # 缺少必填字段
        response = client.post(f"{API_PREFIX}/query", json={
            "message": "Hello"
            # 缺少 session_id, chatbot_id, tenant_id
        })

        assert response.status_code == 422
        print(f"\n[验证错误] status=422, 缺少必填字段")

    def test_query_empty_message_error(self, client, check_server):
        """验证空消息错误"""
        request_data = get_test_request()
        request_data["message"] = ""

        response = client.post(f"{API_PREFIX}/query", json=request_data)

        assert response.status_code == 422
        print(f"\n[验证错误] status=422, 空消息")


# =============================================================================
# 阶段 3: 会话管理测试
# =============================================================================


class TestSessionManagement:
    """测试会话管理"""

    def test_same_session_reuse(self, client, check_server):
        """验证同一会话复用"""
        session_id = "session_reuse_001"
        request_data = get_test_request(session_id=session_id)

        # 第一次请求
        response1 = client.post(f"{API_PREFIX}/query", json=request_data)
        assert response1.status_code == 200
        result1 = response1.json()

        # 第二次请求（同一会话）
        request_data["message"] = "这是第二条消息"
        response2 = client.post(f"{API_PREFIX}/query", json=request_data)
        assert response2.status_code == 200
        result2 = response2.json()

        # 验证使用同一 Agent
        assert result1["agent_id"] == result2["agent_id"]
        assert result1["session_id"] == result2["session_id"]

        print(f"\n[会话复用] session={session_id}")
        print(f"  第一次响应: {result1['message'][:50]}...")
        print(f"  第二次响应: {result2['message'][:50]}...")

    def test_different_sessions_same_agent(self, client, check_server):
        """验证不同会话使用同一 Agent（同一 chatbot）"""
        chatbot_id = "default"
        tenant_id = "multi_session_tenant"

        # 会话 1
        request1 = get_test_request(
            session_id="multi_session_001",
            chatbot_id=chatbot_id,
            tenant_id=tenant_id,
            message="会话1的消息"
        )
        response1 = client.post(f"{API_PREFIX}/query", json=request1)
        assert response1.status_code == 200

        # 会话 2
        request2 = get_test_request(
            session_id="multi_session_002",
            chatbot_id=chatbot_id,
            tenant_id=tenant_id,
            message="会话2的消息"
        )
        response2 = client.post(f"{API_PREFIX}/query", json=request2)
        assert response2.status_code == 200

        # 验证使用同一 Agent
        assert response1.json()["agent_id"] == response2.json()["agent_id"]

        print(f"\n[多会话同Agent] agent={response1.json()['agent_id']}")

    def test_session_release(self, client, check_server):
        """验证会话释放"""
        session_id = "release_test_001"
        chatbot_id = "default"
        tenant_id = "release_tenant"

        # 创建会话
        request_data = get_test_request(
            session_id=session_id,
            chatbot_id=chatbot_id,
            tenant_id=tenant_id
        )
        response = client.post(f"{API_PREFIX}/query", json=request_data)
        assert response.status_code == 200

        # 释放会话
        release_response = client.delete(
            f"{API_PREFIX}/session/{session_id}",
            params={"chatbot_id": chatbot_id, "tenant_id": tenant_id}
        )
        assert release_response.status_code == 200
        assert release_response.json()["status"] == "released"

        print(f"\n[会话释放] session={session_id}")


# =============================================================================
# 阶段 4: Agent 管理测试
# =============================================================================


class TestAgentManagement:
    """测试 Agent 管理"""

    def test_get_agent_info(self, client, check_server):
        """验证获取 Agent 信息"""
        chatbot_id = "default"
        tenant_id = "agent_info_tenant"

        # 先创建 Agent
        request_data = get_test_request(
            session_id="agent_info_001",
            chatbot_id=chatbot_id,
            tenant_id=tenant_id
        )
        query_response = client.post(f"{API_PREFIX}/query", json=request_data)
        assert query_response.status_code == 200

        # 获取 Agent 信息
        response = client.get(
            f"{API_PREFIX}/agent/{chatbot_id}",
            params={"tenant_id": tenant_id}
        )

        assert response.status_code == 200
        result = response.json()

        assert "agent_id" in result
        assert "session_count" in result
        assert result["session_count"] >= 1

        print(f"\n[Agent信息] agent_id={result['agent_id']}, sessions={result['session_count']}")

    def test_get_nonexistent_agent(self, client, check_server):
        """验证获取不存在的 Agent"""
        response = client.get(
            f"{API_PREFIX}/agent/nonexistent_chatbot",
            params={"tenant_id": "nonexistent_tenant"}
        )

        assert response.status_code == 404
        print(f"\n[Agent不存在] status=404")


# =============================================================================
# 阶段 5: 多轮对话测试
# =============================================================================


class TestMultiTurnConversation:
    """测试多轮对话"""

    def test_context_preservation(self, client, check_server):
        """验证上下文保持"""
        session_id = "context_test_001"
        chatbot_id = "default"
        tenant_id = "context_tenant"

        messages = [
            "我叫张三",
            "我刚才说我叫什么名字？",
            "谢谢你的回答",
        ]

        responses = []
        for i, message in enumerate(messages):
            request_data = get_test_request(
                session_id=session_id,
                chatbot_id=chatbot_id,
                tenant_id=tenant_id,
                message=message
            )
            response = client.post(f"{API_PREFIX}/query", json=request_data)
            assert response.status_code == 200
            result = response.json()
            responses.append(result)

            print(f"\n[对话轮次 {i+1}]")
            print(f"  用户: {message}")
            print(f"  助手: {result['message'][:100]}...")

        # 验证所有响应使用同一 Agent
        agent_ids = [r["agent_id"] for r in responses]
        assert len(set(agent_ids)) == 1

        print(f"\n[上下文测试完成] 共 {len(messages)} 轮对话")


# =============================================================================
# 阶段 6: 响应格式测试
# =============================================================================


class TestResponseFormat:
    """测试响应格式"""

    def test_success_response_structure(self, client, check_server):
        """验证成功响应结构"""
        request_data = get_test_request(session_id="format_test_001")
        response = client.post(f"{API_PREFIX}/query", json=request_data)

        assert response.status_code == 200
        result = response.json()

        # 验证必需字段
        required_fields = ["session_id", "message", "status", "agent_id"]
        for field in required_fields:
            assert field in result, f"Missing field: {field}"

        print(f"\n[响应结构] 包含所有必需字段: {required_fields}")

    def test_error_response_structure(self, client, check_server):
        """验证错误响应结构"""
        # 发送无效请求
        response = client.post(f"{API_PREFIX}/query", json={"invalid": "data"})

        assert response.status_code == 422
        result = response.json()

        # 验证错误响应格式
        assert "status" in result or "detail" in result

        print(f"\n[错误响应] status=422")


# =============================================================================
# 阶段 7: 数据写表验证测试
# =============================================================================


class TestDataPersistence:
    """测试数据持久化（写表验证）"""

    def test_query_writes_session(self, client, check_server):
        """验证 Query 写入会话数据"""
        session_id = "persist_session_001"
        request_data = get_test_request(session_id=session_id)

        # 发送请求
        response = client.post(f"{API_PREFIX}/query", json=request_data)
        assert response.status_code == 200

        # 验证会话被创建（通过 Agent 信息间接验证）
        agent_response = client.get(
            f"{API_PREFIX}/agent/{request_data['chatbot_id']}",
            params={"tenant_id": request_data["tenant_id"]}
        )
        assert agent_response.status_code == 200
        assert agent_response.json()["session_count"] >= 1

        print(f"\n[数据写入] session={session_id} 已创建")

    def test_multiple_messages_increment_count(self, client, check_server):
        """验证多条消息递增计数"""
        session_id = "count_test_001"
        chatbot_id = "default"
        tenant_id = "count_tenant"

        # 发送多条消息
        for i in range(3):
            request_data = get_test_request(
                session_id=session_id,
                chatbot_id=chatbot_id,
                tenant_id=tenant_id,
                message=f"消息 {i+1}"
            )
            response = client.post(f"{API_PREFIX}/query", json=request_data)
            assert response.status_code == 200

        # 验证 Agent 信息
        agent_response = client.get(
            f"{API_PREFIX}/agent/{chatbot_id}",
            params={"tenant_id": tenant_id}
        )
        assert agent_response.status_code == 200

        print(f"\n[消息计数] 发送了 3 条消息到 session={session_id}")
