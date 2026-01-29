"""
API 集成测试 - 连接真实运行的服务器

测试覆盖：
- 多租户查询接口
- Agent 管理接口
- 会话管理接口
- 健康检查接口
- 错误处理

使用前请先启动服务器：
    python -m api.main

然后运行测试：
    pytest tests/test_api_optimized.py -v
"""

import pytest
import pytest_asyncio
from httpx import AsyncClient


# =============================================================================
# Configuration
# =============================================================================

# 真实服务器地址（先启动服务器: python -m api.main）
BASE_URL = "http://localhost:8000"


# =============================================================================
# Fixtures
# =============================================================================


@pytest_asyncio.fixture
async def client():
    """Create async HTTP client connecting to real server."""
    async with AsyncClient(base_url=BASE_URL, timeout=60.0) as ac:
        yield ac


# =============================================================================
# Root Endpoint Tests
# =============================================================================


@pytest.mark.asyncio
async def test_root_endpoint(client):
    """Test root endpoint returns welcome message."""
    response = await client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Workflow Agent API"
    assert "version" in data
    assert data["docs"] == "/docs"


# =============================================================================
# Query Endpoint Tests (Multi-tenant)
# =============================================================================


@pytest.mark.asyncio
async def test_query_endpoint_success(client):
    """Test successful query request with full parameters."""
    request_data = {
        "message": "Hello, I need help with my order",
        "customer_id": "cust_123xy",
        "session_id": "test_session_001",
        "tenant_id": "dev-test",
        "chatbot_id": "test_chatbot_001",
        "md5_checksum": "1234567890",
        "source": "bacmk_ui",
        "is_preview": False,
        "autofill_params": {},
        "session_title": "Order Inquiry"
    }

    response = await client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_001"
    assert data["status"] == "success"
    assert data["message"]  # Should have a response
    assert data["agent_id"] is not None


@pytest.mark.asyncio
async def test_query_endpoint_minimal_params(client):
    """Test query with only required parameters."""
    request_data = {
        "message": "Hello",
        "session_id": "test_session_002",
        "chatbot_id": "test_chatbot_002",
        "tenant_id": "test_tenant"
    }

    response = await client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_query_endpoint_missing_required_field(client):
    """Test query with missing required field."""
    request_data = {
        "message": "Hello",
        "session_id": "test_session_003",
        "chatbot_id": "test_chatbot"
        # tenant_id is missing
    }

    response = await client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_query_endpoint_empty_message(client):
    """Test query with empty message."""
    request_data = {
        "message": "",
        "session_id": "test_session_004",
        "chatbot_id": "test_chatbot",
        "tenant_id": "test_tenant"
    }

    response = await client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error (min_length=1)


@pytest.mark.asyncio
async def test_query_endpoint_with_preview_mode(client):
    """Test query with preview mode enabled."""
    request_data = {
        "message": "Test preview",
        "session_id": "preview_session_001",
        "chatbot_id": "test_chatbot_preview",
        "tenant_id": "test_tenant",
        "is_preview": True
    }

    response = await client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


# =============================================================================
# Session Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_release_session_success(client):
    """Test releasing a session after creating it."""
    # First, create a session by querying
    query_request = {
        "message": "Hello",
        "session_id": "session_to_release",
        "chatbot_id": "chatbot_release_test",
        "tenant_id": "tenant_release_test"
    }
    await client.post("/api/v1/query", json=query_request)

    # Now release the session
    response = await client.delete(
        "/api/v1/session/session_to_release",
        params={
            "chatbot_id": "chatbot_release_test",
            "tenant_id": "tenant_release_test"
        }
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "released"
    assert data["session_id"] == "session_to_release"


# =============================================================================
# Agent Management Tests
# =============================================================================


@pytest.mark.asyncio
async def test_get_agent_info_success(client):
    """Test getting agent information after creating it."""
    chatbot_id = "chatbot_info_test"
    tenant_id = "tenant_info_test"

    # First, create an agent by querying
    query_request = {
        "message": "Hello",
        "session_id": "session_for_agent_info",
        "chatbot_id": chatbot_id,
        "tenant_id": tenant_id
    }
    await client.post("/api/v1/query", json=query_request)

    # Now get agent info
    response = await client.get(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == f"{tenant_id}:{chatbot_id}"
    assert data["chatbot_id"] == chatbot_id
    assert data["tenant_id"] == tenant_id
    assert data["session_count"] >= 1


@pytest.mark.asyncio
async def test_get_agent_info_not_found(client):
    """Test getting non-existent agent."""
    response = await client.get(
        "/api/v1/agent/nonexistent_chatbot",
        params={"tenant_id": "nonexistent_tenant"}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


@pytest.mark.asyncio
async def test_delete_agent_success(client):
    """Test deleting an agent after creating it."""
    chatbot_id = "chatbot_delete_test"
    tenant_id = "tenant_delete_test"

    # First, create an agent by querying
    query_request = {
        "message": "Hello",
        "session_id": "session_for_delete",
        "chatbot_id": chatbot_id,
        "tenant_id": tenant_id
    }
    await client.post("/api/v1/query", json=query_request)

    # Now delete the agent
    response = await client.delete(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["chatbot_id"] == chatbot_id
    assert data["tenant_id"] == tenant_id


@pytest.mark.asyncio
async def test_delete_agent_not_found(client):
    """Test deleting non-existent agent."""
    response = await client.delete(
        "/api/v1/agent/nonexistent_chatbot",
        params={"tenant_id": "nonexistent_tenant"}
    )

    assert response.status_code == 404


# =============================================================================
# Health Check Tests
# =============================================================================


@pytest.mark.asyncio
async def test_health_check(client):
    """Test health check endpoint."""
    response = await client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "active_sessions" in data
    assert "active_agents" in data
    assert "version" in data
    assert "uptime" in data


# =============================================================================
# Error Handling Tests
# =============================================================================


@pytest.mark.asyncio
async def test_invalid_json_request(client):
    """Test request with invalid JSON."""
    response = await client.post(
        "/api/v1/query",
        content="invalid json",
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_wrong_http_method(client):
    """Test using wrong HTTP method."""
    # GET instead of POST for query
    response = await client.get("/api/v1/query")

    assert response.status_code == 405  # Method Not Allowed


@pytest.mark.asyncio
async def test_invalid_endpoint(client):
    """Test accessing non-existent endpoint."""
    response = await client.get("/api/v1/nonexistent")

    assert response.status_code == 404


# =============================================================================
# Multi-tenant Isolation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_multi_tenant_isolation(client):
    """Test that different tenants are isolated."""
    # Tenant A
    request_a = {
        "message": "Hello from tenant A",
        "session_id": "session_tenant_a",
        "chatbot_id": "shared_chatbot",
        "tenant_id": "tenant_a"
    }

    response_a = await client.post("/api/v1/query", json=request_a)
    assert response_a.status_code == 200

    # Tenant B (same chatbot_id, different tenant)
    request_b = {
        "message": "Hello from tenant B",
        "session_id": "session_tenant_b",
        "chatbot_id": "shared_chatbot",
        "tenant_id": "tenant_b"
    }

    response_b = await client.post("/api/v1/query", json=request_b)
    assert response_b.status_code == 200

    # Both should succeed independently
    assert response_a.json()["status"] == "success"
    assert response_b.json()["status"] == "success"

    # They should have different agent_ids
    assert response_a.json()["agent_id"] != response_b.json()["agent_id"]


# =============================================================================
# Configuration Change Detection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_config_change_detection(client):
    """Test configuration change detection via md5_checksum."""
    # First request with checksum
    request_1 = {
        "message": "Hello",
        "session_id": "session_config_1",
        "chatbot_id": "chatbot_config_test",
        "tenant_id": "tenant_config_test",
        "md5_checksum": "old_hash_123"
    }

    response_1 = await client.post("/api/v1/query", json=request_1)
    assert response_1.status_code == 200

    # Second request with different checksum (config changed)
    request_2 = {
        "message": "Hello again",
        "session_id": "session_config_2",
        "chatbot_id": "chatbot_config_test",
        "tenant_id": "tenant_config_test",
        "md5_checksum": "new_hash_456"
    }

    response_2 = await client.post("/api/v1/query", json=request_2)
    assert response_2.status_code == 200


# =============================================================================
# Integration Tests
# =============================================================================


@pytest.mark.asyncio
async def test_complete_workflow(client):
    """Test complete workflow: query -> get agent -> release session -> delete agent."""
    chatbot_id = "workflow_chatbot"
    tenant_id = "workflow_tenant"
    session_id = "workflow_session"

    # Step 1: Query
    query_response = await client.post("/api/v1/query", json={
        "message": "Hello",
        "session_id": session_id,
        "chatbot_id": chatbot_id,
        "tenant_id": tenant_id
    })
    assert query_response.status_code == 200
    assert query_response.json()["status"] == "success"

    # Step 2: Get agent info
    agent_response = await client.get(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )
    assert agent_response.status_code == 200
    assert agent_response.json()["session_count"] >= 1

    # Step 3: Release session
    release_response = await client.delete(
        f"/api/v1/session/{session_id}",
        params={"chatbot_id": chatbot_id, "tenant_id": tenant_id}
    )
    assert release_response.status_code == 200

    # Step 4: Delete agent
    delete_response = await client.delete(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )
    assert delete_response.status_code == 200


@pytest.mark.asyncio
async def test_multiple_sequential_queries(client):
    """Test handling multiple queries sequentially."""
    responses = []
    for i in range(5):
        request = {
            "message": f"Message {i}",
            "session_id": f"multi_session_{i}",
            "chatbot_id": "multi_chatbot",
            "tenant_id": "multi_tenant"
        }
        response = await client.post("/api/v1/query", json=request)
        responses.append(response)

    # All should succeed
    for response in responses:
        assert response.status_code == 200
        assert response.json()["status"] == "success"


# =============================================================================
# API Documentation Tests
# =============================================================================


@pytest.mark.asyncio
async def test_openapi_schema_available(client):
    """Test OpenAPI schema is available."""
    response = await client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


@pytest.mark.asyncio
async def test_swagger_docs_available(client):
    """Test Swagger UI is available."""
    response = await client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


@pytest.mark.asyncio
async def test_redoc_docs_available(client):
    """Test ReDoc UI is available."""
    response = await client.get("/redoc")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


# =============================================================================
# Data Model Validation Tests
# =============================================================================


def test_query_request_validation():
    """Test QueryRequest model validation."""
    from api.models import QueryRequest

    # Valid request
    valid_request = QueryRequest(
        message="Test message",
        session_id="session_001",
        chatbot_id="chatbot_001",
        tenant_id="tenant_001",
        customer_id="customer_123"
    )
    assert valid_request.message == "Test message"
    assert valid_request.tenant_id == "tenant_001"

    # Empty message should fail
    with pytest.raises(Exception):  # Pydantic validation error
        QueryRequest(
            message="",
            session_id="session_002",
            chatbot_id="chatbot_001",
            tenant_id="tenant_001"
        )


def test_query_response_model():
    """Test QueryResponse model."""
    from api.models import QueryResponse

    response = QueryResponse(
        session_id="session_001",
        message="Response message",
        status="success",
        agent_id="tenant_001:chatbot_001",
        config_hash="hash_123"
    )

    assert response.session_id == "session_001"
    assert response.agent_id == "tenant_001:chatbot_001"


def test_agent_stats_model():
    """Test AgentStats model."""
    from api.models import AgentStats

    stats = AgentStats(
        agent_id="tenant_001:chatbot_001",
        chatbot_id="chatbot_001",
        tenant_id="tenant_001",
        config_hash="hash_123",
        session_count=5,
        created_at="2026-01-23T10:00:00Z",
        last_active_at="2026-01-23T10:30:00Z"
    )

    assert stats.agent_id == "tenant_001:chatbot_001"
    assert stats.session_count == 5


# =============================================================================
# Real Agent Manager Tests
# =============================================================================


@pytest.mark.asyncio
async def test_agent_manager_stats(client):
    """Test AgentManager statistics are tracked correctly."""
    # Create some agents by querying
    for i in range(3):
        await client.post("/api/v1/query", json={
            "message": f"Hello {i}",
            "session_id": f"stats_session_{i}",
            "chatbot_id": f"stats_chatbot_{i}",
            "tenant_id": "stats_tenant"
        })

    # Check health endpoint shows the stats
    response = await client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["active_agents"] >= 3
    assert data["active_sessions"] >= 3


@pytest.mark.asyncio
async def test_same_session_multiple_queries(client):
    """Test multiple queries in the same session."""
    chatbot_id = "same_session_chatbot"
    tenant_id = "same_session_tenant"
    session_id = "same_session_test"

    # First query
    response_1 = await client.post("/api/v1/query", json={
        "message": "First message",
        "session_id": session_id,
        "chatbot_id": chatbot_id,
        "tenant_id": tenant_id
    })
    assert response_1.status_code == 200

    # Second query in same session
    response_2 = await client.post("/api/v1/query", json={
        "message": "Second message",
        "session_id": session_id,
        "chatbot_id": chatbot_id,
        "tenant_id": tenant_id
    })
    assert response_2.status_code == 200

    # Both should use the same agent
    assert response_1.json()["agent_id"] == response_2.json()["agent_id"]
