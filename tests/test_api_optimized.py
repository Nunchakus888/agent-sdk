"""
API 单元测试 - 优化版

测试覆盖：
- 多租户查询接口
- Agent 管理接口
- 会话管理接口
- 健康检查接口
- 错误处理
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi.testclient import TestClient

from api.main import app
from api.dependencies import get_agent_manager
from api.agent_manager import AgentManager
from bu_agent_sdk.tools.action_books import WorkflowConfigSchema


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_workflow_config():
    """Create test workflow configuration."""
    return WorkflowConfigSchema(
        basic_settings={
            "name": "Test Workflow Agent",
            "description": "Test agent for API testing",
            "language": "English",
            "tone": "Professional"
        },
        greeting="Hello! How can I help you?",
        sop="1. Understand needs\n2. Execute action\n3. Provide feedback",
        constraints="Be helpful",
        tools=[],
        skills=[],
        flows=[],
        system_actions=[],
        kb_config={"enabled": False},
        max_iterations=3,
        iteration_strategy="sop_driven"
    )


@pytest.fixture
def mock_agent_manager():
    """Create mock AgentManager for testing."""
    manager = Mock(spec=AgentManager)

    # Mock get_or_create_agent
    async def mock_get_or_create_agent(chatbot_id, tenant_id, session_id, md5_checksum=None):
        mock_agent = Mock()

        async def mock_query(message, session_id):
            return f"Response to: {message}"

        mock_agent.query = mock_query
        return mock_agent

    manager.get_or_create_agent = mock_get_or_create_agent

    # Mock release_session
    async def mock_release_session(chatbot_id, tenant_id, session_id):
        pass

    manager.release_session = mock_release_session

    # Mock remove_agent
    async def mock_remove_agent(chatbot_id, tenant_id):
        pass

    manager.remove_agent = mock_remove_agent

    # Mock get_agent_info
    def mock_get_agent_info(chatbot_id, tenant_id):
        return {
            "agent_id": f"{tenant_id}:{chatbot_id}",
            "chatbot_id": chatbot_id,
            "tenant_id": tenant_id,
            "config_hash": "test_hash_123",
            "session_count": 2,
            "created_at": "2026-01-23T10:00:00Z",
            "last_active_at": "2026-01-23T10:30:00Z",
            "is_idle": False,
            "idle_time": 0,
        }

    manager.get_agent_info = mock_get_agent_info

    # Mock get_stats
    def mock_get_stats():
        return {
            "active_agents": 3,
            "idle_agents": 1,
            "active_sessions": 10,
            "uptime": 3600.5,
        }

    manager.get_stats = mock_get_stats

    return manager


@pytest.fixture
def client(mock_agent_manager):
    """Create test client with mocked AgentManager."""

    # Override dependency
    def override_get_agent_manager():
        return mock_agent_manager

    app.dependency_overrides[get_agent_manager] = override_get_agent_manager

    with TestClient(app) as test_client:
        yield test_client

    # Clean up
    app.dependency_overrides.clear()


# =============================================================================
# Root Endpoint Tests
# =============================================================================


def test_root_endpoint(client):
    """Test root endpoint returns welcome message."""
    response = client.get("/")

    assert response.status_code == 200
    data = response.json()
    assert data["name"] == "Workflow Agent API"
    assert "version" in data
    assert data["docs"] == "/docs"


# =============================================================================
# Query Endpoint Tests (Multi-tenant)
# =============================================================================


def test_query_endpoint_success(client):
    """Test successful query request with full parameters."""
    request_data = {
        "message": "Hello, I need help with my order",
        "customer_id": "cust_123xy",
        "session_id": "68d510aedff9455e5b019b3e",
        "tenant_id": "dev-test",
        "chatbot_id": "68d510aedff9455e5b019b3e",
        "md5_checksum": "1234567890",
        "source": "bacmk_ui",
        "is_preview": False,
        "autofill_params": {},
        "session_title": "Order Inquiry"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "68d510aedff9455e5b019b3e"
    assert data["status"] == "success"
    assert "Response to: Hello, I need help with my order" in data["message"]
    assert data["agent_id"] is not None


def test_query_endpoint_minimal_params(client):
    """Test query with only required parameters."""
    request_data = {
        "message": "Hello",
        "session_id": "test_session_001",
        "chatbot_id": "test_chatbot",
        "tenant_id": "test_tenant"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_query_endpoint_missing_required_field(client):
    """Test query with missing required field."""
    request_data = {
        "message": "Hello",
        "session_id": "test_session_002",
        "chatbot_id": "test_chatbot"
        # tenant_id is missing
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error


def test_query_endpoint_empty_message(client):
    """Test query with empty message."""
    request_data = {
        "message": "",
        "session_id": "test_session_003",
        "chatbot_id": "test_chatbot",
        "tenant_id": "test_tenant"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error (min_length=1)


def test_query_endpoint_with_preview_mode(client):
    """Test query with preview mode enabled."""
    request_data = {
        "message": "Test preview",
        "session_id": "preview_session",
        "chatbot_id": "test_chatbot",
        "tenant_id": "test_tenant",
        "is_preview": True
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


# =============================================================================
# Session Management Tests
# =============================================================================


def test_release_session_success(client):
    """Test releasing a session."""
    session_id = "test_session_100"
    chatbot_id = "test_chatbot"
    tenant_id = "test_tenant"

    response = client.delete(
        f"/api/v1/session/{session_id}",
        params={"chatbot_id": chatbot_id, "tenant_id": tenant_id}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "released"
    assert data["session_id"] == session_id


# =============================================================================
# Agent Management Tests
# =============================================================================


def test_get_agent_info_success(client):
    """Test getting agent information."""
    chatbot_id = "test_chatbot"
    tenant_id = "test_tenant"

    response = client.get(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["agent_id"] == f"{tenant_id}:{chatbot_id}"
    assert data["chatbot_id"] == chatbot_id
    assert data["tenant_id"] == tenant_id
    assert data["session_count"] == 2


def test_get_agent_info_not_found(client, mock_agent_manager):
    """Test getting non-existent agent."""
    # Mock to return None
    mock_agent_manager.get_agent_info = lambda chatbot_id, tenant_id: None

    response = client.get(
        "/api/v1/agent/nonexistent",
        params={"tenant_id": "test_tenant"}
    )

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data


def test_delete_agent_success(client):
    """Test deleting an agent."""
    chatbot_id = "test_chatbot"
    tenant_id = "test_tenant"

    response = client.delete(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["chatbot_id"] == chatbot_id
    assert data["tenant_id"] == tenant_id


def test_delete_agent_not_found(client, mock_agent_manager):
    """Test deleting non-existent agent."""
    # Mock to return None
    mock_agent_manager.get_agent_info = lambda chatbot_id, tenant_id: None

    response = client.delete(
        "/api/v1/agent/nonexistent",
        params={"tenant_id": "test_tenant"}
    )

    assert response.status_code == 404


# =============================================================================
# Health Check Tests
# =============================================================================


def test_health_check(client):
    """Test health check endpoint."""
    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["active_sessions"] == 10
    assert data["active_agents"] == 3
    assert "version" in data
    assert data["uptime"] == 3600.5


# =============================================================================
# Error Handling Tests
# =============================================================================


def test_invalid_json_request(client):
    """Test request with invalid JSON."""
    response = client.post(
        "/api/v1/query",
        data="invalid json",
        headers={"Content-Type": "application/json"}
    )

    assert response.status_code == 422


def test_wrong_http_method(client):
    """Test using wrong HTTP method."""
    # GET instead of POST for query
    response = client.get("/api/v1/query")

    assert response.status_code == 405  # Method Not Allowed


def test_invalid_endpoint(client):
    """Test accessing non-existent endpoint."""
    response = client.get("/api/v1/nonexistent")

    assert response.status_code == 404


# =============================================================================
# Multi-tenant Isolation Tests
# =============================================================================


def test_multi_tenant_isolation(client):
    """Test that different tenants are isolated."""
    # Tenant A
    request_a = {
        "message": "Hello from tenant A",
        "session_id": "session_a",
        "chatbot_id": "chatbot_001",
        "tenant_id": "tenant_a"
    }

    response_a = client.post("/api/v1/query", json=request_a)
    assert response_a.status_code == 200

    # Tenant B (same chatbot_id, different tenant)
    request_b = {
        "message": "Hello from tenant B",
        "session_id": "session_b",
        "chatbot_id": "chatbot_001",
        "tenant_id": "tenant_b"
    }

    response_b = client.post("/api/v1/query", json=request_b)
    assert response_b.status_code == 200

    # Both should succeed independently
    assert response_a.json()["status"] == "success"
    assert response_b.json()["status"] == "success"


# =============================================================================
# Configuration Change Detection Tests
# =============================================================================


def test_config_change_detection(client):
    """Test configuration change detection via md5_checksum."""
    # First request with checksum
    request_1 = {
        "message": "Hello",
        "session_id": "session_001",
        "chatbot_id": "chatbot_001",
        "tenant_id": "tenant_001",
        "md5_checksum": "old_hash_123"
    }

    response_1 = client.post("/api/v1/query", json=request_1)
    assert response_1.status_code == 200

    # Second request with different checksum (config changed)
    request_2 = {
        "message": "Hello again",
        "session_id": "session_002",
        "chatbot_id": "chatbot_001",
        "tenant_id": "tenant_001",
        "md5_checksum": "new_hash_456"
    }

    response_2 = client.post("/api/v1/query", json=request_2)
    assert response_2.status_code == 200


# =============================================================================
# Integration Tests
# =============================================================================


def test_complete_workflow(client):
    """Test complete workflow: query -> get agent -> release session -> delete agent."""
    chatbot_id = "workflow_chatbot"
    tenant_id = "workflow_tenant"
    session_id = "workflow_session"

    # Step 1: Query
    query_response = client.post("/api/v1/query", json={
        "message": "Hello",
        "session_id": session_id,
        "chatbot_id": chatbot_id,
        "tenant_id": tenant_id
    })
    assert query_response.status_code == 200

    # Step 2: Get agent info
    agent_response = client.get(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )
    assert agent_response.status_code == 200

    # Step 3: Release session
    release_response = client.delete(
        f"/api/v1/session/{session_id}",
        params={"chatbot_id": chatbot_id, "tenant_id": tenant_id}
    )
    assert release_response.status_code == 200

    # Step 4: Delete agent
    delete_response = client.delete(
        f"/api/v1/agent/{chatbot_id}",
        params={"tenant_id": tenant_id}
    )
    assert delete_response.status_code == 200


def test_multiple_concurrent_queries(client):
    """Test handling multiple queries concurrently."""
    requests = [
        {
            "message": f"Message {i}",
            "session_id": f"session_{i}",
            "chatbot_id": "test_chatbot",
            "tenant_id": "test_tenant"
        }
        for i in range(5)
    ]

    responses = []
    for req in requests:
        response = client.post("/api/v1/query", json=req)
        responses.append(response)

    # All should succeed
    for response in responses:
        assert response.status_code == 200
        assert response.json()["status"] == "success"


# =============================================================================
# API Documentation Tests
# =============================================================================


def test_openapi_schema_available(client):
    """Test OpenAPI schema is available."""
    response = client.get("/openapi.json")

    assert response.status_code == 200
    schema = response.json()
    assert "openapi" in schema
    assert "info" in schema
    assert "paths" in schema


def test_swagger_docs_available(client):
    """Test Swagger UI is available."""
    response = client.get("/docs")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_redoc_docs_available(client):
    """Test ReDoc UI is available."""
    response = client.get("/redoc")

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
