"""
Unit tests for FastAPI Web API.

Tests cover:
- Query endpoint
- Session management endpoints
- Health check endpoint
- Error handling
- Dependency injection
"""

import pytest
from unittest.mock import AsyncMock, Mock, patch, MagicMock
from fastapi.testclient import TestClient
from httpx import AsyncClient

from api.main import app
from api.dependencies import get_workflow_agent
from bu_agent_sdk.agent.workflow_agent import WorkflowAgent
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
def mock_workflow_agent(test_workflow_config):
    """Create mock WorkflowAgent for testing."""
    mock_agent = Mock(spec=WorkflowAgent)
    mock_agent.config_hash = "test_hash_123"
    mock_agent._sessions = {}

    # Mock query method
    async def mock_query(message: str, session_id: str):
        return f"Response to: {message}"

    mock_agent.query = mock_query

    return mock_agent


@pytest.fixture
def client(mock_workflow_agent):
    """Create test client with mocked WorkflowAgent."""

    # Override dependency
    def override_get_workflow_agent():
        return mock_workflow_agent

    app.dependency_overrides[get_workflow_agent] = override_get_workflow_agent

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
    assert data["message"] == "Workflow Agent API"
    assert data["version"] is not None
    assert data["docs_url"] == "/docs"


# =============================================================================
# Query Endpoint Tests
# =============================================================================


def test_query_endpoint_success(client):
    """Test successful query request."""
    request_data = {
        "message": "Hello, how are you?",
        "session_id": "test_session_001",
        "user_id": "user_123"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "test_session_001"
    assert data["status"] == "success"
    assert "Response to: Hello, how are you?" in data["message"]


def test_query_endpoint_missing_message(client):
    """Test query with missing message field."""
    request_data = {
        "session_id": "test_session_002"
        # message is missing
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error


def test_query_endpoint_missing_session_id(client):
    """Test query with missing session_id field."""
    request_data = {
        "message": "Hello"
        # session_id is missing
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error


def test_query_endpoint_empty_message(client):
    """Test query with empty message."""
    request_data = {
        "message": "",
        "session_id": "test_session_003"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 422  # Validation error (min_length=1)


def test_query_endpoint_optional_user_id(client):
    """Test query with optional user_id."""
    request_data = {
        "message": "Test message",
        "session_id": "test_session_004",
        "user_id": "user_456"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"


def test_query_endpoint_agent_error(client, mock_workflow_agent):
    """Test query when agent raises an error."""

    # Mock agent to raise exception
    async def mock_query_error(message: str, session_id: str):
        raise ValueError("Test agent error")

    mock_workflow_agent.query = mock_query_error

    request_data = {
        "message": "Test",
        "session_id": "test_session_005"
    }

    response = client.post("/api/v1/query", json=request_data)

    assert response.status_code == 500
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"]


# =============================================================================
# Session Management Tests
# =============================================================================


def test_get_session_success(client, mock_workflow_agent):
    """Test getting existing session."""
    session_id = "test_session_100"

    # Mock session object
    mock_session = Mock()
    mock_session.session_id = session_id
    mock_session.agent_id = "agent_001"
    mock_session.workflow_state = Mock()
    mock_session.workflow_state.config_hash = "hash_123"
    mock_session.workflow_state.need_greeting = False
    mock_session.workflow_state.status = "active"
    mock_session.messages = [{"role": "user", "content": "Hi"}]

    mock_workflow_agent._sessions = {session_id: mock_session}

    response = client.get(f"/api/v1/session/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == session_id
    assert data["agent_id"] == "agent_001"
    assert data["config_hash"] == "hash_123"
    assert data["need_greeting"] is False
    assert data["status"] == "active"
    assert data["message_count"] == 1


def test_get_session_not_found(client, mock_workflow_agent):
    """Test getting non-existent session."""
    mock_workflow_agent._sessions = {}

    response = client.get("/api/v1/session/non_existent_session")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"]
    assert data["detail"]["error"] == "SessionNotFound"


def test_delete_session_success(client, mock_workflow_agent):
    """Test deleting existing session."""
    session_id = "test_session_200"

    # Add session to agent
    mock_workflow_agent._sessions = {session_id: Mock()}

    response = client.delete(f"/api/v1/session/{session_id}")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "deleted"
    assert data["session_id"] == session_id
    assert session_id not in mock_workflow_agent._sessions


def test_delete_session_not_found(client, mock_workflow_agent):
    """Test deleting non-existent session."""
    mock_workflow_agent._sessions = {}

    response = client.delete("/api/v1/session/non_existent_session")

    assert response.status_code == 404
    data = response.json()
    assert "detail" in data
    assert "error" in data["detail"]
    assert data["detail"]["error"] == "SessionNotFound"


def test_delete_session_multiple_sessions(client, mock_workflow_agent):
    """Test deleting one session doesn't affect others."""
    session_1 = "session_001"
    session_2 = "session_002"

    mock_workflow_agent._sessions = {
        session_1: Mock(),
        session_2: Mock()
    }

    # Delete session_1
    response = client.delete(f"/api/v1/session/{session_1}")

    assert response.status_code == 200
    assert session_1 not in mock_workflow_agent._sessions
    assert session_2 in mock_workflow_agent._sessions


# =============================================================================
# Health Check Tests
# =============================================================================


def test_health_check(client, mock_workflow_agent):
    """Test health check endpoint."""
    mock_workflow_agent.config_hash = "test_config_hash"
    mock_workflow_agent._sessions = {
        "session_1": Mock(),
        "session_2": Mock(),
        "session_3": Mock()
    }

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["config_hash"] == "test_config_hash"
    assert data["sessions_count"] == 3
    assert "version" in data


def test_health_check_no_sessions(client, mock_workflow_agent):
    """Test health check with no active sessions."""
    mock_workflow_agent._sessions = {}

    response = client.get("/api/v1/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["sessions_count"] == 0


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
# Integration Tests
# =============================================================================


def test_complete_workflow(client, mock_workflow_agent):
    """Test complete workflow: query -> get session -> delete session."""
    session_id = "workflow_session_001"

    # Step 1: Query
    query_response = client.post("/api/v1/query", json={
        "message": "Hello",
        "session_id": session_id
    })
    assert query_response.status_code == 200

    # Step 2: Add mock session
    mock_session = Mock()
    mock_session.session_id = session_id
    mock_session.agent_id = "agent_001"
    mock_session.workflow_state = Mock(
        config_hash="hash_123",
        need_greeting=False,
        status="active"
    )
    mock_session.messages = [{"role": "user", "content": "Hello"}]
    mock_workflow_agent._sessions = {session_id: mock_session}

    # Step 3: Get session
    get_response = client.get(f"/api/v1/session/{session_id}")
    assert get_response.status_code == 200
    session_data = get_response.json()
    assert session_data["session_id"] == session_id

    # Step 4: Delete session
    delete_response = client.delete(f"/api/v1/session/{session_id}")
    assert delete_response.status_code == 200
    assert session_id not in mock_workflow_agent._sessions


def test_multiple_concurrent_queries(client):
    """Test handling multiple queries concurrently."""
    sessions = [f"session_{i}" for i in range(5)]

    responses = []
    for session_id in sessions:
        response = client.post("/api/v1/query", json={
            "message": f"Message for {session_id}",
            "session_id": session_id
        })
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
# CORS Tests
# =============================================================================


def test_cors_headers(client):
    """Test CORS headers are present."""
    response = client.options("/api/v1/query")

    # CORS headers should be present
    assert "access-control-allow-origin" in response.headers


# =============================================================================
# Dependency Injection Tests
# =============================================================================


@pytest.mark.asyncio
async def test_initialize_agent_creates_singleton():
    """Test that initialize_agent creates singleton instance."""
    from api.dependencies import initialize_agent, _workflow_agent

    with patch('api.dependencies.load_config') as mock_load_config, \
         patch('api.dependencies.WorkflowConfigSchema') as mock_config_schema, \
         patch('api.dependencies.get_llm_decision_llm') as mock_get_llm, \
         patch('api.dependencies.get_session_store_from_config') as mock_get_store, \
         patch('api.dependencies.get_plan_cache_from_config') as mock_get_cache, \
         patch('api.dependencies.WorkflowAgent') as mock_agent_class:

        # Setup mocks
        mock_load_config.return_value = Mock()
        mock_get_llm.return_value = Mock()
        mock_get_store.return_value = Mock()
        mock_get_cache.return_value = Mock()
        mock_agent_instance = Mock()
        mock_agent_class.return_value = mock_agent_instance

        # Initialize agent
        agent = await initialize_agent()

        # Verify singleton
        assert agent is mock_agent_instance
        mock_agent_class.assert_called_once()


def test_get_workflow_agent_raises_when_not_initialized():
    """Test get_workflow_agent raises error when not initialized."""
    from api.dependencies import get_workflow_agent, _workflow_agent

    # Ensure agent is None
    with patch('api.dependencies._workflow_agent', None):
        with pytest.raises(RuntimeError, match="WorkflowAgent not initialized"):
            get_workflow_agent()


# =============================================================================
# Model Validation Tests
# =============================================================================


def test_query_request_model_validation():
    """Test QueryRequest model validation."""
    from api.models import QueryRequest

    # Valid request
    valid_request = QueryRequest(
        message="Test message",
        session_id="session_001",
        user_id="user_123"
    )
    assert valid_request.message == "Test message"
    assert valid_request.session_id == "session_001"
    assert valid_request.user_id == "user_123"

    # Without optional user_id
    minimal_request = QueryRequest(
        message="Test",
        session_id="session_002"
    )
    assert minimal_request.user_id is None

    # Empty message should fail
    with pytest.raises(Exception):  # Pydantic validation error
        QueryRequest(message="", session_id="session_003")


def test_query_response_model():
    """Test QueryResponse model."""
    from api.models import QueryResponse

    response = QueryResponse(
        session_id="session_001",
        message="Response message",
        status="success"
    )

    assert response.session_id == "session_001"
    assert response.message == "Response message"
    assert response.status == "success"


def test_session_info_model():
    """Test SessionInfo model."""
    from api.models import SessionInfo

    session_info = SessionInfo(
        session_id="session_001",
        agent_id="agent_001",
        config_hash="hash_123",
        need_greeting=False,
        status="active",
        message_count=5
    )

    assert session_info.session_id == "session_001"
    assert session_info.agent_id == "agent_001"
    assert session_info.message_count == 5


def test_health_response_model():
    """Test HealthResponse model."""
    from api.models import HealthResponse

    health = HealthResponse(
        status="healthy",
        config_hash="hash_123",
        sessions_count=10,
        version="0.1.0"
    )

    assert health.status == "healthy"
    assert health.sessions_count == 10


def test_error_response_model():
    """Test ErrorResponse model."""
    from api.models import ErrorResponse

    error = ErrorResponse(
        error="ValueError",
        message="Something went wrong",
        details={"field": "value"}
    )

    assert error.error == "ValueError"
    assert error.message == "Something went wrong"
    assert error.details == {"field": "value"}

    # Without optional details
    simple_error = ErrorResponse(
        error="Error",
        message="Message"
    )
    assert simple_error.details is None
