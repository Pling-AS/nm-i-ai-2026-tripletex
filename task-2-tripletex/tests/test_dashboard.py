import pytest
from fastapi.testclient import TestClient
from tripletex_agent.main import app
from tripletex_agent.config import get_settings, Settings


# Helper to override settings
def override_settings(app_api_key=None):
    def _override():
        return Settings(app_api_key=app_api_key)

    return _override


@pytest.fixture
def client():
    return TestClient(app)


def test_dashboard_no_auth(client):
    """Verify dashboard is accessible without auth when APP_API_KEY is not set."""
    app.dependency_overrides[get_settings] = override_settings(app_api_key=None)
    try:
        response = client.get("/dashboard")
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text

        # API endpoints should also be open
        response = client.get("/api/runs")
        assert response.status_code == 200

        # Raw requests should be open
        response = client.get("/api/raw-requests")
        assert response.status_code == 200

    finally:
        app.dependency_overrides = {}


def test_dashboard_with_auth_success(client):
    """Verify dashboard accepts correct credentials when APP_API_KEY is set."""
    secret = "secret123"
    app.dependency_overrides[get_settings] = override_settings(app_api_key=secret)
    try:
        response = client.get("/dashboard", auth=("admin", secret))
        assert response.status_code == 200
        assert "<!DOCTYPE html>" in response.text

        # API endpoints
        response = client.get("/api/runs", auth=("admin", secret))
        assert response.status_code == 200

        # Raw requests
        response = client.get("/api/raw-requests", auth=("admin", secret))
        assert response.status_code == 200
    finally:
        app.dependency_overrides = {}


def test_dashboard_with_auth_failure(client):
    """Verify dashboard rejects missing or incorrect credentials."""
    secret = "secret123"
    app.dependency_overrides[get_settings] = override_settings(app_api_key=secret)
    try:
        # No auth
        response = client.get("/dashboard")
        assert response.status_code == 401

        response = client.get("/api/raw-requests")
        assert response.status_code == 401

        # Wrong password
        response = client.get("/dashboard", auth=("admin", "wrong"))
        assert response.status_code == 401

        # Wrong username (should not matter for Basic Auth standard but implementation ignores username)
        # Our implementation returns username but checks password.
        response = client.get("/dashboard", auth=("wrong_user", secret))
        assert response.status_code == 200
    finally:
        app.dependency_overrides = {}


def test_raw_request_redaction(client):
    """Verify sensitive fields are redacted from raw request logs."""
    import json
    from tripletex_agent.main import RAW_LOG_PATH

    # Ensure clean state
    if RAW_LOG_PATH.exists():
        RAW_LOG_PATH.unlink()

    sensitive_token = "SENSITIVE_12345"
    payload = {
        "prompt": "Test",
        "tripletex_credentials": {
            "session_token": sensitive_token,
            "base_url": "https://example.com",
        },
    }

    # Make a request that triggers logging (POST /solve)
    # Use override to bypass main API key check if needed,
    # but here we can just set no key or use a key
    app.dependency_overrides[get_settings] = override_settings(app_api_key=None)
    try:
        response = client.post("/solve", json=payload)
        # Response might be 400/422/200 depending on agent logic,
        # but middleware runs regardless.
    finally:
        app.dependency_overrides = {}

    # Check log file
    assert RAW_LOG_PATH.exists()
    found = False
    with RAW_LOG_PATH.open("r") as f:
        for line in f:
            entry = json.loads(line)
            if entry["path"] == "/solve" and entry["method"] == "POST":
                found = True
                body_preview = entry.get("body_preview", "")
                assert sensitive_token not in body_preview
                assert "[REDACTED]" in body_preview

    assert found, "Did not find log entry for /solve request"
