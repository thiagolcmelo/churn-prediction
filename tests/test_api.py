"""Tests for the FastAPI app."""

from collections.abc import Generator

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> Generator[TestClient, None, None]:
    """Create a test client for the API."""
    from src.api.main import app

    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# /health endpoint
# ---------------------------------------------------------------------------


def test_health_returns_200(client: TestClient) -> None:
    """GET /health should return HTTP 200."""
    assert client.get("/health").status_code == 200


def test_health_response_body(client: TestClient) -> None:
    """GET /health should return the expected status and model name."""
    body = client.get("/health").json()
    assert body["status"] == "healthy"
    assert body["model"] is not None
