"""Tests for the health endpoint."""
import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_health_returns_ok(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["mode"] in ("cloud", "local")
    assert isinstance(body["capabilities"], list)
    assert "entropy_analysis" in body["capabilities"]
