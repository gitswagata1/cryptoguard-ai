"""Tests for the AI chat endpoint."""
import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_chat_returns_response(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat", json={"message": "What is post-quantum cryptography?"})
    assert r.status_code == 200
    body = r.json()
    assert "response" in body
    assert len(body["response"]) > 0
    assert "model_used" in body


@pytest.mark.asyncio
async def test_chat_with_context(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/chat", json={
            "message": "Is this secure?",
            "context": "We use RSA-1024 for key exchange.",
        })
    assert r.status_code == 200
