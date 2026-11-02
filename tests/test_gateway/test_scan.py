"""Tests for the code scan endpoint."""
import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import create_app

VULNERABLE_PYTHON = '''
SECRET_KEY = "hardcoded_secret_123"

def hash_password(password):
    return hashlib.md5(password.encode()).hexdigest()
'''


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_scan_returns_findings(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/scan", json={"code": VULNERABLE_PYTHON, "language": "python"})
    assert r.status_code == 200
    body = r.json()
    assert "findings" in body
    assert "score" in body
    assert isinstance(body["findings"], list)
    assert len(body["findings"]) > 0
    assert body["score"] < 100


@pytest.mark.asyncio
async def test_scan_clean_code(app):
    clean = "import secrets\ntoken = secrets.token_hex(32)\n"
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/scan", json={"code": clean, "language": "python"})
    assert r.status_code == 200
    body = r.json()
    assert body["score"] == 100
    assert len(body["findings"]) == 0


@pytest.mark.asyncio
async def test_scan_empty_code(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/scan", json={"code": "", "language": "python"})
    assert r.status_code == 200
