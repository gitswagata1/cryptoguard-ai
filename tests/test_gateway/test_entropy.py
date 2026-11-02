"""Tests for the entropy analysis endpoint."""
import os

import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import create_app


@pytest.fixture
def app():
    return create_app()


@pytest.mark.asyncio
async def test_entropy_with_random_data(app):
    hex_data = os.urandom(256).hex()
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/entropy", json={"data": hex_data})
    assert r.status_code == 200
    body = r.json()
    assert "metrics" in body
    assert "analysis" in body
    assert body["metrics"]["shannon_entropy"] > 7.0
    assert body["metrics"]["quality_score"] > 80


@pytest.mark.asyncio
async def test_entropy_with_biased_data(app):
    hex_data = ("00" * 200 + "ff" * 56)
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/entropy", json={"data": hex_data})
    assert r.status_code == 200
    body = r.json()
    assert body["metrics"]["quality_score"] < 50


@pytest.mark.asyncio
async def test_entropy_rejects_invalid_hex(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/entropy", json={"data": "xyz_not_hex"})
    assert r.status_code == 422


@pytest.mark.asyncio
async def test_entropy_rejects_too_short(app):
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
        r = await client.post("/api/entropy", json={"data": "aabb"})
    assert r.status_code == 400
