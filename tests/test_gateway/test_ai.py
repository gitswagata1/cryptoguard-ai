"""Tests for the AI analysis gateway endpoints."""
import pytest
from httpx import ASGITransport, AsyncClient

from gateway.main import create_app

app = create_app()


@pytest.fixture
async def client():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


class TestTokenize:
    async def test_tokenize_returns_tokens(self, client):
        res = await client.post("/api/ai/tokenize", json={"code": "AES ECB key = 256"})
        assert res.status_code == 200
        data = res.json()
        assert "token_count" in data
        assert data["token_count"] > 0
        assert "crypto_features" in data

    async def test_tokenize_empty_rejected(self, client):
        res = await client.post("/api/ai/tokenize", json={"code": ""})
        assert res.status_code == 422


class TestSemanticScan:
    async def test_semantic_scan_finds_matches(self, client):
        res = await client.post("/api/ai/semantic-scan", json={
            "code": "ECB mode no IV deterministic encryption static key"
        })
        assert res.status_code == 200
        data = res.json()
        assert "matches" in data
        assert data["total_matches"] > 0

    async def test_semantic_scan_clean_code(self, client):
        res = await client.post("/api/ai/semantic-scan", json={
            "code": "def add(a, b): return a + b"
        })
        assert res.status_code == 200


class TestClassify:
    async def test_classify_untrained(self, client):
        res = await client.post("/api/ai/classify", json={"code": "AES encrypt(key)"})
        assert res.status_code == 200
        data = res.json()
        assert data["status"] == "model_not_trained"
        assert "features" in data


class TestEntropyFeed:
    async def test_feed_value(self, client):
        res = await client.post("/api/ai/entropy/feed", json={"value": 7.2})
        assert res.status_code == 200
        data = res.json()
        assert "is_anomaly" in data
        assert "health" in data

    async def test_entropy_health(self, client):
        res = await client.get("/api/ai/entropy/health")
        assert res.status_code == 200
        data = res.json()
        assert "total_samples" in data


class TestScanPolicy:
    async def test_get_ranking(self, client):
        res = await client.get("/api/ai/scan-policy/ranking")
        assert res.status_code == 200
        data = res.json()
        assert "ranking" in data
        assert "strategy" in data

    async def test_select_rules(self, client):
        res = await client.get("/api/ai/scan-policy/select?n=3")
        assert res.status_code == 200
        data = res.json()
        assert "selected_rules" in data
        assert len(data["selected_rules"]) <= 3

    async def test_feedback_unknown_rule(self, client):
        res = await client.post("/api/ai/scan-policy/feedback", json={
            "rule_id": "nonexistent_rule_xyz",
            "found_vulnerability": True,
            "severity": 0.8,
        })
        assert res.status_code == 404


class TestFullAnalysis:
    async def test_full_analysis_returns_report(self, client):
        code = """
import hashlib
password = "admin123"
h = hashlib.md5(password.encode())
from Crypto.Cipher import DES
key = b"mysecret"
cipher = DES.new(key, DES.MODE_ECB)
"""
        res = await client.post("/api/ai/full-analysis", json={"code": code, "format": "text"})
        assert res.status_code == 200
        data = res.json()
        assert data["taxonomy_findings"] > 0
        assert "report" in data
        assert len(data["report"]) > 50
        assert "crypto_features" in data
