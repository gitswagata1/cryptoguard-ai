"""CryptoGuard AI — Standalone FastAPI backend with NVIDIA Nemotron on Nebius.

This is the minimal single-file version for quick demos. For the full
production gateway with NIST 800-90B pipeline support, see src/gateway/.

Runs in two modes:
- Cloud mode: set NEBIUS_API_KEY for Nemotron-powered deep analysis
- Local mode: works offline with deterministic cosmo_core analysis
"""
from __future__ import annotations

import importlib
import json
import logging
import math
import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logging.basicConfig(level=os.environ.get("CG_LOG_LEVEL", "INFO").upper())
logger = logging.getLogger("cryptoguard")

NEBIUS_API_KEY = os.environ.get("NEBIUS_API_KEY", "") or os.environ.get("CG_NEBIUS_API_KEY", "")
NIM_BASE_URL = os.environ.get("CG_NIM_BASE_URL", "https://api.studio.nebius.com/v1/")
MODEL_ULTRA = os.environ.get("CG_MODEL_ULTRA", "nvidia/llama-3.1-nemotron-ultra-253b-v1")
MODEL_NANO = os.environ.get("CG_MODEL_NANO", "nvidia/llama-3.1-nemotron-nano-8b-v1")

client = None
if NEBIUS_API_KEY:
    from openai import OpenAI
    client = OpenAI(base_url=NIM_BASE_URL, api_key=NEBIUS_API_KEY)
    logger.info("Cloud mode: Nemotron Ultra + Nano via Nebius AI Cloud")
else:
    logger.info("Local mode: running with deterministic cosmo_core analysis")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
_taxonomy = None
try:
    _taxonomy = importlib.import_module("cosmo_core.taxonomy")
except ImportError:
    logger.warning("cosmo_core.taxonomy not found — local scan disabled")

SYSTEM_PROMPT = (
    "You are CryptoGuard AI, an expert cryptographic security analyst. "
    "You specialize in identifying cryptographic vulnerabilities in code, "
    "analyzing entropy quality and randomness, auditing key management practices, "
    "recommending post-quantum secure alternatives, and explaining complex "
    "cryptographic concepts clearly.\n\n"
    "When analyzing code, check for:\n"
    "1. Hardcoded keys/secrets\n"
    "2. Weak algorithms (MD5, SHA1, DES, RC4, 3DES)\n"
    "3. Insecure random number generation\n"
    "4. Improper key derivation (no salt, low iterations, weak KDF)\n"
    "5. Missing authentication/integrity checks (no AEAD, no HMAC)\n"
    "6. Vulnerable modes (ECB, CBC without HMAC)\n"
    "7. Non-quantum-resistant algorithms\n"
    "8. Timing-unsafe comparisons\n"
    "9. Predictable IVs/nonces\n"
    "10. Insufficient key lengths\n\n"
    "Format findings with severity: CRITICAL, HIGH, MEDIUM, LOW, INFO. "
    "Provide specific line references and actionable fix recommendations."
)

KNOWLEDGE_BASE = {
    "quantum": (
        "Post-quantum cryptography uses algorithms resistant to quantum computer attacks. "
        "NIST has standardized ML-KEM (FIPS 203, based on CRYSTALS-Kyber) for key encapsulation "
        "and ML-DSA (FIPS 204, based on CRYSTALS-Dilithium) for digital signatures. "
        "Migration steps: 1) Inventory all crypto usage, 2) Identify quantum-vulnerable algorithms "
        "(RSA, ECDSA, ECDH, DH), 3) Implement hybrid schemes during transition, "
        "4) Switch to ML-KEM/ML-DSA once libraries are mature."
    ),
    "md5": (
        "MD5 is cryptographically broken. Collision attacks run in seconds on commodity hardware. "
        "For passwords, use Argon2id (memory-hard, side-channel resistant), bcrypt (cost >= 12), "
        "or scrypt. For integrity, use SHA-3-256 or BLAKE2b."
    ),
    "aes": (
        "AES-GCM is the recommended symmetric cipher. Use 256-bit keys for post-quantum safety. "
        "96-bit nonces must NEVER repeat per key. For nonce-misuse resistance, use AES-GCM-SIV "
        "or ChaCha20-Poly1305. Never use ECB mode — it leaks plaintext patterns."
    ),
    "password": (
        "Password storage: use Argon2id (64MB memory, 3 iterations, parallelism 4) with a unique "
        "16+ byte salt per user. Never use SHA-256/MD5 alone — they're too fast for password hashing. "
        "Consider adding a pepper: HMAC(server_secret, password) before hashing."
    ),
    "tls": (
        "TLS 1.3 reduces the handshake to 1-RTT with mandatory perfect forward secrecy (ECDHE). "
        "Only 5 cipher suites allowed. Recommended: TLS_AES_256_GCM_SHA384. "
        "Disable TLS 1.0/1.1, enable HSTS with includeSubDomains, and use Certificate Transparency."
    ),
    "nonce": (
        "A nonce (number used once) must never repeat per key. AES-GCM nonce reuse leaks the "
        "GHASH authentication key, allowing forgery of all past messages. Use random 96-bit nonces "
        "(birthday-safe for ~2^32 messages) or monotonic counters. For safety, prefer AES-GCM-SIV."
    ),
    "timing": (
        "Timing attacks exploit execution time differences to leak secrets. Never compare secrets "
        "with == (early exit leaks position). Use hmac.compare_digest() in Python or "
        "crypto.timingSafeEqual() in Node.js. Avoid branching or indexing based on secret data."
    ),
    "entropy": (
        "NIST SP 800-90B defines entropy source validation. Key metrics: Shannon entropy "
        "(max 8.0 bits/byte), min-entropy (conservative worst-case), chi-squared (uniformity), "
        "and serial correlation (pattern detection). Use os.urandom() or secrets module, "
        "never random.random() for security."
    ),
    "cosmo": (
        "The COSMO pipeline has 4 stages: 1) Ingestion captures raw timestamps via Redis Streams, "
        "2) Extraction applies the Leftover Hash Lemma with SHAKE256, 3) Mixing Pool conditions "
        "via HKDF-SHA3-256 with XOR mixing, 4) Anchoring builds SHA3-256 Merkle trees and "
        "writes hash-chained JSONL ledger entries."
    ),
    "hkdf": (
        "HKDF (RFC 5869) is a two-stage KDF: Extract compresses entropy into a PRK via HMAC, "
        "then Expand derives output key material. CryptoGuard uses HKDF-SHA3-256 for entropy "
        "conditioning. SHA-3's sponge construction prevents length-extension attacks."
    ),
    "jwt": (
        "JWT security: always whitelist algorithms (use RS256 or EdDSA, not HS256 with shared "
        "secrets). Set short expiration (15 min access, 7 days refresh). Validate iss, aud, exp "
        "claims. Store refresh tokens server-side, not in localStorage."
    ),
    "default": (
        "CryptoGuard AI is an AI-powered cryptographic security operations center. "
        "It provides vulnerability scanning across 10 weakness classes, NIST SP 800-90B "
        "entropy analysis with Leftover Hash Lemma extraction, and post-quantum migration guidance. "
        "Ask about: post-quantum crypto, AES-GCM, password hashing, TLS 1.3, timing attacks, "
        "nonce safety, JWT security, entropy testing, or the COSMO pipeline."
    ),
}

app = FastAPI(
    title="CryptoGuard AI",
    description="AI-Powered Cryptographic Security Operations Center",
    version="1.0.0",
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str
    context: str | None = None


class ScanRequest(BaseModel):
    code: str
    language: str = "auto"


class EntropyRequest(BaseModel):
    data: str


@app.get("/api/health")
async def health() -> dict:
    mode = "cloud" if client else "local"
    capabilities = ["code_scanning", "entropy_analysis"]
    if client:
        capabilities.extend(["ai_chat", "deep_analysis"])
    return {
        "status": "ok",
        "mode": mode,
        "api_configured": client is not None,
        "models": [MODEL_ULTRA, MODEL_NANO] if client else [],
        "capabilities": capabilities,
    }


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    if client:
        messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
        if req.context:
            messages.append({"role": "system", "content": f"Context: {req.context}"})
        messages.append({"role": "user", "content": req.message})
        t0 = time.monotonic()
        response = client.chat.completions.create(
            model=MODEL_ULTRA, messages=messages, temperature=0.3, max_tokens=2000,
        )
        latency = (time.monotonic() - t0) * 1000
        return {
            "response": response.choices[0].message.content,
            "model_used": MODEL_ULTRA,
            "latency_ms": round(latency, 1),
        }

    msg_lower = req.message.lower()
    reply = KNOWLEDGE_BASE["default"]
    for key, text in KNOWLEDGE_BASE.items():
        if key != "default" and key in msg_lower:
            reply = text
            break
    return {"response": reply, "model_used": None, "latency_ms": 0}


@app.post("/api/scan")
async def scan_code_endpoint(req: ScanRequest) -> dict:
    findings_local = []
    if _taxonomy and req.code.strip():
        raw = _taxonomy.scan_code(req.code, req.language)
        findings_local = [
            {
                "severity": f.severity,
                "title": f.title,
                "description": f.description,
                "line": f.line,
                "fix": f.fix,
                "vuln_class": f.vuln_class.value if hasattr(f.vuln_class, "value") else str(f.vuln_class),
            }
            for f in raw
        ]

    if client and req.code.strip():
        prompt = (
            f"Analyze this code for cryptographic security vulnerabilities.\n"
            f"Return a JSON object with:\n"
            f'- "score": security score 0-100 (100 = perfectly secure)\n'
            f'- "findings": array of {{severity, title, description, line, fix}}\n'
            f'- "summary": one paragraph summary\n\n'
            f"Code ({req.language}):\n```\n{req.code}\n```\n\n"
            f"Respond ONLY with valid JSON, no markdown fences."
        )
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=MODEL_ULTRA, messages=messages, temperature=0.1, max_tokens=3000,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("\n", 1)[1].rsplit("```", 1)[0]
            llm_result = json.loads(content)
            llm_result["model_used"] = MODEL_ULTRA
            if findings_local:
                llm_result.setdefault("findings", []).extend(findings_local)
            return llm_result
        except Exception:
            logger.exception("LLM scan failed, falling back to local")

    SEVERITY_WEIGHTS = {"CRITICAL": 14, "HIGH": 8, "MEDIUM": 4, "LOW": 1, "INFO": 0}
    penalty = sum(SEVERITY_WEIGHTS.get(f["severity"], 0) for f in findings_local)
    score = max(0, round(100 - penalty))
    n = len(findings_local)
    summary = f"{n} vulnerabilit{'y' if n == 1 else 'ies'} detected (rule-based analysis)" if n else "No vulnerabilities detected"
    return {
        "score": score,
        "findings": findings_local,
        "summary": summary,
        "model_used": None,
    }


@app.post("/api/entropy")
async def analyze_entropy(req: EntropyRequest) -> dict:
    try:
        data = bytes.fromhex(req.data)
    except ValueError:
        raise HTTPException(400, "Invalid hex data")

    n = len(data)
    if n < 4:
        raise HTTPException(400, "Sample too short (minimum 4 bytes)")

    counts = [0] * 256
    for b in data:
        counts[b] += 1

    shannon = 0.0
    for c in counts:
        if c > 0:
            p = c / n
            shannon -= p * math.log2(p)

    p_max = max(counts) / n
    min_ent = -math.log2(p_max) if p_max > 0 else 0.0

    expected = n / 256
    chi_sq = sum((c - expected) ** 2 / expected for c in counts) if expected > 0 else 0.0

    mean = sum(data) / n
    num = sum((data[i] - mean) * (data[i + 1] - mean) for i in range(n - 1))
    den = sum((b - mean) ** 2 for b in data)
    serial_corr = num / den if den != 0 else 0.0

    quality = min(100, int((shannon / 8) * 100))

    analysis = (
        f"Entropy quality: {quality}%. "
        f"Shannon entropy is {shannon:.3f} bits/byte (max 8.0), "
        f"min-entropy is {min_ent:.3f} bits/byte. "
    )
    if quality >= 90:
        analysis += "This sample shows strong randomness suitable for cryptographic use."
    elif quality >= 70:
        analysis += "Acceptable entropy but consider conditioning before cryptographic use."
    else:
        analysis += "Low entropy — not suitable for cryptographic key material without conditioning."

    if client:
        prompt = (
            f"Analyze these entropy metrics for a {n}-byte sample:\n"
            f"- Shannon entropy: {shannon:.4f} bits/byte (max 8.0)\n"
            f"- Min-entropy: {min_ent:.4f} bits/byte\n"
            f"- Chi-squared: {chi_sq:.2f}\n"
            f"- Serial correlation: {serial_corr:.6f} (should be near 0)\n\n"
            f"Give a concise quality assessment (3-4 sentences). Note any concerns."
        )
        try:
            response = client.chat.completions.create(
                model=MODEL_NANO,
                messages=[
                    {"role": "system", "content": "You are a cryptographic entropy analyst. Be concise and precise."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                max_tokens=400,
            )
            analysis = response.choices[0].message.content
        except Exception:
            logger.exception("LLM entropy analysis failed, using local assessment")

    return {
        "metrics": {
            "shannon_entropy": round(shannon, 4),
            "min_entropy": round(min_ent, 4),
            "chi_squared": round(chi_sq, 2),
            "serial_correlation": round(serial_corr, 6),
            "sample_size": n,
            "quality_score": quality,
        },
        "analysis": analysis,
        "quality_score": quality,
        "mode": "cloud" if client else "local",
    }


app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
