"""CryptoGuard AI — Standalone FastAPI backend with NVIDIA Nemotron on Nebius.

This is the minimal single-file version for quick demos. For the full
production gateway with NIST 800-90B pipeline support, see src/gateway/.
"""
from __future__ import annotations

import json
import math
import os

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from pydantic import BaseModel

app = FastAPI(title="CryptoGuard AI")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(
    base_url="https://api.studio.nebius.com/v1/",
    api_key=os.environ.get("NEBIUS_API_KEY", ""),
)

MODEL_ULTRA = "nvidia/llama-3.1-nemotron-ultra-253b-v1"
MODEL_NANO = "nvidia/llama-3.1-nemotron-nano-8b-v1"

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
    has_key = bool(os.environ.get("NEBIUS_API_KEY"))
    return {"status": "ok", "api_configured": has_key, "models": [MODEL_ULTRA, MODEL_NANO]}


@app.post("/api/chat")
async def chat(req: ChatRequest) -> dict:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if req.context:
        messages.append({"role": "system", "content": f"Context: {req.context}"})
    messages.append({"role": "user", "content": req.message})
    response = client.chat.completions.create(
        model=MODEL_ULTRA, messages=messages, temperature=0.3, max_tokens=2000,
    )
    return {"response": response.choices[0].message.content}


@app.post("/api/scan")
async def scan_code(req: ScanRequest) -> dict:
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
    response = client.chat.completions.create(
        model=MODEL_ULTRA, messages=messages, temperature=0.1, max_tokens=3000,
    )
    content = response.choices[0].message.content.strip()
    if content.startswith("```"):
        content = content.split("\n", 1)[1].rsplit("```", 1)[0]
    return json.loads(content)


@app.post("/api/entropy")
async def analyze_entropy(req: EntropyRequest) -> dict:
    try:
        data = bytes.fromhex(req.data)
    except ValueError:
        raise HTTPException(400, "Invalid hex data")

    n = len(data)
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
    chi_sq = sum((c - expected) ** 2 / expected for c in counts)

    mean = sum(data) / n
    num = sum((data[i] - mean) * (data[i + 1] - mean) for i in range(n - 1))
    den = sum((b - mean) ** 2 for b in data)
    serial_corr = num / den if den != 0 else 0.0

    prompt = (
        f"Analyze these entropy metrics for a {n}-byte sample:\n"
        f"- Shannon entropy: {shannon:.4f} bits/byte (max 8.0)\n"
        f"- Min-entropy: {min_ent:.4f} bits/byte\n"
        f"- Chi-squared: {chi_sq:.2f}\n"
        f"- Serial correlation: {serial_corr:.6f} (should be near 0)\n\n"
        f"Give a concise quality assessment (3-4 sentences). Note any concerns."
    )

    response = client.chat.completions.create(
        model=MODEL_NANO,
        messages=[
            {"role": "system", "content": "You are a cryptographic entropy analyst. Be concise and precise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=400,
    )
    quality = min(100, int((shannon / 8) * 100))
    return {
        "metrics": {
            "shannon_entropy": round(shannon, 4),
            "min_entropy": round(min_ent, 4),
            "chi_squared": round(chi_sq, 2),
            "serial_correlation": round(serial_corr, 6),
            "sample_size": n,
        },
        "analysis": response.choices[0].message.content,
        "quality_score": quality,
    }


app.mount("/", StaticFiles(directory="web", html=True), name="web")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
