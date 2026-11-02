"""AI chat endpoint — cryptographic security advisor powered by Nemotron."""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter

from gateway.dependencies import get_ai_client, get_feedback_collector
from gateway.models.requests import ChatRequest
from gateway.models.responses import ChatResponse

logger = logging.getLogger("cosmo.gateway.chat")
router = APIRouter()

SYSTEM_PROMPT = """You are CryptoGuard AI, an expert cryptographic security analyst.
You specialize in identifying cryptographic vulnerabilities in code,
analyzing entropy quality and randomness, auditing key management practices,
recommending post-quantum secure alternatives, and explaining complex
cryptographic concepts clearly.

When analyzing code, check for:
1. Hardcoded keys/secrets
2. Weak algorithms (MD5, SHA1, DES, RC4, 3DES)
3. Insecure random number generation (Math.random, random module without secrets)
4. Improper key derivation (no salt, low iterations, weak KDF)
5. Missing authentication/integrity checks (no AEAD, no HMAC)
6. Vulnerable modes (ECB, CBC without HMAC, unauthenticated encryption)
7. Non-quantum-resistant algorithms that should be upgraded
8. Timing-unsafe comparisons
9. Predictable IVs/nonces
10. Insufficient key lengths

Format findings with severity: CRITICAL, HIGH, MEDIUM, LOW, INFO.
Provide specific line references and actionable fix recommendations."""


@router.post("/api/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    ai = get_ai_client()
    feedback = get_feedback_collector()
    t0 = time.monotonic()

    if not ai:
        return ChatResponse(
            response="System initializing. Please try again.",
            model_used="none",
            latency_ms=0,
        )

    response_text, model_used = await ai.chat(
        message=req.message,
        system_prompt=SYSTEM_PROMPT,
        context=req.context,
    )

    latency_ms = (time.monotonic() - t0) * 1000

    if feedback:
        from gateway.rl.feedback import InferenceRecord
        feedback.record(InferenceRecord(
            request_type="chat",
            input_hash=str(hash(req.message)),
            model_used=model_used,
            routing_decision="cloud" if "nemotron" in model_used else "local",
            response_time_ms=latency_ms,
            output_hash=str(hash(response_text[:100])),
            timestamp=time.time(),
        ))

    return ChatResponse(
        response=response_text,
        model_used=model_used,
        latency_ms=round(latency_ms, 1),
    )
