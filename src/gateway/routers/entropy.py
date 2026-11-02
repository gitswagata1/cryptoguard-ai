"""Entropy analysis endpoint — Patent Claims 3-5.

Integrates NIST SP 800-90B entropy estimation, Leftover Hash Lemma extraction
via SHAKE256 XOF, and HKDF-SHA3-256 conditioning. AI interpretation via
Nemotron Nano when available, deterministic summary in local mode.
"""
from __future__ import annotations

import logging
import time

import numpy as np
from fastapi import APIRouter, HTTPException

from cosmo_core.entropy import compute_all_metrics
from cosmo_core.extraction import (
    compute_inter_arrival_deltas,
    estimate_min_entropy,
    extract_entropy_lhl,
    quantize_deltas,
)
from cosmo_core.fingerprint import sha3_512_fingerprint
from cosmo_core.nist_800_90b import run_nist_90b_sanity_checks
from gateway.dependencies import get_ai_client, get_feedback_collector
from gateway.models.requests import EntropyRequest
from gateway.models.responses import (
    EntropyMetricsResponse,
    EntropyResponse,
    NistCheckResult,
)

logger = logging.getLogger("cosmo.gateway.entropy")
router = APIRouter()


@router.post("/api/entropy", response_model=EntropyResponse)
async def analyze_entropy(req: EntropyRequest):
    ai = get_ai_client()
    feedback = get_feedback_collector()
    t0 = time.monotonic()

    try:
        data = bytes.fromhex(req.data)
    except ValueError:
        raise HTTPException(400, "Invalid hex data")

    if len(data) < 16:
        raise HTTPException(400, "Minimum 16 bytes required for analysis")

    metrics = compute_all_metrics(data)

    data_array = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    nist_warnings = run_nist_90b_sanity_checks(data_array)

    nist_checks = []
    check_names = ["runs_test", "alphabet_size", "dominant_symbol"]
    for name in check_names:
        matching = [w for w in nist_warnings if name.replace("_", " ") in w.lower()
                    or name.split("_")[0] in w.lower()]
        nist_checks.append(NistCheckResult(
            name=name,
            passed=len(matching) == 0,
            detail=matching[0] if matching else "passed",
        ))
    if not nist_warnings:
        nist_checks = [NistCheckResult(name="all_checks", passed=True, detail="All NIST SP 800-90B sanity checks passed")]

    symbols = quantize_deltas(data_array, bin_width=1)
    estimate = estimate_min_entropy(symbols)

    extracted = extract_entropy_lhl(data_array, estimate)
    fingerprint = sha3_512_fingerprint(data)

    metrics_resp = EntropyMetricsResponse(
        shannon_entropy=round(metrics.shannon_entropy, 4),
        min_entropy=round(metrics.min_entropy, 4),
        chi_squared=round(metrics.chi_squared, 2),
        serial_correlation=round(metrics.serial_correlation, 6),
        sample_size=metrics.sample_size,
        quality_score=metrics.quality_score,
    )

    model_used = None
    analysis = ""
    if ai and ai.available:
        try:
            analysis = await ai.analyze_entropy({
                "shannon_entropy": metrics.shannon_entropy,
                "min_entropy": metrics.min_entropy,
                "chi_squared": metrics.chi_squared,
                "serial_correlation": metrics.serial_correlation,
                "sample_size": metrics.sample_size,
                "quality_score": metrics.quality_score,
                "k_eff": estimate.k_eff,
                "extracted_bytes": len(extracted),
                "fingerprint": fingerprint.hex()[:32],
            })
            model_used = "nemotron-nano"
        except Exception:
            logger.exception("AI entropy analysis failed")

    if not analysis:
        q = metrics.quality_score
        analysis = (
            f"Entropy quality: {q}%. Shannon entropy: {metrics.shannon_entropy:.3f} bits/byte "
            f"(max 8.0). Min-entropy: {metrics.min_entropy:.3f} bits/byte. "
            f"Chi-squared: {metrics.chi_squared:.1f}. "
            f"Serial correlation: {metrics.serial_correlation:.5f}. "
            f"LHL extractable: {estimate.k_eff} bits ({len(extracted)} bytes). "
            f"SHA3-512 fingerprint: {fingerprint.hex()[:32]}..."
        )
        if q > 85:
            analysis += " Source suitable for cryptographic key generation."
        elif q > 60:
            analysis += " Consider HKDF-SHA3-256 conditioning before use as key material."
        else:
            analysis += " NOT suitable for cryptographic use without LHL extraction and conditioning."

    latency_ms = (time.monotonic() - t0) * 1000

    if feedback:
        from gateway.rl.feedback import InferenceRecord
        feedback.record(InferenceRecord(
            request_type="entropy",
            input_hash=str(hash(req.data[:64])),
            model_used=model_used or "local",
            routing_decision="cloud" if model_used else "local",
            response_time_ms=latency_ms,
            output_hash=str(metrics.quality_score),
            timestamp=time.time(),
        ))

    return EntropyResponse(
        metrics=metrics_resp,
        nist_checks=nist_checks,
        analysis=analysis,
        mode="cloud" if model_used else "local",
    )
