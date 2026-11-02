"""Code vulnerability scanner — Patent Claim 2: 10-class cryptographic vulnerability taxonomy.

Hybrid approach: deterministic rule-based scanning (always runs) + LLM deep
analysis (when cloud mode is available). Findings are merged and deduplicated.
"""
from __future__ import annotations

import logging
import time

from fastapi import APIRouter

from cosmo_core.taxonomy import Finding as CoreFinding, scan_code as rule_scan
from gateway.dependencies import get_ai_client, get_feedback_collector
from gateway.models.requests import ScanRequest
from gateway.models.responses import Finding, ScanResponse

logger = logging.getLogger("cosmo.gateway.scan")
router = APIRouter()


def _core_to_response(f: CoreFinding) -> Finding:
    return Finding(
        severity=f.severity,
        title=f.title,
        description=f.description,
        line=f.line,
        fix=f.fix,
        vuln_class=f.vuln_class.value if f.vuln_class else None,
    )


def _compute_score(findings: list[Finding]) -> int:
    if not findings:
        return 100
    penalty = sum(
        {"CRITICAL": 25, "HIGH": 15, "MEDIUM": 8, "LOW": 3}.get(f.severity, 0)
        for f in findings
    )
    return max(0, 100 - penalty)


def _merge_findings(rule_findings: list[Finding], llm_findings: list[Finding]) -> list[Finding]:
    seen = set()
    merged = []
    for f in rule_findings:
        key = (f.title.lower(), f.line)
        if key not in seen:
            seen.add(key)
            merged.append(f)
    for f in llm_findings:
        key = (f.title.lower(), f.line)
        if key not in seen:
            seen.add(key)
            merged.append(f)
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3, "INFO": 4}
    merged.sort(key=lambda f: severity_order.get(f.severity, 5))
    return merged


@router.post("/api/scan", response_model=ScanResponse)
async def scan_code_endpoint(req: ScanRequest):
    ai = get_ai_client()
    feedback = get_feedback_collector()
    t0 = time.monotonic()

    rule_findings = [_core_to_response(f) for f in rule_scan(req.code, req.language)]

    llm_findings: list[Finding] = []
    model_used = None

    if ai and ai.available:
        try:
            from cosmo_core.taxonomy import build_taxonomy_prompt
            result = await ai.scan_code(req.code, req.language, build_taxonomy_prompt())
            if result and "findings" in result:
                for f in result["findings"]:
                    llm_findings.append(Finding(
                        severity=f.get("severity", "INFO").upper(),
                        title=f.get("title", "Unknown"),
                        description=f.get("description", ""),
                        line=f.get("line"),
                        fix=f.get("fix"),
                        vuln_class=None,
                    ))
                model_used = "nemotron-ultra"
        except (ImportError, ConnectionError, TimeoutError) as exc:
            logger.warning("LLM scan unavailable (%s), using rule-based results only", exc)
        except Exception:
            logger.exception("LLM scan failed, using rule-based results only")

    findings = _merge_findings(rule_findings, llm_findings)
    score = _compute_score(findings)
    latency_ms = (time.monotonic() - t0) * 1000

    summary = f"{len(findings)} vulnerabilities detected" if findings else "No vulnerabilities found"
    if model_used:
        summary += f" (AI-enhanced via {model_used})"
    else:
        summary += " (rule-based analysis)"

    if feedback:
        from gateway.rl.feedback import InferenceRecord
        feedback.record(InferenceRecord(
            request_type="scan",
            input_hash=str(hash(req.code)),
            model_used=model_used or "local",
            routing_decision="cloud" if model_used else "local",
            response_time_ms=latency_ms,
            output_hash=str(hash(str(findings))),
            timestamp=time.time(),
        ))

    return ScanResponse(
        score=score,
        findings=findings,
        summary=summary,
        model_used=model_used,
    )
