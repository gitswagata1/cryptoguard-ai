"""AI analysis endpoints — ML anomaly detection, DL vulnerability scoring,
NLP semantic scanning, and RL-driven adaptive policy feedback."""
from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from cosmo_core.ml.anomaly import EntropyAnomalyDetector
from cosmo_core.ml.classifier import FeatureExtractor, VulnerabilityClassifier
from cosmo_core.nlp.tokenizer import CryptoCodeTokenizer
from cosmo_core.nlp.semantic_scanner import SemanticVulnerabilityScanner
from cosmo_core.nlp.report_generator import ReportFinding, Severity as ReportSeverity, VulnerabilityReportGenerator
from cosmo_core.rl.scan_policy import AdaptiveScanPolicy
from cosmo_core.taxonomy import TAXONOMY, scan_code

logger = logging.getLogger("cosmo.gateway.ai")
router = APIRouter(prefix="/api/ai")

_anomaly_detector = EntropyAnomalyDetector(window_size=200)
_tokenizer = CryptoCodeTokenizer()
_semantic_scanner = SemanticVulnerabilityScanner()
_feature_extractor = FeatureExtractor()
_classifier = VulnerabilityClassifier(n_estimators=50, learning_rate=0.1)
_report_gen = VulnerabilityReportGenerator()

_rule_names = [r.title for r in TAXONOMY]
_scan_policy = AdaptiveScanPolicy(rule_names=_rule_names, strategy="thompson")

_SEVERITY_MAP = {
    "CRITICAL": ReportSeverity.CRITICAL,
    "HIGH": ReportSeverity.HIGH,
    "MEDIUM": ReportSeverity.MEDIUM,
    "LOW": ReportSeverity.LOW,
    "INFO": ReportSeverity.INFO,
}


class CodeAnalysisRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=50_000)
    language: str = Field(default="python")


class EntropyFeedRequest(BaseModel):
    value: float = Field(..., ge=0.0, le=8.0)


class ScanFeedbackRequest(BaseModel):
    rule_id: str
    found_vulnerability: bool
    severity: float = Field(default=0.5, ge=0.0, le=1.0)


class FullAnalysisRequest(BaseModel):
    code: str = Field(..., min_length=1, max_length=50_000)
    fmt: str = Field(default="markdown", alias="format")


@router.post("/tokenize")
async def tokenize_code(req: CodeAnalysisRequest) -> dict[str, Any]:
    tokens = _tokenizer.tokenize(req.code)
    features = _tokenizer.extract_crypto_features(req.code)
    return {
        "token_count": len(tokens),
        "tokens": [{"type": t.type.name, "value": t.value, "pos": t.position} for t in tokens[:200]],
        "crypto_features": features,
    }


@router.post("/semantic-scan")
async def semantic_scan(req: CodeAnalysisRequest) -> dict[str, Any]:
    matches = _semantic_scanner.scan(req.code)
    return {
        "matches": [
            {
                "pattern_id": m.pattern_id,
                "similarity": round(m.similarity_score, 4),
                "severity": m.severity.name,
                "description": m.description,
                "fix": m.fix_suggestion,
            }
            for m in matches
        ],
        "total_matches": len(matches),
    }


@router.post("/classify")
async def classify_vulnerability(req: CodeAnalysisRequest) -> dict[str, Any]:
    features = _feature_extractor.extract(req.code)
    if not _classifier._fitted:
        return {
            "status": "model_not_trained",
            "features": features.tolist(),
            "message": "Classifier needs training data. Use POST /api/ai/train to fit the model.",
        }
    score = float(_classifier.predict(features.reshape(1, -1))[0])
    proba = float(_classifier.predict_proba(features.reshape(1, -1))[0])
    return {
        "severity_score": round(score, 4),
        "vulnerability_probability": round(proba, 4),
        "features": features.tolist(),
        "feature_importance": _classifier.feature_importance().tolist(),
    }


@router.post("/entropy/feed")
async def feed_entropy(req: EntropyFeedRequest) -> dict[str, Any]:
    result = _anomaly_detector.update(req.value)
    health = _anomaly_detector.get_health()
    return {
        "is_anomaly": result["is_anomaly"],
        "z_score": round(result["z_score"], 4) if result["z_score"] is not None else None,
        "ewma": round(result["ewma"], 4) if result["ewma"] is not None else None,
        "health": health,
    }


@router.get("/entropy/health")
async def entropy_health() -> dict[str, Any]:
    return _anomaly_detector.get_health()


@router.get("/scan-policy/ranking")
async def scan_policy_ranking() -> dict[str, Any]:
    ranking = _scan_policy.get_priority_ranking()
    stats = _scan_policy.get_exploration_stats()
    return {
        "strategy": "thompson",
        "ranking": [{"rule": name, "score": round(score, 4)} for name, score in ranking],
        "exploration_stats": stats,
    }


@router.post("/scan-policy/feedback")
async def scan_policy_feedback(req: ScanFeedbackRequest) -> dict[str, Any]:
    try:
        _scan_policy.report_result(req.rule_id, req.found_vulnerability, req.severity)
    except KeyError:
        raise HTTPException(404, f"Unknown rule: {req.rule_id}")
    return {"updated": True, "rule": req.rule_id}


@router.get("/scan-policy/select")
async def scan_policy_select(n: int = 5) -> dict[str, Any]:
    selected = _scan_policy.select_rules(min(n, len(_rule_names)))
    return {"selected_rules": selected, "n": len(selected)}


@router.post("/full-analysis")
async def full_analysis(req: FullAnalysisRequest) -> dict[str, Any]:
    taxonomy_findings = scan_code(req.code)
    semantic_matches = _semantic_scanner.scan(req.code)
    crypto_features = _tokenizer.extract_crypto_features(req.code)

    report_findings: list[ReportFinding] = []
    for f in taxonomy_findings:
        sev = _SEVERITY_MAP.get(f.severity.name, ReportSeverity.MEDIUM)
        report_findings.append(ReportFinding(
            title=f.title,
            severity=sev,
            vulnerability_class=f.vuln_class.value,
            description=f.description,
            fix_suggestion=f.fix,
            line_number=f.line,
        ))
    for m in semantic_matches:
        sev = _SEVERITY_MAP.get(m.severity.name, ReportSeverity.MEDIUM)
        report_findings.append(ReportFinding(
            title=m.pattern_id,
            severity=sev,
            vulnerability_class=m.vulnerability_class,
            description=m.description,
            fix_suggestion=m.fix_suggestion,
            confidence=m.similarity_score,
        ))

    for f in taxonomy_findings:
        try:
            _scan_policy.report_result(f.title, True, 0.8)
        except (KeyError, ValueError):
            pass

    report = _report_gen.generate_report(report_findings, fmt=req.fmt)

    return {
        "taxonomy_findings": len(taxonomy_findings),
        "semantic_matches": len(semantic_matches),
        "crypto_features": crypto_features,
        "report": report,
        "scan_policy_updated": True,
    }
