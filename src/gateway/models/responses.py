from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Service health check."""

    status: str
    mode: str
    ai_available: bool = False
    pipeline_available: bool = False
    redis_connected: bool = False
    capabilities: list[str]
    models: list[str]


class ChatResponse(BaseModel):
    """AI chat reply."""

    response: str
    model_used: str
    latency_ms: float


class Finding(BaseModel):
    """Single vulnerability finding from a code scan."""

    severity: str
    title: str
    description: str
    line: int | None = None
    fix: str | None = None
    vuln_class: str | None = None


class ScanResponse(BaseModel):
    """Aggregated code-scan results."""

    score: int
    findings: list[Finding]
    summary: str
    model_used: str | None = None


class EntropyMetricsResponse(BaseModel):
    """Raw entropy measurements."""

    shannon_entropy: float
    min_entropy: float
    chi_squared: float
    serial_correlation: float
    sample_size: int
    quality_score: int


class NistCheckResult(BaseModel):
    """Outcome of a single NIST SP 800-22 test."""

    name: str
    passed: bool
    detail: str


class ExtractionResult(BaseModel):
    """LHL extraction output."""

    k_eff: int = 0
    extractable_bytes: int = 0
    q_factor: float = 0.0
    n_samples: int = 0


class EntropyResponse(BaseModel):
    """Full entropy analysis including AI commentary."""

    metrics: EntropyMetricsResponse
    nist_checks: list[NistCheckResult]
    extraction: ExtractionResult | None = None
    fingerprint: str | None = None
    analysis: str
    mode: str


class PipelineStatusResponse(BaseModel):
    """Redis-pipeline connection status."""

    connected: bool
    streams: dict[str, int]


class LedgerEntryResponse(BaseModel):
    """Immutable-ledger batch record."""

    batch_id: int
    merkle_root: str
    n_fingerprints: int
    timestamp: float
    entry_hash: str
