"""CryptoGuard AI -- shared cryptographic core library."""
from __future__ import annotations

from cosmo_core.anchoring import AppendOnlyLedger, compute_merkle_root
from cosmo_core.conditioning import EntropyPool, hkdf_sha3_256_condition
from cosmo_core.entropy import (
    EntropyMetrics,
    chi_squared,
    compute_all_metrics,
    min_entropy,
    quality_score,
    serial_correlation,
    shannon_entropy,
)
from cosmo_core.extraction import (
    BIN_WIDTH_NS,
    EXTRACTION_EPSILON,
    SECURITY_PARAM_BITS,
    EntropyEstimate,
    compute_inter_arrival_deltas,
    estimate_min_entropy,
    extract_entropy_lhl,
    quantize_deltas,
)
from cosmo_core.fingerprint import sha3_512_fingerprint
from cosmo_core.nist_800_90b import run_nist_90b_sanity_checks
from cosmo_core.taxonomy import (
    Finding,
    Severity,
    VulnerabilityClass,
    VulnerabilityRule,
    TAXONOMY,
    build_taxonomy_prompt,
    scan_code,
)

__all__ = [
    # entropy
    "EntropyMetrics",
    "shannon_entropy",
    "min_entropy",
    "chi_squared",
    "serial_correlation",
    "quality_score",
    "compute_all_metrics",
    # extraction
    "EntropyEstimate",
    "compute_inter_arrival_deltas",
    "quantize_deltas",
    "estimate_min_entropy",
    "extract_entropy_lhl",
    "SECURITY_PARAM_BITS",
    "EXTRACTION_EPSILON",
    "BIN_WIDTH_NS",
    # nist
    "run_nist_90b_sanity_checks",
    # conditioning
    "EntropyPool",
    "hkdf_sha3_256_condition",
    # anchoring
    "compute_merkle_root",
    "AppendOnlyLedger",
    # fingerprint
    "sha3_512_fingerprint",
    # taxonomy
    "VulnerabilityClass",
    "VulnerabilityRule",
    "Severity",
    "Finding",
    "TAXONOMY",
    "build_taxonomy_prompt",
    "scan_code",
]
