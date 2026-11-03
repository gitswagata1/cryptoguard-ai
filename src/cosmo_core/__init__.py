"""CryptoGuard AI -- shared cryptographic core library.

Layers:
  - Core: entropy analysis, LHL extraction, HKDF-SHA3-256 conditioning,
    Merkle anchoring, fingerprinting, NIST 800-90B validation, taxonomy scanner.
  - ML: Isolation Forest anomaly detection, gradient-boosted vulnerability classifier.
  - DL: Multi-head self-attention code analysis, LSTM entropy prediction.
  - NLP: Crypto-aware tokenizer, TF-IDF semantic scanner, report generator.
  - RL: Q-learning entropy pool agent, Thompson sampling scan policy.
"""
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

from cosmo_core.ml.anomaly import EntropyAnomalyDetector, IsolationForest
from cosmo_core.ml.classifier import FeatureExtractor, VulnerabilityClassifier
from cosmo_core.dl.attention import CodeVulnerabilityTransformer, MultiHeadSelfAttention
from cosmo_core.dl.entropy_lstm import EntropyPredictor, LSTMCell
from cosmo_core.nlp.tokenizer import CryptoCodeTokenizer
from cosmo_core.nlp.semantic_scanner import SemanticVulnerabilityScanner
from cosmo_core.nlp.report_generator import VulnerabilityReportGenerator
from cosmo_core.rl.entropy_agent import EntropyPoolAgent
from cosmo_core.rl.scan_policy import AdaptiveScanPolicy

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
    # ml
    "IsolationForest",
    "EntropyAnomalyDetector",
    "FeatureExtractor",
    "VulnerabilityClassifier",
    # dl
    "MultiHeadSelfAttention",
    "CodeVulnerabilityTransformer",
    "LSTMCell",
    "EntropyPredictor",
    # nlp
    "CryptoCodeTokenizer",
    "SemanticVulnerabilityScanner",
    "VulnerabilityReportGenerator",
    # rl
    "EntropyPoolAgent",
    "AdaptiveScanPolicy",
]
