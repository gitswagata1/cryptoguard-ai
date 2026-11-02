"""LHL entropy extraction: deltas, quantization, min-entropy, SHAKE256 XOF."""
from __future__ import annotations

import hashlib
import math
import secrets
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

SECURITY_PARAM_BITS: int = 128
EXTRACTION_EPSILON: float = 2.0 ** (-SECURITY_PARAM_BITS)
BIN_WIDTH_NS: int = 1000


@dataclass(frozen=True)
class EntropyEstimate:
    """Result of min-entropy estimation on quantized inter-arrival deltas."""
    min_entropy_per_sample: float
    q_factor: float
    k_eff: int
    n_samples: int
    alphabet_size: int


def compute_inter_arrival_deltas(timestamps_ns: NDArray[np.int64]) -> NDArray[np.int64]:
    """Compute inter-arrival deltas, sorting and filtering zero gaps."""
    if timestamps_ns.size < 2:
        return np.array([], dtype=np.int64)
    sorted_ts = np.sort(timestamps_ns)
    deltas = np.diff(sorted_ts)
    return deltas[deltas > 0]


def quantize_deltas(
    deltas_ns: NDArray[np.int64],
    bin_width: int = BIN_WIDTH_NS,
) -> NDArray[np.int64]:
    """Bin-quantize nanosecond deltas into discrete symbols."""
    return (deltas_ns // bin_width).astype(np.int64)


def estimate_min_entropy(
    symbols: NDArray[np.int64],
) -> EntropyEstimate:
    """NIST SP 800-90B MCV estimator with 99% confidence interval.

    Uses upper bound on p_max to produce a conservative min-entropy estimate.
    Q-factor = H_inf / log2(alphabet_size) measures extraction quality.
    K_eff = n * H_inf - 2*log2(1/epsilon) is the LHL extractable bits.
    """
    n = len(symbols)
    if n == 0:
        return EntropyEstimate(
            min_entropy_per_sample=0.0,
            q_factor=0.0,
            k_eff=0,
            n_samples=0,
            alphabet_size=0,
        )

    unique, counts = np.unique(symbols, return_counts=True)
    alphabet_size = len(unique)

    p_max = float(counts.max()) / n
    z_alpha = 2.576  # 99% confidence
    p_upper = min(1.0, p_max + z_alpha * math.sqrt(p_max * (1 - p_max) / n))

    if p_upper >= 1.0:
        h_inf = 0.0
    else:
        h_inf = -math.log2(p_upper)

    log2_alpha = math.log2(alphabet_size) if alphabet_size > 1 else 1.0
    q_factor = h_inf / log2_alpha if log2_alpha > 0 else 0.0
    k_eff = max(0, int(n * h_inf - 2 * math.log2(1.0 / EXTRACTION_EPSILON)))

    return EntropyEstimate(
        min_entropy_per_sample=h_inf,
        q_factor=q_factor,
        k_eff=k_eff,
        n_samples=n,
        alphabet_size=alphabet_size,
    )


def extract_entropy_lhl(
    deltas_ns: NDArray[np.int64],
    estimate: EntropyEstimate,
    seed: bytes | None = None,
) -> bytes:
    """Extract near-uniform random bits via LHL using SHAKE256 as universal hash.

    seed || raw_deltas -> SHAKE256 XOF -> truncated to floor(K_eff / 8) bytes.
    """
    if seed is None:
        seed = secrets.token_bytes(32)

    output_bytes = max(1, estimate.k_eff // 8)

    raw = deltas_ns.astype(">i8").tobytes()
    shake = hashlib.shake_256(seed + raw)
    return shake.digest(output_bytes)
