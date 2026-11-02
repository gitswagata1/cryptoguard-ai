"""Entropy metrics: Shannon, min-entropy, chi-squared, serial correlation."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

BytesLike = bytes | bytearray | NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class EntropyMetrics:
    """All entropy quality metrics for a byte sequence."""
    shannon_entropy: float
    min_entropy: float
    chi_squared: float
    chi_squared_p_value: float
    serial_correlation: float
    quality_score: int
    sample_size: int


def _to_uint8_array(data: BytesLike) -> NDArray[np.uint8]:
    if isinstance(data, (bytes, bytearray)):
        return np.frombuffer(data, dtype=np.uint8)
    return np.asarray(data, dtype=np.uint8)


def shannon_entropy(data: BytesLike) -> float:
    """Compute Shannon entropy in bits per byte (max 8.0)."""
    arr = _to_uint8_array(data)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256)
    probs = counts[counts > 0] / arr.size
    return -float(np.sum(probs * np.log2(probs)))


def min_entropy(data: BytesLike) -> float:
    """Compute min-entropy: -log2(max probability)."""
    arr = _to_uint8_array(data)
    if arr.size == 0:
        return 0.0
    counts = np.bincount(arr, minlength=256)
    p_max = counts.max() / arr.size
    if p_max == 0:
        return 0.0
    return -math.log2(p_max)


def chi_squared(data: BytesLike) -> tuple[float, float]:
    """Compute chi-squared statistic and p-value for uniform byte distribution."""
    from scipy import stats as sp_stats

    arr = _to_uint8_array(data)
    if arr.size == 0:
        return 0.0, 1.0
    observed = np.bincount(arr, minlength=256).astype(np.float64)
    expected = np.full(256, arr.size / 256.0)
    chi2 = float(np.sum((observed - expected) ** 2 / expected))
    p_value = float(sp_stats.chi2.sf(chi2, df=255))
    return chi2, p_value


def serial_correlation(data: BytesLike) -> float:
    """Compute lag-1 serial correlation coefficient."""
    arr = _to_uint8_array(data).astype(np.float64)
    n = arr.size
    if n < 2:
        return 0.0
    mean = arr.mean()
    centered = arr - mean
    variance = float(np.sum(centered ** 2))
    if variance == 0.0:
        return 0.0
    covariance = float(np.sum(centered[:-1] * centered[1:]))
    return covariance / variance


def quality_score(data: BytesLike) -> float:
    """Map Shannon entropy to a 0-100 quality score (100 = perfect randomness)."""
    h = shannon_entropy(data)
    return min(100.0, max(0.0, (h / 8.0) * 100.0))


def compute_all_metrics(data: BytesLike) -> EntropyMetrics:
    """Compute all entropy metrics for a byte sequence."""
    arr = _to_uint8_array(data)
    h = shannon_entropy(arr)
    h_min = min_entropy(arr)
    chi2, p_val = chi_squared(arr)
    sc = serial_correlation(arr)
    qs = int(quality_score(arr))
    return EntropyMetrics(
        shannon_entropy=h,
        min_entropy=h_min,
        chi_squared=chi2,
        chi_squared_p_value=p_val,
        serial_correlation=sc,
        quality_score=qs,
        sample_size=arr.size,
    )
