"""Tests for cosmo_core.entropy — Shannon, min-entropy, chi-squared, serial correlation."""
import os

import numpy as np
import pytest

from cosmo_core.entropy import (
    EntropyMetrics,
    chi_squared,
    compute_all_metrics,
    min_entropy,
    quality_score,
    serial_correlation,
    shannon_entropy,
)


class TestShannonEntropy:
    def test_max_entropy_random(self):
        data = os.urandom(10_000)
        h = shannon_entropy(data)
        assert h > 7.9

    def test_zero_entropy_constant(self):
        data = b"\x42" * 1000
        h = shannon_entropy(data)
        assert h == 0.0

    def test_one_bit_entropy_two_symbols(self):
        data = bytes([0, 1] * 500)
        h = shannon_entropy(data)
        assert abs(h - 1.0) < 0.01

    def test_empty_data(self):
        assert shannon_entropy(b"") == 0.0


class TestMinEntropy:
    def test_high_for_random(self):
        data = os.urandom(10_000)
        h = min_entropy(data)
        assert h > 7.0

    def test_zero_for_constant(self):
        data = b"\x00" * 1000
        h = min_entropy(data)
        assert h == 0.0


class TestChiSquared:
    def test_good_random_passes(self):
        data = os.urandom(10_000)
        chi2, p = chi_squared(data)
        assert p > 0.01

    def test_biased_data_fails(self):
        data = bytes([0] * 5000 + [1] * 5000)
        chi2, p = chi_squared(data)
        assert p < 0.01


class TestSerialCorrelation:
    def test_low_for_random(self):
        data = os.urandom(10_000)
        sc = serial_correlation(data)
        assert abs(sc) < 0.05

    def test_high_for_pattern(self):
        data = bytes(range(256)) * 40
        sc = serial_correlation(data)
        assert abs(sc) > 0.5


class TestQualityScore:
    def test_high_for_random(self):
        data = os.urandom(10_000)
        q = quality_score(data)
        assert q > 95

    def test_zero_for_constant(self):
        data = b"\x00" * 1000
        q = quality_score(data)
        assert q == 0.0


class TestComputeAllMetrics:
    def test_returns_complete_metrics(self):
        data = os.urandom(1000)
        m = compute_all_metrics(data)
        assert isinstance(m, EntropyMetrics)
        assert m.shannon_entropy > 0
        assert m.min_entropy > 0
        assert m.chi_squared > 0
        assert m.sample_size == 1000
        assert isinstance(m.quality_score, int)

    def test_accepts_numpy_array(self):
        arr = np.frombuffer(os.urandom(500), dtype=np.uint8)
        m = compute_all_metrics(arr)
        assert m.sample_size == 500
