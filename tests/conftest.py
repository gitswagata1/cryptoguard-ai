"""Shared fixtures for the unified test suite."""
from __future__ import annotations

import secrets

import numpy as np
import pytest


@pytest.fixture
def rng42():
    """NumPy RNG seeded at 42 for reproducible tests."""
    return np.random.default_rng(42)


@pytest.fixture
def synthetic_timestamps(rng42):
    """4096 synthetic nanosecond timestamps with exponential inter-arrivals."""
    base = 1_700_000_000_000_000_000
    intervals = rng42.exponential(scale=10_000, size=4096).astype(np.int64)
    intervals = np.clip(intervals, 1, None)
    return base + np.cumsum(intervals)


@pytest.fixture
def random_bytes_256():
    """256 cryptographically random bytes."""
    return secrets.token_bytes(256)
