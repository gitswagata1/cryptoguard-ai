"""NIST SP 800-90B sanity checks: runs test, alphabet size, dominant symbol."""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def check_runs(data: NDArray[np.int64]) -> list[str]:
    """Runs test: flag if observed runs deviate significantly from expected."""
    warnings: list[str] = []
    n = data.size
    if n < 128:
        warnings.append(f"runs_test: insufficient samples ({n} < 128)")
        return warnings

    median = float(np.median(data))
    binary = (data > median).astype(np.int8)
    transitions = int(np.sum(binary[:-1] != binary[1:]))
    n_runs = transitions + 1

    n_above = int(np.sum(binary))
    n_below = n - n_above

    if n_above == 0 or n_below == 0:
        warnings.append("runs_test: all values on one side of median")
        return warnings

    expected_runs = 1.0 + (2.0 * n_above * n_below) / n
    variance = (2.0 * n_above * n_below * (2.0 * n_above * n_below - n)) / (
        n * n * (n - 1.0)
    )

    if variance <= 0:
        warnings.append("runs_test: degenerate variance")
        return warnings

    z = abs(n_runs - expected_runs) / (variance ** 0.5)
    if z > 2.576:
        warnings.append(
            f"runs_test: z-score {z:.2f} exceeds 99% threshold (2.576); "
            f"observed {n_runs} runs, expected {expected_runs:.1f}"
        )
    return warnings


def check_alphabet_size(data: NDArray[np.int64], min_symbols: int = 2) -> list[str]:
    """Warn if alphabet size is below the minimum threshold."""
    warnings: list[str] = []
    k = int(np.unique(data).size)
    if k < min_symbols:
        warnings.append(
            f"alphabet_size: only {k} unique symbol(s); need >= {min_symbols}"
        )
    return warnings


def check_dominant_symbol(
    data: NDArray[np.int64],
    max_frequency: float = 0.99,
) -> list[str]:
    """Warn if any single symbol exceeds the maximum frequency threshold."""
    warnings: list[str] = []
    n = data.size
    if n == 0:
        warnings.append("dominant_symbol: empty data")
        return warnings

    _, counts = np.unique(data, return_counts=True)
    freq = float(counts.max()) / n
    if freq > max_frequency:
        warnings.append(
            f"dominant_symbol: most common symbol has frequency {freq:.4f} "
            f"(> {max_frequency})"
        )
    return warnings


def run_nist_90b_sanity_checks(
    data: NDArray[np.int64],
    min_symbols: int = 4,
    max_dominant_frequency: float = 0.5,
) -> list[str]:
    """Run all NIST SP 800-90B sanity checks and return any warnings."""
    warnings: list[str] = []
    warnings.extend(check_runs(data))
    warnings.extend(check_alphabet_size(data, min_symbols=min_symbols))
    warnings.extend(check_dominant_symbol(data, max_frequency=max_dominant_frequency))
    return warnings
