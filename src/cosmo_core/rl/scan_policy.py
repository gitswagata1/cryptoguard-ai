"""Adaptive scan-rule prioritisation via Thompson sampling and UCB1.

Treats each vulnerability-detection rule as an arm of a multi-armed
bandit.  The policy learns which rules are most likely to surface real
findings and allocates scanning budget accordingly, balancing
exploration of under-tested rules with exploitation of proven ones.
"""
from __future__ import annotations

from typing import Any

import numpy as np


class ThompsonSampler:
    """Beta–Bernoulli Thompson sampling for binary reward arms."""

    def __init__(self, n_arms: int, seed: int | None = None) -> None:
        self.n_arms = n_arms
        self.alpha = np.ones(n_arms, dtype=np.float64)  # successes + 1
        self.beta = np.ones(n_arms, dtype=np.float64)   # failures + 1
        self.rng = np.random.default_rng(seed)

    def select_arm(self) -> int:
        samples = self.rng.beta(self.alpha, self.beta)
        return int(np.argmax(samples))

    def update(self, arm: int, reward: float) -> None:
        """Update posterior with a reward in [0, 1]."""
        reward = float(np.clip(reward, 0.0, 1.0))
        self.alpha[arm] += reward
        self.beta[arm] += 1.0 - reward

    def get_probabilities(self) -> np.ndarray:
        """Expected reward (posterior mean) for each arm."""
        return self.alpha / (self.alpha + self.beta)

    def decay(self, factor: float = 0.99) -> None:
        """Shrink counts toward the prior to handle non-stationarity."""
        self.alpha = 1.0 + (self.alpha - 1.0) * factor
        self.beta = 1.0 + (self.beta - 1.0) * factor


class UCB1:
    """Upper Confidence Bound (UCB1) arm selection."""

    def __init__(self, n_arms: int, c: float = 2.0) -> None:
        self.n_arms = n_arms
        self.c = c
        self.counts = np.zeros(n_arms, dtype=np.float64)
        self.cumulative_rewards = np.zeros(n_arms, dtype=np.float64)
        self._total: int = 0

    def select_arm(self) -> int:
        # Play each arm once before applying UCB formula.
        unplayed = np.where(self.counts == 0)[0]
        if len(unplayed) > 0:
            return int(unplayed[0])

        means = self.cumulative_rewards / self.counts
        exploration = self.c * np.sqrt(np.log(self._total) / self.counts)
        return int(np.argmax(means + exploration))

    def update(self, arm: int, reward: float) -> None:
        reward = float(np.clip(reward, 0.0, 1.0))
        self.counts[arm] += 1
        self.cumulative_rewards[arm] += reward
        self._total += 1

    def get_probabilities(self) -> np.ndarray:
        safe = np.where(self.counts > 0, self.counts, 1.0)
        return self.cumulative_rewards / safe

    def decay(self, factor: float = 0.99) -> None:
        self.cumulative_rewards *= factor
        self.counts *= factor
        self._total = int(np.sum(self.counts))


class AdaptiveScanPolicy:
    """Bandit-based policy that selects which vulnerability rules to run.

    Each rule is an arm.  After a scan, the caller reports whether the
    rule found something and at what severity; the policy converts this
    into a reward and updates the underlying bandit.

    Parameters
    ----------
    rule_names : list[str]
        Names of vulnerability-detection rules (one per arm).
    strategy : ``"thompson"`` | ``"ucb1"``
        Which bandit algorithm to use.
    seed : int | None
        RNG seed for reproducibility.
    """

    SEVERITY_WEIGHTS = {
        "critical": 1.0,
        "high": 0.8,
        "medium": 0.5,
        "low": 0.2,
        "info": 0.05,
    }

    def __init__(
        self,
        rule_names: list[str],
        strategy: str = "thompson",
        seed: int | None = None,
    ) -> None:
        self.rule_names = list(rule_names)
        self._name_to_idx = {name: i for i, name in enumerate(self.rule_names)}
        n = len(self.rule_names)
        self.strategy_name = strategy

        if strategy == "thompson":
            self._bandit: ThompsonSampler | UCB1 = ThompsonSampler(n, seed=seed)
        elif strategy == "ucb1":
            self._bandit = UCB1(n)
        else:
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self._total_scans = np.zeros(n, dtype=np.int64)
        self._total_finds = np.zeros(n, dtype=np.int64)

    # ------------------------------------------------------------------
    # Rule selection
    # ------------------------------------------------------------------

    def select_rules(self, n: int) -> list[str]:
        """Select the top-*n* rules to apply in the next scan pass.

        Uses repeated sampling (without replacement) to produce a
        ranked shortlist.
        """
        n = min(n, len(self.rule_names))
        probs = self._bandit.get_probabilities()
        selected_indices = np.argsort(probs)[::-1][:n]

        # Re-rank via actual bandit draws so exploration still happens.
        scores = np.full(len(self.rule_names), -np.inf)
        for _ in range(max(n * 5, 20)):
            arm = self._bandit.select_arm()
            scores[arm] = max(scores[arm], probs[arm])

        candidates = np.argsort(scores)[::-1]
        seen: set[int] = set()
        result: list[str] = []
        for idx in candidates:
            if idx not in seen:
                seen.add(idx)
                result.append(self.rule_names[idx])
            if len(result) == n:
                break

        # Fall back to probability ranking if draws didn't cover enough.
        if len(result) < n:
            for idx in selected_indices:
                if idx not in seen:
                    result.append(self.rule_names[idx])
                    seen.add(idx)
                if len(result) == n:
                    break

        return result

    # ------------------------------------------------------------------
    # Feedback
    # ------------------------------------------------------------------

    def report_result(
        self,
        rule_name: str,
        found_vulnerability: bool,
        severity: float | str = 0.5,
    ) -> None:
        """Report the outcome of running a rule.

        Parameters
        ----------
        rule_name : str
            The rule that was run.
        found_vulnerability : bool
            Whether it surfaced a finding.
        severity : float | str
            A float in [0, 1] or a label like ``"critical"``, ``"high"``,
            ``"medium"``, ``"low"``, ``"info"``.
        """
        idx = self._name_to_idx[rule_name]
        self._total_scans[idx] += 1

        if isinstance(severity, str):
            sev_weight = self.SEVERITY_WEIGHTS.get(severity.lower(), 0.5)
        else:
            sev_weight = float(np.clip(severity, 0.0, 1.0))

        reward = sev_weight if found_vulnerability else 0.0
        self._bandit.update(idx, reward)

        if found_vulnerability:
            self._total_finds[idx] += 1

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_priority_ranking(self) -> list[tuple[str, float]]:
        """Return rules ranked by current estimated reward, descending."""
        probs = self._bandit.get_probabilities()
        order = np.argsort(probs)[::-1]
        return [(self.rule_names[i], round(float(probs[i]), 6)) for i in order]

    def get_exploration_stats(self) -> dict[str, Any]:
        """Per-rule scan counts, find counts, and hit rate."""
        stats: dict[str, Any] = {}
        for i, name in enumerate(self.rule_names):
            scans = int(self._total_scans[i])
            finds = int(self._total_finds[i])
            stats[name] = {
                "scans": scans,
                "finds": finds,
                "hit_rate": round(finds / max(scans, 1), 4),
            }
        return stats

    def decay(self, factor: float = 0.99) -> None:
        """Decay bandit statistics to adapt to non-stationary environments."""
        self._bandit.decay(factor)
