"""Tests for RL Thompson sampling / UCB scan prioritization policy."""
import pytest

from cosmo_core.rl.scan_policy import AdaptiveScanPolicy


class TestAdaptiveScanPolicy:
    def test_select_rules(self):
        policy = AdaptiveScanPolicy(["a", "b", "c"], strategy="thompson")
        selected = policy.select_rules(2)
        assert len(selected) == 2
        assert all(r in ["a", "b", "c"] for r in selected)

    def test_report_result_updates(self):
        policy = AdaptiveScanPolicy(["a", "b"], strategy="thompson")
        policy.report_result("a", True, 0.9)
        policy.report_result("b", False, 0.0)
        ranking = policy.get_priority_ranking()
        assert ranking[0][0] == "a"

    def test_ucb1_strategy(self):
        policy = AdaptiveScanPolicy(["x", "y", "z"], strategy="ucb1")
        selected = policy.select_rules(2)
        assert len(selected) == 2

    def test_priority_ranking(self):
        policy = AdaptiveScanPolicy(["r1", "r2", "r3"], strategy="thompson")
        ranking = policy.get_priority_ranking()
        assert len(ranking) == 3
        assert all(isinstance(score, float) for _, score in ranking)

    def test_exploration_stats(self):
        policy = AdaptiveScanPolicy(["a", "b"], strategy="thompson")
        policy.report_result("a", True, 1.0)
        stats = policy.get_exploration_stats()
        assert isinstance(stats, dict)

    def test_decay(self):
        policy = AdaptiveScanPolicy(["a", "b"], strategy="thompson")
        policy.report_result("a", True, 1.0)
        policy.report_result("a", True, 1.0)
        policy.decay(factor=0.5)
        ranking = policy.get_priority_ranking()
        assert all(score >= 0 for _, score in ranking)

    def test_unknown_rule_raises(self):
        policy = AdaptiveScanPolicy(["a"], strategy="thompson")
        with pytest.raises((KeyError, ValueError)):
            policy.report_result("nonexistent", True, 1.0)

    def test_select_more_than_available(self):
        policy = AdaptiveScanPolicy(["a", "b"], strategy="thompson")
        selected = policy.select_rules(5)
        assert len(selected) <= 2
