"""Tests for ML anomaly detection — Isolation Forest and EWMA monitoring."""
import numpy as np
import pytest

from cosmo_core.ml.anomaly import EntropyAnomalyDetector, IsolationForest


class TestIsolationForest:
    def test_fit_and_score(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (200, 2))
        forest = IsolationForest(n_estimators=50, max_samples=64, seed=42)
        forest.fit(normal)

        scores = forest.anomaly_score(normal)
        assert scores.shape == (200,)
        assert np.all((scores >= 0) & (scores <= 1))

    def test_anomalies_score_higher(self):
        rng = np.random.default_rng(42)
        normal = rng.normal(0, 1, (200, 2))
        outliers = np.array([[10.0, 10.0], [-10.0, -10.0], [8.0, -8.0]])

        forest = IsolationForest(n_estimators=100, max_samples=128, seed=42)
        forest.fit(normal)

        normal_scores = forest.anomaly_score(normal).mean()
        outlier_scores = forest.anomaly_score(outliers).mean()
        assert outlier_scores > normal_scores

    def test_single_feature(self):
        data = np.random.default_rng(0).normal(0, 1, (100, 1))
        forest = IsolationForest(n_estimators=20, max_samples=50, seed=0)
        forest.fit(data)
        scores = forest.anomaly_score(data)
        assert scores.shape == (100,)


class TestEntropyAnomalyDetector:
    def test_no_anomaly_in_normal_stream(self):
        det = EntropyAnomalyDetector(window_size=50, z_threshold=3.0)
        rng = np.random.default_rng(42)
        anomalies = 0
        for v in rng.uniform(6.5, 7.5, 100):
            r = det.update(float(v))
            if r["is_anomaly"]:
                anomalies += 1
        assert anomalies < 10

    def test_detects_spike_anomaly(self):
        det = EntropyAnomalyDetector(window_size=30, z_threshold=2.5)
        for v in np.linspace(7.0, 7.0, 50):
            det.update(float(v))
        result = det.update(0.5)
        assert result["is_anomaly"]

    def test_ewma_tracks_mean(self):
        det = EntropyAnomalyDetector(window_size=20)
        for _ in range(50):
            det.update(5.0)
        health = det.get_health()
        assert abs(health["ewma"] - 5.0) < 0.5

    def test_health_returns_dict(self):
        det = EntropyAnomalyDetector()
        det.update(7.0)
        h = det.get_health()
        assert "total_samples" in h
        assert "ewma" in h
        assert h["total_samples"] == 1
