"""Tests for ML vulnerability classifier — feature extraction and gradient boosting."""
import numpy as np
import pytest

from cosmo_core.ml.classifier import FeatureExtractor, VulnerabilityClassifier


class TestFeatureExtractor:
    def test_returns_numpy_vector(self):
        fe = FeatureExtractor()
        features = fe.extract("x = encrypt(key)")
        assert isinstance(features, np.ndarray)
        assert features.ndim == 1

    def test_detects_deprecated_algorithms(self):
        fe = FeatureExtractor()
        features = fe.extract("import hashlib; h = hashlib.md5(data); des = DES.new(key)")
        assert features.sum() > 0

    def test_detects_crypto_apis(self):
        fe = FeatureExtractor()
        features = fe.extract("cipher.encrypt(plaintext); signer.sign(message)")
        assert features.sum() > 0

    def test_empty_code(self):
        fe = FeatureExtractor()
        features = fe.extract("")
        assert isinstance(features, np.ndarray)


class TestVulnerabilityClassifier:
    def test_fit_and_predict(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, 10))
        y = (X[:, 0] + X[:, 1] > 0).astype(float)

        clf = VulnerabilityClassifier(n_estimators=20, learning_rate=0.3)
        clf.fit(X, y)
        preds = clf.predict(X)
        assert preds.shape == (100,)

    def test_predict_proba_range(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((50, 5))
        y = (X[:, 0] > 0).astype(float)

        clf = VulnerabilityClassifier(n_estimators=10, learning_rate=0.5)
        clf.fit(X, y)
        proba = clf.predict_proba(X)
        assert np.all((proba >= 0) & (proba <= 1))

    def test_feature_importance(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((80, 8))
        y = (X[:, 0] > 0).astype(float)

        clf = VulnerabilityClassifier(n_estimators=30, learning_rate=0.2)
        clf.fit(X, y)
        imp = clf.feature_importance()
        assert imp.shape == (8,)
        assert imp.sum() > 0

    def test_unfitted_raises(self):
        clf = VulnerabilityClassifier()
        with pytest.raises(RuntimeError):
            clf.feature_importance()

    def test_partial_fit(self):
        rng = np.random.default_rng(42)
        X = rng.standard_normal((40, 5))
        y = (X[:, 0] > 0).astype(float)

        clf = VulnerabilityClassifier(n_estimators=10)
        clf.fit(X[:20], y[:20])
        clf.partial_fit(X[20:], y[20:])
        preds = clf.predict(X)
        assert preds.shape == (40,)
