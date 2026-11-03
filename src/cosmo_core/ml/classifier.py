"""Vulnerability severity classifier — feature extraction and gradient-boosted stumps."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

_CRYPTO_API_PATTERNS: dict[str, re.Pattern[str]] = {
    "encrypt": re.compile(r"\b(encrypt|encipher|aes_encrypt|rsa_encrypt)\b", re.I),
    "decrypt": re.compile(r"\b(decrypt|decipher|aes_decrypt|rsa_decrypt)\b", re.I),
    "sign": re.compile(r"\b(sign|dsa_sign|ecdsa_sign|ed25519_sign)\b", re.I),
    "verify": re.compile(r"\b(verify|dsa_verify|ecdsa_verify|ed25519_verify)\b", re.I),
    "hash": re.compile(r"\b(sha256|sha384|sha512|sha3_256|sha3_512|blake2b|md5|sha1)\b", re.I),
    "kdf": re.compile(r"\b(hkdf|pbkdf2|scrypt|argon2|bcrypt)\b", re.I),
    "mac": re.compile(r"\b(hmac|cmac|gmac|poly1305)\b", re.I),
    "random": re.compile(r"\b(urandom|os\.urandom|secrets|getrandom|rand_bytes|csprng)\b", re.I),
    "tls": re.compile(r"\b(ssl|tls|create_default_context|SSLContext)\b", re.I),
    "key_exchange": re.compile(r"\b(dh|ecdh|diffie.hellman|x25519|x448)\b", re.I),
}

_DEPRECATED_ALGORITHMS: dict[str, re.Pattern[str]] = {
    "des": re.compile(r"\bDES\b|triple.?des|3des", re.I),
    "md5": re.compile(r"\bmd5\b", re.I),
    "sha1": re.compile(r"\bsha.?1\b", re.I),
    "rc4": re.compile(r"\brc4\b|arcfour", re.I),
    "blowfish": re.compile(r"\bblowfish\b", re.I),
    "ecb_mode": re.compile(r"\bECB\b|mode_ecb|\.ECB", re.I),
    "cbc_no_mac": re.compile(r"\bCBC\b", re.I),
    "rsa_1024": re.compile(r"\b1024\b.*rsa|rsa.*\b1024\b", re.I),
    "dsa": re.compile(r"\bDSA\b", re.I),
}

_KEY_SIZE_PATTERN = re.compile(r"\b(64|128|192|256|384|512|1024|2048|3072|4096|8192)\b")

FEATURE_NAMES: list[str] = (
    [f"api_{k}" for k in _CRYPTO_API_PATTERNS]
    + [f"deprecated_{k}" for k in _DEPRECATED_ALGORITHMS]
    + ["key_size_min", "key_size_max", "key_size_count"]
    + ["line_count", "max_nesting", "import_count", "func_count", "class_count"]
)
N_FEATURES: int = len(FEATURE_NAMES)


class FeatureExtractor:
    """Extract a fixed-length feature vector from source code for classification."""

    def extract(self, source: str) -> np.ndarray:
        """Return a 1-D feature vector of shape ``(N_FEATURES,)``."""
        features: list[float] = []

        for pattern in _CRYPTO_API_PATTERNS.values():
            features.append(float(len(pattern.findall(source))))

        for pattern in _DEPRECATED_ALGORITHMS.values():
            features.append(float(len(pattern.findall(source))))

        sizes = [int(m) for m in _KEY_SIZE_PATTERN.findall(source)]
        features.append(float(min(sizes)) if sizes else 0.0)
        features.append(float(max(sizes)) if sizes else 0.0)
        features.append(float(len(sizes)))

        lines = source.splitlines()
        features.append(float(len(lines)))

        max_indent = 0
        for line in lines:
            stripped = line.lstrip()
            if stripped:
                indent = len(line) - len(stripped)
                spaces = indent if line[0] == " " else indent * 4
                max_indent = max(max_indent, spaces // 4)
        features.append(float(max_indent))

        features.append(float(len(re.findall(r"^\s*(import |from .+ import )", source, re.M))))
        features.append(float(len(re.findall(r"\bdef\s+\w+", source))))
        features.append(float(len(re.findall(r"\bclass\s+\w+", source))))

        return np.array(features, dtype=np.float64)

    def extract_batch(self, sources: list[str]) -> np.ndarray:
        return np.vstack([self.extract(s) for s in sources])


# ---------------------------------------------------------------------------
# Gradient-boosted decision stumps
# ---------------------------------------------------------------------------

@dataclass
class _DecisionStump:
    """Axis-aligned decision stump — the weak learner."""
    feature: int = 0
    threshold: float = 0.0
    left_value: float = 0.0
    right_value: float = 0.0

    def predict(self, X: np.ndarray) -> np.ndarray:
        out = np.where(X[:, self.feature] <= self.threshold, self.left_value, self.right_value)
        return out


def _fit_stump(X: np.ndarray, residuals: np.ndarray) -> _DecisionStump:
    """Greedy search over all features and candidate thresholds."""
    n_samples, n_features = X.shape
    best_loss = np.inf
    best = _DecisionStump()

    for f in range(n_features):
        col = X[:, f]
        thresholds = np.unique(col)
        if len(thresholds) > 50:
            thresholds = np.percentile(col, np.linspace(0, 100, 50))

        for t in thresholds:
            left_mask = col <= t
            right_mask = ~left_mask
            if not left_mask.any() or not right_mask.any():
                continue
            lv = residuals[left_mask].mean()
            rv = residuals[right_mask].mean()

            preds = np.where(left_mask, lv, rv)
            loss = float(np.sum((residuals - preds) ** 2))
            if loss < best_loss:
                best_loss = loss
                best = _DecisionStump(feature=f, threshold=float(t), left_value=float(lv), right_value=float(rv))
    return best


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-x))


class VulnerabilityClassifier:
    """Gradient-boosted decision stumps for vulnerability severity prediction.

    Fits an additive ensemble of axis-aligned stumps to minimise squared-error
    on severity scores (0.0 = benign, 1.0 = critical).  Supports online
    ``partial_fit`` for continuous learning from new scan results.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self._rng = np.random.default_rng(seed)
        self._stumps: list[_DecisionStump] = []
        self._base_prediction: float = 0.0
        self._fitted = False
        self._feature_importance: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> VulnerabilityClassifier:
        """Train the ensemble on labelled data.

        Parameters
        ----------
        X : array of shape (n_samples, n_features)
        y : array of shape (n_samples,) with values in [0, 1]
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        self._base_prediction = float(y.mean())
        self._stumps = []
        self._feature_importance = np.zeros(X.shape[1], dtype=np.float64)

        predictions = np.full(len(y), self._base_prediction)

        for _ in range(self.n_estimators):
            residuals = y - predictions
            stump = _fit_stump(X, residuals)
            self._stumps.append(stump)
            self._feature_importance[stump.feature] += 1.0
            predictions += self.learning_rate * stump.predict(X)

        total = self._feature_importance.sum()
        if total > 0:
            self._feature_importance /= total
        self._fitted = True
        return self

    def partial_fit(self, X: np.ndarray, y: np.ndarray, n_rounds: int = 10) -> None:
        """Online update — fit additional stumps on new data."""
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()

        if not self._fitted:
            self.fit(X, y)
            return

        predictions = self._raw_predict(X)
        if self._feature_importance is None:
            self._feature_importance = np.zeros(X.shape[1], dtype=np.float64)

        for _ in range(n_rounds):
            residuals = y - predictions
            stump = _fit_stump(X, residuals)
            self._stumps.append(stump)
            self._feature_importance[stump.feature] += 1.0
            predictions += self.learning_rate * stump.predict(X)

        total = self._feature_importance.sum()
        if total > 0:
            self._feature_importance /= total

    def _raw_predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        preds = np.full(X.shape[0], self._base_prediction)
        for stump in self._stumps:
            preds += self.learning_rate * stump.predict(X)
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return severity scores clipped to [0, 1]."""
        return np.clip(self._raw_predict(X), 0.0, 1.0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly probabilities via sigmoid transform."""
        return _sigmoid(self._raw_predict(X))

    def feature_importance(self) -> np.ndarray:
        """Normalised feature importance (fraction of stumps using each feature)."""
        if self._feature_importance is None:
            raise RuntimeError("Classifier has not been fitted yet")
        return self._feature_importance.copy()
