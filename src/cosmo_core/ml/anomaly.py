"""Entropy stream anomaly detection — Isolation Forest and EWMA-based monitoring."""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field

import numpy as np


def _c(n: int) -> float:
    """Average path length of unsuccessful search in a BST (normalization factor)."""
    if n <= 1:
        return 0.0
    if n == 2:
        return 1.0
    euler = 0.5772156649
    return 2.0 * (math.log(n - 1) + euler) - 2.0 * (n - 1) / n


@dataclass
class _IsolationNode:
    """Internal node of an isolation tree."""
    feature: int = 0
    threshold: float = 0.0
    left: _IsolationNode | _IsolationLeaf | None = None
    right: _IsolationNode | _IsolationLeaf | None = None


@dataclass
class _IsolationLeaf:
    """Leaf node storing the subsample size for path-length correction."""
    size: int = 1


class _IsolationTree:
    """Single isolation tree — recursively partitions data with random splits."""

    def __init__(self, max_depth: int, rng: np.random.Generator) -> None:
        self._max_depth = max_depth
        self._rng = rng
        self.root: _IsolationNode | _IsolationLeaf | None = None

    def fit(self, X: np.ndarray) -> None:
        self.root = self._build(X, depth=0)

    def _build(self, X: np.ndarray, depth: int) -> _IsolationNode | _IsolationLeaf:
        n_samples, n_features = X.shape

        if n_samples <= 1 or depth >= self._max_depth:
            return _IsolationLeaf(size=n_samples)

        feature = int(self._rng.integers(0, n_features))
        col = X[:, feature]
        lo, hi = float(col.min()), float(col.max())

        if lo == hi:
            return _IsolationLeaf(size=n_samples)

        threshold = float(self._rng.uniform(lo, hi))

        left_mask = col < threshold
        right_mask = ~left_mask

        if not left_mask.any() or not right_mask.any():
            return _IsolationLeaf(size=n_samples)

        node = _IsolationNode(feature=feature, threshold=threshold)
        node.left = self._build(X[left_mask], depth + 1)
        node.right = self._build(X[right_mask], depth + 1)
        return node

    def path_length(self, x: np.ndarray) -> float:
        """Compute the path length for a single sample."""
        return self._traverse(x, self.root, 0)

    def _traverse(
        self, x: np.ndarray, node: _IsolationNode | _IsolationLeaf | None, depth: int
    ) -> float:
        if node is None:
            return float(depth)
        if isinstance(node, _IsolationLeaf):
            return depth + _c(node.size)
        if x[node.feature] < node.threshold:
            return self._traverse(x, node.left, depth + 1)
        return self._traverse(x, node.right, depth + 1)


class IsolationForest:
    """Isolation Forest for unsupervised anomaly detection (pure numpy).

    Anomalous points are isolated in fewer splits, yielding shorter average
    path lengths across the ensemble.  Scores are normalised to [0, 1] where
    values closer to 1 indicate anomalies.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_samples: int = 256,
        contamination: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_samples = max_samples
        self.contamination = contamination
        self._rng = np.random.default_rng(seed)
        self._trees: list[_IsolationTree] = []
        self._threshold: float = 0.5
        self._n_train: int = 0

    def fit(self, X: np.ndarray) -> IsolationForest:
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        n = X.shape[0]
        self._n_train = n
        subsample = min(self.max_samples, n)
        max_depth = int(math.ceil(math.log2(max(subsample, 2))))

        self._trees = []
        for _ in range(self.n_estimators):
            idx = self._rng.choice(n, size=subsample, replace=False) if n > subsample else np.arange(n)
            tree = _IsolationTree(max_depth=max_depth, rng=self._rng)
            tree.fit(X[idx])
            self._trees.append(tree)

        scores = self.anomaly_score(X)
        k = max(1, int(self.contamination * n))
        self._threshold = float(np.partition(scores, -k)[-k])
        return self

    def anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """Return anomaly scores in [0, 1]; higher = more anomalous."""
        X = np.asarray(X, dtype=np.float64)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        avg_path = np.zeros(X.shape[0], dtype=np.float64)
        for tree in self._trees:
            for i in range(X.shape[0]):
                avg_path[i] += tree.path_length(X[i])
        avg_path /= len(self._trees)

        norm = _c(self.max_samples if self._n_train > self.max_samples else self._n_train)
        if norm == 0:
            return np.ones(X.shape[0])
        return np.power(2, -avg_path / norm)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return 1 for anomalies, 0 for normal."""
        return (self.anomaly_score(X) >= self._threshold).astype(int)


class EntropyAnomalyDetector:
    """Real-time entropy stream monitor combining z-score, EWMA, and Isolation Forest.

    Feed entropy quality values via ``update()``; the detector tracks trends,
    flags statistical outliers, and (once enough multivariate samples have
    been collected) runs Isolation Forest for richer anomaly detection.
    """

    def __init__(
        self,
        window_size: int = 200,
        z_threshold: float = 3.0,
        ewma_alpha: float = 0.1,
        iforest_min_samples: int = 128,
        seed: int | None = None,
    ) -> None:
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.ewma_alpha = ewma_alpha
        self.iforest_min_samples = iforest_min_samples

        self._window: deque[float] = deque(maxlen=window_size)
        self._ewma: float | None = None
        self._ewma_var: float = 0.0
        self._count: int = 0
        self._anomaly_count: int = 0

        self._mv_buffer: list[np.ndarray] = []
        self._iforest: IsolationForest | None = None
        self._seed = seed

    def update(self, value: float, features: np.ndarray | None = None) -> dict:
        """Ingest a new entropy quality reading.

        Parameters
        ----------
        value : float
            Scalar entropy quality metric (e.g. Shannon entropy in bits).
        features : np.ndarray, optional
            Additional feature vector for multivariate anomaly detection.

        Returns
        -------
        dict with keys: ``is_anomaly``, ``z_score``, ``ewma``, ``iforest_score``.
        """
        self._count += 1
        self._window.append(value)

        # EWMA update
        if self._ewma is None:
            self._ewma = value
        else:
            self._ewma = self.ewma_alpha * value + (1 - self.ewma_alpha) * self._ewma
        diff = value - self._ewma
        self._ewma_var = (1 - self.ewma_alpha) * (self._ewma_var + self.ewma_alpha * diff * diff)

        # Z-score over sliding window
        z_score = 0.0
        z_anomaly = False
        if len(self._window) >= 10:
            arr = np.array(self._window)
            mu = arr.mean()
            sigma = arr.std()
            if sigma > 1e-12:
                z_score = (value - mu) / sigma
                z_anomaly = abs(z_score) > self.z_threshold

        # Isolation Forest (multivariate)
        iforest_score = 0.0
        iforest_anomaly = False
        if features is not None:
            row = np.concatenate([[value], np.asarray(features, dtype=np.float64).ravel()])
            self._mv_buffer.append(row)

            if len(self._mv_buffer) >= self.iforest_min_samples:
                if self._iforest is None or self._count % self.iforest_min_samples == 0:
                    data = np.vstack(self._mv_buffer[-self.window_size * 2:])
                    self._iforest = IsolationForest(
                        n_estimators=50, max_samples=min(256, len(data)), seed=self._seed
                    )
                    self._iforest.fit(data)

                if self._iforest is not None:
                    iforest_score = float(self._iforest.anomaly_score(row.reshape(1, -1))[0])
                    iforest_anomaly = self._iforest.predict(row.reshape(1, -1))[0] == 1

        is_anomaly = z_anomaly or iforest_anomaly
        if is_anomaly:
            self._anomaly_count += 1

        return {
            "is_anomaly": is_anomaly,
            "z_score": float(z_score),
            "ewma": float(self._ewma),
            "ewma_std": float(math.sqrt(max(self._ewma_var, 0))),
            "iforest_score": float(iforest_score),
        }

    def get_health(self) -> dict:
        """Return a summary of the detector's state."""
        arr = np.array(self._window) if self._window else np.array([0.0])
        return {
            "total_samples": self._count,
            "anomaly_count": self._anomaly_count,
            "anomaly_rate": self._anomaly_count / max(self._count, 1),
            "window_mean": float(arr.mean()),
            "window_std": float(arr.std()),
            "ewma": float(self._ewma) if self._ewma is not None else None,
            "iforest_fitted": self._iforest is not None,
        }
