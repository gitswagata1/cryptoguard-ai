"""Multi-head self-attention transformer for code vulnerability detection — pure numpy."""
from __future__ import annotations

import numpy as np


def _xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=axis, keepdims=True)


def _layer_norm(x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    mean = np.mean(x, axis=-1, keepdims=True)
    var = np.var(x, axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps)


def positional_encoding(seq_len: int, d_model: int) -> np.ndarray:
    """Sinusoidal positional encoding (Vaswani et al., 2017)."""
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len)[:, np.newaxis]
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))
    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


def scaled_dot_product_attention(
    Q: np.ndarray, K: np.ndarray, V: np.ndarray, mask: np.ndarray | None = None
) -> np.ndarray:
    """Scaled dot-product attention: softmax(QK^T / sqrt(dk)) V."""
    dk = K.shape[-1]
    scores = Q @ K.swapaxes(-2, -1) / np.sqrt(dk)
    if mask is not None:
        scores = np.where(mask == 0, -1e9, scores)
    weights = _softmax(scores, axis=-1)
    return weights @ V


class MultiHeadSelfAttention:
    """Multi-head self-attention with learned projections."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = _xavier_uniform(d_model, d_model)
        self.W_k = _xavier_uniform(d_model, d_model)
        self.W_v = _xavier_uniform(d_model, d_model)
        self.W_o = _xavier_uniform(d_model, d_model)

    def _split_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, seq, d_model) -> (batch, n_heads, seq, d_k)"""
        batch, seq, _ = x.shape
        x = x.reshape(batch, seq, self.n_heads, self.d_k)
        return x.transpose(0, 2, 1, 3)

    def _merge_heads(self, x: np.ndarray) -> np.ndarray:
        """(batch, n_heads, seq, d_k) -> (batch, seq, d_model)"""
        batch, _, seq, _ = x.shape
        x = x.transpose(0, 2, 1, 3)
        return x.reshape(batch, seq, self.d_model)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        """Apply multi-head self-attention.

        Args:
            x: (batch, seq_len, d_model)
            mask: optional (batch, 1, 1, seq_len) or broadcastable
        Returns:
            (batch, seq_len, d_model)
        """
        Q = self._split_heads(x @ self.W_q)
        K = self._split_heads(x @ self.W_k)
        V = self._split_heads(x @ self.W_v)

        attn_out = scaled_dot_product_attention(Q, K, V, mask)
        merged = self._merge_heads(attn_out)
        return merged @ self.W_o


class FeedForward:
    """Position-wise feed-forward network: Linear → ReLU → Linear."""

    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        d_ff = d_ff or d_model * 4
        self.W1 = _xavier_uniform(d_model, d_ff)
        self.b1 = np.zeros(d_ff)
        self.W2 = _xavier_uniform(d_ff, d_model)
        self.b2 = np.zeros(d_model)

    def forward(self, x: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0, x @ self.W1 + self.b1)  # ReLU
        return hidden @ self.W2 + self.b2


class TransformerEncoderLayer:
    """Single transformer encoder block: self-attention + FFN with residual + layer norm."""

    def __init__(self, d_model: int, n_heads: int, d_ff: int | None = None) -> None:
        self.attention = MultiHeadSelfAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)

    def forward(self, x: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        attn_out = self.attention.forward(x, mask)
        x = _layer_norm(x + attn_out)
        ffn_out = self.ffn.forward(x)
        return _layer_norm(x + ffn_out)


class CodeVulnerabilityTransformer:
    """Transformer encoder for classifying cryptographic vulnerability patterns.

    Tokenised code → embedding + positional encoding → N transformer layers
    → mean pooling → linear classifier → per-class vulnerability score.
    """

    def __init__(
        self,
        vocab_size: int,
        n_classes: int = 6,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        max_seq_len: int = 512,
    ) -> None:
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        self.embedding = _xavier_uniform(vocab_size, d_model)
        self.pe = positional_encoding(max_seq_len, d_model)

        self.layers = [
            TransformerEncoderLayer(d_model, n_heads) for _ in range(n_layers)
        ]

        self.cls_weight = _xavier_uniform(d_model, n_classes)
        self.cls_bias = np.zeros(n_classes)

    @staticmethod
    def _sigmoid(x: np.ndarray) -> np.ndarray:
        return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))

    def forward(self, token_ids: np.ndarray) -> np.ndarray:
        """Run a forward pass.

        Args:
            token_ids: (batch, seq_len) integer token indices.
        Returns:
            (batch, n_classes) vulnerability scores in [0, 1].
        """
        batch, seq_len = token_ids.shape
        seq_len = min(seq_len, self.max_seq_len)
        token_ids = token_ids[:, :seq_len]

        x = self.embedding[token_ids] * np.sqrt(self.d_model)
        x = x + self.pe[:seq_len][np.newaxis, :, :]

        padding_mask = (token_ids != 0).astype(np.float32)
        mask = padding_mask[:, np.newaxis, np.newaxis, :]

        for layer in self.layers:
            x = layer.forward(x, mask)

        pooled = (x * padding_mask[:, :, np.newaxis]).sum(axis=1)
        counts = padding_mask.sum(axis=1, keepdims=True).clip(min=1)
        pooled = pooled / counts

        logits = pooled @ self.cls_weight + self.cls_bias
        return self._sigmoid(logits)
