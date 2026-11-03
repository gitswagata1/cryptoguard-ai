"""Tests for DL transformer — multi-head self-attention and vulnerability scoring."""
import numpy as np
import pytest

from cosmo_core.dl.attention import (
    CodeVulnerabilityTransformer,
    MultiHeadSelfAttention,
    positional_encoding,
    scaled_dot_product_attention,
)


class TestPositionalEncoding:
    def test_shape(self):
        pe = positional_encoding(100, 64)
        assert pe.shape == (100, 64)

    def test_values_bounded(self):
        pe = positional_encoding(50, 32)
        assert np.all(pe >= -1.0) and np.all(pe <= 1.0)

    def test_different_positions(self):
        pe = positional_encoding(10, 16)
        assert not np.allclose(pe[0], pe[1])


class TestScaledDotProductAttention:
    def test_output_shape(self):
        Q = np.random.randn(2, 5, 8)
        K = np.random.randn(2, 5, 8)
        V = np.random.randn(2, 5, 8)
        out = scaled_dot_product_attention(Q, K, V)
        assert out.shape == (2, 5, 8)

    def test_with_mask(self):
        Q = np.random.randn(1, 4, 8)
        K = np.random.randn(1, 4, 8)
        V = np.random.randn(1, 4, 8)
        mask = np.zeros((1, 1, 4))
        mask[0, 0, 3] = -1e9
        out = scaled_dot_product_attention(Q, K, V, mask)
        assert out.shape == (1, 4, 8)


class TestMultiHeadSelfAttention:
    def test_output_shape(self):
        mhsa = MultiHeadSelfAttention(d_model=32, n_heads=4)
        x = np.random.randn(2, 10, 32)
        out = mhsa.forward(x)
        assert out.shape == (2, 10, 32)

    def test_single_head(self):
        mhsa = MultiHeadSelfAttention(d_model=16, n_heads=1)
        x = np.random.randn(1, 5, 16)
        out = mhsa.forward(x)
        assert out.shape == (1, 5, 16)


class TestCodeVulnerabilityTransformer:
    def test_forward_shape(self):
        model = CodeVulnerabilityTransformer(
            vocab_size=100, d_model=32, n_heads=4, n_layers=1, n_classes=5
        )
        token_ids = np.array([[1, 2, 3, 4, 5]])
        out = model.forward(token_ids)
        assert out.shape == (1, 5)

    def test_output_probabilities(self):
        model = CodeVulnerabilityTransformer(
            vocab_size=200, d_model=16, n_heads=2, n_layers=1, n_classes=3
        )
        token_ids = np.array([[10, 20, 30]])
        out = model.forward(token_ids)
        assert np.all((out >= 0) & (out <= 1))

    def test_batch_processing(self):
        model = CodeVulnerabilityTransformer(
            vocab_size=100, d_model=32, n_heads=4, n_layers=1, n_classes=2
        )
        token_ids = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        out = model.forward(token_ids)
        assert out.shape == (3, 2)

    def test_different_inputs_different_outputs(self):
        model = CodeVulnerabilityTransformer(
            vocab_size=100, d_model=32, n_heads=4, n_layers=1, n_classes=2
        )
        out1 = model.forward(np.array([[1, 2, 3]]))
        out2 = model.forward(np.array([[50, 60, 70]]))
        assert not np.allclose(out1, out2)
