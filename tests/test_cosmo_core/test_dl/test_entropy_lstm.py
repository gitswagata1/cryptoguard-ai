"""Tests for DL LSTM — entropy quality time-series prediction."""
import numpy as np
import pytest

from cosmo_core.dl.entropy_lstm import EntropyPredictor, LSTMCell


class TestLSTMCell:
    def test_output_shapes(self):
        cell = LSTMCell(input_size=4, hidden_size=8)
        x = np.random.randn(2, 4)
        h = np.zeros((2, 8))
        c = np.zeros((2, 8))
        h_new, c_new = cell.forward(x, h, c)
        assert h_new.shape == (2, 8)
        assert c_new.shape == (2, 8)

    def test_hidden_bounded(self):
        cell = LSTMCell(input_size=3, hidden_size=5)
        x = np.random.randn(1, 3)
        h = np.zeros((1, 5))
        c = np.zeros((1, 5))
        h_new, _ = cell.forward(x, h, c)
        assert np.all(np.abs(h_new) <= 1.0)


class TestEntropyPredictor:
    def test_predict_2d_input(self):
        pred = EntropyPredictor(input_features=4, hidden_size=8, n_layers=1)
        seq = np.random.rand(5, 4)
        result = pred.predict(seq)
        assert result.ndim == 1

    def test_predict_3d_input(self):
        pred = EntropyPredictor(input_features=4, hidden_size=8, n_layers=1)
        seq = np.random.rand(3, 5, 4)
        result = pred.predict(seq)
        assert result.shape == (3, 1)

    def test_different_sequences_different_predictions(self):
        pred = EntropyPredictor(input_features=4, hidden_size=16, n_layers=1)
        seq1 = np.ones((10, 4)) * 0.9
        seq2 = np.ones((10, 4)) * 0.1
        p1 = pred.predict(seq1)
        p2 = pred.predict(seq2)
        assert not np.allclose(p1, p2)

    def test_multi_horizon(self):
        pred = EntropyPredictor(input_features=4, hidden_size=8, n_layers=1, horizon=5)
        seq = np.random.rand(10, 4)
        result = pred.predict(seq)
        assert result.shape == (5,)

    def test_train_step_reduces_loss(self):
        np.random.seed(42)
        pred = EntropyPredictor(input_features=2, hidden_size=8, n_layers=1, horizon=1)
        seq = np.random.rand(1, 5, 2)
        target = np.array([[0.7]])

        loss1 = pred.train_step(seq, target, lr=0.01)
        for _ in range(20):
            loss = pred.train_step(seq, target, lr=0.01)
        assert loss < loss1
