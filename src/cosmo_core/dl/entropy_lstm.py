"""LSTM-based entropy quality predictor — pure numpy implementation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _xavier_uniform(fan_in: int, fan_out: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (fan_in + fan_out))
    return np.random.uniform(-limit, limit, (fan_in, fan_out))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


def _tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


class LSTMCell:
    """Single LSTM cell with forget, input, candidate, and output gates."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size

        concat_size = input_size + hidden_size

        self.W_f = _xavier_uniform(concat_size, hidden_size)
        self.b_f = np.ones(hidden_size)  # bias-init to 1 to avoid vanishing gradients early on

        self.W_i = _xavier_uniform(concat_size, hidden_size)
        self.b_i = np.zeros(hidden_size)

        self.W_c = _xavier_uniform(concat_size, hidden_size)
        self.b_c = np.zeros(hidden_size)

        self.W_o = _xavier_uniform(concat_size, hidden_size)
        self.b_o = np.zeros(hidden_size)

    def forward(
        self, x: np.ndarray, h_prev: np.ndarray, c_prev: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Single time-step forward pass.

        Args:
            x: (batch, input_size)
            h_prev: (batch, hidden_size)
            c_prev: (batch, hidden_size)
        Returns:
            (h_next, c_next) each (batch, hidden_size)
        """
        combined = np.concatenate([x, h_prev], axis=-1)

        f = _sigmoid(combined @ self.W_f + self.b_f)
        i = _sigmoid(combined @ self.W_i + self.b_i)
        c_hat = _tanh(combined @ self.W_c + self.b_c)
        o = _sigmoid(combined @ self.W_o + self.b_o)

        c_next = f * c_prev + i * c_hat
        h_next = o * _tanh(c_next)

        return h_next, c_next


class StackedLSTM:
    """Multi-layer LSTM built from individual LSTMCell instances."""

    def __init__(self, input_size: int, hidden_size: int, n_layers: int = 1) -> None:
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.cells: list[LSTMCell] = []
        for layer_idx in range(n_layers):
            in_sz = input_size if layer_idx == 0 else hidden_size
            self.cells.append(LSTMCell(in_sz, hidden_size))

    def forward(
        self, sequence: np.ndarray
    ) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
        """Process an entire sequence.

        Args:
            sequence: (batch, seq_len, input_size)
        Returns:
            outputs: (batch, seq_len, hidden_size) — top-layer hidden states
            final_states: list of (h, c) per layer, each (batch, hidden_size)
        """
        batch, seq_len, _ = sequence.shape
        states = [
            (
                np.zeros((batch, self.hidden_size)),
                np.zeros((batch, self.hidden_size)),
            )
            for _ in range(self.n_layers)
        ]

        outputs = np.zeros((batch, seq_len, self.hidden_size))

        for t in range(seq_len):
            x_t = sequence[:, t, :]
            for layer_idx, cell in enumerate(self.cells):
                h_prev, c_prev = states[layer_idx]
                h_next, c_next = cell.forward(x_t, h_prev, c_prev)
                states[layer_idx] = (h_next, c_next)
                x_t = h_next
            outputs[:, t, :] = x_t

        return outputs, states


@dataclass
class _TrainCache:
    """Intermediate values cached during forward pass for BPTT."""
    inputs: list[np.ndarray] = field(default_factory=list)
    outputs: np.ndarray | None = None


class EntropyPredictor:
    """LSTM model that forecasts entropy quality from sensor metric time series.

    Input features per timestep: [entropy_value, min_entropy, chi_squared, serial_correlation]
    Output: predicted quality score(s) for the next `horizon` steps.
    """

    def __init__(
        self,
        input_features: int = 4,
        hidden_size: int = 32,
        n_layers: int = 2,
        horizon: int = 1,
    ) -> None:
        self.input_features = input_features
        self.hidden_size = hidden_size
        self.horizon = horizon

        self.lstm = StackedLSTM(input_features, hidden_size, n_layers)

        self.W_out = _xavier_uniform(hidden_size, horizon)
        self.b_out = np.zeros(horizon)

        self._cache = _TrainCache()

    def predict(self, sequence: np.ndarray) -> np.ndarray:
        """Predict future entropy quality.

        Args:
            sequence: (seq_len, input_features) or (batch, seq_len, input_features)
        Returns:
            (horizon,) or (batch, horizon) predicted quality values
        """
        squeezed = False
        if sequence.ndim == 2:
            sequence = sequence[np.newaxis, :, :]
            squeezed = True
        outputs, _ = self.lstm.forward(sequence)
        last_hidden = outputs[:, -1, :]
        result = last_hidden @ self.W_out + self.b_out
        if squeezed:
            result = result[0]
        return result

    def train_step(
        self, sequence: np.ndarray, target: np.ndarray, lr: float = 0.001
    ) -> float:
        """Single training step with numerical-gradient BPTT and MSE loss.

        Args:
            sequence: (batch, seq_len, input_features)
            target: (batch, horizon) ground-truth future quality
            lr: learning rate
        Returns:
            scalar MSE loss value
        """
        prediction = self.predict(sequence)
        error = prediction - target
        loss = float(np.mean(error ** 2))

        batch = sequence.shape[0]
        d_pred = (2.0 / (batch * self.horizon)) * error  # (batch, horizon)

        outputs, _ = self.lstm.forward(sequence)
        last_hidden = outputs[:, -1, :]  # (batch, hidden_size)

        d_W_out = last_hidden.T @ d_pred
        d_b_out = d_pred.sum(axis=0)

        max_grad_norm = 5.0
        for grad in (d_W_out, d_b_out):
            norm = np.linalg.norm(grad)
            if norm > max_grad_norm:
                grad *= max_grad_norm / norm

        self.W_out -= lr * d_W_out
        self.b_out -= lr * d_b_out

        return loss
