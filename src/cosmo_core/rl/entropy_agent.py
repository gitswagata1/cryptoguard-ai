"""Q-learning agent for adaptive entropy pool management.

The agent learns optimal actions (mixing ratio adjustments, conditioning
triggers, source switching) by observing entropy quality, pool state,
and source reliability — then maximising a reward signal that balances
output quality against operational cost.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Any

import numpy as np


class Action(IntEnum):
    DO_NOTHING = 0
    INCREASE_MIX_RATIO = 1
    DECREASE_MIX_RATIO = 2
    TRIGGER_CONDITIONING = 3
    SWITCH_SOURCE = 4


N_ACTIONS = len(Action)
STATE_DIM = 4  # entropy_quality, pool_fullness, source_reliability, conditioning_staleness

ACTION_COSTS = np.array([0.0, 0.3, 0.3, 0.3, 0.3], dtype=np.float64)


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool


class ExperienceReplay:
    """Fixed-capacity circular buffer for experience replay."""

    def __init__(self, capacity: int = 10_000) -> None:
        self._capacity = capacity
        self._buffer: list[Transition] = []
        self._pos: int = 0

    def __len__(self) -> int:
        return len(self._buffer)

    @property
    def capacity(self) -> int:
        return self._capacity

    def add(self, transition: Transition) -> None:
        if len(self._buffer) < self._capacity:
            self._buffer.append(transition)
        else:
            self._buffer[self._pos] = transition
        self._pos = (self._pos + 1) % self._capacity

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> list[Transition]:
        rng = rng or np.random.default_rng()
        n = min(batch_size, len(self._buffer))
        indices = rng.choice(len(self._buffer), size=n, replace=False)
        return [self._buffer[i] for i in indices]


def compute_reward(
    state: np.ndarray,
    next_state: np.ndarray,
    action: int,
) -> float:
    """Shaped reward combining quality improvement, threshold bonuses, and action cost.

    Components
    ----------
    * +1.0 × (entropy_quality improvement)
    * +0.5  if next quality > 0.8
    * -ACTION_COST for non-trivial actions
    * -1.0  if next quality drops below 0.5
    """
    quality_now = float(state[0])
    quality_next = float(next_state[0])
    improvement = quality_next - quality_now

    reward = 1.0 * improvement

    if quality_next > 0.8:
        reward += 0.5
    if quality_next < 0.5:
        reward -= 1.0

    reward -= ACTION_COSTS[action]
    return reward


class EntropyPoolAgent:
    """Tabular Q-learning agent with experience replay for entropy pool management.

    Continuous state is discretised into ``n_bins`` per dimension so a
    standard Q-table can be used.  Epsilon-greedy exploration decays
    towards a minimum over training steps.
    """

    def __init__(
        self,
        n_bins: int = 10,
        learning_rate: float = 0.1,
        discount: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10_000,
        seed: int | None = None,
    ) -> None:
        self.n_bins = n_bins
        self.lr = learning_rate
        self.gamma = discount
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

        q_shape = tuple([n_bins] * STATE_DIM + [N_ACTIONS])
        self.q_table = np.zeros(q_shape, dtype=np.float64)

        self.replay = ExperienceReplay(capacity=replay_capacity)
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # State discretisation
    # ------------------------------------------------------------------

    def _discretize(self, state: np.ndarray) -> tuple[int, ...]:
        """Map continuous state in [0, 1]^4 to bin indices."""
        clipped = np.clip(state, 0.0, 1.0 - 1e-9)
        indices = (clipped * self.n_bins).astype(int)
        return tuple(indices.tolist())

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(N_ACTIONS))
        idx = self._discretize(state)
        q_values = self.q_table[idx]
        return int(np.argmax(q_values))

    def greedy_action(self, state: np.ndarray) -> int:
        """Pure exploitation — always pick the best-known action."""
        idx = self._discretize(state)
        return int(np.argmax(self.q_table[idx]))

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def update(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool = False,
    ) -> float:
        """Single-step Q-learning update and store transition in replay buffer.

        Returns the TD error.
        """
        transition = Transition(
            state=np.asarray(state, dtype=np.float64),
            action=action,
            reward=reward,
            next_state=np.asarray(next_state, dtype=np.float64),
            done=done,
        )
        self.replay.add(transition)

        td_error = self._q_update(transition)

        self._step_count += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return td_error

    def _q_update(self, t: Transition) -> float:
        s_idx = self._discretize(t.state)
        ns_idx = self._discretize(t.next_state)

        current_q = self.q_table[s_idx + (t.action,)]
        next_max = 0.0 if t.done else float(np.max(self.q_table[ns_idx]))
        target = t.reward + self.gamma * next_max
        td_error = target - current_q

        self.q_table[s_idx + (t.action,)] += self.lr * td_error
        return td_error

    def train_batch(self, batch_size: int = 32) -> float:
        """Sample a minibatch from the replay buffer and perform batch Q-updates.

        Returns mean absolute TD error across the batch.
        """
        if len(self.replay) < batch_size:
            return 0.0

        batch = self.replay.sample(batch_size, rng=self.rng)
        errors = [abs(self._q_update(t)) for t in batch]
        return float(np.mean(errors))

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_policy_summary(self) -> dict[str, Any]:
        """Return statistics about the learned Q-table and exploration state."""
        best_actions = np.argmax(self.q_table, axis=-1).ravel()
        action_counts = np.bincount(best_actions, minlength=N_ACTIONS)
        action_dist = {Action(i).name: int(c) for i, c in enumerate(action_counts)}

        nonzero_mask = np.any(self.q_table != 0.0, axis=-1)
        visited_states = int(np.sum(nonzero_mask))
        total_states = int(np.prod(self.q_table.shape[:-1]))

        return {
            "step_count": self._step_count,
            "epsilon": round(self.epsilon, 6),
            "replay_size": len(self.replay),
            "visited_states": visited_states,
            "total_states": total_states,
            "coverage": round(visited_states / max(total_states, 1), 4),
            "preferred_action_distribution": action_dist,
            "q_table_mean": round(float(np.mean(self.q_table)), 6),
            "q_table_std": round(float(np.std(self.q_table)), 6),
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            q_table=self.q_table,
            epsilon=np.array([self.epsilon]),
            step_count=np.array([self._step_count]),
            n_bins=np.array([self.n_bins]),
            lr=np.array([self.lr]),
            gamma=np.array([self.gamma]),
        )

    def load(self, path: str | Path) -> None:
        data = np.load(path)
        self.q_table = data["q_table"]
        self.epsilon = float(data["epsilon"][0])
        self._step_count = int(data["step_count"][0])
        self.n_bins = int(data["n_bins"][0])
        self.lr = float(data["lr"][0])
        self.gamma = float(data["gamma"][0])
