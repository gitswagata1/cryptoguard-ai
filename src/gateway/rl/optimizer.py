from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass
class Episode:
    """Single RL episode: a routing decision and its outcome."""

    state: np.ndarray  # input features (complexity, token_count, crypto_density, ...)
    action: int  # model choice index (0=ULTRA, 1=NANO, 2=NEVA, 3=LOCAL)
    reward: float
    advantage: float = 0.0


class RoutingPolicy:
    """Softmax policy mapping input features to model-selection probabilities."""

    def __init__(self, n_features: int, n_actions: int = 4) -> None:
        self.n_features = n_features
        self.n_actions = n_actions
        self.weights = np.zeros((n_features, n_actions), dtype=np.float64)

    def probabilities(self, state: np.ndarray) -> np.ndarray:
        """Compute action probabilities for a given state via softmax."""
        logits = state @ self.weights
        logits = logits - np.max(logits)  # numerical stability
        exp_logits = np.exp(logits)
        return exp_logits / np.sum(exp_logits)

    def log_prob(self, state: np.ndarray, action: int) -> float:
        """Log-probability of taking an action in a given state."""
        probs = self.probabilities(state)
        return float(np.log(probs[action] + 1e-10))

    def select_action(self, state: np.ndarray) -> int:
        """Sample an action from the policy distribution."""
        probs = self.probabilities(state)
        return int(np.random.choice(self.n_actions, p=probs))

    def clone(self) -> RoutingPolicy:
        """Create an independent copy of this policy."""
        copy = RoutingPolicy(self.n_features, self.n_actions)
        copy.weights = self.weights.copy()
        return copy


class PolicyOptimizer:
    """PPO/DPO framework for optimising the routing policy."""

    def __init__(self, learning_rate: float = 3e-4) -> None:
        self.learning_rate = learning_rate

    @staticmethod
    def compute_advantages(
        episodes: list[Episode],
        gamma: float = 0.99,
        lam: float = 0.95,
    ) -> list[Episode]:
        """Generalised Advantage Estimation (GAE)."""
        n = len(episodes)
        if n == 0:
            return episodes

        rewards = np.array([e.reward for e in episodes], dtype=np.float64)

        # Simple value baseline: running mean of future rewards
        values = np.zeros(n + 1, dtype=np.float64)
        for t in reversed(range(n)):
            values[t] = rewards[t] + gamma * values[t + 1]

        # GAE computation
        advantages = np.zeros(n, dtype=np.float64)
        gae = 0.0
        for t in reversed(range(n)):
            delta = rewards[t] + gamma * values[t + 1] - values[t]
            gae = delta + gamma * lam * gae
            advantages[t] = gae

        # Normalise advantages
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        advantages = (advantages - adv_mean) / adv_std

        updated = []
        for i, ep in enumerate(episodes):
            updated.append(Episode(
                state=ep.state,
                action=ep.action,
                reward=ep.reward,
                advantage=float(advantages[i]),
            ))
        return updated

    def ppo_step(
        self,
        episodes: list[Episode],
        old_policy: RoutingPolicy,
        clip_epsilon: float = 0.2,
    ) -> RoutingPolicy:
        """Execute a single PPO update step."""
        new_policy = old_policy.clone()

        if not episodes:
            return new_policy

        # Compute policy gradient with clipping
        grad = np.zeros_like(new_policy.weights)

        for ep in episodes:
            old_log_p = old_policy.log_prob(ep.state, ep.action)
            new_log_p = new_policy.log_prob(ep.state, ep.action)

            ratio = math.exp(new_log_p - old_log_p)
            clipped_ratio = max(
                1.0 - clip_epsilon,
                min(ratio, 1.0 + clip_epsilon),
            )

            surrogate = min(ratio * ep.advantage, clipped_ratio * ep.advantage)

            # Policy gradient: d/dw log pi(a|s) * surrogate
            probs = new_policy.probabilities(ep.state)
            one_hot = np.zeros(new_policy.n_actions, dtype=np.float64)
            one_hot[ep.action] = 1.0
            grad_log = np.outer(ep.state, one_hot - probs)
            grad += grad_log * surrogate

        grad /= len(episodes)
        new_policy.weights += self.learning_rate * grad
        return new_policy

    @staticmethod
    def export_dpo_pairs(
        episodes: list[Episode],
    ) -> list[tuple[Episode, Episode]]:
        """Extract preference pairs for Direct Preference Optimisation."""
        pairs: list[tuple[Episode, Episode]] = []

        # Group episodes by state similarity (same state vector)
        state_groups: dict[str, list[Episode]] = {}
        for ep in episodes:
            key = ep.state.tobytes().hex()
            state_groups.setdefault(key, []).append(ep)

        for group in state_groups.values():
            if len(group) < 2:
                continue
            # Sort by reward descending
            sorted_group = sorted(group, key=lambda e: e.reward, reverse=True)
            # Create pairs: (preferred, rejected)
            for i in range(len(sorted_group) - 1):
                if sorted_group[i].reward > sorted_group[i + 1].reward:
                    pairs.append((sorted_group[i], sorted_group[i + 1]))

        return pairs
