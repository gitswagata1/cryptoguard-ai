"""Tests for RL Q-learning entropy pool management agent."""
import numpy as np
import pytest

from cosmo_core.rl.entropy_agent import Action, EntropyPoolAgent, compute_reward


class TestComputeReward:
    def test_quality_improvement_positive(self):
        state = np.array([0.5, 0.5, 0.8, 0.2])
        next_state = np.array([0.8, 0.5, 0.8, 0.1])
        reward = compute_reward(state, next_state, Action.DO_NOTHING)
        assert reward > 0

    def test_quality_drop_penalized(self):
        state = np.array([0.8, 0.5, 0.8, 0.2])
        next_state = np.array([0.3, 0.5, 0.8, 0.3])
        reward = compute_reward(state, next_state, Action.TRIGGER_CONDITIONING)
        assert reward < 0

    def test_do_nothing_no_action_cost(self):
        state = np.array([0.7, 0.5, 0.8, 0.2])
        reward = compute_reward(state, state, Action.DO_NOTHING)
        assert reward >= compute_reward(state, state, Action.TRIGGER_CONDITIONING)


class TestEntropyPoolAgent:
    def test_select_action_valid(self):
        agent = EntropyPoolAgent(n_bins=5)
        state = np.array([0.5, 0.5, 0.5, 0.5])
        action = agent.select_action(state)
        assert 0 <= action < len(Action)

    def test_update_changes_q_table(self):
        agent = EntropyPoolAgent(n_bins=5, epsilon_start=0.0, learning_rate=0.5)
        state = np.array([0.5, 0.5, 0.5, 0.5])
        q_before = agent.q_table.copy()
        agent.update(state, 0, 1.0, state)
        assert not np.array_equal(agent.q_table, q_before)

    def test_epsilon_decay(self):
        agent = EntropyPoolAgent(epsilon_start=1.0, epsilon_decay=0.9, epsilon_min=0.01)
        state = np.array([0.5, 0.5, 0.5, 0.5])
        for _ in range(10):
            agent.update(state, 0, 1.0, state)
        assert agent.epsilon < 1.0

    def test_q_table_shape(self):
        agent = EntropyPoolAgent(n_bins=5)
        expected = (5,) * 4 + (len(Action),)
        assert agent.q_table.shape == expected

    def test_policy_summary(self):
        agent = EntropyPoolAgent(n_bins=3)
        state = np.array([0.5, 0.5, 0.5, 0.5])
        agent.update(state, 0, 1.0, state)
        summary = agent.get_policy_summary()
        assert isinstance(summary, dict)
        assert "total_updates" in summary or "episodes" in summary or len(summary) > 0
