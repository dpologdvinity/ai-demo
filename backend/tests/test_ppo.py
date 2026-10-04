"""Smoke tests for PPO (Proximal Policy Optimization) implementation."""

from algorithms.reinforcement_learning.ppo import PPOModel
from algorithms.reinforcement_learning.ppo_schema import PPORequest


def test_ppo_train_success():
    """Test PPO training with minimal parameters."""
    model = PPOModel()
    result = model.train(
        learning_rate=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=1,
        episodes=100,
        gae_lambda=0.95,
        batch_size=8,
        hidden_size=64,
        random_state=42
    )

    assert result['success']
    assert 'metrics' in result
    assert 'avg_reward_last_100' in result['metrics']
    assert 'execution_time_ms' in result
    assert result['execution_time_ms'] > 0


def test_ppo_request_defaults():
    """Test PPO request default values."""
    request = PPORequest()

    assert request.learning_rate == 0.0003
    assert request.gamma == 0.99
    assert request.clip_epsilon == 0.2
    assert request.episodes == 500
