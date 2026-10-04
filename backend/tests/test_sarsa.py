"""Smoke tests for SARSA implementation."""

from algorithms.reinforcement_learning.sarsa import SARSAModel
from algorithms.reinforcement_learning.sarsa_schema import SARSARequest


def test_sarsa_train_success():
    """Test SARSA training with minimal parameters."""
    model = SARSAModel()
    result = model.train(
        environment="CartPole-v1",
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        epsilon_decay=0.995,
        episodes=100,
        random_state=42
    )

    assert result['success']
    assert 'metrics' in result
    assert 'avg_reward_last_100' in result['metrics']
    assert result['execution_time_ms'] > 0


def test_sarsa_request_defaults():
    """Test SARSA request default values."""
    request = SARSARequest()

    assert request.environment == "CartPole-v1"
    assert request.learning_rate == 0.1
    assert request.discount_factor == 0.99
    assert request.epsilon == 0.1
