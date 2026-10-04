"""Smoke tests for Actor-Critic implementation."""

from algorithms.reinforcement_learning.actor_critic import ActorCriticModel
from algorithms.reinforcement_learning.actor_critic_schema import ActorCriticRequest


def test_actor_critic_train_success():
    """Test Actor-Critic training with minimal parameters."""
    model = ActorCriticModel()
    result = model.train(
        actor_lr=0.001,
        critic_lr=0.005,
        gamma=0.99,
        episodes=100,
        hidden_size=64,
        random_state=42
    )

    assert result['success']
    assert 'metrics' in result
    assert 'avg_reward_last_100' in result['metrics']
    assert 'max_reward' in result['metrics']
    assert 'success_rate' in result['metrics']
    assert 'visualization_data' in result
    assert result['execution_time_ms'] > 0
    assert 'parameters_used' in result


def test_actor_critic_request_defaults():
    """Test Actor-Critic request default values."""
    request = ActorCriticRequest()

    assert request.actor_lr == 0.001
    assert request.critic_lr == 0.005
    assert request.gamma == 0.99
    assert request.episodes == 1000
