"""Smoke tests for DDPG (Deep Deterministic Policy Gradient) implementation."""

from algorithms.reinforcement_learning.ddpg import DDPGModel
from algorithms.reinforcement_learning.ddpg_schema import DDPGRequest


def test_ddpg_train_success():
    """Test DDPG training with minimal parameters."""
    model = DDPGModel()
    result = model.train(
        actor_lr=0.0001,
        critic_lr=0.001,
        gamma=0.99,
        tau=0.005,
        episodes=50,
        buffer_size=10000,
        batch_size=32,
        random_state=42
    )

    assert result['success']
    assert 'metrics' in result
    assert 'avg_reward_last_100' in result['metrics']
    assert result['execution_time_ms'] > 0


def test_ddpg_request_defaults():
    """Test DDPG request default values."""
    request = DDPGRequest()

    assert request.actor_lr == 0.0001
    assert request.critic_lr == 0.001
    assert request.gamma == 0.99
    assert request.tau == 0.005
