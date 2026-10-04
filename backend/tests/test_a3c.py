"""Smoke tests for A3C (Asynchronous Advantage Actor-Critic) implementation."""

from algorithms.reinforcement_learning.a3c import A3CModel, ActorCriticNetwork
from algorithms.reinforcement_learning.a3c_schema import A3CRequest
import torch


def test_a3c_train_success():
    """Test A3C training with minimal parameters for fast feedback."""
    model = A3CModel()
    result = model.train(
        num_workers=2,
        actor_lr=0.001,
        critic_lr=0.005,
        gamma=0.99,
        episodes_per_worker=50,
        entropy_coef=0.01,
        hidden_size=64,
        random_state=42
    )

    assert result['success'] is True
    assert isinstance(result['metrics'], dict)
    assert 'avg_reward' in result['metrics']
    assert 'avg_length' in result['metrics']
    assert 'success_rate' in result['metrics']
    assert result['execution_time_ms'] >= 0.0
    assert isinstance(result['visualization_data'], dict)
    assert 'worker_rewards' in result['visualization_data']
    assert 'worker_stats' in result['visualization_data']


def test_a3c_network_forward():
    """Test ActorCriticNetwork forward pass."""
    network = ActorCriticNetwork(state_dim=4, action_dim=2, hidden_size=64)
    state = torch.randn(1, 4)

    action_probs, value = network(state)

    assert action_probs.shape == (1, 2)
    assert value.shape == (1, 1)
    assert torch.allclose(action_probs.sum(dim=1), torch.ones(1), atol=1e-5)


def test_a3c_request_defaults():
    """Test A3C request default values."""
    request = A3CRequest()

    assert request.num_workers == 4
    assert request.gamma == 0.99
    assert request.episodes_per_worker == 100
