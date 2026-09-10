"""
Unit tests for DQN (Deep Q-Network) algorithm.

This module contains tests for the DQN implementation, including:
- Training functionality
- API endpoints
- Parameter validation
- Visualization data generation
"""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import torch

from algorithms.reinforcement_learning import DQNModel, ReplayBuffer, QNetwork
from algorithms.reinforcement_learning import DQNRequest, DQNResponse


class TestReplayBuffer:
    """Test cases for ReplayBuffer class."""

    def test_buffer_initialization(self):
        """Test that buffer initializes correctly."""
        buffer = ReplayBuffer(max_size=100)
        assert len(buffer) == 0
        assert buffer.max_size == 100

    def test_buffer_push(self):
        """Test adding transitions to buffer."""
        buffer = ReplayBuffer(max_size=100)

        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = 1
        reward = 1.0
        next_state = np.array([0.2, 0.3, 0.4, 0.5])
        done = False

        buffer.push(state, action, reward, next_state, done)
        assert len(buffer) == 1

    def test_buffer_max_size(self):
        """Test that buffer respects max size."""
        buffer = ReplayBuffer(max_size=10)

        for i in range(20):
            state = np.array([i, i, i, i], dtype=float)
            buffer.push(state, 0, 1.0, state, False)

        assert len(buffer) == 10

    def test_buffer_sample(self):
        """Test sampling from buffer."""
        buffer = ReplayBuffer(max_size=100)

        # Add transitions
        for i in range(50):
            state = np.array([i, i, i, i], dtype=float)
            buffer.push(state, i % 2, 1.0, state, False)

        # Sample batch
        batch_size = 32
        states, actions, rewards, next_states, dones = buffer.sample(batch_size)

        assert states.shape == (batch_size, 4)
        assert actions.shape == (batch_size,)
        assert rewards.shape == (batch_size,)
        assert next_states.shape == (batch_size, 4)
        assert dones.shape == (batch_size,)


class TestQNetwork:
    """Test cases for QNetwork class."""

    def test_network_initialization(self):
        """Test that network initializes correctly."""
        network = QNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        assert network.fc1.in_features == 4
        assert network.fc3.out_features == 2

    def test_network_forward_pass(self):
        """Test forward pass through network."""
        network = QNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        q_values = network(state)
        assert q_values.shape == (1, 2)

    def test_network_output_range(self):
        """Test that network produces reasonable Q-values."""
        network = QNetwork(state_dim=4, action_dim=2, hidden_dim=64)
        state = torch.FloatTensor([[0.1, 0.2, 0.3, 0.4]])

        q_values = network(state)
        # Q-values should be finite
        assert torch.isfinite(q_values).all()


class TestDQNModel:
    """Test cases for DQNModel class."""

    def test_model_initialization(self):
        """Test that model initializes correctly."""
        model = DQNModel()
        assert model.env_name == "CartPole-v1"
        assert model.q_network is None
        assert model.training_time_ms == 0.0

    def test_model_train_success(self):
        """Test successful training with minimal episodes."""
        model = DQNModel()
        result = model.train(
            learning_rate=0.001,
            gamma=0.99,
            epsilon=0.1,
            episodes=100,  # Minimal for faster testing
            replay_buffer_size=1000,
            batch_size=32,
            random_state=42
        )

        assert result['success'] is True
        assert 'metrics' in result
        assert 'visualization_data' in result
        assert result['execution_time_ms'] > 0

    def test_model_train_metrics(self):
        """Test that training produces expected metrics."""
        model = DQNModel()
        result = model.train(
            episodes=100,
            random_state=42
        )

        metrics = result['metrics']
        assert 'avg_reward_last_100' in metrics
        assert 'avg_length_last_100' in metrics
        assert 'max_reward' in metrics
        assert 'success_rate' in metrics
        assert 'total_episodes' in metrics
        assert metrics['total_episodes'] == 100

    def test_model_train_visualization_data(self):
        """Test that training produces visualization data."""
        model = DQNModel()
        result = model.train(
            episodes=100,
            random_state=42
        )

        viz_data = result['visualization_data']
        assert 'reward_data' in viz_data
        assert 'loss_data' in viz_data
        assert 'length_data' in viz_data
        assert 'trajectories' in viz_data
        assert 'environment' in viz_data

        # Check reward data structure
        assert len(viz_data['reward_data']) == 100
        assert 'episode' in viz_data['reward_data'][0]
        assert 'reward' in viz_data['reward_data'][0]
        assert 'moving_avg' in viz_data['reward_data'][0]

    def test_model_parameter_validation(self):
        """Test parameter validation."""
        model = DQNModel()

        # Invalid learning rate
        result = model.train(learning_rate=0.0)
        assert result['success'] is False
        assert 'error' in result

        # Invalid gamma
        result = model.train(gamma=1.5)
        assert result['success'] is False
        assert 'error' in result

        # Invalid epsilon
        result = model.train(epsilon=-0.1)
        assert result['success'] is False
        assert 'error' in result

        # Invalid episodes
        result = model.train(episodes=50)
        assert result['success'] is False
        assert 'error' in result

    def test_model_get_action(self):
        """Test getting action from trained model."""
        model = DQNModel()
        result = model.train(episodes=100, random_state=42)

        assert result['success'] is True

        # Get action for a random state
        state = np.array([0.1, 0.2, 0.3, 0.4])
        action = model.get_action(state)

        assert isinstance(action, int)
        assert 0 <= action < 2  # CartPole has 2 actions

    def test_model_get_action_before_training(self):
        """Test that get_action raises error before training."""
        model = DQNModel()

        with pytest.raises(RuntimeError):
            state = np.array([0.1, 0.2, 0.3, 0.4])
            model.get_action(state)


class TestDQNRequest:
    """Test cases for DQNRequest schema."""

    def test_request_default_values(self):
        """Test that request has correct default values."""
        request = DQNRequest()
        assert request.learning_rate == 0.001
        assert request.gamma == 0.99
        assert request.epsilon == 0.1
        assert request.episodes == 500
        assert request.replay_buffer_size == 10000
        assert request.batch_size == 32
        assert request.random_state == 42

    def test_request_custom_values(self):
        """Test request with custom values."""
        request = DQNRequest(
            learning_rate=0.0005,
            gamma=0.95,
            epsilon=0.2,
            episodes=1000,
            replay_buffer_size=5000,
            batch_size=64
        )
        assert request.learning_rate == 0.0005
        assert request.gamma == 0.95
        assert request.epsilon == 0.2
        assert request.episodes == 1000
        assert request.replay_buffer_size == 5000
        assert request.batch_size == 64

    def test_request_validation(self):
        """Test request parameter validation."""
        # Valid request
        request = DQNRequest(learning_rate=0.001)
        assert request.learning_rate == 0.001

        # Invalid learning rate (too low)
        with pytest.raises(ValueError):
            DQNRequest(learning_rate=0.00001)

        # Invalid gamma (too high)
        with pytest.raises(ValueError):
            DQNRequest(gamma=1.0)

        # Invalid episodes (too low)
        with pytest.raises(ValueError):
            DQNRequest(episodes=50)


@pytest.mark.asyncio
class TestDQNAPIEndpoints:
    """Test cases for DQN API endpoints."""

    @pytest.fixture
    def client(self):
        """Create test client."""
        from main import app
        return TestClient(app)

    def test_train_endpoint_success(self, client):
        """Test successful training via API."""
        response = client.post(
            "/api/reinforcement-learning/dqn/train",
            json={
                "learning_rate": 0.001,
                "gamma": 0.99,
                "epsilon": 0.1,
                "episodes": 100,
                "replay_buffer_size": 1000,
                "batch_size": 32,
                "random_state": 42
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True
        assert 'metrics' in data
        assert 'visualization_data' in data

    def test_train_endpoint_default_params(self, client):
        """Test training with default parameters."""
        response = client.post(
            "/api/reinforcement-learning/dqn/train",
            json={"episodes": 100}  # Override episodes for faster test
        )

        assert response.status_code == 200
        data = response.json()
        assert data['success'] is True

    def test_train_endpoint_invalid_params(self, client):
        """Test training with invalid parameters."""
        response = client.post(
            "/api/reinforcement-learning/dqn/train",
            json={
                "learning_rate": 0.0,  # Invalid
                "episodes": 100
            }
        )

        assert response.status_code == 422 or response.status_code == 400

    def test_info_endpoint(self, client):
        """Test DQN info endpoint."""
        response = client.get("/api/reinforcement-learning/dqn/info")

        assert response.status_code == 200
        data = response.json()
        assert 'metadata' in data
        assert 'environment_info' in data
        assert data['metadata']['slug'] == 'dqn'
        assert data['environment_info']['type'] == 'CartPole-v1'

    def test_list_algorithms_includes_dqn(self, client):
        """Test that DQN appears in algorithm list."""
        response = client.get("/api/reinforcement-learning/algorithms")

        assert response.status_code == 200
        algorithms = response.json()
        dqn_found = any(algo['slug'] == 'dqn' for algo in algorithms)
        assert dqn_found is True
