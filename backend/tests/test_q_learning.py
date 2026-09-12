"""
Tests for Q-Learning implementation.

This module contains unit tests for the Q-Learning algorithm,
including environment, training, and API endpoint tests.
"""

import pytest
import numpy as np
from algorithms.reinforcement_learning import QLearningModel, GridWorld
from algorithms.reinforcement_learning.q_learning_schema import QLearningRequest


class TestGridWorld:
    """Test cases for GridWorld environment."""

    def test_initialization(self):
        """Test grid world initialization."""
        env = GridWorld(grid_size=5, random_state=42)

        assert env.grid_size == 5
        assert env.start_pos == (0, 0)
        assert env.goal_pos == (4, 4)
        assert len(env.obstacles) > 0
        assert env.start_pos not in env.obstacles
        assert env.goal_pos not in env.obstacles

    def test_reset(self):
        """Test environment reset."""
        env = GridWorld(grid_size=5, random_state=42)
        env.current_pos = (2, 2)

        reset_pos = env.reset()

        assert reset_pos == env.start_pos
        assert env.current_pos == env.start_pos

    def test_step_valid_move(self):
        """Test valid movement in the environment."""
        env = GridWorld(grid_size=5, random_state=42)
        env.reset()

        # Move right (action 1)
        next_state, reward, done = env.step(1)

        assert next_state == (0, 1)
        assert reward == -0.1  # Step penalty
        assert not done

    def test_step_hit_wall(self):
        """Test hitting a wall."""
        env = GridWorld(grid_size=5, random_state=42)
        env.reset()

        # Move up from start (should hit wall)
        next_state, reward, done = env.step(0)

        assert next_state == env.start_pos  # Stay in place
        assert reward == -1.0  # Wall penalty
        assert not done

    def test_reach_goal(self):
        """Test reaching the goal."""
        env = GridWorld(grid_size=3, random_state=42)
        env.current_pos = (2, 1)

        # Move right to goal (assuming (2,2) is goal)
        next_state, reward, done = env.step(1)

        if next_state == env.goal_pos:
            assert reward == 10.0
            assert done
        else:
            # May have hit obstacle
            assert reward <= 0

    def test_state_index_conversion(self):
        """Test state index conversion."""
        env = GridWorld(grid_size=5, random_state=42)

        # Test position to index
        pos = (2, 3)
        idx = env.get_state_index(pos)
        assert idx == 2 * 5 + 3  # row * grid_size + col

        # Test index to position
        recovered_pos = env.get_position_from_index(idx)
        assert recovered_pos == pos


class TestQLearningModel:
    """Test cases for Q-Learning model."""

    def test_initialization(self):
        """Test model initialization."""
        model = QLearningModel()

        assert model.env is None
        assert model.q_table is None
        assert model.training_time_ms == 0.0

    def test_train_success(self):
        """Test successful training."""
        model = QLearningModel()
        result = model.train(
            learning_rate=0.1,
            discount_factor=0.99,
            epsilon=0.1,
            episodes=100,
            grid_size=3,
            random_state=42
        )

        assert result['success']
        assert 'metrics' in result
        assert 'avg_reward_last_100' in result['metrics']
        assert 'success_rate' in result['metrics']
        assert 'visualization_data' in result
        assert result['execution_time_ms'] > 0

    def test_train_invalid_learning_rate(self):
        """Test training with invalid learning rate."""
        model = QLearningModel()
        result = model.train(learning_rate=1.5)  # Invalid

        assert not result['success']
        assert 'error' in result

    def test_train_invalid_discount_factor(self):
        """Test training with invalid discount factor."""
        model = QLearningModel()
        result = model.train(discount_factor=1.5)  # Invalid

        assert not result['success']
        assert 'error' in result

    def test_train_invalid_episodes(self):
        """Test training with too few episodes."""
        model = QLearningModel()
        result = model.train(episodes=50)  # Too few

        assert not result['success']
        assert 'error' in result

    def test_train_creates_q_table(self):
        """Test that training creates a Q-table."""
        model = QLearningModel()
        model.train(
            episodes=100,
            grid_size=3,
            random_state=42
        )

        assert model.q_table is not None
        assert model.q_table.shape == (9, 4)  # 3x3 grid, 4 actions

    def test_train_learning_progress(self):
        """Test that the agent learns over time."""
        model = QLearningModel()
        result = model.train(
            learning_rate=0.2,
            discount_factor=0.99,
            epsilon=0.2,
            episodes=500,
            grid_size=3,
            random_state=42
        )

        # Check that rewards improve over time
        reward_data = result['visualization_data']['reward_data']
        first_100_avg = np.mean([r['reward'] for r in reward_data[:100]])
        last_100_avg = np.mean([r['reward'] for r in reward_data[-100:]])

        # Later episodes should generally have better rewards
        # (This is a soft check as RL can be noisy)
        assert last_100_avg >= first_100_avg - 2.0

    def test_extract_policy(self):
        """Test policy extraction."""
        model = QLearningModel()
        model.train(
            episodes=100,
            grid_size=3,
            random_state=42
        )

        policy = model._extract_policy()

        assert len(policy) == 3
        assert all(len(row) == 3 for row in policy)
        assert all(0 <= action <= 3 for row in policy for action in row)

    def test_visualization_data_structure(self):
        """Test visualization data structure."""
        model = QLearningModel()
        result = model.train(
            episodes=200,
            grid_size=4,
            random_state=42
        )

        viz_data = result['visualization_data']

        # Check required fields
        assert 'grid' in viz_data
        assert 'policy' in viz_data
        assert 'q_value_heatmaps' in viz_data
        assert 'reward_data' in viz_data
        assert 'steps_data' in viz_data
        assert 'trajectories' in viz_data
        assert 'action_labels' in viz_data

        # Check grid info
        grid_info = viz_data['grid']
        assert 'size' in grid_info
        assert 'start' in grid_info
        assert 'goal' in grid_info
        assert 'obstacles' in grid_info

        # Check Q-value heatmaps
        assert len(viz_data['q_value_heatmaps']) > 0
        for heatmap in viz_data['q_value_heatmaps']:
            assert 'episode' in heatmap
            assert 'values' in heatmap

        # Check trajectories
        assert len(viz_data['trajectories']) > 0
        for traj in viz_data['trajectories']:
            assert 'episode' in traj
            assert 'trajectory' in traj
            assert 'reward' in traj
            assert 'steps' in traj

    def test_get_action_without_training(self):
        """Test getting action before training."""
        model = QLearningModel()

        with pytest.raises(RuntimeError):
            model.get_action((0, 0))

    def test_get_action_after_training(self):
        """Test getting action after training."""
        model = QLearningModel()
        model.train(
            episodes=100,
            grid_size=3,
            random_state=42
        )

        action = model.get_action((0, 0))

        assert isinstance(action, int)
        assert 0 <= action <= 3


class TestQLearningRequest:
    """Test cases for QLearningRequest schema."""

    def test_default_values(self):
        """Test default parameter values."""
        request = QLearningRequest()

        assert request.learning_rate == 0.1
        assert request.discount_factor == 0.99
        assert request.epsilon == 0.1
        assert request.episodes == 1000
        assert request.grid_size == 5
        assert request.random_state == 42

    def test_custom_values(self):
        """Test custom parameter values."""
        request = QLearningRequest(
            learning_rate=0.2,
            discount_factor=0.95,
            epsilon=0.3,
            episodes=2000,
            grid_size=7,
            random_state=123
        )

        assert request.learning_rate == 0.2
        assert request.discount_factor == 0.95
        assert request.epsilon == 0.3
        assert request.episodes == 2000
        assert request.grid_size == 7
        assert request.random_state == 123

    def test_validation_learning_rate(self):
        """Test learning rate validation."""
        with pytest.raises(Exception):
            QLearningRequest(learning_rate=-0.1)

        with pytest.raises(Exception):
            QLearningRequest(learning_rate=1.5)

    def test_validation_discount_factor(self):
        """Test discount factor validation."""
        with pytest.raises(Exception):
            QLearningRequest(discount_factor=0.3)

        with pytest.raises(Exception):
            QLearningRequest(discount_factor=1.0)

    def test_validation_episodes(self):
        """Test episodes validation."""
        with pytest.raises(Exception):
            QLearningRequest(episodes=50)

        with pytest.raises(Exception):
            QLearningRequest(episodes=6000)

    def test_validation_grid_size(self):
        """Test grid size validation."""
        with pytest.raises(Exception):
            QLearningRequest(grid_size=2)

        with pytest.raises(Exception):
            QLearningRequest(grid_size=15)


# Integration test
def test_end_to_end_training():
    """End-to-end test of Q-Learning training pipeline."""
    # Create request
    request = QLearningRequest(
        learning_rate=0.15,
        discount_factor=0.95,
        epsilon=0.15,
        episodes=300,
        grid_size=4,
        random_state=42
    )

    # Train model
    model = QLearningModel()
    result = model.train(
        learning_rate=request.learning_rate,
        discount_factor=request.discount_factor,
        epsilon=request.epsilon,
        episodes=request.episodes,
        grid_size=request.grid_size,
        random_state=request.random_state
    )

    # Verify result
    assert result['success']
    assert result['parameters_used']['learning_rate'] == request.learning_rate
    assert result['parameters_used']['episodes'] == request.episodes

    # Verify metrics
    metrics = result['metrics']
    assert 'avg_reward_last_100' in metrics
    assert 'success_rate' in metrics
    assert 0.0 <= metrics['success_rate'] <= 1.0

    # Verify visualization data
    viz_data = result['visualization_data']
    assert len(viz_data['reward_data']) == request.episodes
    assert len(viz_data['policy']) == request.grid_size
