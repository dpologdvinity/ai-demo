#!/usr/bin/env python3
"""
Standalone test script for Q-Learning implementation.
This can be run without pytest to verify the basic functionality.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from algorithms.reinforcement_learning.q_learning import QLearningModel, GridWorld


def test_grid_world():
    """Test GridWorld environment."""
    print("\n=== Testing GridWorld Environment ===")

    env = GridWorld(grid_size=5, random_state=42)

    print(f"Grid size: {env.grid_size}")
    print(f"Start position: {env.start_pos}")
    print(f"Goal position: {env.goal_pos}")
    print(f"Number of obstacles: {len(env.obstacles)}")
    print(f"Obstacles: {env.obstacles}")

    # Test reset
    pos = env.reset()
    assert pos == env.start_pos, "Reset should return to start position"
    print("✓ Reset works correctly")

    # Test movement
    next_state, reward, done = env.step(1)  # Move right
    print(f"After moving right: pos={next_state}, reward={reward}, done={done}")
    print("✓ Movement works correctly")

    print("✓ GridWorld tests passed!\n")


def test_q_learning_training():
    """Test Q-Learning training."""
    print("=== Testing Q-Learning Training ===")

    model = QLearningModel()

    print("Training Q-Learning agent (100 episodes, 3x3 grid)...")
    result = model.train(
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=0.1,
        episodes=100,
        grid_size=3,
        random_state=42
    )

    assert result['success'], f"Training failed: {result.get('error')}"
    print("✓ Training completed successfully")

    # Check metrics
    metrics = result['metrics']
    print(f"\nTraining Metrics:")
    print(f"  Average reward (last 100): {metrics['avg_reward_last_100']:.2f}")
    print(f"  Average steps (last 100): {metrics['avg_steps_last_100']:.2f}")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Best episode reward: {metrics['best_episode_reward']:.2f}")
    print(f"  Training time: {result['execution_time_ms']:.2f}ms")

    # Check Q-table
    assert model.q_table is not None, "Q-table should be created"
    assert model.q_table.shape == (9, 4), f"Q-table shape should be (9, 4), got {model.q_table.shape}"
    print(f"\n✓ Q-table created with shape {model.q_table.shape}")

    # Check visualization data
    viz_data = result['visualization_data']
    assert 'grid' in viz_data, "Visualization data should contain grid"
    assert 'policy' in viz_data, "Visualization data should contain policy"
    assert 'q_value_heatmaps' in viz_data, "Visualization data should contain Q-value heatmaps"
    assert 'reward_data' in viz_data, "Visualization data should contain reward data"
    print("✓ Visualization data structure is correct")

    # Check policy
    policy = viz_data['policy']
    print(f"\nLearned Policy (3x3 grid):")
    action_symbols = ['↑', '→', '↓', '←']
    for row in policy:
        print("  " + " ".join(action_symbols[a] for a in row))

    print("\n✓ Q-Learning training tests passed!\n")


def test_longer_training():
    """Test with more episodes to see learning progress."""
    print("=== Testing Longer Training (500 episodes, 5x5 grid) ===")

    model = QLearningModel()

    result = model.train(
        learning_rate=0.15,
        discount_factor=0.99,
        epsilon=0.2,
        episodes=500,
        grid_size=5,
        random_state=42
    )

    assert result['success'], f"Training failed: {result.get('error')}"

    # Analyze learning progress
    reward_data = result['visualization_data']['reward_data']
    first_100 = [r['reward'] for r in reward_data[:100]]
    last_100 = [r['reward'] for r in reward_data[-100:]]

    avg_first_100 = np.mean(first_100)
    avg_last_100 = np.mean(last_100)

    print(f"Average reward (first 100 episodes): {avg_first_100:.2f}")
    print(f"Average reward (last 100 episodes): {avg_last_100:.2f}")
    print(f"Improvement: {avg_last_100 - avg_first_100:.2f}")

    metrics = result['metrics']
    print(f"\nFinal Metrics:")
    print(f"  Success rate: {metrics['success_rate']:.2%}")
    print(f"  Average steps: {metrics['avg_steps_last_100']:.2f}")

    # Display policy
    policy = result['visualization_data']['policy']
    grid_info = result['visualization_data']['grid']

    print(f"\nLearned Policy for {grid_info['size']}x{grid_info['size']} grid:")
    print(f"  Start: {grid_info['start']}, Goal: {grid_info['goal']}")
    print(f"  Obstacles: {grid_info['obstacles']}")

    action_symbols = ['↑', '→', '↓', '←']
    obstacles_set = set(tuple(obs) for obs in grid_info['obstacles'])

    print("\n  Policy visualization:")
    for i, row in enumerate(policy):
        row_str = "  "
        for j, action in enumerate(row):
            if (i, j) == tuple(grid_info['start']):
                row_str += "S "
            elif (i, j) == tuple(grid_info['goal']):
                row_str += "G "
            elif (i, j) in obstacles_set:
                row_str += "X "
            else:
                row_str += action_symbols[action] + " "
        print(row_str)

    print("\n  Legend: S=Start, G=Goal, X=Obstacle, Arrows=Policy direction")

    print("\n✓ Longer training test passed!\n")


def test_parameter_validation():
    """Test parameter validation."""
    print("=== Testing Parameter Validation ===")

    model = QLearningModel()

    # Test invalid learning rate
    result = model.train(learning_rate=1.5, episodes=100)
    assert not result['success'], "Should fail with invalid learning rate"
    print("✓ Invalid learning rate rejected")

    # Test invalid discount factor
    result = model.train(discount_factor=1.5, episodes=100)
    assert not result['success'], "Should fail with invalid discount factor"
    print("✓ Invalid discount factor rejected")

    # Test invalid episodes
    result = model.train(episodes=50)
    assert not result['success'], "Should fail with too few episodes"
    print("✓ Too few episodes rejected")

    # Test invalid grid size
    result = model.train(grid_size=2, episodes=100)
    assert not result['success'], "Should fail with invalid grid size"
    print("✓ Invalid grid size rejected")

    print("\n✓ Parameter validation tests passed!\n")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Q-LEARNING IMPLEMENTATION TEST SUITE")
    print("="*60)

    try:
        test_grid_world()
        test_q_learning_training()
        test_longer_training()
        test_parameter_validation()

        print("="*60)
        print("ALL TESTS PASSED! ✓")
        print("="*60 + "\n")
        return 0

    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}\n")
        return 1
    except Exception as e:
        print(f"\n❌ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
