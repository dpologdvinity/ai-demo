"""
Quick test script for PPO implementation.
"""

import sys
import os

# Add backend to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from algorithms.reinforcement_learning import PPOModel

def test_ppo():
    """Test PPO model with minimal parameters."""
    print("Testing PPO implementation...")
    print("-" * 50)

    # Create model
    model = PPOModel(env_name="CartPole-v1")

    # Train with minimal episodes for quick test
    result = model.train(
        learning_rate=0.0003,
        gamma=0.99,
        clip_epsilon=0.2,
        epochs=4,
        episodes=50,  # Minimal for testing
        gae_lambda=0.95,
        batch_size=64,
        hidden_size=128,
        random_state=42
    )

    # Print results
    print(f"Success: {result['success']}")

    if result['success']:
        print("\nMetrics:")
        for key, value in result['metrics'].items():
            print(f"  {key}: {value}")

        print(f"\nExecution time: {result['execution_time_ms']:.2f} ms")

        print("\nVisualization data keys:")
        for key in result['visualization_data'].keys():
            print(f"  - {key}")

        # Check specific data
        reward_data = result['visualization_data']['reward_data']
        print(f"\nNumber of episodes recorded: {len(reward_data)}")
        print(f"Final episode reward: {reward_data[-1]['reward']:.2f}")

        # Check losses
        loss_data = result['visualization_data']['loss_data']
        if loss_data:
            print(f"Number of updates: {len(loss_data)}")
            print(f"Final policy loss: {loss_data[-1]['policy_loss']:.4f}")
            print(f"Final value loss: {loss_data[-1]['value_loss']:.4f}")

        # Check clip and KL data
        clip_data = result['visualization_data']['clip_data']
        kl_data = result['visualization_data']['kl_data']
        if clip_data:
            print(f"Average clip fraction: {sum(d['clip_fraction'] for d in clip_data) / len(clip_data):.4f}")
        if kl_data:
            print(f"Average KL divergence: {sum(d['kl_divergence'] for d in kl_data) / len(kl_data):.6f}")

        print("\n✓ PPO implementation test passed!")
    else:
        print(f"\n✗ Training failed: {result.get('error', 'Unknown error')}")
        return False

    return True

if __name__ == "__main__":
    try:
        success = test_ppo()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
