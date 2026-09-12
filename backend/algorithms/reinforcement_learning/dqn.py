"""
Deep Q-Network (DQN) implementation for reinforcement learning.

This module implements the DQN algorithm with experience replay and target network
for the CartPole-v1 environment from OpenAI Gym.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random


class ReplayBuffer:
    """Experience replay buffer for DQN.

    Stores transitions (state, action, reward, next_state, done) and
    samples random minibatches for training.

    Attributes:
        buffer: Deque storing transitions
        max_size: Maximum buffer size
    """

    def __init__(self, max_size: int = 10000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones) as tensors
        """
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.FloatTensor(np.array(states)),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(dones)
        )

    def __len__(self) -> int:
        """Get current buffer size.

        Returns:
            Number of transitions in buffer
        """
        return len(self.buffer)


class QNetwork(nn.Module):
    """Q-Network for DQN.

    Simple multi-layer perceptron that maps states to Q-values for each action.

    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        """Initialize Q-Network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_dim: Hidden layer dimension
        """
        super(QNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            Q-values for each action
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DQNModel:
    """Deep Q-Network algorithm for reinforcement learning.

    Implements DQN with experience replay and target network for the
    CartPole-v1 environment.

    Attributes:
        env_name: Gym environment name
        state_dim: Dimension of state space
        action_dim: Number of actions
        q_network: Main Q-network
        target_network: Target Q-network (updated periodically)
        optimizer: Optimizer for Q-network
        replay_buffer: Experience replay buffer
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize the DQN model.

        Args:
            env_name: OpenAI Gym environment name
        """
        self.env_name = env_name
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.q_network: Optional[QNetwork] = None
        self.target_network: Optional[QNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.replay_buffer: Optional[ReplayBuffer] = None
        self.training_time_ms: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 0.1,
        episodes: int = 500,
        replay_buffer_size: int = 10000,
        batch_size: int = 32,
        target_update_freq: int = 10,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the DQN agent.

        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
            episodes: Number of training episodes
            replay_buffer_size: Maximum size of replay buffer
            batch_size: Batch size for training
            target_update_freq: Frequency (in episodes) to update target network
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics
                - visualization_data: Episode rewards, loss curve, sample trajectory
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training
        """
        try:
            # Import gym here to avoid import errors if not installed
            import gymnasium as gym

            # Validate parameters
            if not 0.0001 <= learning_rate <= 0.01:
                raise ValueError("learning_rate must be in [0.0001, 0.01]")
            if not 0.9 <= gamma <= 0.999:
                raise ValueError("gamma must be in [0.9, 0.999]")
            if not 0.0 <= epsilon <= 1.0:
                raise ValueError("epsilon must be in [0, 1]")
            if not 100 <= episodes <= 2000:
                raise ValueError("episodes must be in [100, 2000]")
            if not 1000 <= replay_buffer_size <= 50000:
                raise ValueError("replay_buffer_size must be in [1000, 50000]")
            if not 16 <= batch_size <= 128:
                raise ValueError("batch_size must be in [16, 128]")

            # Set random seeds
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)

            # Initialize environment
            env = gym.make(self.env_name)
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.n

            # Initialize networks
            self.q_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network = QNetwork(self.state_dim, self.action_dim).to(self.device)
            self.target_network.load_state_dict(self.q_network.state_dict())

            # Initialize optimizer and replay buffer
            self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
            self.replay_buffer = ReplayBuffer(max_size=replay_buffer_size)

            # Training tracking
            start_time = time.time()
            episode_rewards = []
            episode_lengths = []
            training_losses = []

            # Epsilon decay
            epsilon_start = max(epsilon, 1.0)
            epsilon_end = epsilon
            epsilon_decay = 0.995
            current_epsilon = epsilon_start

            # Sample trajectory tracking (record first, middle, last episodes)
            sample_episodes = [0, episodes // 2, episodes - 1]
            sample_trajectories = []

            # Train for specified episodes
            for episode in range(episodes):
                state, _ = env.reset(seed=random_state + episode)
                episode_reward = 0.0
                episode_length = 0
                episode_losses = []
                done = False
                truncated = False

                # Track trajectory for sample episodes
                trajectory = [] if episode in sample_episodes else None
                if trajectory is not None:
                    trajectory.append(state.tolist())

                while not (done or truncated):
                    # Epsilon-greedy action selection
                    if np.random.random() < current_epsilon:
                        action = env.action_space.sample()  # Explore
                    else:
                        with torch.no_grad():
                            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                            q_values = self.q_network(state_tensor)
                            action = q_values.argmax().item()  # Exploit

                    # Take action in environment
                    next_state, reward, done, truncated, info = env.step(action)

                    # Store transition in replay buffer
                    self.replay_buffer.push(state, action, reward, next_state, float(done))

                    # Update state
                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                    if trajectory is not None:
                        trajectory.append(state.tolist())

                    # Train if enough samples in buffer
                    if len(self.replay_buffer) >= batch_size:
                        loss = self._train_step(batch_size, gamma)
                        episode_losses.append(loss)

                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                # Average loss for episode
                if episode_losses:
                    avg_loss = np.mean(episode_losses)
                    training_losses.append(avg_loss)
                else:
                    training_losses.append(0.0)

                # Decay epsilon
                current_epsilon = max(epsilon_end, current_epsilon * epsilon_decay)

                # Update target network periodically
                if episode % target_update_freq == 0:
                    self.target_network.load_state_dict(self.q_network.state_dict())

                # Save sample trajectories
                if trajectory is not None:
                    sample_trajectories.append({
                        'episode': episode,
                        'trajectory': trajectory,
                        'reward': episode_reward,
                        'length': episode_length
                    })

            env.close()
            end_time = time.time()
            self.training_time_ms = (end_time - start_time) * 1000

            # Calculate metrics
            avg_reward_last_100 = np.mean(episode_rewards[-100:])
            avg_length_last_100 = np.mean(episode_lengths[-100:])
            max_reward = max(episode_rewards)

            # Success rate (CartPole is solved when avg reward >= 195 over 100 episodes)
            success_threshold = 195.0
            success_rate = sum(1 for r in episode_rewards[-100:] if r >= success_threshold) / min(100, len(episode_rewards))

            # Convergence episode (first time avg reward >= 195)
            convergence_episode = None
            window_size = 100
            for i in range(window_size - 1, len(episode_rewards)):
                avg = np.mean(episode_rewards[i - window_size + 1:i + 1])
                if avg >= success_threshold:
                    convergence_episode = i
                    break

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                episode_rewards,
                training_losses,
                episode_lengths,
                sample_trajectories
            )

            return {
                'success': True,
                'metrics': {
                    'avg_reward_last_100': float(avg_reward_last_100),
                    'avg_length_last_100': float(avg_length_last_100),
                    'max_reward': float(max_reward),
                    'success_rate': float(success_rate),
                    'convergence_episode': int(convergence_episode) if convergence_episode is not None else None,
                    'final_episode_reward': float(episode_rewards[-1]),
                    'total_episodes': int(episodes)
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'learning_rate': learning_rate,
                    'gamma': gamma,
                    'epsilon': epsilon,
                    'episodes': episodes,
                    'replay_buffer_size': replay_buffer_size,
                    'batch_size': batch_size,
                    'target_update_freq': target_update_freq,
                    'random_state': random_state
                }
            }

        except Exception as e:
            return {
                'success': False,
                'metrics': {},
                'visualization_data': {},
                'execution_time_ms': 0.0,
                'parameters_used': {},
                'error': str(e)
            }

    def _train_step(self, batch_size: int, gamma: float) -> float:
        """Perform a single training step.

        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor

        Returns:
            Training loss
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Compute current Q-values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute target Q-values using target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + gamma * next_q_values * (1 - dones)

        # Compute loss
        loss = nn.MSELoss()(current_q_values, target_q_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def _prepare_visualization_data(
        self,
        episode_rewards: List[float],
        training_losses: List[float],
        episode_lengths: List[int],
        sample_trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            episode_rewards: Rewards for each episode
            training_losses: Training loss for each episode
            episode_lengths: Episode lengths
            sample_trajectories: Sample agent trajectories

        Returns:
            Dictionary with visualization data
        """
        # Calculate moving average for rewards
        window_size = 50
        moving_avg_rewards = []
        for i in range(len(episode_rewards)):
            start_idx = max(0, i - window_size + 1)
            avg = np.mean(episode_rewards[start_idx:i+1])
            moving_avg_rewards.append(float(avg))

        # Prepare reward chart data
        reward_data = [
            {
                'episode': i,
                'reward': float(r),
                'moving_avg': moving_avg_rewards[i]
            }
            for i, r in enumerate(episode_rewards)
        ]

        # Prepare loss chart data
        loss_data = [
            {
                'episode': i,
                'loss': float(l)
            }
            for i, l in enumerate(training_losses)
        ]

        # Prepare length chart data
        length_data = [
            {
                'episode': i,
                'length': int(l)
            }
            for i, l in enumerate(episode_lengths)
        ]

        # Environment info
        environment_info = {
            'name': self.env_name,
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'state_labels': ['Cart Position', 'Cart Velocity', 'Pole Angle', 'Pole Angular Velocity'],
            'action_labels': ['Push Left', 'Push Right'],
            'success_threshold': 195.0,
            'description': 'Balance a pole on a cart by moving left or right'
        }

        return {
            'reward_data': reward_data,
            'loss_data': loss_data,
            'length_data': length_data,
            'trajectories': sample_trajectories,
            'environment': environment_info
        }

    def get_action(self, state: np.ndarray) -> int:
        """Get the best action for a given state using the learned policy.

        Args:
            state: Current state

        Returns:
            Best action

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.q_network is None:
            raise RuntimeError("Model must be trained before getting actions")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
