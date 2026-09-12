"""
DDPG (Deep Deterministic Policy Gradient) implementation for reinforcement learning.

This module implements the DDPG algorithm for continuous control tasks using
actor-critic architecture with experience replay and target networks.
Designed for continuous action spaces like Pendulum-v1.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random


class OUNoise:
    """Ornstein-Uhlenbeck process for exploration noise.

    Generates temporally correlated noise for exploration in continuous
    action spaces, which is more effective than uncorrelated Gaussian noise.

    Attributes:
        mu: Mean of the noise process (typically 0)
        theta: Rate of mean reversion
        sigma: Volatility/scale of the noise
        state: Current state of the noise process
    """

    def __init__(self, action_dim: int, mu: float = 0.0, theta: float = 0.15, sigma: float = 0.2):
        """Initialize OU noise process.

        Args:
            action_dim: Dimension of action space
            mu: Mean of the noise
            theta: Mean reversion rate
            sigma: Volatility parameter
        """
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu

    def reset(self):
        """Reset noise to mean."""
        self.state = np.ones(self.action_dim) * self.mu

    def sample(self) -> np.ndarray:
        """Generate noise sample.

        Returns:
            Noise vector for exploration
        """
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(self.action_dim)
        self.state = self.state + dx
        return self.state


class DDPGReplayBuffer:
    """Experience replay buffer for DDPG.

    Stores transitions (s, a, r, s', done) and provides random sampling
    for training, which breaks temporal correlations and improves stability.

    Attributes:
        buffer: Deque storing transitions
        max_size: Maximum buffer capacity
    """

    def __init__(self, max_size: int = 100000):
        """Initialize replay buffer.

        Args:
            max_size: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=max_size)
        self.max_size = max_size

    def add(self, state: np.ndarray, action: np.ndarray, reward: float,
            next_state: np.ndarray, done: bool):
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode ended
        """
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> Tuple[np.ndarray, ...]:
        """Sample a random batch of transitions.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Tuple of (states, actions, rewards, next_states, dones)
        """
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards, dtype=np.float32),
            np.array(next_states),
            np.array(dones, dtype=np.float32)
        )

    def __len__(self) -> int:
        """Get current buffer size."""
        return len(self.buffer)


class DDPGActor(nn.Module):
    """Actor network for DDPG (deterministic policy).

    Maps states to continuous actions using a deterministic policy.
    Uses tanh activation to bound actions to [-1, 1] range.

    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer (action values)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256,
                 action_low: float = -1.0, action_high: float = 1.0):
        """Initialize Actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Hidden layer dimension
            action_low: Lower bound of action space
            action_high: Upper bound of action space
        """
        super(DDPGActor, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

        self.action_low = action_low
        self.action_high = action_high

        # Initialize weights with smaller values for stability
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            Continuous action(s) scaled to action space bounds
        """
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))

        # Scale to action bounds
        action_range = (self.action_high - self.action_low) / 2.0
        action_center = (self.action_high + self.action_low) / 2.0
        x = x * action_range + action_center

        return x


class DDPGCritic(nn.Module):
    """Critic network for DDPG (Q-function approximator).

    Estimates Q(s, a) - the value of taking action 'a' in state 's'.
    Takes both state and action as input.

    Attributes:
        fc1: First fully connected layer (state)
        fc2: Second fully connected layer (concatenated state+action)
        fc3: Output layer (Q-value)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 256):
        """Initialize Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_size: Hidden layer dimension
        """
        super(DDPGCritic, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size + action_dim, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

        # Initialize weights with smaller values for stability
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.uniform_(self.fc3.bias, -3e-3, 3e-3)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            state: Input state tensor
            action: Input action tensor

        Returns:
            Q-value estimate
        """
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class DDPGModel:
    """DDPG (Deep Deterministic Policy Gradient) algorithm for continuous control.

    Implements DDPG with:
    - Deterministic actor policy
    - Q-function critic
    - Experience replay
    - Target networks with soft updates
    - Ornstein-Uhlenbeck noise for exploration

    Attributes:
        env_name: Gym environment name
        state_dim: Dimension of state space
        action_dim: Dimension of action space
        action_low: Lower bound of action space
        action_high: Upper bound of action space
        actor: Actor network (policy)
        critic: Critic network (Q-function)
        actor_target: Target actor network
        critic_target: Target critic network
        actor_optimizer: Optimizer for actor network
        critic_optimizer: Optimizer for critic network
        replay_buffer: Experience replay buffer
        noise: OU noise process for exploration
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self, env_name: str = "Pendulum-v1"):
        """Initialize the DDPG model.

        Args:
            env_name: OpenAI Gym environment name (must have continuous actions)
        """
        self.env_name = env_name
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.action_low: Optional[float] = None
        self.action_high: Optional[float] = None

        self.actor: Optional[DDPGActor] = None
        self.critic: Optional[DDPGCritic] = None
        self.actor_target: Optional[DDPGActor] = None
        self.critic_target: Optional[DDPGCritic] = None

        self.actor_optimizer: Optional[optim.Optimizer] = None
        self.critic_optimizer: Optional[optim.Optimizer] = None
        self.replay_buffer: Optional[DDPGReplayBuffer] = None
        self.noise: Optional[OUNoise] = None

        self.training_time_ms: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        actor_lr: float = 0.0001,
        critic_lr: float = 0.001,
        gamma: float = 0.99,
        tau: float = 0.005,
        episodes: int = 200,
        buffer_size: int = 100000,
        batch_size: int = 64,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the DDPG agent.

        Args:
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            gamma: Discount factor for future rewards
            tau: Soft update coefficient for target networks
            episodes: Number of training episodes
            buffer_size: Replay buffer size
            batch_size: Training batch size
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics
                - visualization_data: Episode rewards, losses, action distributions, Q-values
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training
        """
        try:
            # Import gym here to avoid import errors if not installed
            import gymnasium as gym

            # Validate parameters
            if not 0.00001 <= actor_lr <= 0.001:
                raise ValueError("actor_lr must be in [0.00001, 0.001]")
            if not 0.0001 <= critic_lr <= 0.01:
                raise ValueError("critic_lr must be in [0.0001, 0.01]")
            if not 0.9 <= gamma <= 0.999:
                raise ValueError("gamma must be in [0.9, 0.999]")
            if not 0.001 <= tau <= 0.01:
                raise ValueError("tau must be in [0.001, 0.01]")
            if not 50 <= episodes <= 500:
                raise ValueError("episodes must be in [50, 500]")
            if not 10000 <= buffer_size <= 1000000:
                raise ValueError("buffer_size must be in [10000, 1000000]")
            if not 32 <= batch_size <= 256:
                raise ValueError("batch_size must be in [32, 256]")

            # Set random seeds
            torch.manual_seed(random_state)
            np.random.seed(random_state)
            random.seed(random_state)

            # Initialize environment
            env = gym.make(self.env_name)
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.shape[0]
            self.action_low = float(env.action_space.low[0])
            self.action_high = float(env.action_space.high[0])

            # Initialize networks
            hidden_size = 256
            self.actor = DDPGActor(
                self.state_dim,
                self.action_dim,
                hidden_size,
                self.action_low,
                self.action_high
            ).to(self.device)

            self.critic = DDPGCritic(
                self.state_dim,
                self.action_dim,
                hidden_size
            ).to(self.device)

            # Initialize target networks
            self.actor_target = DDPGActor(
                self.state_dim,
                self.action_dim,
                hidden_size,
                self.action_low,
                self.action_high
            ).to(self.device)

            self.critic_target = DDPGCritic(
                self.state_dim,
                self.action_dim,
                hidden_size
            ).to(self.device)

            # Copy weights to target networks
            self.actor_target.load_state_dict(self.actor.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())

            # Initialize optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

            # Initialize replay buffer and noise
            self.replay_buffer = DDPGReplayBuffer(max_size=buffer_size)
            self.noise = OUNoise(self.action_dim)

            # Training tracking
            start_time = time.time()
            episode_rewards = []
            episode_lengths = []
            actor_losses = []
            critic_losses = []
            q_values_history = []
            action_distributions_history = []

            # Minimum buffer size before training
            min_buffer_size = min(1000, batch_size * 10)

            # Train for specified episodes
            for episode in range(episodes):
                state, _ = env.reset(seed=random_state + episode)
                self.noise.reset()

                episode_reward = 0.0
                episode_length = 0
                episode_actor_losses = []
                episode_critic_losses = []
                episode_q_values = []
                episode_actions = []

                done = False
                truncated = False

                while not (done or truncated):
                    # Select action with exploration noise
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    with torch.no_grad():
                        action = self.actor(state_tensor).cpu().numpy()[0]

                    # Add exploration noise
                    noise = self.noise.sample()
                    action = np.clip(action + noise, self.action_low, self.action_high)

                    # Take action in environment
                    next_state, reward, done, truncated, info = env.step(action)

                    # Store transition in replay buffer
                    self.replay_buffer.add(state, action, reward, next_state, done or truncated)

                    # Update networks if buffer has enough samples
                    if len(self.replay_buffer) >= min_buffer_size:
                        actor_loss, critic_loss = self._update_networks(
                            batch_size, gamma, tau
                        )
                        episode_actor_losses.append(actor_loss)
                        episode_critic_losses.append(critic_loss)

                        # Track Q-values periodically
                        if episode_length % 10 == 0:
                            with torch.no_grad():
                                q_val = self.critic(
                                    state_tensor,
                                    torch.FloatTensor(action).unsqueeze(0).to(self.device)
                                ).item()
                                episode_q_values.append(q_val)

                    # Track actions for distribution analysis
                    episode_actions.append(action.copy())

                    # Update state
                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)

                if len(episode_actor_losses) > 0:
                    actor_losses.append(np.mean(episode_actor_losses))
                    critic_losses.append(np.mean(episode_critic_losses))
                else:
                    actor_losses.append(0.0)
                    critic_losses.append(0.0)

                # Track Q-values and actions periodically
                if episode % max(1, episodes // 20) == 0:
                    if len(episode_q_values) > 0:
                        q_values_history.append({
                            'episode': episode,
                            'mean_q': float(np.mean(episode_q_values)),
                            'max_q': float(np.max(episode_q_values)),
                            'min_q': float(np.min(episode_q_values))
                        })

                    if len(episode_actions) > 0:
                        actions_array = np.array(episode_actions)
                        action_distributions_history.append({
                            'episode': episode,
                            'mean_action': float(np.mean(actions_array)),
                            'std_action': float(np.std(actions_array)),
                            'min_action': float(np.min(actions_array)),
                            'max_action': float(np.max(actions_array))
                        })

            env.close()
            end_time = time.time()
            self.training_time_ms = (end_time - start_time) * 1000

            # Calculate metrics
            avg_reward_last_100 = np.mean(episode_rewards[-min(100, len(episode_rewards)):])
            avg_length_last_100 = np.mean(episode_lengths[-min(100, len(episode_lengths)):])
            max_reward = max(episode_rewards)
            min_reward = min(episode_rewards)

            # Calculate improvement (compare first 20% vs last 20%)
            early_avg = np.mean(episode_rewards[:max(1, len(episode_rewards) // 5)])
            late_avg = np.mean(episode_rewards[-max(1, len(episode_rewards) // 5):])
            improvement = late_avg - early_avg

            # Convergence detection (when reward stabilizes)
            convergence_episode = None
            window_size = min(50, len(episode_rewards) // 4)
            if window_size >= 10:
                for i in range(window_size, len(episode_rewards)):
                    window_rewards = episode_rewards[i-window_size:i]
                    if np.std(window_rewards) < abs(np.mean(window_rewards)) * 0.1:
                        convergence_episode = i
                        break

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                episode_rewards,
                actor_losses,
                critic_losses,
                episode_lengths,
                q_values_history,
                action_distributions_history
            )

            return {
                'success': True,
                'metrics': {
                    'avg_reward_last_100': float(avg_reward_last_100),
                    'avg_length_last_100': float(avg_length_last_100),
                    'max_reward': float(max_reward),
                    'min_reward': float(min_reward),
                    'improvement': float(improvement),
                    'convergence_episode': int(convergence_episode) if convergence_episode is not None else None,
                    'final_episode_reward': float(episode_rewards[-1]),
                    'total_episodes': int(episodes),
                    'final_actor_loss': float(actor_losses[-1]) if len(actor_losses) > 0 else 0.0,
                    'final_critic_loss': float(critic_losses[-1]) if len(critic_losses) > 0 else 0.0,
                    'buffer_size': len(self.replay_buffer)
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,
                    'gamma': gamma,
                    'tau': tau,
                    'episodes': episodes,
                    'buffer_size': buffer_size,
                    'batch_size': batch_size,
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

    def _update_networks(
        self,
        batch_size: int,
        gamma: float,
        tau: float
    ) -> Tuple[float, float]:
        """Update actor and critic networks using sampled batch.

        Args:
            batch_size: Number of transitions to sample
            gamma: Discount factor
            tau: Soft update coefficient

        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Sample batch from replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Update Critic
        with torch.no_grad():
            # Compute target Q-value: y = r + γ * Q_target(s', μ_target(s'))
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            target_q = rewards + (1 - dones) * gamma * target_q

        # Compute current Q-value
        current_q = self.critic(states, actions)

        # Critic loss: MSE between current and target Q-values
        critic_loss = F.mse_loss(current_q, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        # Update Actor
        # Actor loss: -mean(Q(s, μ(s))) (maximize Q-value)
        actor_loss = -self.critic(states, self.actor(states)).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Soft update target networks
        self._soft_update(self.actor, self.actor_target, tau)
        self._soft_update(self.critic, self.critic_target, tau)

        return actor_loss.item(), critic_loss.item()

    def _soft_update(self, source: nn.Module, target: nn.Module, tau: float):
        """Soft update target network parameters.

        θ_target = τ * θ_source + (1 - τ) * θ_target

        Args:
            source: Source network
            target: Target network
            tau: Update coefficient
        """
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(
                tau * source_param.data + (1.0 - tau) * target_param.data
            )

    def _prepare_visualization_data(
        self,
        episode_rewards: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
        episode_lengths: List[int],
        q_values_history: List[Dict[str, Any]],
        action_distributions_history: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            episode_rewards: Rewards for each episode
            actor_losses: Actor loss for each episode
            critic_losses: Critic loss for each episode
            episode_lengths: Episode lengths
            q_values_history: Q-value estimates over training
            action_distributions_history: Action distributions over training

        Returns:
            Dictionary with visualization data
        """
        # Calculate moving average for rewards
        window_size = min(50, len(episode_rewards) // 4)
        if window_size < 5:
            window_size = 5

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

        # Prepare loss chart data (combined actor and critic)
        loss_data = [
            {
                'episode': i,
                'actor_loss': float(actor_losses[i]),
                'critic_loss': float(critic_losses[i])
            }
            for i in range(len(actor_losses))
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
            'action_space': f"Continuous [{self.action_low}, {self.action_high}]",
            'state_labels': ['Cos(θ)', 'Sin(θ)', 'Angular Velocity'] if self.env_name == "Pendulum-v1" else None,
            'action_labels': ['Torque'] if self.env_name == "Pendulum-v1" else None,
            'description': 'Swing pendulum to upright position' if self.env_name == "Pendulum-v1" else 'Continuous control task'
        }

        return {
            'reward_data': reward_data,
            'loss_data': loss_data,
            'length_data': length_data,
            'q_values': q_values_history,
            'action_distributions': action_distributions_history,
            'environment': environment_info
        }

    def get_action(self, state: np.ndarray, add_noise: bool = False) -> np.ndarray:
        """Get an action for a given state using the learned policy.

        Args:
            state: Current state
            add_noise: Whether to add exploration noise

        Returns:
            Selected action

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.actor is None:
            raise RuntimeError("Model must be trained before getting actions")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action = self.actor(state_tensor).cpu().numpy()[0]

            if add_noise and self.noise is not None:
                noise = self.noise.sample()
                action = np.clip(action + noise, self.action_low, self.action_high)

            return action

    def get_q_value(self, state: np.ndarray, action: np.ndarray) -> float:
        """Get Q-value estimate for a state-action pair.

        Args:
            state: Current state
            action: Action to evaluate

        Returns:
            Q-value estimate

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.critic is None:
            raise RuntimeError("Model must be trained before getting Q-values")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_tensor = torch.FloatTensor(action).unsqueeze(0).to(self.device)
            q_value = self.critic(state_tensor, action_tensor)
            return q_value.item()
