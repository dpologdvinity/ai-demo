"""
Actor-Critic implementation for reinforcement learning.

This module implements the Actor-Critic algorithm with separate actor and critic networks
for the CartPole-v1 environment from OpenAI Gym.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    """Actor network for policy approximation.

    Maps states to action probability distributions.

    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer (action logits)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        """Initialize Actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_size: Hidden layer dimension
        """
        super(ActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            Action probability distribution (softmax over logits)
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return F.softmax(x, dim=-1)


class CriticNetwork(nn.Module):
    """Critic network for value function approximation.

    Maps states to state values.

    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer (state value)
    """

    def __init__(self, state_dim: int, hidden_size: int = 128):
        """Initialize Critic network.

        Args:
            state_dim: Dimension of state space
            hidden_size: Hidden layer dimension
        """
        super(CriticNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            State value estimate
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ActorCriticModel:
    """Actor-Critic algorithm for reinforcement learning.

    Implements Actor-Critic with separate actor (policy) and critic (value)
    networks for the CartPole-v1 environment.

    Attributes:
        env_name: Gym environment name
        state_dim: Dimension of state space
        action_dim: Number of actions
        actor: Actor network (policy)
        critic: Critic network (value function)
        actor_optimizer: Optimizer for actor network
        critic_optimizer: Optimizer for critic network
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize the Actor-Critic model.

        Args:
            env_name: OpenAI Gym environment name
        """
        self.env_name = env_name
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.actor: Optional[ActorNetwork] = None
        self.critic: Optional[CriticNetwork] = None
        self.actor_optimizer: Optional[optim.Optimizer] = None
        self.critic_optimizer: Optional[optim.Optimizer] = None
        self.training_time_ms: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        actor_lr: float = 0.001,
        critic_lr: float = 0.005,
        gamma: float = 0.99,
        episodes: int = 1000,
        hidden_size: int = 128,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the Actor-Critic agent.

        Args:
            actor_lr: Learning rate for actor network
            critic_lr: Learning rate for critic network
            gamma: Discount factor for future rewards
            episodes: Number of training episodes
            hidden_size: Hidden layer size for both networks
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics
                - visualization_data: Episode rewards, losses, value estimates, policy distributions
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training
        """
        try:
            # Import gym here to avoid import errors if not installed
            import gymnasium as gym

            # Validate parameters
            if not 0.0001 <= actor_lr <= 0.01:
                raise ValueError("actor_lr must be in [0.0001, 0.01]")
            if not 0.0001 <= critic_lr <= 0.01:
                raise ValueError("critic_lr must be in [0.0001, 0.01]")
            if not 0.9 <= gamma <= 0.999:
                raise ValueError("gamma must be in [0.9, 0.999]")
            if not 100 <= episodes <= 3000:
                raise ValueError("episodes must be in [100, 3000]")
            if not 64 <= hidden_size <= 256:
                raise ValueError("hidden_size must be in [64, 256]")

            # Set random seeds
            torch.manual_seed(random_state)
            np.random.seed(random_state)

            # Initialize environment
            env = gym.make(self.env_name)
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.n

            # Initialize networks
            self.actor = ActorNetwork(
                self.state_dim,
                self.action_dim,
                hidden_size
            ).to(self.device)
            self.critic = CriticNetwork(
                self.state_dim,
                hidden_size
            ).to(self.device)

            # Initialize optimizers
            self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
            self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=critic_lr)

            # Training tracking
            start_time = time.time()
            episode_rewards = []
            episode_lengths = []
            actor_losses = []
            critic_losses = []

            # Sample trajectory tracking (record first, middle, last episodes)
            sample_episodes = [0, episodes // 2, episodes - 1]
            sample_trajectories = []

            # Value estimates tracking
            value_estimates_history = []
            policy_distributions_history = []

            # Train for specified episodes
            for episode in range(episodes):
                state, _ = env.reset(seed=random_state + episode)
                episode_reward = 0.0
                episode_length = 0
                episode_actor_losses = []
                episode_critic_losses = []

                done = False
                truncated = False

                # Track trajectory for sample episodes
                trajectory = [] if episode in sample_episodes else None
                if trajectory is not None:
                    trajectory.append(state.tolist())

                # Storage for episode transitions
                log_probs = []
                values = []
                rewards = []

                # Sample states for policy/value tracking
                sample_states = []

                while not (done or truncated):
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    # Get action probabilities from actor
                    action_probs = self.actor(state_tensor)

                    # Get value estimate from critic
                    value = self.critic(state_tensor)

                    # Sample action from policy
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)

                    # Take action in environment
                    next_state, reward, done, truncated, info = env.step(action.item())

                    # Store transition data
                    log_probs.append(log_prob)
                    values.append(value)
                    rewards.append(reward)

                    # Store sample states for visualization
                    if episode_length % 10 == 0 and len(sample_states) < 5:
                        sample_states.append(state.copy())

                    # Update state
                    state = next_state
                    episode_reward += reward
                    episode_length += 1

                    if trajectory is not None:
                        trajectory.append(state.tolist())

                # Episode finished - compute returns and update networks
                actor_loss, critic_loss = self._update_networks(
                    log_probs, values, rewards, gamma, done or truncated
                )

                episode_actor_losses.append(actor_loss)
                episode_critic_losses.append(critic_loss)

                # Store episode metrics
                episode_rewards.append(episode_reward)
                episode_lengths.append(episode_length)
                actor_losses.append(np.mean(episode_actor_losses))
                critic_losses.append(np.mean(episode_critic_losses))

                # Save sample trajectories
                if trajectory is not None:
                    sample_trajectories.append({
                        'episode': episode,
                        'trajectory': trajectory,
                        'reward': episode_reward,
                        'length': episode_length
                    })

                # Track value estimates and policy distributions periodically
                if episode % (episodes // 10) == 0 and len(sample_states) > 0:
                    with torch.no_grad():
                        sample_state_tensor = torch.FloatTensor(sample_states[0]).unsqueeze(0).to(self.device)
                        value_est = self.critic(sample_state_tensor).item()
                        policy_dist = self.actor(sample_state_tensor).cpu().numpy()[0]

                        value_estimates_history.append({
                            'episode': episode,
                            'value': value_est,
                            'state': sample_states[0].tolist()
                        })

                        policy_distributions_history.append({
                            'episode': episode,
                            'distribution': policy_dist.tolist(),
                            'state': sample_states[0].tolist()
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
                actor_losses,
                critic_losses,
                episode_lengths,
                sample_trajectories,
                value_estimates_history,
                policy_distributions_history
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
                    'total_episodes': int(episodes),
                    'final_actor_loss': float(actor_losses[-1]),
                    'final_critic_loss': float(critic_losses[-1])
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,
                    'gamma': gamma,
                    'episodes': episodes,
                    'hidden_size': hidden_size,
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
        log_probs: List[torch.Tensor],
        values: List[torch.Tensor],
        rewards: List[float],
        gamma: float,
        done: bool
    ) -> Tuple[float, float]:
        """Update actor and critic networks using accumulated episode data.

        Args:
            log_probs: Log probabilities of actions taken
            values: Value estimates from critic
            rewards: Rewards received
            gamma: Discount factor
            done: Whether episode ended

        Returns:
            Tuple of (actor_loss, critic_loss)
        """
        # Compute discounted returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.FloatTensor(returns).to(self.device)

        # Normalize returns for stability
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Convert lists to tensors
        log_probs = torch.stack(log_probs)
        values = torch.stack(values).squeeze()

        # Compute advantages (TD error)
        advantages = returns - values.detach()

        # Actor loss: -log_prob * advantage (policy gradient)
        actor_loss = -(log_probs * advantages).mean()

        # Critic loss: MSE between predicted values and returns
        critic_loss = F.mse_loss(values, returns)

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1.0)
        self.actor_optimizer.step()

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
        self.critic_optimizer.step()

        return actor_loss.item(), critic_loss.item()

    def _prepare_visualization_data(
        self,
        episode_rewards: List[float],
        actor_losses: List[float],
        critic_losses: List[float],
        episode_lengths: List[int],
        sample_trajectories: List[Dict[str, Any]],
        value_estimates: List[Dict[str, Any]],
        policy_distributions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            episode_rewards: Rewards for each episode
            actor_losses: Actor loss for each episode
            critic_losses: Critic loss for each episode
            episode_lengths: Episode lengths
            sample_trajectories: Sample agent trajectories
            value_estimates: Value estimates over training
            policy_distributions: Policy distributions over training

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
            'value_estimates': value_estimates,
            'policy_distributions': policy_distributions,
            'environment': environment_info
        }

    def get_action(self, state: np.ndarray) -> int:
        """Get an action for a given state using the learned policy.

        Args:
            state: Current state

        Returns:
            Selected action

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.actor is None:
            raise RuntimeError("Model must be trained before getting actions")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs = self.actor(state_tensor)
            action = action_probs.argmax().item()
            return action

    def get_value(self, state: np.ndarray) -> float:
        """Get value estimate for a given state.

        Args:
            state: Current state

        Returns:
            Value estimate

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.critic is None:
            raise RuntimeError("Model must be trained before getting values")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            value = self.critic(state_tensor)
            return value.item()
