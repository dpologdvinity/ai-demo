"""
Proximal Policy Optimization (PPO) implementation for reinforcement learning.

This module implements PPO with clipped surrogate objective and Generalized Advantage Estimation (GAE)
for the CartPole-v1 environment from OpenAI Gym.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque


class PPOActorNetwork(nn.Module):
    """Actor network for PPO policy approximation.

    Maps states to action probability distributions.

    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer (action logits)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        """Initialize PPO Actor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_size: Hidden layer dimension
        """
        super(PPOActorNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            Action logits (before softmax)
        """
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x

    def get_action_probs(self, x: torch.Tensor) -> torch.Tensor:
        """Get action probabilities.

        Args:
            x: Input state tensor

        Returns:
            Action probability distribution
        """
        logits = self.forward(x)
        return F.softmax(logits, dim=-1)


class PPOCriticNetwork(nn.Module):
    """Critic network for PPO value function approximation.

    Maps states to state values.

    Attributes:
        fc1: First fully connected layer
        fc2: Second fully connected layer
        fc3: Output layer (state value)
    """

    def __init__(self, state_dim: int, hidden_size: int = 128):
        """Initialize PPO Critic network.

        Args:
            state_dim: Dimension of state space
            hidden_size: Hidden layer dimension
        """
        super(PPOCriticNetwork, self).__init__()

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
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class RolloutBuffer:
    """Buffer for storing trajectory rollouts for PPO training.

    Stores states, actions, rewards, log probabilities, values, and done flags
    for computing advantages and training PPO.

    Attributes:
        states: List of states
        actions: List of actions
        rewards: List of rewards
        log_probs: List of action log probabilities
        values: List of value estimates
        dones: List of done flags
    """

    def __init__(self):
        """Initialize rollout buffer."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        log_prob: float,
        value: float,
        done: bool
    ):
        """Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            log_prob: Log probability of action
            value: Value estimate
            done: Whether episode ended
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.dones.append(done)

    def compute_advantages(self, gamma: float, gae_lambda: float, next_value: float = 0.0) -> Tuple[np.ndarray, np.ndarray]:
        """Compute advantages using Generalized Advantage Estimation (GAE).

        Args:
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
            next_value: Value estimate for final state (if not done)

        Returns:
            Tuple of (advantages, returns)
        """
        advantages = []
        gae = 0

        # Compute GAE backwards
        values_ext = self.values + [next_value]
        for t in reversed(range(len(self.rewards))):
            if self.dones[t]:
                delta = self.rewards[t] - values_ext[t]
                gae = delta
            else:
                delta = self.rewards[t] + gamma * values_ext[t + 1] - values_ext[t]
                gae = delta + gamma * gae_lambda * gae
            advantages.insert(0, gae)

        advantages = np.array(advantages)
        returns = advantages + np.array(self.values)

        return advantages, returns

    def get_data(self) -> Dict[str, np.ndarray]:
        """Get all buffer data.

        Returns:
            Dictionary with arrays of states, actions, rewards, log_probs, values, dones
        """
        return {
            'states': np.array(self.states),
            'actions': np.array(self.actions),
            'rewards': np.array(self.rewards),
            'log_probs': np.array(self.log_probs),
            'values': np.array(self.values),
            'dones': np.array(self.dones)
        }

    def clear(self):
        """Clear all buffer data."""
        self.states = []
        self.actions = []
        self.rewards = []
        self.log_probs = []
        self.values = []
        self.dones = []


class PPOModel:
    """Proximal Policy Optimization (PPO) algorithm for reinforcement learning.

    Implements PPO with clipped surrogate objective and GAE for the CartPole-v1 environment.

    Attributes:
        env_name: Gym environment name
        state_dim: Dimension of state space
        action_dim: Number of actions
        actor: Actor network (policy)
        critic: Critic network (value function)
        optimizer: Shared optimizer for both networks
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize the PPO model.

        Args:
            env_name: OpenAI Gym environment name
        """
        self.env_name = env_name
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.actor: Optional[PPOActorNetwork] = None
        self.critic: Optional[PPOCriticNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_time_ms: float = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        learning_rate: float = 0.0003,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        epochs: int = 4,
        episodes: int = 500,
        gae_lambda: float = 0.95,
        batch_size: int = 64,
        hidden_size: int = 128,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the PPO agent.

        Args:
            learning_rate: Learning rate for optimizer
            gamma: Discount factor for future rewards
            clip_epsilon: PPO clipping parameter
            epochs: Number of PPO epochs per rollout
            episodes: Number of training episodes
            gae_lambda: GAE lambda parameter
            batch_size: Minibatch size for PPO updates
            hidden_size: Hidden layer size for both networks
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics
                - visualization_data: Episode rewards, losses, clip fractions, KL divergences
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training
        """
        try:
            # Import gym here to avoid import errors if not installed
            import gymnasium as gym

            # Validate parameters
            if not 0.0001 <= learning_rate <= 0.001:
                raise ValueError("learning_rate must be in [0.0001, 0.001]")
            if not 0.9 <= gamma <= 0.999:
                raise ValueError("gamma must be in [0.9, 0.999]")
            if not 0.1 <= clip_epsilon <= 0.3:
                raise ValueError("clip_epsilon must be in [0.1, 0.3]")
            if not 1 <= epochs <= 10:
                raise ValueError("epochs must be in [1, 10]")
            if not 100 <= episodes <= 2000:
                raise ValueError("episodes must be in [100, 2000]")
            if not 0.9 <= gae_lambda <= 0.99:
                raise ValueError("gae_lambda must be in [0.9, 0.99]")

            # Set random seeds
            torch.manual_seed(random_state)
            np.random.seed(random_state)

            # Initialize environment
            env = gym.make(self.env_name)
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.n

            # Initialize networks
            self.actor = PPOActorNetwork(
                self.state_dim,
                self.action_dim,
                hidden_size
            ).to(self.device)
            self.critic = PPOCriticNetwork(
                self.state_dim,
                hidden_size
            ).to(self.device)

            # Initialize optimizer (shared for both networks)
            self.optimizer = optim.Adam(
                list(self.actor.parameters()) + list(self.critic.parameters()),
                lr=learning_rate
            )

            # Training tracking
            start_time = time.time()
            episode_rewards = []
            episode_lengths = []
            policy_losses = []
            value_losses = []
            clip_fractions = []
            kl_divergences = []

            # Buffer for collecting rollouts
            buffer = RolloutBuffer()

            # Episode counter
            episode = 0
            state, _ = env.reset(seed=random_state)
            episode_reward = 0.0
            episode_length = 0

            # Sample trajectory tracking (record first, middle, last episodes)
            sample_episodes = [0, episodes // 2, episodes - 1]
            sample_trajectories = []
            current_trajectory = []

            # Track training steps
            total_steps = 0
            steps_per_update = 2048  # Collect this many steps before update

            while episode < episodes:
                # Collect rollout
                with torch.no_grad():
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                    # Get action from policy
                    logits = self.actor(state_tensor)
                    action_probs = F.softmax(logits, dim=-1)
                    action_dist = torch.distributions.Categorical(action_probs)
                    action = action_dist.sample()
                    log_prob = action_dist.log_prob(action)

                    # Get value estimate
                    value = self.critic(state_tensor)

                # Take action in environment
                next_state, reward, done, truncated, info = env.step(action.item())

                # Store in buffer
                buffer.add(
                    state,
                    action.item(),
                    reward,
                    log_prob.item(),
                    value.item(),
                    done or truncated
                )

                # Track trajectory for sample episodes
                if episode in sample_episodes:
                    current_trajectory.append(state.tolist())

                # Update state and metrics
                state = next_state
                episode_reward += reward
                episode_length += 1
                total_steps += 1

                # Episode finished
                if done or truncated:
                    episode_rewards.append(episode_reward)
                    episode_lengths.append(episode_length)

                    # Save sample trajectory
                    if episode in sample_episodes:
                        current_trajectory.append(state.tolist())
                        sample_trajectories.append({
                            'episode': episode,
                            'trajectory': current_trajectory,
                            'reward': episode_reward,
                            'length': episode_length
                        })
                        current_trajectory = []

                    # Reset environment
                    episode += 1
                    state, _ = env.reset(seed=random_state + episode)
                    episode_reward = 0.0
                    episode_length = 0

                # Perform PPO update
                if total_steps % steps_per_update == 0 or episode >= episodes:
                    # Compute advantages
                    with torch.no_grad():
                        next_state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                        next_value = self.critic(next_state_tensor).item() if not (done or truncated) else 0.0

                    advantages, returns = buffer.compute_advantages(gamma, gae_lambda, next_value)

                    # Normalize advantages
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                    # Get buffer data
                    data = buffer.get_data()
                    states = torch.FloatTensor(data['states']).to(self.device)
                    actions = torch.LongTensor(data['actions']).to(self.device)
                    old_log_probs = torch.FloatTensor(data['log_probs']).to(self.device)
                    advantages_tensor = torch.FloatTensor(advantages).to(self.device)
                    returns_tensor = torch.FloatTensor(returns).to(self.device)

                    # PPO update for multiple epochs
                    epoch_policy_losses = []
                    epoch_value_losses = []
                    epoch_clip_fractions = []
                    epoch_kl_divs = []

                    for ppo_epoch in range(epochs):
                        # Create minibatches
                        num_samples = states.shape[0]
                        indices = np.random.permutation(num_samples)

                        for start_idx in range(0, num_samples, batch_size):
                            end_idx = min(start_idx + batch_size, num_samples)
                            batch_indices = indices[start_idx:end_idx]

                            # Get batch
                            batch_states = states[batch_indices]
                            batch_actions = actions[batch_indices]
                            batch_old_log_probs = old_log_probs[batch_indices]
                            batch_advantages = advantages_tensor[batch_indices]
                            batch_returns = returns_tensor[batch_indices]

                            # Compute new log probs and values
                            logits = self.actor(batch_states)
                            action_probs = F.softmax(logits, dim=-1)
                            action_dist = torch.distributions.Categorical(action_probs)
                            new_log_probs = action_dist.log_prob(batch_actions)
                            entropy = action_dist.entropy()
                            values = self.critic(batch_states).squeeze()

                            # Compute ratio
                            ratio = torch.exp(new_log_probs - batch_old_log_probs)

                            # Compute clipped surrogate loss
                            surr1 = ratio * batch_advantages
                            surr2 = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * batch_advantages
                            policy_loss = -torch.min(surr1, surr2).mean()

                            # Compute value loss
                            value_loss = F.mse_loss(values, batch_returns)

                            # Total loss with entropy bonus
                            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                            # Update networks
                            self.optimizer.zero_grad()
                            loss.backward()
                            torch.nn.utils.clip_grad_norm_(
                                list(self.actor.parameters()) + list(self.critic.parameters()),
                                0.5
                            )
                            self.optimizer.step()

                            # Track metrics
                            with torch.no_grad():
                                clip_frac = torch.mean((torch.abs(ratio - 1) > clip_epsilon).float()).item()
                                kl_div = (batch_old_log_probs - new_log_probs).mean().item()

                            epoch_policy_losses.append(policy_loss.item())
                            epoch_value_losses.append(value_loss.item())
                            epoch_clip_fractions.append(clip_frac)
                            epoch_kl_divs.append(kl_div)

                    # Store average metrics for this update
                    if epoch_policy_losses:
                        policy_losses.append(np.mean(epoch_policy_losses))
                        value_losses.append(np.mean(epoch_value_losses))
                        clip_fractions.append(np.mean(epoch_clip_fractions))
                        kl_divergences.append(np.mean(epoch_kl_divs))

                    # Clear buffer
                    buffer.clear()

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
                policy_losses,
                value_losses,
                clip_fractions,
                kl_divergences,
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
                    'total_episodes': int(episodes),
                    'final_policy_loss': float(policy_losses[-1]) if policy_losses else 0.0,
                    'final_value_loss': float(value_losses[-1]) if value_losses else 0.0,
                    'avg_clip_fraction': float(np.mean(clip_fractions)) if clip_fractions else 0.0,
                    'avg_kl_divergence': float(np.mean(kl_divergences)) if kl_divergences else 0.0
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'learning_rate': learning_rate,
                    'gamma': gamma,
                    'clip_epsilon': clip_epsilon,
                    'epochs': epochs,
                    'episodes': episodes,
                    'gae_lambda': gae_lambda,
                    'batch_size': batch_size,
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

    def _prepare_visualization_data(
        self,
        episode_rewards: List[float],
        policy_losses: List[float],
        value_losses: List[float],
        clip_fractions: List[float],
        kl_divergences: List[float],
        episode_lengths: List[int],
        sample_trajectories: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            episode_rewards: Rewards for each episode
            policy_losses: Policy loss over training
            value_losses: Value loss over training
            clip_fractions: Clip fractions over training
            kl_divergences: KL divergences over training
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
        # Map update indices to episode indices (approximate)
        updates_per_episode = len(policy_losses) / len(episode_rewards) if episode_rewards else 1
        loss_data = []
        for i in range(len(policy_losses)):
            approx_episode = int(i / updates_per_episode) if updates_per_episode > 0 else i
            loss_data.append({
                'update': i,
                'episode': approx_episode,
                'policy_loss': float(policy_losses[i]),
                'value_loss': float(value_losses[i])
            })

        # Prepare clip fraction data
        clip_data = [
            {
                'update': i,
                'episode': int(i / updates_per_episode) if updates_per_episode > 0 else i,
                'clip_fraction': float(cf)
            }
            for i, cf in enumerate(clip_fractions)
        ]

        # Prepare KL divergence data
        kl_data = [
            {
                'update': i,
                'episode': int(i / updates_per_episode) if updates_per_episode > 0 else i,
                'kl_divergence': float(kl)
            }
            for i, kl in enumerate(kl_divergences)
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
            'clip_data': clip_data,
            'kl_data': kl_data,
            'length_data': length_data,
            'trajectories': sample_trajectories,
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
            action_probs = self.actor.get_action_probs(state_tensor)
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
