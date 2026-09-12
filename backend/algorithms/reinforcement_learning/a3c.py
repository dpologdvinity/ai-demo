"""
A3C (Asynchronous Advantage Actor-Critic) implementation for reinforcement learning.

This module implements the A3C algorithm with parallel workers sharing a global model
for the CartPole-v1 environment from OpenAI Gym.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.multiprocessing as mp
from collections import deque


class ActorCriticNetwork(nn.Module):
    """Combined actor-critic network for A3C.

    Maps states to both action probability distributions (actor)
    and state values (critic).

    Attributes:
        fc1: First shared fully connected layer
        fc2: Second shared fully connected layer
        actor_head: Actor output layer (action logits)
        critic_head: Critic output layer (state value)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_size: int = 128):
        """Initialize Actor-Critic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Number of actions
            hidden_size: Hidden layer dimension
        """
        super(ActorCriticNetwork, self).__init__()

        # Shared layers
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        # Actor head (policy)
        self.actor_head = nn.Linear(hidden_size, action_dim)

        # Critic head (value function)
        self.critic_head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through network.

        Args:
            x: Input state tensor

        Returns:
            Tuple of (action_probs, value)
                - action_probs: Action probability distribution
                - value: State value estimate
        """
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # Actor output (policy)
        action_logits = self.actor_head(x)
        action_probs = F.softmax(action_logits, dim=-1)

        # Critic output (value)
        value = self.critic_head(x)

        return action_probs, value


class Worker:
    """A3C worker process.

    Each worker runs its own environment and periodically updates
    the shared global model with gradients.

    Attributes:
        worker_id: Unique worker identifier
        global_model: Shared global actor-critic model
        local_model: Worker's local copy of the model
        optimizer: Global optimizer shared across workers
        env_name: Gym environment name
        device: CPU/GPU device
    """

    def __init__(
        self,
        worker_id: int,
        global_model: ActorCriticNetwork,
        optimizer: optim.Optimizer,
        env_name: str,
        device: torch.device
    ):
        """Initialize worker.

        Args:
            worker_id: Worker ID
            global_model: Shared global model
            optimizer: Shared global optimizer
            env_name: Gym environment name
            device: Device to run on
        """
        self.worker_id = worker_id
        self.global_model = global_model
        self.local_model = ActorCriticNetwork(
            global_model.fc1.in_features,
            global_model.actor_head.out_features,
            global_model.fc1.out_features
        ).to(device)
        self.optimizer = optimizer
        self.env_name = env_name
        self.device = device

    def run(
        self,
        gamma: float,
        entropy_coef: float,
        episodes_per_worker: int,
        random_state: int,
        shared_dict: Dict,
        worker_results: mp.Queue
    ):
        """Run worker training loop.

        Args:
            gamma: Discount factor
            entropy_coef: Entropy coefficient for exploration
            episodes_per_worker: Number of episodes to run
            random_state: Random seed base
            shared_dict: Shared dictionary for synchronization
            worker_results: Queue to store worker results
        """
        import gymnasium as gym

        # Set worker-specific random seed
        torch.manual_seed(random_state + self.worker_id)
        np.random.seed(random_state + self.worker_id)

        # Create environment
        env = gym.make(self.env_name)

        # Worker-specific tracking
        episode_rewards = []
        episode_lengths = []
        advantage_estimates = []
        entropy_history = []

        for episode in range(episodes_per_worker):
            # Sync local model with global model
            self.local_model.load_state_dict(self.global_model.state_dict())

            state, _ = env.reset(seed=random_state + self.worker_id + episode * 1000)
            episode_reward = 0.0
            episode_length = 0

            # Episode storage
            log_probs = []
            values = []
            rewards = []
            entropies = []

            done = False
            truncated = False

            while not (done or truncated):
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

                # Get action probabilities and value from local model
                action_probs, value = self.local_model(state_tensor)

                # Sample action
                action_dist = torch.distributions.Categorical(action_probs)
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

                # Calculate entropy for exploration bonus
                entropy = action_dist.entropy()

                # Take action
                next_state, reward, done, truncated, _ = env.step(action.item())

                # Store transition
                log_probs.append(log_prob)
                values.append(value)
                rewards.append(reward)
                entropies.append(entropy)

                state = next_state
                episode_reward += reward
                episode_length += 1

            # Compute returns and advantages
            returns = []
            R = 0
            for r in reversed(rewards):
                R = r + gamma * R
                returns.insert(0, R)

            returns = torch.FloatTensor(returns).to(self.device)

            # Normalize returns
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)

            # Convert to tensors
            log_probs = torch.stack(log_probs)
            values = torch.stack(values).squeeze()
            entropies = torch.stack(entropies)

            # Compute advantages (TD error)
            advantages = returns - values.detach()

            # Store average advantage for this episode
            advantage_estimates.append(advantages.mean().item())

            # Compute losses
            actor_loss = -(log_probs * advantages).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy_loss = -entropies.mean()  # Negative because we want to maximize entropy

            # Total loss with entropy bonus
            total_loss = actor_loss + 0.5 * critic_loss + entropy_coef * entropy_loss

            # Backprop and update global model
            self.optimizer.zero_grad()
            total_loss.backward()

            # Clip gradients to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(self.local_model.parameters(), 1.0)

            # Copy gradients to global model and update
            for local_param, global_param in zip(
                self.local_model.parameters(),
                self.global_model.parameters()
            ):
                if global_param.grad is None:
                    global_param.grad = local_param.grad.clone()
                else:
                    global_param.grad += local_param.grad

            self.optimizer.step()

            # Track metrics
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            entropy_history.append(entropies.mean().item())

        env.close()

        # Store worker results
        worker_results.put({
            'worker_id': self.worker_id,
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'advantage_estimates': advantage_estimates,
            'entropy_history': entropy_history
        })


class A3CModel:
    """A3C (Asynchronous Advantage Actor-Critic) algorithm.

    Implements A3C with multiple parallel workers sharing a global model
    for the CartPole-v1 environment.

    Attributes:
        env_name: Gym environment name
        state_dim: Dimension of state space
        action_dim: Number of actions
        global_model: Shared global actor-critic model
        optimizer: Global optimizer
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self, env_name: str = "CartPole-v1"):
        """Initialize the A3C model.

        Args:
            env_name: OpenAI Gym environment name
        """
        self.env_name = env_name
        self.state_dim: Optional[int] = None
        self.action_dim: Optional[int] = None
        self.global_model: Optional[ActorCriticNetwork] = None
        self.optimizer: Optional[optim.Optimizer] = None
        self.training_time_ms: float = 0.0
        self.device = torch.device("cpu")  # A3C typically uses CPU for parallelism

    def train(
        self,
        num_workers: int = 4,
        actor_lr: float = 0.001,
        critic_lr: float = 0.005,
        gamma: float = 0.99,
        episodes_per_worker: int = 100,
        entropy_coef: float = 0.01,
        hidden_size: int = 128,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the A3C agent with multiple parallel workers.

        Args:
            num_workers: Number of parallel workers
            actor_lr: Learning rate for actor (policy)
            critic_lr: Learning rate for critic (value)
            gamma: Discount factor for future rewards
            episodes_per_worker: Episodes each worker runs
            entropy_coef: Entropy coefficient for exploration
            hidden_size: Hidden layer size
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics
                - visualization_data: Rewards per worker, advantages, entropy
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training
        """
        try:
            # Import gym here to avoid import errors if not installed
            import gymnasium as gym

            # Validate parameters
            if not 2 <= num_workers <= 8:
                raise ValueError("num_workers must be in [2, 8]")
            if not 0.0001 <= actor_lr <= 0.01:
                raise ValueError("actor_lr must be in [0.0001, 0.01]")
            if not 0.0001 <= critic_lr <= 0.01:
                raise ValueError("critic_lr must be in [0.0001, 0.01]")
            if not 0.9 <= gamma <= 0.999:
                raise ValueError("gamma must be in [0.9, 0.999]")
            if not 50 <= episodes_per_worker <= 500:
                raise ValueError("episodes_per_worker must be in [50, 500]")
            if not 0.0 <= entropy_coef <= 0.1:
                raise ValueError("entropy_coef must be in [0.0, 0.1]")

            # Set random seeds
            torch.manual_seed(random_state)
            np.random.seed(random_state)

            # Initialize environment to get dimensions
            env = gym.make(self.env_name)
            self.state_dim = env.observation_space.shape[0]
            self.action_dim = env.action_space.n
            env.close()

            # Initialize global model (shared across workers)
            self.global_model = ActorCriticNetwork(
                self.state_dim,
                self.action_dim,
                hidden_size
            ).to(self.device)
            self.global_model.share_memory()  # Share model across processes

            # Use a combined learning rate (average of actor and critic)
            combined_lr = (actor_lr + critic_lr) / 2
            self.optimizer = optim.Adam(self.global_model.parameters(), lr=combined_lr)

            # Shared dictionary for synchronization
            manager = mp.Manager()
            shared_dict = manager.dict()
            worker_results_queue = manager.Queue()

            # Start training
            start_time = time.time()

            # Create and start workers
            processes = []
            for worker_id in range(num_workers):
                worker = Worker(
                    worker_id,
                    self.global_model,
                    self.optimizer,
                    self.env_name,
                    self.device
                )

                # Create process
                p = mp.Process(
                    target=worker.run,
                    args=(
                        gamma,
                        entropy_coef,
                        episodes_per_worker,
                        random_state,
                        shared_dict,
                        worker_results_queue
                    )
                )
                p.start()
                processes.append(p)

            # Wait for all workers to complete
            for p in processes:
                p.join()

            end_time = time.time()
            self.training_time_ms = (end_time - start_time) * 1000

            # Collect results from all workers
            worker_results = []
            while not worker_results_queue.empty():
                worker_results.append(worker_results_queue.get())

            # Sort by worker_id for consistent ordering
            worker_results = sorted(worker_results, key=lambda x: x['worker_id'])

            # Aggregate metrics
            all_rewards = []
            all_lengths = []
            all_advantages = []
            all_entropies = []

            for result in worker_results:
                all_rewards.extend(result['episode_rewards'])
                all_lengths.extend(result['episode_lengths'])
                all_advantages.extend(result['advantage_estimates'])
                all_entropies.extend(result['entropy_history'])

            # Calculate metrics
            avg_reward = np.mean(all_rewards)
            max_reward = np.max(all_rewards)
            avg_length = np.mean(all_lengths)
            avg_advantage = np.mean(all_advantages)
            avg_entropy = np.mean(all_entropies)

            # Success rate (CartPole is solved when avg reward >= 195)
            success_threshold = 195.0
            success_rate = sum(1 for r in all_rewards if r >= success_threshold) / len(all_rewards)

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                worker_results,
                episodes_per_worker,
                num_workers
            )

            return {
                'success': True,
                'metrics': {
                    'avg_reward': float(avg_reward),
                    'max_reward': float(max_reward),
                    'avg_length': float(avg_length),
                    'avg_advantage': float(avg_advantage),
                    'avg_entropy': float(avg_entropy),
                    'success_rate': float(success_rate),
                    'total_episodes': int(len(all_rewards)),
                    'num_workers': int(num_workers)
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'num_workers': num_workers,
                    'actor_lr': actor_lr,
                    'critic_lr': critic_lr,
                    'gamma': gamma,
                    'episodes_per_worker': episodes_per_worker,
                    'entropy_coef': entropy_coef,
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
        worker_results: List[Dict[str, Any]],
        episodes_per_worker: int,
        num_workers: int
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            worker_results: Results from all workers
            episodes_per_worker: Episodes per worker
            num_workers: Number of workers

        Returns:
            Dictionary with visualization data
        """
        # Prepare per-worker reward data
        worker_reward_data = []
        for result in worker_results:
            worker_id = result['worker_id']
            rewards = result['episode_rewards']

            for episode_idx, reward in enumerate(rewards):
                worker_reward_data.append({
                    'worker_id': worker_id,
                    'episode': episode_idx,
                    'reward': float(reward)
                })

        # Prepare advantage estimates data
        advantage_data = []
        for result in worker_results:
            worker_id = result['worker_id']
            advantages = result['advantage_estimates']

            for episode_idx, advantage in enumerate(advantages):
                advantage_data.append({
                    'worker_id': worker_id,
                    'episode': episode_idx,
                    'advantage': float(advantage)
                })

        # Prepare entropy data (exploration measure)
        entropy_data = []
        for result in worker_results:
            worker_id = result['worker_id']
            entropies = result['entropy_history']

            for episode_idx, entropy in enumerate(entropies):
                entropy_data.append({
                    'worker_id': worker_id,
                    'episode': episode_idx,
                    'entropy': float(entropy)
                })

        # Calculate worker statistics
        worker_stats = []
        for result in worker_results:
            worker_id = result['worker_id']
            rewards = result['episode_rewards']
            lengths = result['episode_lengths']

            worker_stats.append({
                'worker_id': worker_id,
                'avg_reward': float(np.mean(rewards)),
                'max_reward': float(np.max(rewards)),
                'min_reward': float(np.min(rewards)),
                'avg_length': float(np.mean(lengths)),
                'success_rate': float(sum(1 for r in rewards if r >= 195.0) / len(rewards))
            })

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
            'worker_rewards': worker_reward_data,
            'advantages': advantage_data,
            'entropy': entropy_data,
            'worker_stats': worker_stats,
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
        if self.global_model is None:
            raise RuntimeError("Model must be trained before getting actions")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action_probs, _ = self.global_model(state_tensor)
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
        if self.global_model is None:
            raise RuntimeError("Model must be trained before getting values")

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            _, value = self.global_model(state_tensor)
            return value.item()
