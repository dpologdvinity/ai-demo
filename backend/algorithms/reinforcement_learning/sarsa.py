"""
SARSA implementation for reinforcement learning.

This module implements the SARSA (State-Action-Reward-State-Action) algorithm,
an on-policy TD control algorithm for learning action-value functions. SARSA is
safer than Q-Learning as it learns about the policy it follows, making it more
conservative in uncertain environments.
"""

import time
from typing import Dict, List, Any, Optional, Tuple, Union
import numpy as np
import gymnasium as gym


class SARSAGridWorld:
    """Simple grid world environment for SARSA learning.

    The agent navigates a grid to reach a goal while avoiding obstacles.
    This can be the same environment as Q-Learning but provides a comparison
    between on-policy (SARSA) and off-policy (Q-Learning) methods.

    Attributes:
        grid_size: Size of the square grid (grid_size x grid_size)
        start_pos: Starting position of the agent (row, col)
        goal_pos: Goal position (row, col)
        obstacles: Set of obstacle positions
        current_pos: Current position of the agent
    """

    def __init__(self, grid_size: int = 5, random_state: int = 42):
        """Initialize the grid world environment.

        Args:
            grid_size: Size of the square grid
            random_state: Random seed for reproducibility
        """
        self.grid_size = grid_size
        self.rng = np.random.RandomState(random_state)

        # Define start and goal positions
        self.start_pos = (0, 0)
        self.goal_pos = (grid_size - 1, grid_size - 1)

        # Generate random obstacles (avoid start and goal)
        self.obstacles = set()
        num_obstacles = max(1, grid_size // 2)

        while len(self.obstacles) < num_obstacles:
            pos = (self.rng.randint(0, grid_size), self.rng.randint(0, grid_size))
            if pos != self.start_pos and pos != self.goal_pos:
                self.obstacles.add(pos)

        self.current_pos = self.start_pos

    def reset(self) -> Tuple[int, int]:
        """Reset the environment to the starting position.

        Returns:
            Starting position (row, col)
        """
        self.current_pos = self.start_pos
        return self.current_pos

    def step(self, action: int) -> Tuple[Tuple[int, int], float, bool]:
        """Take a step in the environment.

        Args:
            action: Action to take (0=up, 1=right, 2=down, 3=left)

        Returns:
            Tuple of (next_state, reward, done)
        """
        # Define action effects
        actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # up, right, down, left

        # Calculate next position
        delta = actions[action]
        next_pos = (self.current_pos[0] + delta[0], self.current_pos[1] + delta[1])

        # Check boundaries
        if not (0 <= next_pos[0] < self.grid_size and 0 <= next_pos[1] < self.grid_size):
            # Hit wall - stay in place, negative reward
            return self.current_pos, -1.0, False

        # Check obstacles
        if next_pos in self.obstacles:
            # Hit obstacle - stay in place, negative reward
            return self.current_pos, -1.0, False

        # Valid move
        self.current_pos = next_pos

        # Check if goal reached
        if self.current_pos == self.goal_pos:
            return self.current_pos, 10.0, True  # Large positive reward for reaching goal

        # Small negative reward for each step (encourages shorter paths)
        return self.current_pos, -0.1, False

    def get_state_index(self, pos: Tuple[int, int]) -> int:
        """Convert position to state index.

        Args:
            pos: Position (row, col)

        Returns:
            State index
        """
        return pos[0] * self.grid_size + pos[1]

    def get_position_from_index(self, index: int) -> Tuple[int, int]:
        """Convert state index to position.

        Args:
            index: State index

        Returns:
            Position (row, col)
        """
        return (index // self.grid_size, index % self.grid_size)


class SARSAModel:
    """SARSA algorithm for reinforcement learning.

    SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm.
    Unlike Q-Learning which is off-policy and learns the optimal policy,
    SARSA learns about the policy it follows (including exploration).
    This makes SARSA more conservative and safer in environments with risks.

    Key difference from Q-Learning:
    - Q-Learning: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
    - SARSA: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]

    SARSA uses the actual next action a' (from epsilon-greedy policy),
    not the maximum Q-value, making it on-policy.

    Attributes:
        env: Gymnasium environment (CartPole or FrozenLake)
        gym_env: Actual gym environment instance
        q_table: Q-value table (states x actions)
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        epsilon: Exploration rate
        epsilon_decay: Epsilon decay rate
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self):
        """Initialize the SARSA model."""
        self.env: Optional[SARSAGridWorld] = None
        self.gym_env: Optional[gym.Env] = None
        self.q_table: Optional[np.ndarray] = None
        self.learning_rate: float = 0.1
        self.discount_factor: float = 0.99
        self.epsilon: float = 0.1
        self.epsilon_decay: float = 0.995
        self.training_time_ms: float = 0.0
        self.environment_name: str = "CartPole-v1"

    def _discretize_cartpole_state(self, state: np.ndarray, bins: Tuple[int, ...] = (6, 6, 6, 6)) -> int:
        """Discretize continuous CartPole state into discrete bins.

        Args:
            state: Continuous state [cart_pos, cart_vel, pole_angle, pole_vel]
            bins: Number of bins for each dimension

        Returns:
            Discrete state index
        """
        # Define bounds for each state dimension
        bounds = [
            (-2.4, 2.4),      # cart position
            (-3.0, 3.0),      # cart velocity
            (-0.25, 0.25),    # pole angle (radians)
            (-2.0, 2.0)       # pole angular velocity
        ]

        state_idx = 0
        multiplier = 1

        for i in range(len(state)):
            # Clip state to bounds
            val = np.clip(state[i], bounds[i][0], bounds[i][1])
            # Discretize to bin
            bin_idx = int(np.floor((val - bounds[i][0]) / (bounds[i][1] - bounds[i][0]) * bins[i]))
            bin_idx = min(bin_idx, bins[i] - 1)  # Handle edge case
            state_idx += bin_idx * multiplier
            multiplier *= bins[i]

        return state_idx

    def _epsilon_greedy_action(self, state_idx: int, n_actions: int) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state_idx: Current state index
            n_actions: Number of available actions

        Returns:
            Selected action
        """
        if np.random.random() < self.epsilon:
            return np.random.randint(0, n_actions)  # Explore
        else:
            return int(np.argmax(self.q_table[state_idx]))  # Exploit

    def train(
        self,
        environment: str = 'CartPole-v1',
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        epsilon_decay: float = 0.995,
        episodes: int = 500,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the SARSA agent.

        Args:
            environment: Environment name ('CartPole-v1' or 'FrozenLake-v1')
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Initial exploration rate for epsilon-greedy policy
            epsilon_decay: Epsilon decay rate per episode
            episodes: Number of training episodes
            random_state: Random seed for reproducibility

        Returns:
            Dictionary containing:
                - success: Whether training was successful
                - metrics: Performance metrics
                - visualization_data: Q-table evolution, rewards, trajectories
                - execution_time_ms: Training time in milliseconds
                - parameters_used: Actual parameters used for training
        """
        try:
            # Validate parameters
            if not 0.01 <= learning_rate <= 1.0:
                raise ValueError("learning_rate must be in [0.01, 1.0]")
            if not 0.8 <= discount_factor <= 1.0:
                raise ValueError("discount_factor must be in [0.8, 1.0]")
            if not 0.0 <= epsilon <= 1.0:
                raise ValueError("epsilon must be in [0, 1]")
            if not 0.9 <= epsilon_decay <= 1.0:
                raise ValueError("epsilon_decay must be in [0.9, 1.0]")
            if episodes < 100:
                raise ValueError("episodes must be at least 100")
            if environment not in ['CartPole-v1', 'FrozenLake-v1']:
                raise ValueError("environment must be 'CartPole-v1' or 'FrozenLake-v1'")

            # Set random seed
            np.random.seed(random_state)

            # Initialize environment
            self.environment_name = environment
            self.gym_env = gym.make(environment)
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.epsilon = epsilon
            self.epsilon_decay = epsilon_decay

            # Determine state and action space sizes
            if environment == 'CartPole-v1':
                # Discretize CartPole state space
                bins_per_dim = (6, 6, 6, 6)
                n_states = np.prod(bins_per_dim)
                n_actions = 2  # left, right
            else:  # FrozenLake-v1
                n_states = self.gym_env.observation_space.n
                n_actions = self.gym_env.action_space.n

            self.q_table = np.zeros((n_states, n_actions))

            # Training tracking
            start_time = time.time()
            rewards_per_episode = []
            steps_per_episode = []
            q_table_snapshots = []
            epsilon_history = []

            # Snapshot intervals for Q-table evolution
            snapshot_intervals = [0, episodes // 4, episodes // 2, 3 * episodes // 4, episodes - 1]

            current_epsilon = epsilon
            max_steps = 500 if environment == 'CartPole-v1' else 100

            # Train for specified episodes
            for episode in range(episodes):
                state, info = self.gym_env.reset(seed=random_state + episode)

                # Get state index
                if environment == 'CartPole-v1':
                    state_idx = self._discretize_cartpole_state(state)
                else:
                    state_idx = state

                # SARSA: Select initial action using epsilon-greedy policy
                action = self._epsilon_greedy_action(state_idx, n_actions)

                episode_reward = 0.0
                steps = 0
                done = False
                truncated = False

                while not (done or truncated) and steps < max_steps:
                    # Take action
                    next_state, reward, done, truncated, info = self.gym_env.step(action)

                    # Get next state index
                    if environment == 'CartPole-v1':
                        next_state_idx = self._discretize_cartpole_state(next_state)
                    else:
                        next_state_idx = next_state

                    # SARSA: Select next action using epsilon-greedy policy
                    next_action = self._epsilon_greedy_action(next_state_idx, n_actions)

                    # SARSA update: Use the actual next action (not max Q-value)
                    # Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
                    old_q = self.q_table[state_idx, action]
                    next_q = self.q_table[next_state_idx, next_action]
                    new_q = old_q + learning_rate * (reward + discount_factor * next_q - old_q)
                    self.q_table[state_idx, action] = new_q

                    # Update state and action for next iteration
                    state_idx = next_state_idx
                    action = next_action  # SARSA uses the selected next action

                    episode_reward += reward
                    steps += 1

                rewards_per_episode.append(episode_reward)
                steps_per_episode.append(steps)
                epsilon_history.append(current_epsilon)

                # Decay epsilon
                current_epsilon *= epsilon_decay
                self.epsilon = current_epsilon

                # Save Q-table snapshots at intervals
                if episode in snapshot_intervals:
                    q_table_snapshots.append({
                        'episode': episode,
                        'q_table': self.q_table.copy()
                    })

            end_time = time.time()
            self.training_time_ms = (end_time - start_time) * 1000

            # Close environment
            self.gym_env.close()

            # Calculate metrics
            window = min(100, len(rewards_per_episode))
            avg_reward_last_100 = float(np.mean(rewards_per_episode[-window:]))
            avg_steps_last_100 = float(np.mean(steps_per_episode[-window:]))

            # Calculate success rate based on environment
            if environment == 'CartPole-v1':
                success_rate = sum(1 for r in rewards_per_episode[-window:] if r >= 195) / window
            else:  # FrozenLake-v1
                success_rate = sum(1 for r in rewards_per_episode[-window:] if r > 0) / window

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data_gym(
                q_table_snapshots,
                rewards_per_episode,
                steps_per_episode,
                epsilon_history,
                environment
            )

            return {
                'success': True,
                'metrics': {
                    'avg_reward_last_100': avg_reward_last_100,
                    'avg_steps_last_100': avg_steps_last_100,
                    'success_rate': float(success_rate),
                    'total_episodes': episodes,
                    'final_episode_reward': float(rewards_per_episode[-1]),
                    'best_episode_reward': float(max(rewards_per_episode)),
                    'final_epsilon': float(current_epsilon)
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'environment': environment,
                    'learning_rate': learning_rate,
                    'discount_factor': discount_factor,
                    'epsilon': epsilon,
                    'epsilon_decay': epsilon_decay,
                    'episodes': episodes,
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

    def _prepare_visualization_data_gym(
        self,
        q_table_snapshots: List[Dict[str, Any]],
        rewards_per_episode: List[float],
        steps_per_episode: List[int],
        epsilon_history: List[float],
        environment: str
    ) -> Dict[str, Any]:
        """Prepare visualization data for gym environments.

        Args:
            q_table_snapshots: Q-table at different training stages
            rewards_per_episode: Rewards for each episode
            steps_per_episode: Steps taken in each episode
            epsilon_history: Epsilon values over episodes
            environment: Environment name

        Returns:
            Dictionary with visualization data
        """
        # Calculate smoothed rewards
        window_size = min(50, len(rewards_per_episode))
        smoothed_rewards = []
        for i in range(len(rewards_per_episode)):
            start_idx = max(0, i - window_size + 1)
            avg = np.mean(rewards_per_episode[start_idx:i+1])
            smoothed_rewards.append(float(avg))

        # Prepare reward chart data
        reward_data = [
            {
                'episode': i,
                'reward': float(r),
                'smoothed': smoothed_rewards[i]
            }
            for i, r in enumerate(rewards_per_episode)
        ]

        # Prepare steps chart data
        steps_data = [
            {
                'episode': i,
                'steps': int(s)
            }
            for i, s in enumerate(steps_per_episode)
        ]

        # Prepare epsilon decay data
        epsilon_data = [
            {
                'episode': i,
                'epsilon': float(e)
            }
            for i, e in enumerate(epsilon_history)
        ]

        # Prepare Q-value statistics
        q_value_stats = []
        for snapshot in q_table_snapshots:
            q_table = snapshot['q_table']
            q_value_stats.append({
                'episode': snapshot['episode'],
                'mean_q': float(np.mean(q_table)),
                'max_q': float(np.max(q_table)),
                'min_q': float(np.min(q_table)),
                'std_q': float(np.std(q_table))
            })

        # Extract policy (best action for each state)
        policy = [int(np.argmax(self.q_table[s])) for s in range(self.q_table.shape[0])]

        # Calculate convergence metrics
        convergence_window = min(20, len(rewards_per_episode) // 10)
        if len(rewards_per_episode) >= convergence_window * 2:
            recent_avg = np.mean(rewards_per_episode[-convergence_window:])
            earlier_avg = np.mean(rewards_per_episode[-2*convergence_window:-convergence_window])
            improvement = float((recent_avg - earlier_avg) / max(abs(earlier_avg), 1e-6))
        else:
            improvement = 0.0

        return {
            'environment': environment,
            'reward_data': reward_data,
            'steps_data': steps_data,
            'epsilon_data': epsilon_data,
            'q_value_stats': q_value_stats,
            'policy': policy,
            'algorithm_type': 'SARSA (On-Policy)',
            'convergence_improvement': improvement,
            'comparison_note': (
                'SARSA learns about the policy it follows (on-policy), '
                'making it more conservative than Q-Learning (off-policy). '
                'SARSA considers exploration in its updates, leading to safer behavior '
                'in environments with risks or cliffs.'
            )
        }

    def _extract_policy(self) -> List[List[int]]:
        """Extract the learned policy from Q-table.

        Returns:
            2D grid of best actions for each state
        """
        if self.q_table is None or self.env is None:
            return []

        grid_size = self.env.grid_size
        policy = []

        for row in range(grid_size):
            policy_row = []
            for col in range(grid_size):
                state_idx = self.env.get_state_index((row, col))
                best_action = int(np.argmax(self.q_table[state_idx]))
                policy_row.append(best_action)
            policy.append(policy_row)

        return policy

    def _prepare_visualization_data(
        self,
        q_table_snapshots: List[Dict[str, Any]],
        rewards_per_episode: List[float],
        steps_per_episode: List[int],
        sample_trajectories: List[Dict[str, Any]],
        policy: List[List[int]]
    ) -> Dict[str, Any]:
        """Prepare data for visualization.

        Args:
            q_table_snapshots: Q-table at different training stages
            rewards_per_episode: Rewards for each episode
            steps_per_episode: Steps taken in each episode
            sample_trajectories: Sample agent trajectories
            policy: Learned policy grid

        Returns:
            Dictionary with visualization data
        """
        if self.env is None:
            return {}

        # Convert Q-table to max Q-values per state for heatmap
        q_value_heatmaps = []
        for snapshot in q_table_snapshots:
            q_table = snapshot['q_table']
            max_q_values = np.max(q_table, axis=1).reshape(self.env.grid_size, self.env.grid_size)
            q_value_heatmaps.append({
                'episode': snapshot['episode'],
                'values': max_q_values.tolist()
            })

        # Prepare grid world info
        grid_info = {
            'size': self.env.grid_size,
            'start': list(self.env.start_pos),
            'goal': list(self.env.goal_pos),
            'obstacles': [list(pos) for pos in self.env.obstacles]
        }

        # Calculate cumulative rewards (smoothed)
        window_size = 50
        cumulative_rewards = []
        smoothed_rewards = []
        cumsum = 0
        for i, reward in enumerate(rewards_per_episode):
            cumsum += reward
            cumulative_rewards.append(cumsum)

            # Smoothed average
            start_idx = max(0, i - window_size + 1)
            avg = np.mean(rewards_per_episode[start_idx:i+1])
            smoothed_rewards.append(float(avg))

        # Prepare reward chart data
        reward_data = [
            {
                'episode': i,
                'reward': float(r),
                'smoothed': smoothed_rewards[i]
            }
            for i, r in enumerate(rewards_per_episode)
        ]

        # Prepare steps chart data
        steps_data = [
            {
                'episode': i,
                'steps': int(s)
            }
            for i, s in enumerate(steps_per_episode)
        ]

        return {
            'grid': grid_info,
            'policy': policy,
            'q_value_heatmaps': q_value_heatmaps,
            'reward_data': reward_data,
            'steps_data': steps_data,
            'trajectories': sample_trajectories,
            'action_labels': ['Up', 'Right', 'Down', 'Left'],
            'algorithm_type': 'SARSA (On-Policy)',
            'comparison_note': (
                'SARSA learns about the policy it follows (on-policy), '
                'making it more conservative than Q-Learning (off-policy). '
                'SARSA considers exploration in its updates, leading to safer behavior '
                'in environments with risks or cliffs.'
            )
        }

    def get_action(self, state: Tuple[int, int]) -> int:
        """Get the best action for a given state using the learned policy.

        Args:
            state: Current state (row, col)

        Returns:
            Best action (0=up, 1=right, 2=down, 3=left)

        Raises:
            RuntimeError: If model hasn't been trained yet
        """
        if self.q_table is None or self.env is None:
            raise RuntimeError("Model must be trained before getting actions")

        state_idx = self.env.get_state_index(state)
        return int(np.argmax(self.q_table[state_idx]))
