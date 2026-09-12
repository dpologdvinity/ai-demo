"""
Q-Learning implementation for reinforcement learning.

This module implements the Q-Learning algorithm with a simple grid world
environment for agent navigation.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
import numpy as np


class GridWorld:
    """Simple grid world environment for Q-Learning.

    The agent navigates a grid to reach a goal while avoiding obstacles.

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


class QLearningModel:
    """Q-Learning algorithm for reinforcement learning.

    This class implements the Q-Learning algorithm with epsilon-greedy
    exploration strategy.

    Attributes:
        env: GridWorld environment
        q_table: Q-value table (states x actions)
        learning_rate: Learning rate (alpha)
        discount_factor: Discount factor (gamma)
        epsilon: Exploration rate
        training_time_ms: Time taken to train in milliseconds
    """

    def __init__(self):
        """Initialize the Q-Learning model."""
        self.env: Optional[GridWorld] = None
        self.q_table: Optional[np.ndarray] = None
        self.learning_rate: float = 0.1
        self.discount_factor: float = 0.99
        self.epsilon: float = 0.1
        self.training_time_ms: float = 0.0

    def train(
        self,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 0.1,
        episodes: int = 1000,
        grid_size: int = 5,
        random_state: int = 42
    ) -> Dict[str, Any]:
        """Train the Q-Learning agent.

        Args:
            learning_rate: Learning rate (alpha) for Q-value updates
            discount_factor: Discount factor (gamma) for future rewards
            epsilon: Exploration rate for epsilon-greedy policy
            episodes: Number of training episodes
            grid_size: Size of the grid world
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
            if not 0 < learning_rate <= 1.0:
                raise ValueError("learning_rate must be in (0, 1]")
            if not 0.5 <= discount_factor < 1.0:
                raise ValueError("discount_factor must be in [0.5, 1)")
            if not 0.0 <= epsilon <= 1.0:
                raise ValueError("epsilon must be in [0, 1]")
            if episodes < 100:
                raise ValueError("episodes must be at least 100")
            if not 3 <= grid_size <= 10:
                raise ValueError("grid_size must be between 3 and 10")

            # Initialize environment and Q-table
            self.env = GridWorld(grid_size=grid_size, random_state=random_state)
            self.learning_rate = learning_rate
            self.discount_factor = discount_factor
            self.epsilon = epsilon

            n_states = grid_size * grid_size
            n_actions = 4  # up, right, down, left
            self.q_table = np.zeros((n_states, n_actions))

            # Training tracking
            start_time = time.time()
            rewards_per_episode = []
            steps_per_episode = []
            q_table_snapshots = []
            sample_trajectories = []

            # Snapshot intervals for Q-table evolution
            snapshot_intervals = [0, episodes // 4, episodes // 2, 3 * episodes // 4, episodes - 1]

            # Train for specified episodes
            for episode in range(episodes):
                state = self.env.reset()
                state_idx = self.env.get_state_index(state)

                episode_reward = 0.0
                steps = 0
                done = False
                trajectory = [state] if episode in [0, episodes // 2, episodes - 1] else None

                while not done and steps < grid_size * grid_size * 2:  # Limit steps per episode
                    # Epsilon-greedy action selection
                    if np.random.random() < epsilon:
                        action = np.random.randint(0, n_actions)  # Explore
                    else:
                        action = np.argmax(self.q_table[state_idx])  # Exploit

                    # Take action
                    next_state, reward, done = self.env.step(action)
                    next_state_idx = self.env.get_state_index(next_state)

                    # Q-learning update
                    old_q = self.q_table[state_idx, action]
                    next_max_q = np.max(self.q_table[next_state_idx])
                    new_q = old_q + learning_rate * (reward + discount_factor * next_max_q - old_q)
                    self.q_table[state_idx, action] = new_q

                    # Update state
                    state = next_state
                    state_idx = next_state_idx
                    episode_reward += reward
                    steps += 1

                    if trajectory is not None:
                        trajectory.append(state)

                rewards_per_episode.append(episode_reward)
                steps_per_episode.append(steps)

                # Save Q-table snapshots at intervals
                if episode in snapshot_intervals:
                    q_table_snapshots.append({
                        'episode': episode,
                        'q_table': self.q_table.copy()
                    })

                # Save sample trajectories
                if trajectory is not None:
                    sample_trajectories.append({
                        'episode': episode,
                        'trajectory': trajectory,
                        'reward': episode_reward,
                        'steps': steps
                    })

            end_time = time.time()
            self.training_time_ms = (end_time - start_time) * 1000

            # Calculate metrics
            avg_reward_last_100 = np.mean(rewards_per_episode[-100:])
            avg_steps_last_100 = np.mean(steps_per_episode[-100:])
            success_rate = sum(1 for r in rewards_per_episode[-100:] if r > 5) / 100.0

            # Extract learned policy
            policy = self._extract_policy()

            # Prepare visualization data
            visualization_data = self._prepare_visualization_data(
                q_table_snapshots,
                rewards_per_episode,
                steps_per_episode,
                sample_trajectories,
                policy
            )

            return {
                'success': True,
                'metrics': {
                    'avg_reward_last_100': float(avg_reward_last_100),
                    'avg_steps_last_100': float(avg_steps_last_100),
                    'success_rate': float(success_rate),
                    'total_episodes': int(episodes),
                    'final_episode_reward': float(rewards_per_episode[-1]),
                    'best_episode_reward': float(max(rewards_per_episode))
                },
                'visualization_data': visualization_data,
                'execution_time_ms': self.training_time_ms,
                'parameters_used': {
                    'learning_rate': learning_rate,
                    'discount_factor': discount_factor,
                    'epsilon': epsilon,
                    'episodes': episodes,
                    'grid_size': grid_size,
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
            'action_labels': ['Up', 'Right', 'Down', 'Left']
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
