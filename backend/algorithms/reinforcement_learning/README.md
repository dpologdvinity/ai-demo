# Q-Learning Implementation

This directory contains the Q-Learning reinforcement learning algorithm implementation for the AI Algorithms Demo website.

## Overview

Q-Learning is a model-free reinforcement learning algorithm that learns the optimal policy by estimating the value of state-action pairs. The agent learns to navigate a grid world environment from a start position to a goal while avoiding obstacles.

## Files

- **`q_learning.py`**: Core Q-Learning algorithm implementation
  - `GridWorld`: Simple grid environment with start, goal, and obstacles
  - `QLearningModel`: Q-Learning agent with training and policy extraction

- **`q_learning_schema.py`**: Pydantic schemas for API requests/responses
  - `QLearningRequest`: Training parameters
  - `QLearningResponse`: Training results and visualization data
  - `QLearningInfoResponse`: Algorithm metadata

- **`__init__.py`**: Package exports

## Algorithm Details

### Environment: GridWorld

A simple 2D grid world where:
- Agent starts at position (0, 0)
- Goal is at position (grid_size-1, grid_size-1)
- Random obstacles are placed in the grid
- Agent can move in 4 directions: up, right, down, left

**Rewards:**
- +10.0: Reaching the goal
- -0.1: Each step (encourages shorter paths)
- -1.0: Hitting walls or obstacles

### Q-Learning Algorithm

**Update Rule:**
```
Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)]
```

Where:
- `α` (alpha): Learning rate
- `γ` (gamma): Discount factor
- `r`: Immediate reward
- `s`: Current state
- `a`: Action taken
- `s'`: Next state

**Exploration Strategy:**
- Epsilon-greedy policy
- With probability ε: take random action (explore)
- With probability 1-ε: take best known action (exploit)

## Parameters

| Parameter | Type | Default | Range | Description |
|-----------|------|---------|-------|-------------|
| `learning_rate` | float | 0.1 | 0.01-1.0 | How much to update Q-values each step |
| `discount_factor` | float | 0.99 | 0.5-0.99 | Weight of future rewards |
| `epsilon` | float | 0.1 | 0.0-1.0 | Exploration rate |
| `episodes` | int | 1000 | 100-5000 | Number of training episodes |
| `grid_size` | int | 5 | 3-10 | Size of the grid world |
| `random_state` | int | 42 | - | Random seed for reproducibility |

## API Endpoints

### Train Q-Learning Agent

**Endpoint:** `POST /api/reinforcement-learning/q-learning/train`

**Request Body:**
```json
{
  "learning_rate": 0.1,
  "discount_factor": 0.99,
  "epsilon": 0.1,
  "episodes": 1000,
  "grid_size": 5,
  "random_state": 42
}
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "avg_reward_last_100": 8.5,
    "avg_steps_last_100": 12.3,
    "success_rate": 0.95,
    "total_episodes": 1000,
    "final_episode_reward": 9.2,
    "best_episode_reward": 9.9
  },
  "visualization_data": {
    "grid": {
      "size": 5,
      "start": [0, 0],
      "goal": [4, 4],
      "obstacles": [[1, 2], [2, 3]]
    },
    "policy": [[1, 1, 2, 2, 2], ...],
    "q_value_heatmaps": [...],
    "reward_data": [...],
    "steps_data": [...],
    "trajectories": [...]
  },
  "execution_time_ms": 125.3,
  "parameters_used": {...}
}
```

### Get Algorithm Info

**Endpoint:** `GET /api/reinforcement-learning/q-learning/info`

**Response:**
```json
{
  "metadata": {
    "id": "q-learning",
    "name": "Q-Learning",
    "category": "reinforcement_learning",
    "description": "Model-free RL algorithm...",
    "difficulty": "Intermediate",
    ...
  },
  "environment_info": {
    "type": "GridWorld",
    "state_space": "Discrete (grid positions)",
    "action_space": "Discrete (4 actions)",
    ...
  }
}
```

## Visualization Data

The training results include comprehensive visualization data:

### 1. Grid World
- Grid dimensions
- Start and goal positions
- Obstacle locations

### 2. Learned Policy
- 2D array showing best action for each state
- Actions: 0=up, 1=right, 2=down, 3=left

### 3. Q-Value Heatmaps
- Snapshots of Q-values at different training stages (episodes 0, 250, 500, 750, 999)
- Shows how Q-values evolve during training

### 4. Reward Curves
- Reward per episode
- Smoothed average (50-episode window)
- Shows learning progress over time

### 5. Steps per Episode
- Number of steps taken in each episode
- Indicates policy efficiency

### 6. Sample Trajectories
- Agent paths at beginning, middle, and end of training
- Shows improvement in navigation

## Usage Example

```python
from algorithms.reinforcement_learning import QLearningModel

# Create and train model
model = QLearningModel()
result = model.train(
    learning_rate=0.1,
    discount_factor=0.99,
    epsilon=0.1,
    episodes=1000,
    grid_size=5,
    random_state=42
)

# Check results
if result['success']:
    print(f"Success rate: {result['metrics']['success_rate']:.2%}")
    print(f"Average reward: {result['metrics']['avg_reward_last_100']:.2f}")

    # Get learned policy
    policy = result['visualization_data']['policy']

    # Use model for inference
    action = model.get_action((0, 0))
```

## Testing

Run the standalone test suite:
```bash
cd backend
python test_q_learning_standalone.py
```

Or use pytest (requires dependencies):
```bash
cd backend
pytest tests/test_q_learning.py -v
```

## Implementation Notes

### Complexity
- **Time Complexity:** O(episodes × steps_per_episode)
- **Space Complexity:** O(states × actions) = O(grid_size² × 4)

### Convergence
The agent typically shows learning progress within 100-500 episodes on small grids (3x3 to 5x5). Larger grids or more obstacles require more episodes for optimal convergence.

### Hyperparameter Tuning
- **High learning rate (>0.3):** Fast learning but may be unstable
- **Low learning rate (<0.05):** Stable but slow learning
- **High epsilon (>0.3):** More exploration, slower convergence
- **Low epsilon (<0.05):** More exploitation, may get stuck in local optima
- **High discount factor (>0.95):** Focus on long-term rewards
- **Low discount factor (<0.8):** Focus on immediate rewards

## Future Enhancements

Potential improvements:
- [ ] Epsilon decay schedule
- [ ] Experience replay
- [ ] Double Q-Learning
- [ ] SARSA comparison
- [ ] Continuous state spaces with function approximation
- [ ] Multi-goal environments
- [ ] Dynamic obstacles
