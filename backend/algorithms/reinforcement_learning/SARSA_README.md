# SARSA Implementation

## Overview

SARSA (State-Action-Reward-State-Action) is an on-policy temporal-difference control algorithm for reinforcement learning. This implementation has been updated to support modern Gymnasium environments (CartPole-v1 and FrozenLake-v1) with epsilon decay for improved exploration-exploitation balance.

## Algorithm Details

- **Name**: SARSA
- **Slug**: `sarsa`
- **Category**: Reinforcement Learning
- **Description**: On-policy TD control algorithm that learns Q-values while following the current policy
- **Difficulty**: Intermediate
- **Complexity**: 
  - Time: O(episodes × steps)
  - Space: O(states × actions)

## Key Features

### On-Policy Learning
Unlike Q-Learning (off-policy), SARSA learns about the policy it actually follows, including exploration. This makes it more conservative and safer in environments with risks or cliffs.

### Update Rule
```
Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]
```

Where:
- `α` (alpha) = learning rate
- `γ` (gamma) = discount factor
- `r` = reward
- `s'` = next state
- `a'` = actual next action taken (not max Q-value)

### Epsilon Decay
The implementation includes epsilon decay, which gradually reduces exploration over time:
```
ε(t+1) = ε(t) × decay_rate
```

## Supported Environments

### CartPole-v1
- **Description**: Balance a pole on a cart by moving left or right
- **State Space**: Continuous 4D (cart position, velocity, pole angle, angular velocity) - discretized into bins
- **Action Space**: Discrete 2 actions (push left, push right)
- **Success Criterion**: Average reward ≥ 195 over 100 consecutive episodes

### FrozenLake-v1
- **Description**: Navigate a frozen lake from start to goal, avoiding holes
- **State Space**: Discrete 16 states in 4×4 grid
- **Action Space**: Discrete 4 actions (left, down, right, up)
- **Success Criterion**: Reach the goal without falling into holes

## Parameters

1. **environment** (select)
   - Default: `'CartPole-v1'`
   - Options: `['CartPole-v1', 'FrozenLake-v1']`
   - Description: Gymnasium environment for training

2. **learning_rate** (range)
   - Default: 0.1
   - Range: [0.01, 1.0]
   - Description: Learning rate for Q-value updates

3. **discount_factor** (range)
   - Default: 0.99
   - Range: [0.8, 1.0]
   - Description: Discount factor for future rewards

4. **epsilon** (range)
   - Default: 0.1
   - Range: [0.0, 1.0]
   - Description: Initial exploration rate

5. **epsilon_decay** (range)
   - Default: 0.995
   - Range: [0.9, 1.0]
   - Description: Epsilon decay rate per episode

6. **episodes** (range)
   - Default: 500
   - Range: [100, 2000]
   - Description: Number of training episodes

## API Endpoints

### POST `/api/reinforcement-learning/sarsa/train`

Train a SARSA agent in the specified environment.

**Request Body:**
```json
{
  "environment": "CartPole-v1",
  "learning_rate": 0.1,
  "discount_factor": 0.99,
  "epsilon": 0.1,
  "epsilon_decay": 0.995,
  "episodes": 500,
  "random_state": 42
}
```

**Response:**
```json
{
  "success": true,
  "metrics": {
    "avg_reward_last_100": 195.5,
    "avg_steps_last_100": 195.5,
    "success_rate": 0.95,
    "total_episodes": 500,
    "final_episode_reward": 200.0,
    "best_episode_reward": 200.0,
    "final_epsilon": 0.0822
  },
  "visualization_data": {
    "environment": "CartPole-v1",
    "reward_data": [...],
    "steps_data": [...],
    "epsilon_data": [...],
    "q_value_stats": [...],
    "policy": [...],
    "algorithm_type": "SARSA (On-Policy)",
    "convergence_improvement": 0.156,
    "comparison_note": "..."
  },
  "execution_time_ms": 2534.5,
  "parameters_used": {
    "environment": "CartPole-v1",
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon": 0.1,
    "epsilon_decay": 0.995,
    "episodes": 500,
    "random_state": 42
  }
}
```

### GET `/api/reinforcement-learning/sarsa/info`

Get SARSA algorithm information and metadata.

**Response:**
```json
{
  "metadata": {
    "id": "sarsa",
    "name": "SARSA",
    "slug": "sarsa",
    "category": "reinforcement_learning",
    "description": "On-policy TD control algorithm that learns Q-values while following the current policy",
    "difficulty": "Intermediate",
    "tags": [...],
    "use_cases": [...],
    "complexity": {...},
    "parameters": [...],
    "theory": "...",
    "pros": [...],
    "cons": [...]
  },
  "environment_info": {
    "type": "CartPole-v1 / FrozenLake-v1",
    "cartpole": {...},
    "frozenlake": {...},
    "algorithm_features": {...},
    "comparison": {...}
  }
}
```

## Visualization Data

The training results include comprehensive visualization data:

1. **Reward Data**: Episode rewards over time with smoothed moving average
2. **Steps Data**: Number of steps per episode
3. **Epsilon Data**: Epsilon decay curve showing exploration reduction
4. **Q-Value Statistics**: Mean, max, min, and standard deviation of Q-values at different training stages
5. **Policy**: Learned policy (best action for each state)
6. **Convergence Metrics**: Improvement rate over recent episodes

## Implementation Details

### State Discretization (CartPole)

For CartPole's continuous state space, the implementation discretizes each dimension into bins:

```python
bins_per_dim = (6, 6, 6, 6)  # 6 bins for each of 4 dimensions
total_states = 6^4 = 1296 discrete states
```

Bounds for discretization:
- Cart position: [-2.4, 2.4]
- Cart velocity: [-3.0, 3.0]
- Pole angle: [-0.25, 0.25] radians
- Pole angular velocity: [-2.0, 2.0]

### Epsilon-Greedy Policy

```python
def epsilon_greedy_action(state_idx, n_actions):
    if random() < epsilon:
        return random_action()  # Explore
    else:
        return argmax(Q[state_idx])  # Exploit
```

### Training Loop

```python
for episode in episodes:
    state = env.reset()
    action = epsilon_greedy(state)  # Initial action
    
    while not done:
        next_state, reward, done = env.step(action)
        next_action = epsilon_greedy(next_state)  # Select next action
        
        # SARSA update using next_action (not max Q-value)
        Q[state, action] += α * (reward + γ * Q[next_state, next_action] - Q[state, action])
        
        state = next_state
        action = next_action  # Use the selected action
    
    epsilon *= epsilon_decay  # Decay exploration rate
```

## Use Cases

1. **Robot Navigation**: Safe navigation in environments with obstacles
2. **Game Playing**: Learning game strategies while being cautious
3. **Control Problems**: Balancing, stabilization, and control tasks
4. **Sequential Decision Making**: Multi-step decision problems
5. **On-Policy Learning Scenarios**: When you want to learn about the policy being followed

## Advantages (Pros)

- On-policy learning about the actual policy being followed
- More conservative and safer than Q-Learning in risky environments
- Considers exploration in learning, leading to safer policies
- Simple to implement and understand
- Works well for discrete state and action spaces
- Better suited for online learning where safety matters

## Limitations (Cons)

- Slower convergence than Q-Learning in some environments
- May not find the optimal policy if exploration is suboptimal
- Doesn't scale well to large or continuous state spaces without discretization
- Q-table memory requirements grow with state-action space size
- Requires balancing exploration vs exploitation carefully
- May be overly conservative in safe environments

## Comparison with Q-Learning

| Aspect | SARSA (On-Policy) | Q-Learning (Off-Policy) |
|--------|-------------------|-------------------------|
| Update Rule | Uses actual next action a' | Uses max Q-value |
| Learning | Learns about policy it follows | Learns optimal policy |
| Safety | More conservative | More aggressive |
| Convergence | May be slower | Often faster |
| Use Case | Risky environments | Safe environments |

## File Structure

```
backend/
├── algorithms/
│   └── reinforcement_learning/
│       ├── sarsa.py              # Main SARSA implementation
│       ├── sarsa_schema.py       # Pydantic schemas
│       └── __init__.py           # Package exports
├── api/
│   └── routes/
│       └── reinforcement_learning.py  # API endpoints and metadata
└── utils/
    └── algorithm_metadata.py     # Metadata utilities
```

## Testing

```bash
cd backend
source venv/bin/activate

# Test CartPole
curl -X POST http://localhost:8000/api/reinforcement-learning/sarsa/train \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "CartPole-v1",
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon": 0.1,
    "epsilon_decay": 0.995,
    "episodes": 500
  }'

# Test FrozenLake
curl -X POST http://localhost:8000/api/reinforcement-learning/sarsa/train \
  -H "Content-Type: application/json" \
  -d '{
    "environment": "FrozenLake-v1",
    "learning_rate": 0.1,
    "discount_factor": 0.99,
    "epsilon": 0.2,
    "epsilon_decay": 0.995,
    "episodes": 1000
  }'

# Get algorithm info
curl http://localhost:8000/api/reinforcement-learning/sarsa/info
```

## Dependencies

- `numpy`: Numerical computations and Q-table management
- `gymnasium`: OpenAI Gym successor for RL environments
- `fastapi`: Web framework for API endpoints
- `pydantic`: Data validation and schemas

## Future Enhancements

1. **Expected SARSA**: Use expected Q-value instead of sampled Q(s',a')
2. **N-Step SARSA**: Use n-step returns for faster learning
3. **Function Approximation**: Replace Q-table with neural network for large state spaces
4. **Eligibility Traces**: SARSA(λ) for better credit assignment
5. **Additional Environments**: Support for more Gymnasium environments
6. **Adaptive Learning Rate**: Dynamic learning rate scheduling
7. **Visualization Improvements**: Interactive policy visualization, trajectory playback

## References

1. Sutton, R. S., & Barto, A. G. (2018). Reinforcement Learning: An Introduction (2nd ed.)
2. Rummery, G. A., & Niranjan, M. (1994). On-line Q-learning using connectionist systems
3. Singh, S., et al. (2000). Convergence results for single-step on-policy reinforcement-learning algorithms

## Notes

- The implementation uses state discretization for CartPole to make the continuous state space tractable with tabular Q-learning
- Epsilon decay helps transition from exploration to exploitation over training
- For CartPole, success is typically achieved when the average reward exceeds 195 over 100 episodes
- FrozenLake is stochastic by default, which can make learning more challenging
- SARSA is particularly useful in environments where making risky moves during training should be avoided
