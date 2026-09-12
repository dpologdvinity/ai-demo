# PPO (Proximal Policy Optimization) Implementation

## Overview
This implementation provides a complete PPO algorithm for reinforcement learning, integrated into the AI algorithms demonstration website.

## Files Created

### 1. `/backend/algorithms/reinforcement_learning/ppo.py`
**Main PPO Implementation:**
- `PPOActorNetwork`: Actor network for policy approximation
  - Maps states to action probability distributions
  - 2-layer fully connected network with tanh activations
  
- `PPOCriticNetwork`: Critic network for value function approximation
  - Maps states to state values
  - 2-layer fully connected network with tanh activations
  
- `RolloutBuffer`: Buffer for storing trajectory rollouts
  - Stores states, actions, rewards, log probabilities, values, and done flags
  - Implements GAE (Generalized Advantage Estimation) for computing advantages
  
- `PPOModel`: Main PPO algorithm class
  - Implements PPO with clipped surrogate objective
  - Uses GAE for advantage estimation
  - Multiple epochs of minibatch updates on collected data
  - Tracks policy loss, value loss, clip fraction, and KL divergence

**Key Features:**
- Clipped surrogate objective: `L(θ) = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)`
- GAE for variance reduction in advantage estimation
- Multiple epochs per update for sample efficiency
- Entropy bonus for exploration (coefficient: 0.01)
- Gradient clipping (max norm: 0.5)
- Collects rollouts of 2048 steps before each update

### 2. `/backend/algorithms/reinforcement_learning/ppo_schema.py`
**Pydantic Schemas for API:**
- `PPORequest`: Request schema with parameter validation
- `PPOResponse`: Response schema with results, metrics, and visualizations
- `PPOInfoResponse`: Algorithm metadata and environment information

## Algorithm Parameters

| Parameter | Default | Range | Description |
|-----------|---------|-------|-------------|
| learning_rate | 0.0003 | [0.0001, 0.001] | Learning rate for optimizer |
| gamma | 0.99 | [0.9, 0.999] | Discount factor for future rewards |
| clip_epsilon | 0.2 | [0.1, 0.3] | PPO clipping parameter |
| epochs | 4 | [1, 10] | Number of PPO epochs per update |
| episodes | 500 | [100, 2000] | Number of training episodes |
| gae_lambda | 0.95 | [0.9, 0.99] | GAE lambda parameter |
| batch_size | 64 | [32, 256] | Minibatch size for PPO updates |
| hidden_size | 128 | [64, 256] | Hidden layer size for networks |

## Metrics Tracked

1. **Training Metrics:**
   - Average reward (last 100 episodes)
   - Average episode length (last 100 episodes)
   - Maximum reward achieved
   - Success rate (episodes with reward >= 195)
   - Convergence episode (first time avg >= 195)
   - Final episode reward
   - Total episodes

2. **Loss Metrics:**
   - Policy loss (clipped surrogate objective)
   - Value loss (MSE between predicted and actual returns)
   - Final policy and value losses

3. **PPO-Specific Metrics:**
   - **Clip Fraction:** Percentage of updates hitting the clip boundary
     - Indicates how aggressive policy updates are
     - High values suggest policy is changing rapidly
   - **KL Divergence:** Measures policy change magnitude
     - Monitors how much the policy changes per update
     - Helps ensure stable training

## Visualization Data

### 1. Reward Data
- Episode-by-episode rewards
- 50-episode moving average
- Used for line chart showing learning progress

### 2. Loss Data
- Policy loss over training updates
- Value loss over training updates
- Dual line chart showing both losses

### 3. Clip Fraction Data
- Clip fraction over training updates
- Shows percentage of updates that were clipped
- Indicates training stability

### 4. KL Divergence Data
- KL divergence over training updates
- Measures policy change magnitude
- Monitors training stability

### 5. Episode Lengths
- Episode length for each episode
- Shows how long the agent survives

### 6. Sample Trajectories
- First, middle, and last episode trajectories
- Shows state progression during episodes

## API Endpoints

### POST `/reinforcement-learning/ppo/train`
Train a PPO agent on CartPole-v1 environment.

**Request Body:** PPORequest with training parameters

**Response:** PPOResponse with:
- success: bool
- metrics: training and loss metrics
- visualization_data: charts and trajectory data
- execution_time_ms: training time
- parameters_used: actual parameters used

### GET `/reinforcement-learning/ppo/info`
Get PPO algorithm metadata and environment information.

**Response:** PPOInfoResponse with:
- metadata: algorithm details, complexity, parameters
- environment_info: CartPole environment details

## Integration

### Updated Files:
1. `/backend/algorithms/reinforcement_learning/__init__.py`
   - Added PPO imports to module exports

2. `/backend/api/routes/reinforcement_learning.py`
   - Added PPO imports
   - Registered PPO metadata with AlgorithmRegistry
   - Added `/ppo/train` endpoint
   - Added `/ppo/info` endpoint

## Environment

**CartPole-v1:**
- State space: 4D continuous (cart position, cart velocity, pole angle, pole angular velocity)
- Action space: 2 discrete actions (push left, push right)
- Success criterion: Average reward >= 195 over 100 consecutive episodes
- Max steps per episode: 500
- Reward: +1 for every step the pole remains balanced

## Algorithm Theory

PPO is a state-of-the-art policy gradient method that improves training stability by limiting policy changes:

1. **Clipped Objective:** Prevents destructively large policy updates
   - `L(θ) = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A)`
   - `r(θ)` is the probability ratio between new and old policies
   - `A` is the advantage estimate
   - `ε` is the clipping parameter

2. **GAE (Generalized Advantage Estimation):**
   - Reduces variance in advantage estimates
   - Balances bias-variance tradeoff with λ parameter
   - `A_t = Σ(γλ)^l δ_{t+l}` where `δ_t = r_t + γV(s_{t+1}) - V(s_t)`

3. **Multiple Epochs:**
   - Reuses collected trajectories for multiple gradient updates
   - Improves sample efficiency
   - Each update uses minibatches of collected data

4. **Key Metrics:**
   - **Clip Fraction:** Monitors aggressive policy changes
   - **KL Divergence:** Ensures policy doesn't change too much
   
## Advantages

1. Excellent balance of performance, stability, and simplicity
2. Clipped objective prevents destructive updates
3. Sample efficient through multiple epochs
4. Works for both continuous and discrete action spaces
5. Robust to hyperparameter choices
6. Industry-standard algorithm (widely used)
7. GAE reduces variance in advantage estimates

## Disadvantages

1. On-policy (less sample efficient than off-policy methods)
2. Requires careful tuning of clip epsilon
3. Can be slower than A3C (sequential rollout collection)
4. May get stuck in local optima
5. Requires larger batch sizes for stable training
6. Computational overhead from multiple epochs

## Testing

A test script is provided at `/backend/test_ppo.py` that:
- Creates a PPO model
- Trains for 50 episodes (quick test)
- Validates all metrics and visualization data
- Checks clip fraction and KL divergence tracking

## Dependencies

All dependencies are already in `/backend/requirements.txt`:
- torch==2.4.1 (PyTorch for neural networks)
- gymnasium==0.29.1 (RL environment)
- numpy==2.1.1 (numerical operations)
- fastapi==0.115.0 (API framework)
- pydantic==2.9.2 (data validation)

## Next Steps

To use the PPO implementation:

1. Ensure virtual environment is activated and dependencies are installed
2. Start the FastAPI server
3. Send POST request to `/reinforcement-learning/ppo/train` with parameters
4. Retrieve training results including:
   - Episode rewards with moving average
   - Policy and value loss curves
   - Clip fraction over time
   - KL divergence tracking
   - Success rate and convergence metrics

## Frontend Integration Requirements

The frontend should implement the following visualizations:

1. **Line Chart: Episode Rewards**
   - X-axis: Episode number
   - Y-axis: Reward
   - Two lines: actual rewards and 50-episode moving average
   - Shows learning progress

2. **Dual Line Chart: Losses**
   - X-axis: Update step (or approximate episode)
   - Y-axis: Loss value
   - Two lines: policy loss (clipped objective) and value loss
   - Different colors for each loss type

3. **Line Chart: Clip Fraction**
   - X-axis: Update step
   - Y-axis: Clip fraction (0-1)
   - Shows what percentage of updates hit the clip boundary
   - Higher values indicate more aggressive policy changes

4. **Line Chart: KL Divergence**
   - X-axis: Update step
   - Y-axis: KL divergence
   - Shows magnitude of policy changes
   - Monitors training stability

5. **Success Rate Meter**
   - Displays success_rate as percentage
   - Shows if agent solves the task (>= 195 reward)

6. **Hyperparameter Controls**
   - Sliders for all parameters (learning_rate, gamma, clip_epsilon, epochs, episodes, gae_lambda)
   - With default values and ranges specified in the metadata

## Related Algorithms

- Actor-Critic (simpler baseline)
- A3C (parallel training variant)
- TRPO (predecessor with KL constraint)
- Policy Gradient (vanilla baseline)
