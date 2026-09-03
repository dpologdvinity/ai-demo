from fastapi import APIRouter, HTTPException
import time
from typing import Dict, Any

from algorithms.reinforcement_learning import (
    QLearningModel,
    QLearningRequest,
    QLearningResponse,
    QLearningInfoResponse,
    DQNModel,
    DQNRequest,
    DQNResponse,
    DQNInfoResponse,
    ActorCriticModel,
    ActorCriticRequest,
    ActorCriticResponse,
    ActorCriticInfoResponse,
    A3CModel,
    A3CRequest,
    A3CResponse,
    A3CInfoResponse,
    SARSAModel,
    SARSARequest,
    SARSAResponse,
    SARSAInfoResponse,
    PPOModel,
    PPORequest,
    PPOResponse,
    PPOInfoResponse,
    DDPGModel,
    DDPGRequest,
    DDPGResponse,
    DDPGInfoResponse
)
from utils.algorithm_metadata import (
    AlgorithmMetadata,
    AlgorithmParameter,
    AlgorithmComplexity,
    AlgorithmCategory,
    DifficultyLevel,
    AlgorithmRegistry,
)

router = APIRouter(prefix="/reinforcement-learning", tags=["Reinforcement Learning"])


# Register Q-Learning metadata
q_learning_metadata = AlgorithmMetadata(
    id="q-learning",
    name="Q-Learning",
    slug="q-learning",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="Model-free reinforcement learning algorithm that learns Q-values for state-action pairs",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["reinforcement-learning", "q-learning", "model-free", "value-based"],
    use_cases=[
        "Game AI",
        "Robot navigation",
        "Resource allocation",
        "Traffic control"
    ],
    complexity=AlgorithmComplexity(
        time="O(episodes × steps_per_episode)",
        space="O(states × actions)"
    ),
    parameters=[
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate (α)",
            type="range",
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            description="Learning rate for Q-value updates (higher = faster learning, less stability)"
        ),
        AlgorithmParameter(
            name="discount_factor",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.5,
            max=0.99,
            step=0.01,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="epsilon",
            label="Exploration Rate (ε)",
            type="range",
            default=0.1,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Probability of random exploration vs exploitation (higher = more exploration)"
        ),
        AlgorithmParameter(
            name="episodes",
            label="Training Episodes",
            type="range",
            default=1000,
            min=100,
            max=5000,
            step=100,
            description="Number of training episodes (higher = more learning time)"
        ),
        AlgorithmParameter(
            name="grid_size",
            label="Grid Size",
            type="range",
            default=5,
            min=3,
            max=10,
            step=1,
            description="Size of the grid world environment (larger = more complex)"
        ),
    ],
    dataset_name="grid_world",
    visualization_type="grid_world",
    theory=(
        "Q-Learning is a model-free reinforcement learning algorithm that learns "
        "the value of taking actions in different states. It maintains a Q-table "
        "where each entry Q(s, a) represents the expected cumulative reward of taking "
        "action 'a' in state 's'. The algorithm uses the Bellman equation to iteratively "
        "update Q-values: Q(s,a) ← Q(s,a) + α[r + γ·max(Q(s',a')) - Q(s,a)], "
        "where α is the learning rate, γ is the discount factor, r is the immediate reward, "
        "and s' is the next state. The agent follows an epsilon-greedy policy, balancing "
        "exploration (random actions) and exploitation (best known actions)."
    ),
    pros=[
        "Model-free: doesn't require knowledge of environment dynamics",
        "Off-policy: can learn optimal policy while following exploratory policy",
        "Simple to implement and understand",
        "Guaranteed to converge to optimal policy given sufficient exploration",
        "Works well for discrete state and action spaces"
    ],
    cons=[
        "Doesn't scale well to large or continuous state spaces",
        "Requires significant exploration to learn good policies",
        "May converge slowly in complex environments",
        "Q-table memory requirements grow with state-action space size",
        "No generalization between similar states"
    ],
    related_algorithms=["sarsa", "deep-q-learning", "policy-gradient"]
)

AlgorithmRegistry.register(q_learning_metadata)


# Register SARSA metadata
sarsa_metadata = AlgorithmMetadata(
    id="sarsa",
    name="SARSA",
    slug="sarsa",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="On-policy TD control algorithm that learns Q-values while following the current policy",
    difficulty=DifficultyLevel.INTERMEDIATE,
    tags=["reinforcement-learning", "sarsa", "on-policy", "temporal-difference"],
    use_cases=[
        "Robot navigation",
        "Game playing",
        "Control problems",
        "Sequential decision making",
        "On-policy learning scenarios"
    ],
    complexity=AlgorithmComplexity(
        time="O(episodes × steps)",
        space="O(states × actions)"
    ),
    parameters=[
        AlgorithmParameter(
            name="environment",
            label="Environment",
            type="select",
            default="CartPole-v1",
            options=[
                {"label": "CartPole-v1", "value": "CartPole-v1"},
                {"label": "FrozenLake-v1", "value": "FrozenLake-v1"}
            ],
            description="Gymnasium environment for training"
        ),
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate (α)",
            type="range",
            default=0.1,
            min=0.01,
            max=1.0,
            step=0.01,
            description="Learning rate for Q-value updates (higher = faster learning, less stability)"
        ),
        AlgorithmParameter(
            name="discount_factor",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.8,
            max=1.0,
            step=0.01,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="epsilon",
            label="Initial Exploration Rate (ε)",
            type="range",
            default=0.1,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Initial probability of random exploration vs exploitation"
        ),
        AlgorithmParameter(
            name="epsilon_decay",
            label="Epsilon Decay Rate",
            type="range",
            default=0.995,
            min=0.9,
            max=1.0,
            step=0.001,
            description="Epsilon decay rate per episode (higher = slower decay)"
        ),
        AlgorithmParameter(
            name="episodes",
            label="Training Episodes",
            type="range",
            default=500,
            min=100,
            max=2000,
            step=50,
            description="Number of training episodes (higher = more learning time)"
        ),
    ],
    dataset_name="cartpole",
    visualization_type="sarsa_rl",
    theory=(
        "SARSA (State-Action-Reward-State-Action) is an on-policy TD control algorithm "
        "that learns the action-value function Q(s,a). Unlike Q-Learning which is off-policy "
        "and learns the optimal policy, SARSA learns about the policy it actually follows, "
        "including exploration. The update rule is: Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)], "
        "where α is the learning rate, γ is the discount factor, r is the immediate reward, "
        "s' is the next state, and a' is the actual action taken (not the maximum). "
        "This makes SARSA more conservative than Q-Learning, as it considers the exploration "
        "policy when updating values. SARSA is particularly useful in environments with risks "
        "or 'cliffs' where a conservative policy is preferred over an optimal but risky one."
    ),
    pros=[
        "On-policy: learns about the actual policy being followed",
        "More conservative and safer than Q-Learning in risky environments",
        "Considers exploration in learning, leading to safer policies",
        "Simple to implement and understand",
        "Works well for discrete state and action spaces",
        "Better suited for online learning where safety matters"
    ],
    cons=[
        "Slower convergence than Q-Learning in some environments",
        "May not find the optimal policy if exploration is suboptimal",
        "Doesn't scale well to large or continuous state spaces",
        "Q-table memory requirements grow with state-action space size",
        "Requires balancing exploration vs exploitation carefully",
        "May be overly conservative in safe environments"
    ],
    related_algorithms=["q-learning", "expected-sarsa", "n-step-sarsa"]
)

AlgorithmRegistry.register(sarsa_metadata)


# Register DQN metadata
dqn_metadata = AlgorithmMetadata(
    id="dqn",
    name="Deep Q-Network (DQN)",
    slug="dqn",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="Deep RL algorithm combining Q-learning with neural networks",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["reinforcement-learning", "deep-learning", "q-learning", "value-based"],
    use_cases=[
        "Game playing (Atari)",
        "Robot control",
        "Resource allocation",
        "Autonomous navigation",
        "Trading strategies"
    ],
    complexity=AlgorithmComplexity(
        time="O(episodes × steps × forward_passes)",
        space="O(replay_buffer + network_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for optimizer (higher = faster learning, less stability)"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.9,
            max=0.999,
            step=0.001,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="epsilon",
            label="Exploration Rate (ε)",
            type="range",
            default=0.1,
            min=0.0,
            max=1.0,
            step=0.05,
            description="Probability of random exploration vs exploitation (higher = more exploration)"
        ),
        AlgorithmParameter(
            name="episodes",
            label="Training Episodes",
            type="range",
            default=500,
            min=100,
            max=2000,
            step=50,
            description="Number of training episodes (higher = more learning time)"
        ),
        AlgorithmParameter(
            name="replay_buffer_size",
            label="Replay Buffer Size",
            type="range",
            default=10000,
            min=1000,
            max=50000,
            step=1000,
            description="Experience replay buffer size (larger = more memory, better generalization)"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="range",
            default=32,
            min=16,
            max=128,
            step=16,
            description="Training batch size (larger = more stable updates, slower training)"
        ),
    ],
    dataset_name="cartpole",
    visualization_type="cartpole_rl",
    theory=(
        "Deep Q-Network (DQN) extends Q-learning to high-dimensional state spaces by "
        "using deep neural networks to approximate Q-values. Key innovations include: "
        "(1) Experience Replay - storing transitions in a buffer and sampling random "
        "minibatches for training, breaking temporal correlations; (2) Target Network - "
        "using a separate network with frozen parameters for computing target Q-values, "
        "stabilizing training. The Q-network learns to predict Q(s,a) for all actions "
        "given a state. Updates follow: Q(s,a) ← Q(s,a) + α[r + γ·max(Q_target(s',a')) - Q(s,a)]. "
        "This allows DQN to learn directly from raw sensory input (pixels) and master "
        "complex tasks like Atari games at superhuman levels."
    ),
    pros=[
        "Scales to high-dimensional state spaces (images, continuous states)",
        "Learns directly from raw sensory input",
        "Experience replay improves data efficiency",
        "Target network stabilizes training",
        "Can learn complex policies from sparse rewards",
        "Proven success on challenging benchmarks (Atari games)"
    ],
    cons=[
        "Requires careful hyperparameter tuning",
        "Can be unstable during training",
        "Sample inefficient compared to some methods",
        "Overestimation bias in Q-values",
        "Limited to discrete action spaces (without modifications)",
        "Requires significant computational resources"
    ],
    related_algorithms=["q-learning", "double-dqn", "dueling-dqn", "rainbow-dqn"]
)

AlgorithmRegistry.register(dqn_metadata)


# Register Actor-Critic metadata
actor_critic_metadata = AlgorithmMetadata(
    id="actor-critic",
    name="Actor-Critic",
    slug="actor-critic",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="RL algorithm combining policy gradient (actor) with value function (critic)",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["reinforcement-learning", "policy-based", "value-based", "actor-critic"],
    use_cases=[
        "Continuous control",
        "Game playing",
        "Robotics",
        "Autonomous driving",
        "Resource management"
    ],
    complexity=AlgorithmComplexity(
        time="O(episodes × steps × network_forward)",
        space="O(actor_params + critic_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="actor_lr",
            label="Actor Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for actor (policy) network"
        ),
        AlgorithmParameter(
            name="critic_lr",
            label="Critic Learning Rate",
            type="range",
            default=0.005,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for critic (value) network"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.9,
            max=0.999,
            step=0.001,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="episodes",
            label="Training Episodes",
            type="range",
            default=1000,
            min=100,
            max=3000,
            step=50,
            description="Number of training episodes (higher = more learning time)"
        ),
        AlgorithmParameter(
            name="hidden_size",
            label="Network Hidden Size",
            type="range",
            default=128,
            min=64,
            max=256,
            step=32,
            description="Hidden layer size for both actor and critic networks"
        ),
    ],
    dataset_name="cartpole",
    visualization_type="actor_critic_rl",
    theory=(
        "Actor-Critic algorithms combine the benefits of policy-based and value-based "
        "reinforcement learning. The algorithm maintains two separate neural networks: "
        "(1) Actor - learns the policy π(a|s) that maps states to action probabilities, "
        "(2) Critic - learns the value function V(s) that estimates expected cumulative reward. "
        "During training, the actor generates actions based on the current policy, while the "
        "critic evaluates these actions by computing the advantage A(s,a) = r + γV(s') - V(s). "
        "The actor is updated using policy gradient: ∇J(θ) = E[∇log π(a|s) * A(s,a)], "
        "improving actions that lead to higher-than-expected returns. The critic is updated "
        "using temporal difference learning to minimize the TD error. This dual architecture "
        "reduces variance (via critic baseline) while maintaining policy gradient's ability "
        "to handle continuous action spaces."
    ),
    pros=[
        "Combines strengths of policy-based and value-based methods",
        "Lower variance than pure policy gradient methods",
        "Can handle continuous action spaces",
        "More sample efficient than policy gradient alone",
        "Natural extension to actor-critic variants (A3C, A2C, PPO)",
        "Online learning without replay buffer"
    ],
    cons=[
        "Requires tuning two separate networks and learning rates",
        "Can be unstable during training without careful tuning",
        "Sensitive to hyperparameters",
        "Slower to converge than Q-learning for simple tasks",
        "Requires more computational resources than tabular methods",
        "Potential for divergence if critic is poorly trained"
    ],
    related_algorithms=["policy-gradient", "dqn", "a3c", "ppo"]
)

AlgorithmRegistry.register(actor_critic_metadata)


# Register A3C metadata
a3c_metadata = AlgorithmMetadata(
    id="a3c",
    name="A3C (Asynchronous Advantage Actor-Critic)",
    slug="a3c",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="Parallel actor-critic with advantage estimation for efficient RL training",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["reinforcement-learning", "actor-critic", "asynchronous", "parallel", "advantage"],
    use_cases=[
        "Real-time game playing",
        "Robot control",
        "Continuous control tasks",
        "Parallel training for faster convergence",
        "Exploration in large state spaces"
    ],
    complexity=AlgorithmComplexity(
        time="O(workers × episodes × steps × forward)",
        space="O(shared_model + worker_models)"
    ),
    parameters=[
        AlgorithmParameter(
            name="num_workers",
            label="Number of Workers",
            type="range",
            default=4,
            min=2,
            max=8,
            step=1,
            description="Number of parallel workers for asynchronous training"
        ),
        AlgorithmParameter(
            name="actor_lr",
            label="Actor Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for actor network (policy)"
        ),
        AlgorithmParameter(
            name="critic_lr",
            label="Critic Learning Rate",
            type="range",
            default=0.005,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for critic network (value function)"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.9,
            max=0.999,
            step=0.001,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="episodes_per_worker",
            label="Episodes per Worker",
            type="range",
            default=100,
            min=50,
            max=500,
            step=10,
            description="Number of episodes each worker runs"
        ),
        AlgorithmParameter(
            name="entropy_coef",
            label="Entropy Coefficient",
            type="range",
            default=0.01,
            min=0.0,
            max=0.1,
            step=0.001,
            description="Entropy coefficient for exploration bonus (higher = more exploration)"
        ),
    ],
    dataset_name="cartpole",
    visualization_type="a3c_rl",
    theory=(
        "A3C (Asynchronous Advantage Actor-Critic) is a parallel reinforcement learning algorithm "
        "that uses multiple workers running asynchronously to train a shared global model. Each worker "
        "has its own environment and local copy of the model, collecting experience independently. "
        "After each episode, workers compute gradients and asynchronously update the shared global model. "
        "Key innovations: (1) Asynchronous updates - workers train in parallel without waiting for each other, "
        "decorrelating experiences naturally; (2) Advantage estimation - uses TD error (returns - value) "
        "as advantages for policy gradient updates; (3) Entropy regularization - adds entropy bonus to "
        "encourage exploration. The actor-critic architecture uses a shared network with two heads: "
        "actor (policy) outputs action probabilities, critic (value) estimates state values. "
        "This parallel approach provides faster training and better exploration than sequential methods."
    ),
    pros=[
        "Parallel training significantly speeds up learning",
        "Natural experience decorrelation through asynchronous workers",
        "Advantage estimation reduces variance in policy gradient",
        "Entropy regularization promotes exploration",
        "On-policy learning with immediate gradient updates",
        "Works well for both discrete and continuous action spaces",
        "More stable than vanilla policy gradients"
    ],
    cons=[
        "Requires more computational resources (multiple CPU cores)",
        "Asynchronous updates can lead to stale gradients",
        "Hyperparameter tuning can be challenging",
        "May not fully utilize GPU acceleration",
        "Workers can learn conflicting policies temporarily",
        "Requires careful gradient clipping to prevent instability"
    ],
    related_algorithms=["actor-critic", "a2c", "ppo", "policy-gradient"]
)

AlgorithmRegistry.register(a3c_metadata)


# Register PPO metadata
ppo_metadata = AlgorithmMetadata(
    id="ppo",
    name="PPO (Proximal Policy Optimization)",
    slug="ppo",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="Policy gradient method with clipped objective for stable training",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["reinforcement-learning", "policy-gradient", "on-policy", "clipped-objective"],
    use_cases=[
        "Robot control",
        "Game AI",
        "Autonomous driving",
        "Continuous control tasks",
        "Resource optimization"
    ],
    complexity=AlgorithmComplexity(
        time="O(episodes × steps × epochs × minibatches)",
        space="O(buffer_size + model_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="learning_rate",
            label="Learning Rate",
            type="range",
            default=0.0003,
            min=0.0001,
            max=0.001,
            step=0.0001,
            description="Learning rate for optimizer (higher = faster learning, less stability)"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.9,
            max=0.999,
            step=0.001,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="clip_epsilon",
            label="Clip Epsilon (ε)",
            type="range",
            default=0.2,
            min=0.1,
            max=0.3,
            step=0.01,
            description="PPO clipping parameter (controls how much policy can change per update)"
        ),
        AlgorithmParameter(
            name="epochs",
            label="PPO Epochs",
            type="range",
            default=4,
            min=1,
            max=10,
            step=1,
            description="Number of epochs per PPO update (more epochs = better sample efficiency)"
        ),
        AlgorithmParameter(
            name="episodes",
            label="Training Episodes",
            type="range",
            default=500,
            min=100,
            max=2000,
            step=50,
            description="Number of training episodes (higher = more learning time)"
        ),
        AlgorithmParameter(
            name="gae_lambda",
            label="GAE Lambda (λ)",
            type="range",
            default=0.95,
            min=0.9,
            max=0.99,
            step=0.01,
            description="GAE lambda parameter (higher = lower bias, higher variance)"
        ),
    ],
    dataset_name="cartpole",
    visualization_type="ppo_rl",
    theory=(
        "Proximal Policy Optimization (PPO) is a state-of-the-art policy gradient method that "
        "improves training stability by limiting how much the policy can change in a single update. "
        "PPO uses a clipped surrogate objective: L(θ) = min(r(θ)·A, clip(r(θ), 1-ε, 1+ε)·A), "
        "where r(θ) is the probability ratio between new and old policies, A is the advantage, "
        "and ε is the clipping parameter. This clipping prevents large policy updates that could "
        "destabilize training. PPO also uses Generalized Advantage Estimation (GAE) for computing "
        "advantages with reduced variance. The algorithm collects trajectories using the current "
        "policy, then performs multiple epochs of minibatch updates on the collected data. "
        "Key metrics tracked: (1) Clip fraction - percentage of updates that hit the clip boundary, "
        "indicating aggressive policy changes; (2) KL divergence - measures policy change magnitude. "
        "PPO achieves strong performance while being simpler and more stable than other policy "
        "gradient methods like TRPO."
    ),
    pros=[
        "Excellent balance of performance, stability, and simplicity",
        "Clipped objective prevents destructively large policy updates",
        "Sample efficient through multiple epochs on collected data",
        "Works well for both continuous and discrete action spaces",
        "Robust to hyperparameter choices",
        "Industry-standard algorithm (widely used in practice)",
        "GAE reduces variance in advantage estimates"
    ],
    cons=[
        "On-policy algorithm (less sample efficient than off-policy methods)",
        "Requires careful tuning of clip epsilon",
        "Can be slower than A3C due to sequential rollout collection",
        "May get stuck in local optima",
        "Requires larger batch sizes for stable training",
        "Computational overhead from multiple epochs per update"
    ],
    related_algorithms=["actor-critic", "a3c", "trpo", "policy-gradient"]
)

AlgorithmRegistry.register(ppo_metadata)


# Register DDPG metadata
ddpg_metadata = AlgorithmMetadata(
    id="ddpg",
    name="DDPG (Deep Deterministic Policy Gradient)",
    slug="ddpg",
    category=AlgorithmCategory.REINFORCEMENT_LEARNING,
    description="Actor-critic algorithm for continuous action spaces",
    difficulty=DifficultyLevel.ADVANCED,
    tags=["reinforcement-learning", "actor-critic", "continuous-control", "off-policy"],
    use_cases=[
        "Robot control",
        "Continuous control tasks",
        "Industrial automation",
        "Autonomous vehicles",
        "Manipulation tasks"
    ],
    complexity=AlgorithmComplexity(
        time="O(episodes × steps × network_forward)",
        space="O(replay_buffer + actor_params + critic_params)"
    ),
    parameters=[
        AlgorithmParameter(
            name="actor_lr",
            label="Actor Learning Rate",
            type="range",
            default=0.0001,
            min=0.00001,
            max=0.001,
            step=0.00001,
            description="Learning rate for actor (policy) network"
        ),
        AlgorithmParameter(
            name="critic_lr",
            label="Critic Learning Rate",
            type="range",
            default=0.001,
            min=0.0001,
            max=0.01,
            step=0.0001,
            description="Learning rate for critic (Q-function) network"
        ),
        AlgorithmParameter(
            name="gamma",
            label="Discount Factor (γ)",
            type="range",
            default=0.99,
            min=0.9,
            max=0.999,
            step=0.001,
            description="Discount factor for future rewards (higher = more focus on long-term rewards)"
        ),
        AlgorithmParameter(
            name="tau",
            label="Soft Update Coefficient (τ)",
            type="range",
            default=0.005,
            min=0.001,
            max=0.01,
            step=0.001,
            description="Soft update coefficient for target networks (higher = faster target updates)"
        ),
        AlgorithmParameter(
            name="episodes",
            label="Training Episodes",
            type="range",
            default=200,
            min=50,
            max=500,
            step=10,
            description="Number of training episodes (higher = more learning time)"
        ),
        AlgorithmParameter(
            name="buffer_size",
            label="Replay Buffer Size",
            type="range",
            default=100000,
            min=10000,
            max=1000000,
            step=10000,
            description="Experience replay buffer size (larger = more memory, better generalization)"
        ),
        AlgorithmParameter(
            name="batch_size",
            label="Batch Size",
            type="range",
            default=64,
            min=32,
            max=256,
            step=16,
            description="Training batch size (larger = more stable updates, slower training)"
        ),
    ],
    dataset_name="pendulum",
    visualization_type="ddpg_rl",
    theory=(
        "DDPG (Deep Deterministic Policy Gradient) is an off-policy actor-critic algorithm "
        "designed for continuous action spaces. It combines ideas from DPG (Deterministic Policy Gradient) "
        "and DQN (Deep Q-Network). Key components: (1) Actor Network - learns a deterministic policy "
        "μ(s) that directly maps states to actions; (2) Critic Network - learns Q(s,a) to evaluate "
        "the actor's actions; (3) Experience Replay - stores transitions (s,a,r,s') in a buffer and "
        "samples random minibatches for training, breaking temporal correlations; (4) Target Networks - "
        "maintains separate target networks for both actor and critic with soft updates (θ' ← τθ + (1-τ)θ'), "
        "providing stable learning targets; (5) Ornstein-Uhlenbeck Noise - adds temporally correlated "
        "noise to actions for exploration in continuous spaces. The critic is updated using the Bellman "
        "equation: L = E[(Q(s,a) - (r + γQ'(s',μ'(s'))))²]. The actor is updated using the deterministic "
        "policy gradient: ∇J = E[∇_a Q(s,a)|a=μ(s) ∇_θ μ(s)], maximizing expected Q-value."
    ),
    pros=[
        "Handles continuous action spaces directly (no discretization needed)",
        "Off-policy learning enables high sample efficiency",
        "Experience replay improves stability and data efficiency",
        "Target networks prevent divergence during training",
        "Deterministic policy enables efficient gradient computation",
        "Proven success on robotics and control tasks",
        "Can learn complex continuous control policies"
    ],
    cons=[
        "Sensitive to hyperparameter choices (especially learning rates)",
        "Exploration can be challenging in continuous spaces",
        "May converge to local optima",
        "Requires careful tuning of noise process",
        "Can be unstable during training without proper initialization",
        "Overestimation bias in Q-values (like DQN)",
        "Requires significant memory for replay buffer"
    ],
    related_algorithms=["actor-critic", "dqn", "td3", "sac"]
)

AlgorithmRegistry.register(ddpg_metadata)


@router.get("/")
async def reinforcement_learning_root():
    return {"message": "Reinforcement Learning algorithms endpoint"}


@router.get("/algorithms")
async def list_rl_algorithms():
    """Get all registered Reinforcement Learning algorithms.

    Returns:
        List of algorithm metadata for all registered RL algorithms
    """
    algorithms = AlgorithmRegistry.get_by_category(AlgorithmCategory.REINFORCEMENT_LEARNING)
    return [algo.model_dump() for algo in algorithms]


@router.post("/q-learning/train", response_model=QLearningResponse)
async def train_q_learning(request: QLearningRequest):
    """Train a Q-Learning agent in a grid world environment.

    This endpoint trains a Q-Learning agent to navigate a grid world from
    start to goal while avoiding obstacles. Returns training metrics,
    learned policy, and visualization data.

    Args:
        request: QLearningRequest containing training parameters

    Returns:
        QLearningResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = QLearningModel()
        result = model.train(
            learning_rate=request.learning_rate,
            discount_factor=request.discount_factor,
            epsilon=request.epsilon,
            episodes=request.episodes,
            grid_size=request.grid_size,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return QLearningResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/q-learning/info", response_model=QLearningInfoResponse)
async def get_q_learning_info():
    """Get Q-Learning algorithm information and metadata.

    Returns:
        QLearningInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("q-learning")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "GridWorld",
        "description": "2D grid world with start, goal, and obstacles",
        "state_space": "Discrete (grid positions)",
        "action_space": "Discrete (4 actions: up, right, down, left)",
        "reward_structure": {
            "goal": 10.0,
            "step": -0.1,
            "wall_or_obstacle": -1.0
        }
    }

    return QLearningInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )


@router.post("/dqn/train", response_model=DQNResponse)
async def train_dqn(request: DQNRequest):
    """Train a DQN agent in the CartPole-v1 environment.

    This endpoint trains a Deep Q-Network agent to balance a pole on a cart
    using deep reinforcement learning with experience replay and target networks.

    Args:
        request: DQNRequest containing training parameters

    Returns:
        DQNResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = DQNModel()
        result = model.train(
            learning_rate=request.learning_rate,
            gamma=request.gamma,
            epsilon=request.epsilon,
            episodes=request.episodes,
            replay_buffer_size=request.replay_buffer_size,
            batch_size=request.batch_size,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return DQNResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/dqn/info", response_model=DQNInfoResponse)
async def get_dqn_info():
    """Get DQN algorithm information and metadata.

    Returns:
        DQNInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("dqn")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "CartPole-v1",
        "description": "Balance a pole on a cart by moving left or right",
        "state_space": "Continuous (4D: cart position, velocity, pole angle, angular velocity)",
        "action_space": "Discrete (2 actions: push left, push right)",
        "success_criterion": "Average reward >= 195 over 100 consecutive episodes",
        "max_steps": 500,
        "reward_structure": {
            "step": 1.0,
            "termination": 0.0,
            "description": "+1 for every step the pole remains balanced"
        }
    }

    return DQNInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )


@router.post("/actor-critic/train", response_model=ActorCriticResponse)
async def train_actor_critic(request: ActorCriticRequest):
    """Train an Actor-Critic agent in the CartPole-v1 environment.

    This endpoint trains an Actor-Critic agent to balance a pole on a cart
    using separate actor (policy) and critic (value) networks.

    Args:
        request: ActorCriticRequest containing training parameters

    Returns:
        ActorCriticResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = ActorCriticModel()
        result = model.train(
            actor_lr=request.actor_lr,
            critic_lr=request.critic_lr,
            gamma=request.gamma,
            episodes=request.episodes,
            hidden_size=request.hidden_size,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return ActorCriticResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/actor-critic/info", response_model=ActorCriticInfoResponse)
async def get_actor_critic_info():
    """Get Actor-Critic algorithm information and metadata.

    Returns:
        ActorCriticInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("actor-critic")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "CartPole-v1",
        "description": "Balance a pole on a cart by moving left or right",
        "state_space": "Continuous (4D: cart position, velocity, pole angle, angular velocity)",
        "action_space": "Discrete (2 actions: push left, push right)",
        "success_criterion": "Average reward >= 195 over 100 consecutive episodes",
        "max_steps": 500,
        "reward_structure": {
            "step": 1.0,
            "termination": 0.0,
            "description": "+1 for every step the pole remains balanced"
        },
        "networks": {
            "actor": "Policy network mapping states to action probabilities",
            "critic": "Value network mapping states to state values"
        }
    }

    return ActorCriticInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )


@router.post("/a3c/train", response_model=A3CResponse)
async def train_a3c(request: A3CRequest):
    """Train an A3C agent with multiple parallel workers.

    This endpoint trains an Asynchronous Advantage Actor-Critic agent using
    multiple workers running in parallel, each with its own environment and
    local model. Workers asynchronously update a shared global model.

    Args:
        request: A3CRequest containing training parameters

    Returns:
        A3CResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = A3CModel()
        result = model.train(
            num_workers=request.num_workers,
            actor_lr=request.actor_lr,
            critic_lr=request.critic_lr,
            gamma=request.gamma,
            episodes_per_worker=request.episodes_per_worker,
            entropy_coef=request.entropy_coef,
            hidden_size=request.hidden_size,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return A3CResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/a3c/info", response_model=A3CInfoResponse)
async def get_a3c_info():
    """Get A3C algorithm information and metadata.

    Returns:
        A3CInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("a3c")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "CartPole-v1",
        "description": "Balance a pole on a cart by moving left or right",
        "state_space": "Continuous (4D: cart position, velocity, pole angle, angular velocity)",
        "action_space": "Discrete (2 actions: push left, push right)",
        "success_criterion": "Average reward >= 195 over 100 consecutive episodes",
        "max_steps": 500,
        "reward_structure": {
            "step": 1.0,
            "termination": 0.0,
            "description": "+1 for every step the pole remains balanced"
        },
        "architecture": {
            "shared_network": "Global model shared across all workers",
            "actor_head": "Policy network mapping states to action probabilities",
            "critic_head": "Value network mapping states to state values",
            "workers": "Multiple parallel workers with local model copies"
        },
        "training_mechanism": {
            "asynchronous": "Workers train independently and update global model asynchronously",
            "advantage_estimation": "Uses TD error (returns - value) for policy gradient",
            "entropy_regularization": "Adds entropy bonus to encourage exploration"
        }
    }

    return A3CInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )


@router.post("/sarsa/train", response_model=SARSAResponse)
async def train_sarsa(request: SARSARequest):
    """Train a SARSA agent in a grid world environment.

    This endpoint trains a SARSA (State-Action-Reward-State-Action) agent,
    an on-policy TD control algorithm. SARSA learns about the policy it follows,
    making it more conservative than Q-Learning in risky environments.

    Args:
        request: SARSARequest containing training parameters

    Returns:
        SARSAResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = SARSAModel()
        result = model.train(
            environment=request.environment,
            learning_rate=request.learning_rate,
            discount_factor=request.discount_factor,
            epsilon=request.epsilon,
            epsilon_decay=request.epsilon_decay,
            episodes=request.episodes,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return SARSAResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/sarsa/info", response_model=SARSAInfoResponse)
async def get_sarsa_info():
    """Get SARSA algorithm information and metadata.

    Returns:
        SARSAInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("sarsa")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "CartPole-v1 / FrozenLake-v1",
        "cartpole": {
            "description": "Balance a pole on a cart by moving left or right",
            "state_space": "Continuous (4D: cart position, velocity, pole angle, angular velocity) - discretized",
            "action_space": "Discrete (2 actions: push left, push right)",
            "success_criterion": "Average reward >= 195 over 100 consecutive episodes"
        },
        "frozenlake": {
            "description": "Navigate a frozen lake from start to goal, avoiding holes",
            "state_space": "Discrete (16 states in 4x4 grid)",
            "action_space": "Discrete (4 actions: left, down, right, up)",
            "success_criterion": "Reach the goal without falling into holes"
        },
        "algorithm_features": {
            "epsilon_decay": "Gradual reduction of exploration over time",
            "on_policy": "Learns about the policy it actually follows",
            "sarsa_update": "Q(s,a) ← Q(s,a) + α[r + γ·Q(s',a') - Q(s,a)]"
        },
        "comparison": {
            "vs_qlearning": "SARSA is on-policy (learns about policy it follows), Q-Learning is off-policy (learns optimal policy)",
            "safety": "SARSA is more conservative and safer in environments with risks or cliffs",
            "convergence": "Q-Learning may converge faster but SARSA provides safer learning"
        }
    }

    return SARSAInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )


@router.post("/ppo/train", response_model=PPOResponse)
async def train_ppo(request: PPORequest):
    """Train a PPO agent in the CartPole-v1 environment.

    This endpoint trains a Proximal Policy Optimization agent with clipped
    surrogate objective and Generalized Advantage Estimation (GAE) to balance
    a pole on a cart.

    Args:
        request: PPORequest containing training parameters

    Returns:
        PPOResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = PPOModel()
        result = model.train(
            learning_rate=request.learning_rate,
            gamma=request.gamma,
            clip_epsilon=request.clip_epsilon,
            epochs=request.epochs,
            episodes=request.episodes,
            gae_lambda=request.gae_lambda,
            batch_size=request.batch_size,
            hidden_size=request.hidden_size,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return PPOResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/ppo/info", response_model=PPOInfoResponse)
async def get_ppo_info():
    """Get PPO algorithm information and metadata.

    Returns:
        PPOInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("ppo")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "CartPole-v1",
        "description": "Balance a pole on a cart by moving left or right",
        "state_space": "Continuous (4D: cart position, velocity, pole angle, angular velocity)",
        "action_space": "Discrete (2 actions: push left, push right)",
        "success_criterion": "Average reward >= 195 over 100 consecutive episodes",
        "max_steps": 500,
        "reward_structure": {
            "step": 1.0,
            "termination": 0.0,
            "description": "+1 for every step the pole remains balanced"
        },
        "algorithm_specifics": {
            "clipped_objective": "Limits policy updates to prevent destructive changes",
            "gae": "Generalized Advantage Estimation for reduced variance",
            "multiple_epochs": "Reuses collected data for sample efficiency",
            "clip_fraction": "Tracks percentage of updates hitting clip boundary",
            "kl_divergence": "Measures policy change magnitude per update"
        }
    }

    return PPOInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )


@router.post("/ddpg/train", response_model=DDPGResponse)
async def train_ddpg(request: DDPGRequest):
    """Train a DDPG agent in the Pendulum-v1 environment.

    This endpoint trains a Deep Deterministic Policy Gradient agent to control
    a pendulum in continuous action space using actor-critic architecture with
    experience replay and target networks.

    Args:
        request: DDPGRequest containing training parameters

    Returns:
        DDPGResponse with training results, metrics, and visualizations

    Raises:
        HTTPException: If training fails due to invalid parameters
    """
    try:
        # Initialize and train model
        model = DDPGModel()
        result = model.train(
            actor_lr=request.actor_lr,
            critic_lr=request.critic_lr,
            gamma=request.gamma,
            tau=request.tau,
            episodes=request.episodes,
            buffer_size=request.buffer_size,
            batch_size=request.batch_size,
            random_state=request.random_state
        )

        if not result['success']:
            raise HTTPException(
                status_code=400,
                detail=result.get('error', 'Training failed')
            )

        return DDPGResponse(**result)

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.get("/ddpg/info", response_model=DDPGInfoResponse)
async def get_ddpg_info():
    """Get DDPG algorithm information and metadata.

    Returns:
        DDPGInfoResponse with algorithm metadata and environment info
    """
    metadata = AlgorithmRegistry.get("ddpg")
    if metadata is None:
        raise HTTPException(status_code=404, detail="Algorithm metadata not found")

    environment_info = {
        "type": "Pendulum-v1",
        "description": "Swing a pendulum to upright position and keep it there",
        "state_space": "Continuous (3D: cos(θ), sin(θ), angular velocity)",
        "action_space": "Continuous (1D: torque in [-2.0, 2.0])",
        "reward_structure": {
            "angle_cost": "-(θ²)",
            "velocity_cost": "-0.1 * (θ_dot²)",
            "action_cost": "-0.001 * (action²)",
            "description": "Higher reward for keeping pendulum upright with minimal effort"
        },
        "success_criterion": "Higher average reward over episodes (reward range: approximately -1600 to 0)",
        "max_steps": 200,
        "networks": {
            "actor": "Deterministic policy network μ(s) mapping states to continuous actions",
            "critic": "Q-function network Q(s,a) mapping state-action pairs to values",
            "target_networks": "Slowly updated copies of actor and critic for stable learning"
        },
        "exploration": {
            "method": "Ornstein-Uhlenbeck process",
            "description": "Temporally correlated noise for smooth exploration in continuous space"
        }
    }

    return DDPGInfoResponse(
        metadata=metadata.model_dump(),
        environment_info=environment_info
    )
