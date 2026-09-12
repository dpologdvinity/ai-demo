"""
Pydantic schemas for PPO API endpoints.

This module defines the request and response models for the PPO
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class PPORequest(BaseModel):
    """Request schema for PPO training.

    Attributes:
        learning_rate: Learning rate for optimizer (0.0001-0.001)
        gamma: Discount factor for future rewards (0.9-0.999)
        clip_epsilon: PPO clipping parameter (0.1-0.3)
        epochs: Number of PPO epochs per update (1-10)
        episodes: Number of training episodes (100-2000)
        gae_lambda: GAE lambda parameter (0.9-0.99)
        batch_size: Minibatch size for PPO updates
        hidden_size: Hidden layer size for networks (64-256)
        random_state: Random seed for reproducibility

    Example:
        >>> request = PPORequest(
        ...     learning_rate=0.0003,
        ...     gamma=0.99,
        ...     clip_epsilon=0.2,
        ...     epochs=4,
        ...     episodes=500,
        ...     gae_lambda=0.95
        ... )
    """

    learning_rate: float = Field(
        default=0.0003,
        ge=0.0001,
        le=0.001,
        description="Learning rate for optimizer"
    )
    gamma: float = Field(
        default=0.99,
        ge=0.9,
        le=0.999,
        description="Discount factor for future rewards"
    )
    clip_epsilon: float = Field(
        default=0.2,
        ge=0.1,
        le=0.3,
        description="PPO clipping parameter"
    )
    epochs: int = Field(
        default=4,
        ge=1,
        le=10,
        description="Number of PPO epochs per update"
    )
    episodes: int = Field(
        default=500,
        ge=100,
        le=2000,
        description="Number of training episodes"
    )
    gae_lambda: float = Field(
        default=0.95,
        ge=0.9,
        le=0.99,
        description="GAE lambda parameter"
    )
    batch_size: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Minibatch size for PPO updates"
    )
    hidden_size: int = Field(
        default=128,
        ge=64,
        le=256,
        description="Hidden layer size for both actor and critic networks"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class PPOResponse(BaseModel):
    """Response schema for PPO training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards, success rate, convergence, losses, clip fraction, KL divergence
        visualization_data: Episode rewards, policy/value losses, clip fractions, KL divergences
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = PPOResponse(
        ...     success=True,
        ...     metrics={'avg_reward_last_100': 195.5, 'success_rate': 0.85, 'avg_clip_fraction': 0.15},
        ...     visualization_data={...},
        ...     execution_time_ms=85230.5,
        ...     parameters_used={'learning_rate': 0.0003, 'episodes': 500}
        ... )
    """

    success: bool
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class PPOInfoResponse(BaseModel):
    """Response schema for PPO algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the CartPole environment

    Example:
        >>> info = PPOInfoResponse(
        ...     metadata={...},
        ...     environment_info={'name': 'CartPole-v1', 'actions': 2}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
