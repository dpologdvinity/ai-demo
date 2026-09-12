"""
Pydantic schemas for Actor-Critic API endpoints.

This module defines the request and response models for the Actor-Critic
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class ActorCriticRequest(BaseModel):
    """Request schema for Actor-Critic training.

    Attributes:
        actor_lr: Actor learning rate (0.0001-0.01)
        critic_lr: Critic learning rate (0.0001-0.01)
        gamma: Discount factor for future rewards (0.9-0.999)
        episodes: Number of training episodes (100-3000)
        hidden_size: Hidden layer size for networks (64-256)
        random_state: Random seed for reproducibility

    Example:
        >>> request = ActorCriticRequest(
        ...     actor_lr=0.001,
        ...     critic_lr=0.005,
        ...     gamma=0.99,
        ...     episodes=1000,
        ...     hidden_size=128
        ... )
    """

    actor_lr: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Actor learning rate"
    )
    critic_lr: float = Field(
        default=0.005,
        ge=0.0001,
        le=0.01,
        description="Critic learning rate"
    )
    gamma: float = Field(
        default=0.99,
        ge=0.9,
        le=0.999,
        description="Discount factor for future rewards"
    )
    episodes: int = Field(
        default=1000,
        ge=100,
        le=3000,
        description="Number of training episodes"
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


class ActorCriticResponse(BaseModel):
    """Response schema for Actor-Critic training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards, success rate, convergence, losses
        visualization_data: Episode rewards, actor/critic losses, value estimates, policy distributions
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = ActorCriticResponse(
        ...     success=True,
        ...     metrics={'avg_reward_last_100': 195.5, 'success_rate': 0.85},
        ...     visualization_data={...},
        ...     execution_time_ms=65230.5,
        ...     parameters_used={'actor_lr': 0.001, 'episodes': 1000}
        ... )
    """

    success: bool
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class ActorCriticInfoResponse(BaseModel):
    """Response schema for Actor-Critic algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the CartPole environment

    Example:
        >>> info = ActorCriticInfoResponse(
        ...     metadata={...},
        ...     environment_info={'name': 'CartPole-v1', 'actions': 2}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
