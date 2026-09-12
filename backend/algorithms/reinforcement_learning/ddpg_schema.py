"""
Pydantic schemas for DDPG API endpoints.

This module defines the request and response models for the DDPG
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class DDPGRequest(BaseModel):
    """Request schema for DDPG training.

    Attributes:
        actor_lr: Actor learning rate (0.00001-0.001)
        critic_lr: Critic learning rate (0.0001-0.01)
        gamma: Discount factor for future rewards (0.9-0.999)
        tau: Soft update coefficient for target networks (0.001-0.01)
        episodes: Number of training episodes (50-500)
        buffer_size: Replay buffer size (10000-1000000)
        batch_size: Training batch size (32-256)
        random_state: Random seed for reproducibility

    Example:
        >>> request = DDPGRequest(
        ...     actor_lr=0.0001,
        ...     critic_lr=0.001,
        ...     gamma=0.99,
        ...     tau=0.005,
        ...     episodes=200,
        ...     buffer_size=100000,
        ...     batch_size=64
        ... )
    """

    actor_lr: float = Field(
        default=0.0001,
        ge=0.00001,
        le=0.001,
        description="Actor learning rate"
    )
    critic_lr: float = Field(
        default=0.001,
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
    tau: float = Field(
        default=0.005,
        ge=0.001,
        le=0.01,
        description="Soft update coefficient for target networks"
    )
    episodes: int = Field(
        default=200,
        ge=50,
        le=500,
        description="Number of training episodes"
    )
    buffer_size: int = Field(
        default=100000,
        ge=10000,
        le=1000000,
        description="Replay buffer size"
    )
    batch_size: int = Field(
        default=64,
        ge=32,
        le=256,
        description="Training batch size"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class DDPGResponse(BaseModel):
    """Response schema for DDPG training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards, improvement, convergence, losses
        visualization_data: Episode rewards, actor/critic losses, Q-values, action distributions
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = DDPGResponse(
        ...     success=True,
        ...     metrics={'avg_reward_last_100': -150.5, 'improvement': 800.2},
        ...     visualization_data={...},
        ...     execution_time_ms=125430.5,
        ...     parameters_used={'actor_lr': 0.0001, 'episodes': 200}
        ... )
    """

    success: bool
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class DDPGInfoResponse(BaseModel):
    """Response schema for DDPG algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the Pendulum environment

    Example:
        >>> info = DDPGInfoResponse(
        ...     metadata={...},
        ...     environment_info={'name': 'Pendulum-v1', 'action_space': 'continuous'}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
