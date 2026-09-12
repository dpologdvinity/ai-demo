"""
Pydantic schemas for A3C API endpoints.

This module defines the request and response models for the A3C
API endpoints, ensuring type safety and validation.
"""

from typing import Dict, List, Any, Optional
from pydantic import BaseModel, Field


class A3CRequest(BaseModel):
    """Request schema for A3C training.

    Attributes:
        num_workers: Number of parallel workers (2-8)
        actor_lr: Learning rate for actor network (0.0001-0.01)
        critic_lr: Learning rate for critic network (0.0001-0.01)
        gamma: Discount factor for future rewards (0.9-0.999)
        episodes_per_worker: Episodes each worker runs (50-500)
        entropy_coef: Entropy coefficient for exploration (0.0-0.1)
        hidden_size: Hidden layer size (64-256)
        random_state: Random seed for reproducibility

    Example:
        >>> request = A3CRequest(
        ...     num_workers=4,
        ...     actor_lr=0.001,
        ...     critic_lr=0.005,
        ...     gamma=0.99,
        ...     episodes_per_worker=100,
        ...     entropy_coef=0.01
        ... )
    """

    num_workers: int = Field(
        default=4,
        ge=2,
        le=8,
        description="Number of parallel workers"
    )
    actor_lr: float = Field(
        default=0.001,
        ge=0.0001,
        le=0.01,
        description="Learning rate for actor network"
    )
    critic_lr: float = Field(
        default=0.005,
        ge=0.0001,
        le=0.01,
        description="Learning rate for critic network"
    )
    gamma: float = Field(
        default=0.99,
        ge=0.9,
        le=0.999,
        description="Discount factor for future rewards"
    )
    episodes_per_worker: int = Field(
        default=100,
        ge=50,
        le=500,
        description="Episodes each worker runs"
    )
    entropy_coef: float = Field(
        default=0.01,
        ge=0.0,
        le=0.1,
        description="Entropy coefficient for exploration"
    )
    hidden_size: int = Field(
        default=128,
        ge=64,
        le=256,
        description="Hidden layer size for networks"
    )
    random_state: int = Field(
        default=42,
        description="Random seed for reproducibility"
    )


class A3CResponse(BaseModel):
    """Response schema for A3C training results.

    Attributes:
        success: Whether training completed successfully
        metrics: Performance metrics including rewards, advantages, entropy
        visualization_data: Worker rewards, advantages, entropy, worker stats
        execution_time_ms: Training time in milliseconds
        parameters_used: Actual parameters used for training
        error: Error message if training failed

    Example:
        >>> response = A3CResponse(
        ...     success=True,
        ...     metrics={'avg_reward': 195.5, 'success_rate': 0.85},
        ...     visualization_data={...},
        ...     execution_time_ms=45230.5,
        ...     parameters_used={'num_workers': 4, 'episodes_per_worker': 100}
        ... )
    """

    success: bool
    metrics: Dict[str, Any]
    visualization_data: Dict[str, Any] = Field(default_factory=dict)
    execution_time_ms: float
    parameters_used: Dict[str, Any]
    error: Optional[str] = None


class A3CInfoResponse(BaseModel):
    """Response schema for A3C algorithm information.

    Attributes:
        metadata: Complete algorithm metadata
        environment_info: Information about the CartPole environment

    Example:
        >>> info = A3CInfoResponse(
        ...     metadata={...},
        ...     environment_info={'name': 'CartPole-v1', 'actions': 2}
        ... )
    """

    metadata: Dict[str, Any]
    environment_info: Dict[str, Any]
